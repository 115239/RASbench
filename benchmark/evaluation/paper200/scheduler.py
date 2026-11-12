"""Memory-admitted concurrency across all GPUs, without a tasks-per-GPU limit."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from .common import PROJECT, NATIVE, read, write, now
from .queue import connect, report

INITIAL_PEAK_MIB={'groot_agent_on':30000,'groot_agent_off':30000,'pi05_agent_off':24000}
RESERVE_MIB=4096
DISK_FLOOR_BYTES=100*1024**3
INITIAL_DISK_PEAK_BYTES=32*1024**3


def directory_allocated_bytes(path):
    """Private attempt usage, excluding symlink targets and shared model weights."""
    total=0;pending=[Path(path)]
    while pending:
        directory=pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():continue
                        if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):total+=entry.stat(follow_symlinks=False).st_blocks*512
                    except FileNotFoundError:continue
        except FileNotFoundError:continue
    return total


def predicted_disk_peak(profile):
    # Observed successful v2 cache residue is 20.4 GiB, before final artifacts.
    # Keep a 32 GiB floor; larger measured peaks receive 25% plus 4 GiB margin.
    observed=profile.get('observed_disk_peak_bytes',0)
    return max(INITIAL_DISK_PEAK_BYTES,math.ceil(observed*1.25)+4*1024**3)


def disk_admission(free_bytes,new_peak_bytes,running,reserve_bytes=DISK_FLOOR_BYTES):
    remaining=sum(max(0,r.get('predicted_disk_peak_bytes',INITIAL_DISK_PEAK_BYTES)-r.get('current_disk_bytes',0)) for r in running)
    return free_bytes-reserve_bytes-remaining-new_peak_bytes


def measured_envelope(peak,samples,system):
    if samples>=5:return peak+max(1024,math.ceil(peak*.05))
    measured=math.ceil(peak*1.15)+1024
    return max(16000,measured) if samples>=2 else max(INITIAL_PEAK_MIB[system],measured)


def profile_key(system,row,networks=None,purpose='main'):
    if system.startswith('pi05'):tag='pi05'
    else:
        if networks is None:
            networks=[]
            if any(s.startswith('NAV_') for s in row['stages']):networks.append('unified')
            if any(not s.startswith('NAV_') for s in row['stages']):networks.append('grasp')
        tag='+'.join(sorted(networks))
    return ':'.join([system,row['scene'],row['task_type'],tag])


def predicted_peak(peaks,system,row,purpose='main'):
    def envelope(record):
        # Old profile files stored only an already conservative prediction.
        # Retain that prediction until an actual peak observation is available.
        if 'observed_peak_mib' not in record:return record['predicted_peak_mib']
        return measured_envelope(record['observed_peak_mib'],record['samples'],system)
    key=profile_key(system,row,purpose=purpose)
    exact=peaks.get(key,{})
    if exact.get('samples',0)>=2:return envelope(exact)
    # Earlier complete pilots loaded the same heads or a superset of them.
    # Their measured envelope can seed a split profile, never an assisted run
    # that introduces an additional navigation network.
    legacy=peaks.get(':'.join([system,row['scene'],row['task_type']]),{})
    if purpose not in ('assisted','assisted_pilot') and legacy.get('samples',0)>=2:
        return max(envelope(legacy),math.ceil(exact.get('observed_peak_mib',0)*1.15)+1024)
    return exact.get('predicted_peak_mib',INITIAL_PEAK_MIB[system])


class ExistingWorker:
    def __init__(self,pid):
        self.pid=pid
        self.identity=self._identity()
    def _identity(self):
        try:
            raw=Path('/proc/%d/stat'%self.pid).read_text()
            return raw[raw.rfind(')')+2:].split()[19]
        except (FileNotFoundError,ProcessLookupError):return None
    def poll(self):
        return None if self.identity is not None and self._identity()==self.identity else 0


def admission(free_mib,new_peak_mib,running,reserve_mib=RESERVE_MIB):
    """Do not spend VRAM reserved for another task's still-unreached peak."""
    headroom=sum(max(0,r['predicted_peak_mib']-r['current_mib']) for r in running)
    return free_mib-headroom-reserve_mib-new_peak_mib


def devices():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],
                          check=True,stdout=subprocess.PIPE,encoding='utf-8')
    return [dict(index=int(i),uuid=u,total_mib=int(t),used_mib=int(m))
            for i,u,t,m in (line.strip().split(', ') for line in result.stdout.splitlines())]


def process_memory():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
                          check=True,stdout=subprocess.PIPE,encoding='utf-8')
    rows=[]
    for line in result.stdout.splitlines():
        try:
            pid,uuid,memory=line.split(', ');rows.append((int(pid),uuid,int(memory)))
        except ValueError:continue
    return rows


def descendants(pid):
    result={pid};parents={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            raw=(p/'stat').read_text();ppid=int(raw[raw.rfind(')')+2:].split()[1]);parents[int(p.name)]=ppid
        except (FileNotFoundError,ProcessLookupError,ValueError,PermissionError):continue
    changed=True
    while changed:
        children={p for p,parent in parents.items() if parent in result};changed=not children<=result;result|=children
    return result


def select_gpu(gpus,active,peak):
    candidates=[]
    for gpu in gpus:
        on_gpu=[r for r in active if r['gpu']==gpu['index']]
        available=admission(gpu['total_mib']-gpu['used_mib'],peak,on_gpu)
        if available>=0:candidates.append((available,gpu['index']))
    return max(candidates)[1] if candidates else None


def run(root,purpose='main'):
    root=Path(root).resolve();lock=(root/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if purpose in ('main','assisted') and not read(root/'reports/MAIN_RELEASE_GATE.json').get('passed'):
            raise RuntimeError('main release gate not passed')
        if purpose in ('main','assisted') and (root/'manifests/source_drain.json').exists():
            from .resume_repaired import verify_prepared
            verify_prepared(root)
    except BaseException:
        lock.close()
        raise
    active=[];peaks=read(root/'reports/resource_profiles.json') if (root/'reports/resource_profiles.json').exists() else {}
    resident_enabled=(root/'manifests/resident_services.json').exists()
    controller=None
    if (root/'manifests/throughput_policy.json').exists():
        from .throughput_control import Controller
        policy=read(root/'manifests/throughput_policy.json')
        controller=Controller(root,purpose,window_s=policy.get('trial_window_s',120),initial=policy.get('initial_concurrency',8))
    previous=read(root/'reports/scheduler.json') if (root/'reports/scheduler.json').exists() else {}
    previous_active={r['run_id']:r for r in previous.get('active',[])}
    rows={q['question']:q for q in read(root/'manifests/questions.json')}
    with connect(root/'queue.sqlite3') as db:
        existing=[dict(r) for r in db.execute("SELECT * FROM runs WHERE status='running'")]
    for job in existing:
        info=read(Path(job['output'])/'processes.json')
        process=ExistingWorker(info['launcher'])
        if process.poll() is not None:raise RuntimeError('orphan running job needs recovery: '+job['run_id'])
        prior=previous_active.get(job['run_id'],{})
        active.append({'process':process,'log':None,'run_id':job['run_id'],'system':job['system'],
            'scene':rows[job['question']]['scene'],'task_type':rows[job['question']]['task_type'],'gpu':info['gpu'],
            'predicted_peak_mib':max(predicted_peak(peaks,job['system'],rows[job['question']],job['purpose']),
                math.ceil(prior.get('observed_peak_mib',0)*1.05)+1024),'current_mib':0,
            'observed_peak_mib':prior.get('observed_peak_mib',0),'adopted':True})
    stopping=False;last_summary=0.
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            with connect(root/'queue.sqlite3') as db:
                disk_rows={j['run_id']:dict(j) for j in db.execute("SELECT * FROM runs WHERE status='running'")}
            memories=process_memory();gpus=devices();uuids={g['index']:g['uuid'] for g in gpus}
            from .vision_pool import reservations
            service_reservations=reservations(root,memories)
            resident_reservations=[]
            if resident_enabled:
                from .resident_admission import reservations as resident_memory
                resident_reservations=resident_memory(root,memories,gpus)
                service_reservations+=resident_reservations
            for item in list(active):
                pids=descendants(item['process'].pid)
                used=sum(m for pid,u,m in memories if pid in pids and u==uuids[item['gpu']])
                owned_kit=next((s for s in resident_reservations if s.get('active_output') and s.get('active_output')==(disk_rows.get(item['run_id']) or {}).get('output') and s.get('role')=='resident_kit'),None)
                if resident_enabled and owned_kit is not None:
                    item['predicted_peak_mib']=1024 # Shared Kit reservation is charged once above.
                item['current_mib']=used;item['observed_peak_mib']=max(item['observed_peak_mib'],used+(owned_kit['current_mib'] if owned_kit else 0))
                disk_job=disk_rows.get(item['run_id'])
                current_disk=directory_allocated_bytes(disk_job['output']) if disk_job and disk_job['output'] else 0
                item['current_disk_bytes']=current_disk
                item['observed_disk_peak_bytes']=max(item.get('observed_disk_peak_bytes',0),current_disk)
                item['predicted_disk_peak_bytes']=max(item.get('predicted_disk_peak_bytes',INITIAL_DISK_PEAK_BYTES),predicted_disk_peak(item))
                if item['process'].poll() is not None:
                    if item['log'] is not None:item['log'].close()
                    active.remove(item)
                    with connect(root/'queue.sqlite3') as db:
                        job=dict(db.execute('SELECT * FROM runs WHERE run_id=?',(item['run_id'],)).fetchone())
                    metadata_path=Path(job['output'])/'model/READY.json'
                    metadata=read(metadata_path) if metadata_path.exists() else {}
                    write(Path(job['output'])/'RESOURCE_USAGE.json',{
                        'observed_peak_mib':item['observed_peak_mib'],'gpu':item['gpu'],
                        'complete_scheduler_observation':not item.get('adopted',False),
                        'sampling_interval_s':5,'observed_disk_peak_bytes':item.get('observed_disk_peak_bytes',0),'time':now()})
                    key=profile_key(item['system'],rows[job['question']],metadata.get('loaded_networks',['grasp','unified']))
                    old=peaks.get(key,{});peak=max(old.get('observed_peak_mib',0),item['observed_peak_mib'])
                    with connect(root/'queue.sqlite3') as db:
                        completed=db.execute('SELECT status FROM runs WHERE run_id=?',(item['run_id'],)).fetchone()[0]=='collected'
                    samples=old.get('samples',0)+int(completed and not item.get('adopted'))
                    peaks[key]={'observed_peak_mib':peak,'predicted_peak_mib':measured_envelope(peak,samples,item['system']),
                                'samples':samples,'observed_disk_peak_bytes':max(old.get('observed_disk_peak_bytes',0),item.get('observed_disk_peak_bytes',0))}
                    write(root/'reports/resource_profiles.json',peaks)
            if controller is not None:
                with connect(root/'queue.sqlite3') as db:controller.observe(db,active)
            status=report(root)
            if purpose in ('main','assisted') and time.monotonic()-last_summary>=60.:
                from .collection_report import build
                build(root)
                last_summary=time.monotonic()
            with connect(root/'queue.sqlite3') as db:
                purposes=('pilot','probe','assisted_pilot') if purpose=='pilot' else (purpose,)
                pending=[dict(r) for r in db.execute("SELECT * FROM runs WHERE status='pending' AND purpose IN ("+','.join('?' for _ in purposes)+") ORDER BY attempts,CAST(substr(question,2) AS INTEGER),system,variant",purposes)]
            pending=[r for r in pending if r['run_id'] not in {a['run_id'] for a in active}]
            if resident_enabled:
                pending.sort(key=lambda j:(rows[j['question']]['scene'],int(j['question'][1:]),j['variant'],j['system']))
            if purpose=='pilot':
                repaired={read(p)['question'] for p in (root/'manifests/revisions').glob('spawn_*/revision.json')}
                pending.sort(key=lambda j:0 if j['variant']=='clean' and j['question'] in repaired else 1)
            rows={q['question']:q for q in read(root/'manifests/questions.json')}
            free_disk=shutil.disk_usage(root).free
            disk_can_admit=disk_admission(free_disk,INITIAL_DISK_PEAK_BYTES,active)>=0
            admission_lock=(root/'gpu_admission.lock').open('a')
            fcntl.flock(admission_lock,fcntl.LOCK_EX)
            # Provisioning a vision replica uses the same admission lock.
            gpus=devices();admission_memories=process_memory();service_reservations=reservations(root,admission_memories)
            if resident_enabled:
                service_reservations+=resident_memory(root,admission_memories,gpus)
            if not stopping and not (root/'STOP').exists() and disk_can_admit:
                for job in pending:
                    if controller is not None and len(active)>=controller.target:break
                    scene=rows[job['question']]['scene'];kind=rows[job['question']]['task_type'];key=profile_key(job['system'],rows[job['question']],purpose=job['purpose'])
                    peak=predicted_peak(peaks,job['system'],rows[job['question']],job['purpose'])
                    disk_peak=predicted_disk_peak(peaks.get(key,{}))
                    if disk_admission(free_disk,disk_peak,active)<0:continue
                    reuse_worker=None
                    if resident_enabled:
                        from .resident_admission import choose as choose_resident_gpu
                        measured=peaks.get(key,{})
                        private_peak=max(12000,int(measured.get('observed_peak_mib',0)*1.15)+1024) if measured.get('samples',0)>=2 else 18000
                        from .resident_pool import compatible_idle_workers
                        compatible={w['worker_id'] for w in compatible_idle_workers(root,job['question'])}
                        gpu,peak,reuse_worker=choose_resident_gpu(gpus,active,service_reservations,scene,private_peak,compatible_worker_ids=compatible)
                    else:gpu=select_gpu(gpus,active+service_reservations,peak)
                    if gpu is None:continue
                    command=[str(NATIVE/'python.sh'),'-m','paper200.cli','worker','--root',str(root),
                             '--gpu',str(gpu),'--run-id',job['run_id'],'--purpose',job['purpose']]
                    log_path=root/'reports/scheduler_logs'/(job['run_id']+'_%d.log'%time.time_ns())
                    log_path.parent.mkdir(exist_ok=True);log=log_path.open('w')
                    entry=root/'runtime/collection_code' if (root/'manifests/FROZEN.json').exists() else PROJECT/'experiment'
                    env=dict(os.environ,PYTHONPATH=str(entry),PAPER200_PROJECT_ROOT=str(PROJECT),PYTHONUTF8='1')
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append({'process':process,'log':log,'run_id':job['run_id'],'system':job['system'],
                                   'scene':scene,'task_type':kind,'gpu':gpu,'predicted_peak_mib':peak,'current_mib':0,'observed_peak_mib':0,'predicted_disk_peak_bytes':disk_peak,'current_disk_bytes':0,'observed_disk_peak_bytes':0,'resident_reuse_worker_id':reuse_worker})
            write(root/'reports/scheduler.json',{'updated':now(),'pid':os.getpid(),'purpose':purpose,
                   'policy':'all GPUs; no per-GPU task cap; reserve predicted unobserved peaks + 4096 MiB',
                   'gpus':gpus,'shared_service_reservations':service_reservations,'active':[{k:v for k,v in a.items() if k not in ('process','log')}|{'pid':a['process'].pid} for a in active],
                   'pending':len(pending),'main_collected':status['main_collected'],'stopping':stopping,
                   'throughput_admission_target':controller.target if controller else None})
            admission_lock.close()
            write(root/'reports/storage_admission.json',{'free_bytes':free_disk,'admit_new_jobs':disk_admission(free_disk,INITIAL_DISK_PEAK_BYTES,active)>=0,
                'floor_bytes':DISK_FLOOR_BYTES,'incoming_default_peak_bytes':INITIAL_DISK_PEAK_BYTES,
                'remaining_active_peak_bytes':sum(max(0,a.get('predicted_disk_peak_bytes',INITIAL_DISK_PEAK_BYTES)-a.get('current_disk_bytes',0)) for a in active),
                'scope':'continuous admission guard; not a certificate that the entire batch fits','updated':now()})
            if not active and (not pending or stopping or (root/'STOP').exists()):break
            time.sleep(5)
    finally:
        # Graceful stop drains active jobs. Unexpected scheduler errors leave
        # separately registered workers queryable; never blindly duplicate them.
        lock.close()
    return report(root)
