"""Isolated local vision replicas and episode-sticky admission."""
import fcntl,json,os,subprocess,time,urllib.request,socket,re,uuid,shutil
from pathlib import Path
from .common import read,write,now,PROJECT,sha
from .queue import connect


def services(root):
    config=read(root/'manifests/vision_service.json')
    return config.get('services',[config])


def reservations(root,memories):
    from .scheduler import descendants
    result=[]
    for s in services(root):
        pid=s.get('host_pid')
        if pid is None:continue
        # Container init can be a shell; CUDA belongs to its Python child.
        # Count the full owned process tree to avoid reserving already-used
        # service memory a second time.
        pids=descendants(pid)
        used=sum(m for p,u,m in memories if p in pids)
        result.append({'gpu':s['gpu'],'predicted_peak_mib':s.get('reserved_peak_mib',14000),'current_mib':used})
    return result


def choose(root,out):
    with (root/'vision_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        endpoints=[s for s in services(root) if s.get('status','READY')=='READY']
        if not endpoints:raise RuntimeError('no ready vision replica')
        counts={s['endpoint']:0 for s in endpoints}
        with connect(root/'queue.sqlite3') as db:
            paths=[Path(j['output'])/'VISION_BINDING.json' for j in db.execute("SELECT output FROM runs WHERE status='running' AND system='groot_agent_on'")]
        for path in paths:
            if path.exists():
                endpoint=read(path)['endpoint']
                if endpoint in counts:counts[endpoint]+=1
        selected=min(endpoints,key=lambda s:(counts[s['endpoint']],s['endpoint']))
        with urllib.request.urlopen(selected['endpoint']+'/health',timeout=15) as response:health=json.load(response)
        if health.get('status')!='READY':raise RuntimeError('selected vision service is not ready')
        write(out/'VISION_BINDING.json',selected|{'selected':now(),'episode_sticky':True})
        return selected['endpoint']


def _health(service):
    with urllib.request.urlopen(service['endpoint']+'/health',timeout=15) as response:
        health=json.load(response)
    if health.get('status')!='READY':raise RuntimeError('vision service is not ready: '+service['endpoint'])
    return health


def _isolated_identity(root,index):
    # Unique names and OS-selected free ports keep versions and failed attempts
    # independent. Docker failure is retained; nothing deletes old containers.
    token=uuid.uuid4().hex[:12]
    prefix=re.sub('[^a-zA-Z0-9_.-]', '-', root.name)
    name=prefix+'-vision-%d-'%index+token
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    return name,port,root/'pilot'/('vision_replica_'+token)


def _frozen_service_source(root,config):
    source=Path(config.get('frozen_source',str(root/'runtime/agent'))).resolve()
    # A replica may only use this collection's immutable source snapshot.
    if root.resolve() not in source.parents:raise ValueError('vision source belongs to another collection')
    expected=config.get('source_sha256')
    actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    if expected is not None and actual!=expected:raise ValueError('vision service source changed')
    if expected is None:
        target=root/'runtime'/('vision_source_'+uuid.uuid4().hex[:12])
        if (root/'manifests/FROZEN.json').exists():
            raise ValueError('frozen collection requires an existing source-attested vision service')
        shutil.copytree(source,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source=target;actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    return source,actual


def provision(root,replicas=3):
    from .scheduler import devices,process_memory,select_gpu
    root=Path(root).resolve()
    if replicas<1:raise ValueError('at least one vision replica is required')
    # Serialize whole pool provisioning so concurrent scheduler starts cannot
    # race manifest updates or accidentally provision the same replica twice.
    with (root/'vision_provision.lock').open('a') as provision_lock:
        fcntl.flock(provision_lock,fcntl.LOCK_EX)
        existing=services(root)
        if not existing:raise RuntimeError('a source-attested vision service is required')
        for s in existing:
            state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',s['container']],encoding='utf-8'))
            if not state['Running']:raise RuntimeError('existing vision service is stopped: '+s['container'])
            _health(s)
            s.update(host_pid=state['Pid'],reserved_peak_mib=s.get('reserved_peak_mib',14000),status='READY')
        config=existing[0]
        source,source_hashes=_frozen_service_source(root,config)
        config.update(frozen_source=str(source),source_sha256=source_hashes)
        def save():
            # Retain diagnostic attempt/provenance fields from the initial service.
            manifest=read(root/'manifests/vision_service.json')
            manifest.update(services=existing,endpoint=existing[0]['endpoint'])
            write(root/'manifests/vision_service.json',manifest)
        save()
        while len(existing)<replicas:
            with (root/'gpu_admission.lock').open('a') as admission_lock:
                fcntl.flock(admission_lock,fcntl.LOCK_EX)
                snapshot_path=root/'reports/scheduler.json'
                active=read(snapshot_path).get('active',[]) if snapshot_path.exists() else []
                peak=config.get('reserved_peak_mib',14000)
                gpu=select_gpu(devices(),active+reservations(root,process_memory()),peak)
                if gpu is None:raise RuntimeError('not enough GPU peak headroom for another vision replica; retry after active jobs drain')
                name,port,destination=_isolated_identity(root,len(existing)+1)
                destination.mkdir(parents=True)
                cpus=sorted(os.sched_getaffinity(0));assigned=cpus[gpu*32:(gpu+1)*32] or cpus[:32]
                command=['docker','run','-d','--name',name,'--gpus','device=%d'%gpu,'--network','host',
                    '--user','%d:%d'%(os.getuid(),os.getgid()),'--cpuset-cpus',','.join(map(str,assigned)),
                    '-v',str(source)+':/agent:ro','-v',str(destination)+':/output:rw',
                    '-v',str(PROJECT/'mc/safe_agent_vision_models')+':/vision-models:ro',
                    '-e','PYTHONPATH=/agent','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
                    '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4',
                    '--entrypoint','/opt/groot-env/bin/python',config['image'],'-m','safety_agent.vision_service',
                    '--qwen','/vision-models/qwen','--sam','/vision-models/sam','--port',str(port),'--device','cuda:0',
                    '--output','/output/service']
                identifier=subprocess.check_output(command,encoding='utf-8').strip()
                pid=int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',identifier],encoding='utf-8'))
                item={'endpoint':'http://127.0.0.1:%d'%port,'container':name,'gpu':gpu,'image':config['image'],
                    'host_pid':pid,'reserved_peak_mib':peak,'created':now(),'status':'STARTING',
                    'frozen_source':str(source),'source_sha256':source_hashes,
                    'model_source':config.get('model_source','shared verified vision weights read-only')}
                existing.append(item);save()
            deadline=time.monotonic()+600
            while True:
                try:
                    _health(item);break
                except (OSError,RuntimeError):pass
                if time.monotonic()>deadline:
                    item['status']='READINESS_TIMEOUT';save()
                    raise RuntimeError('vision replica readiness timeout: '+name)
                time.sleep(2)
            item['status']='READY';save()
        return {'services':existing}
