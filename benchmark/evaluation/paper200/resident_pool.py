"""Private outer Runner processes submit to a separately owned serial Kit worker."""
import fcntl,os,socket,subprocess,time,contextlib
from pathlib import Path
from .common import read,write,now,NATIVE,digest


def process_token(pid):
    try:
        text=Path('/proc/%d/stat'%pid).read_text();fields=text[text.rfind(')')+2:].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None


def list_workers(root):
    records=[]
    for path in sorted((Path(root)/'.resident').glob('*/WORKER.json')):
        record=read(path);record['alive']=process_token(record['pid'])==record['process_start_token']
        records.append(record)
    return records


def compatible_idle_workers(root,question):
    from .resident_scene import physical_group
    group=physical_group(root,question);records=[]
    for worker in list_workers(root):
        key=digest({'scene':group['identity']['scene'],'gpu':worker['gpu'],'runtime':group['identity']['runtime_implementation']})
        if worker['alive'] and worker['status']=='idle' and worker['pool_identity']==key:records.append(worker)
    return sorted(records,key=lambda w:w['last_used_unix_s'],reverse=True)


def ensure_worker(root,group,gpu,environment,slot):
    root=Path(root).resolve();identity={'scene':group['identity']['scene'],'gpu':int(gpu),
        'runtime':group['identity']['runtime_implementation']}
    worker_id='gpu%d-%s-slot%d'%(gpu,digest(identity)[:16],slot);directory=root/'.resident'/worker_id
    directory.mkdir(parents=True,exist_ok=True)
    # AF_UNIX pathname max108: keep the long dataset path out of the socket.
    socket_path=Path('/tmp')/('paper200-%s-%s.sock'%(os.getuid(),digest(str(directory))[:24]))
    with (directory/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=directory/'WORKER.json'
        if state.exists():
            prior=read(state)
            if process_token(prior['pid'])==prior['process_start_token'] and socket_path.exists():return prior
            if process_token(prior['pid'])==prior['process_start_token']:
                raise RuntimeError('live resident worker has no ready socket; inspect '+str(directory))
            state.rename(directory/('WORKER.ended.%d.json'%time.time_ns()))
            if socket_path.exists():socket_path.unlink()
        from .runtime_cache import prepare_worker_cache
        cache=prepare_worker_cache(root,worker_id,identity,template_scene=group['identity']['scene'])
        env=dict(environment);env.update(PAPER200_RESIDENT_WORKER_ID=worker_id,
            PAPER200_RESIDENT_CACHE=str(cache),BASELINE_PYTHON_KIND='isaac')
        log=(directory/'worker.log').open('ab')
        process=subprocess.Popen([str(NATIVE/'isaac_python.sh'),'-m','paper200.resident_scene','serve','--socket',str(socket_path)],
            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        record={'worker_id':worker_id,'pid':process.pid,'process_start_token':process_token(process.pid),
            'socket':str(socket_path),'gpu':int(gpu),'physical_group_key':group['key'],
            'scene':identity['scene'],'pool_identity':digest(identity),'slot':slot,
            'status':'idle','reserved_peak_mib':int(environment.get('PAPER200_RESIDENT_PEAK_MIB','18000')),
            'directory':str(directory),'cache':str(cache),'created':now(),'last_used_unix_s':time.time(),
            'ownership':'resident service; never kill via per-episode outer process-group cleanup'}
        write(state,record)
        deadline=time.monotonic()+60
        while not socket_path.exists():
            if process.poll() is not None:raise RuntimeError('resident daemon startup failed; see '+str(directory/'worker.log'))
            if time.monotonic()>deadline:raise TimeoutError('resident daemon socket readiness')
            time.sleep(.1)
        return record


@contextlib.contextmanager
def acquire_worker(root,group,gpu,environment,out):
    """Choose a free compatible slot; never queue behind a busy worker."""
    root=Path(root).resolve();pool=root/'.resident';pool.mkdir(parents=True,exist_ok=True)
    identity={'scene':group['identity']['scene'],'gpu':int(gpu),'runtime':group['identity']['runtime_implementation']}
    key=digest(identity);lease=None;record=None
    with (pool/('gpu%d.allocate.lock'%gpu)).open('a') as allocation:
        fcntl.flock(allocation,fcntl.LOCK_EX)
        records=[read(p) for p in pool.glob('gpu%d-*/WORKER.json'%gpu)]
        live=[r for r in records if process_token(r['pid'])==r['process_start_token']]
        for candidate in sorted((r for r in live if r.get('pool_identity')==key),key=lambda r:r['last_used_unix_s'],reverse=True):
            candidate_lease=(Path(candidate['directory'])/'lease.lock').open('a')
            try:fcntl.flock(candidate_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate_lease.close();continue
            if candidate.get('status')=='busy':
                candidate_lease.close();continue
            lease=candidate_lease;record=candidate;break
        if record is None:
            # Retire idle incompatible scene workers before allocating another
            # Kit process. Caches stay on disk; active episodes are untouched.
            for candidate in sorted((r for r in live if r.get('pool_identity')!=key),key=lambda r:r['last_used_unix_s']):
                candidate_lease=(Path(candidate['directory'])/'lease.lock').open('a')
                try:fcntl.flock(candidate_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate_lease.close();continue
                try:
                    if candidate.get('status')=='busy':continue
                    from .resident_scene import submit
                    submit(candidate['socket'],{'operation':'shutdown'})
                    deadline=time.monotonic()+20
                    while process_token(candidate['pid'])==candidate['process_start_token'] and time.monotonic()<deadline:time.sleep(.1)
                    if process_token(candidate['pid'])==candidate['process_start_token']:raise RuntimeError('idle resident did not stop')
                    candidate['status']='retired';write(Path(candidate['directory'])/'WORKER.json',candidate)
                finally:candidate_lease.close()
            live=[r for r in live if process_token(r['pid'])==r['process_start_token']]
            limit=int(environment.get('PAPER200_RESIDENT_MAX_SLOTS_PER_GPU','8'))
            if len(live)>=limit:raise RuntimeError('RESIDENT_CAPACITY_BUSY; no compatible idle slot, retry after scheduler admission')
            occupied={r['slot'] for r in live if r.get('pool_identity')==key}
            slot=next(n for n in range(limit) if n not in occupied)
            record=ensure_worker(root,group,gpu,environment,slot)
            lease=(Path(record['directory'])/'lease.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record.update(status='busy',active_output=str(Path(out).resolve()),
            episode_launcher_pid=os.getpid(),episode_launcher_token=process_token(os.getpid()),last_used_unix_s=time.time())
        write(Path(record['directory'])/'WORKER.json',record)
    try:yield record
    finally:
        record.update(status='idle' if process_token(record['pid'])==record['process_start_token'] else 'exited',
            active_output=None,last_used_unix_s=time.time())
        write(Path(record['directory'])/'WORKER.json',record);lease.close()


def shim(runtime_argv):
    """The question wrapper supplies its ordinary runtime script + arguments."""
    from .resident_scene import physical_group,submit
    config=Path(os.environ['THREE_SCENES_RUN_CONFIG']).resolve();cfg=read(config)
    root=Path(cfg['collection_root']);out=Path(cfg['output'])
    if os.environ.get('PAPER200_FRESH_SCENE_COMPARISON')=='1':
        if not cfg.get('purpose','').startswith('control') or not (root/'STOP').exists():
            raise ValueError('fresh-vs-resident comparison is diagnostic only')
        write(out/'FRESH_SCENE_COMPARISON.json',{'mode':'fresh_kit_fresh_stage','time':now(),
            'scope':'same diagnostic runtime/model/config; independently started Kit comparison'})
        native=os.environ.get('PAPER200_NATIVE_ISAAC_PYTHON',str(NATIVE/'isaac_python.sh'))
        return subprocess.call([native,*runtime_argv],env=dict(os.environ))
    question='Q%d'%int(cfg['task']['question'][1:]);group=physical_group(root,question)
    args=list(runtime_argv);gpu=int(args[args.index('--gpu')+1]) if '--gpu' in args else 0
    with acquire_worker(root,group,gpu,dict(os.environ),out) as worker:
        env=dict(os.environ);cache=Path(worker['cache'])
        env.update(XDG_CACHE_HOME=str(cache/'cache'),XDG_CONFIG_HOME=str(cache/'config'),
            XDG_RUNTIME_DIR=str(cache/'tmp'),TMPDIR=str(cache/'tmp'),CUDA_CACHE_PATH=str(cache/'cache/cuda'),
            PAPER200_RESIDENT_WORKER_ID=worker['worker_id'])
        if '--portable-root' in args:args[args.index('--portable-root')+1]=str(cache/'kit')
        else:args+=['--portable-root',str(cache/'kit')]
        write(out/'RESIDENT_WORKER.json',{**worker,'episode_launcher_pid':os.getpid(),
            'lease_acquired':now(),'release_mode':'close socket after completed result; worker persists'})
        result=submit(worker['socket'],{'config_path':str(config),'question':question,
            'physical_group_key':group['key'],'runtime_script':args[0],'argv':args[1:],
            'environment':env,'cwd':str(Path.cwd()),'client_pid':os.getpid(),'client_token':process_token(os.getpid())})
        return int(result['exit_code'])


def abort(out,reason='episode deadline or caller canceled'):
    """Only the executing request's watchdog may kill the resident worker.

    A per-output cancellation token cannot terminate a subsequent unrelated
    episode even if the caller discovers the failure after the lease changed.
    """
    out=Path(out)
    if not (out/'RESIDENT_WORKER.json').exists() or (out/'RESIDENT_EPISODE_RESULT.json').exists():return False
    write(out/'RESIDENT_CANCEL.json',{'created':now(),'reason':reason,'output':str(out.resolve())})
    return True


if __name__=='__main__':
    import sys
    raise SystemExit(shim(sys.argv[1:]))
