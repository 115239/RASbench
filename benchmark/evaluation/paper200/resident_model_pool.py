"""Cross-worker lifecycle for persistent GPU daemons and disposable proxies."""
import fcntl,hashlib,json,os,signal,socket,subprocess,time
from pathlib import Path
from .common import NATIVE,write,read
from .resident_model_service import source_manifest


def available_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def process_alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False


def implementation_identity(code_root=None):
    package=Path(code_root)/'paper200' if code_root else Path(__file__).parent
    paths=[package/'resident_model_service.py',package/'model_service.py',package.parent/'live_models.py']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class SessionHandle:
    def __init__(self,proxy,port,service):self.proxy=proxy;self.port=port;self.service=service;self.service_pid=service['pid']
    def close(self):
        if self.proxy.poll() is None:
            os.killpg(self.proxy.pid,signal.SIGTERM)
            try:self.proxy.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(self.proxy.pid,signal.SIGKILL);self.proxy.wait()


def acquire(root,cfg,out,gpu,*,model_python,model_env,cpu_prefix=(),timeout_s=900,code_root=None,service_gpu=None):
    """Return handle.port, handle.proxy/Popen and handle.service; close never kills bank.

    out is the episode attempt directory (model evidence goes in out/model).
    Caller excludes service_pid from per-episode cleanup and charges its GPU
    memory once. An exclusive lock prevents simultaneous duplicate bank loads.
    """
    root=Path(root).resolve();out=Path(out).resolve();simulator_gpu=int(gpu)
    plan=root/'manifests/resident_services.json'
    if service_gpu is None and plan.exists():service_gpu=read(plan)['backends'][cfg['backend']]['gpu']
    gpu=int(service_gpu) if service_gpu is not None else simulator_gpu
    model_env=dict(model_env,CUDA_VISIBLE_DEVICES=str(gpu))
    runtime=Path(cfg.get('runtime_snapshot',str(root/'runtime')))
    code_root=Path(code_root) if code_root else (out/'code' if (out/'code/paper200/resident_model_service.py').exists() else Path(__file__).parent.parent)
    code_identity=implementation_identity(code_root);session_cfg=dict(cfg,resident_code_identity=code_identity)
    signature={'backend':cfg['backend'],'gpu':int(gpu),'models':cfg['model_bindings'],'runtime_sources':source_manifest(runtime,cfg['backend']),'implementation':code_identity}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    service_root=root/'services/models'/('%s_gpu%d_%s'%(cfg['backend'],gpu,digest[:16]));service_root.mkdir(parents=True,exist_ok=True)
    with (service_root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state_path=service_root/'process.json';service=read(state_path) if state_path.exists() else None
        ready=service_root/'READY.json'
        healthy=False
        if service and process_alive(service['pid']) and ready.exists():
            try:
                with socket.create_connection(('127.0.0.1',service['port']),timeout=1):healthy=True
            except OSError:pass
        if not healthy:
            if service and process_alive(service['pid']):raise RuntimeError('RESIDENT_EXISTING_PROCESS_UNHEALTHY; preserve for diagnosis')
            if ready.exists():ready.rename(service_root/('READY.previous.%d.json'%time.time_ns()))
            port=available_port();write(service_root/'config.json',session_cfg);write(service_root/'signature.json',signature)
            env=dict(model_env);env['PYTHONPATH']=str(code_root)+':'+env.get('PYTHONPATH','')
            # Persistent caches belong to the service, never a released episode.
            for key,name in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax')]:
                path=service_root/name;path.mkdir(exist_ok=True);env[key]=str(path)
            started=time.monotonic()
            command=list(cpu_prefix)+[str(model_python),'-m','paper200.resident_model_service','--config',str(service_root/'config.json'),'--out',str(service_root),'--port',str(port)]
            with (service_root/'service.log').open('a') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            service={'pid':process.pid,'port':port,'signature_sha256':digest,'root':str(service_root),'gpu':int(gpu),'backend':cfg['backend'],'command':command,'persistent':True}
            write(state_path,service)
            while not ready.exists():
                if process.poll() is not None:raise RuntimeError('RESIDENT_MODEL_START_FAILED: '+str(service_root/'service.log'))
                if time.monotonic()-started>timeout_s:raise RuntimeError('RESIDENT_MODEL_START_TIMEOUT; process retained for diagnosis')
                time.sleep(.25)
            write(service_root/'LOAD_TIMING.json',{'wall_seconds':time.monotonic()-started,'outside_episode_simulation':True,'not_inference_or_task_success':True})
        service['ready']=read(ready)
    port=available_port();proxy_cfg=out/'resident_session_config.json';write(proxy_cfg,session_cfg)
    proxy_env=dict(model_env,BASELINE_PYTHON_KIND='model');proxy_env['PYTHONPATH']=str(code_root)+':'+proxy_env.get('PYTHONPATH','')
    # Native model interpreter is used only for small msgpack/websocket codec;
    # proxy imports no Torch/JAX/model service modules.
    command=list(cpu_prefix)+[str(NATIVE/'python.sh'),'-m','paper200.resident_model_proxy','--config',str(proxy_cfg),'--out',str(out/'model'),'--port',str(port),'--resident-uri','ws://127.0.0.1:%d'%service['port']]
    started=time.monotonic()
    with (out/'model.log').open('a') as log:
        proxy=subprocess.Popen(command,env=proxy_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    handle=SessionHandle(proxy,port,service)
    try:
        while not (out/'model/READY.json').exists():
            if proxy.poll() is not None:raise RuntimeError('RESIDENT_PROXY_START_FAILED; see model.log and service.log')
            if time.monotonic()-started>timeout_s:raise RuntimeError('RESIDENT_PROXY_START_TIMEOUT')
            time.sleep(.1)
        write(out/'RESIDENT_MODEL_BINDING.json',{'service':service,'proxy_pid':proxy.pid,'local_port':port,'proxy_command':command,'simulator_gpu':simulator_gpu,'model_service_gpu':gpu,'registration_wall_seconds':time.monotonic()-started,'cleanup_scope':'proxy/session only; shared weight daemon persists'})
        return handle
    except BaseException:handle.close();raise
