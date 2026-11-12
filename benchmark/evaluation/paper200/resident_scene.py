"""Serial Kit workers with fresh stage/PhysX/Collector per episode.

This intentionally does not claim in-place scene reset. Reusing mutable scenes
is forbidden until a separate reset-parity acceptance has proved that path.
"""
import argparse,contextlib,gc,json,os,runpy,socket,sys,time,traceback,threading
from pathlib import Path
from .common import read,write,sha,digest,now

# ``python -m`` otherwise creates a second copy on runtime_entry's canonical
# import, losing the active session and accidentally closing the real Kit.
if __name__=='__main__':sys.modules['paper200.resident_scene']=sys.modules[__name__]
_SESSION=None


def physical_group(root,question):
    """Conservative physical configuration, never merely a scene label."""
    root=Path(root).resolve();package=root/'benchmark/task'/question
    task=read(package/'data/task/task.json')
    # Keep unknown/new keys conservatively. Only text, identifiers and pure
    # stopping/scoring definitions are excluded from physical construction.
    excluded={'instruction','policy_instruction','manipulation_prompt','task_id','question_id',
        'scenario_id','rubric_source','termination','maximum_vla_actions','seed'}
    physical={k:v for k,v in task.items() if k not in excluded}
    assets={str(p.relative_to(package)):sha(p) for directory in ('data/scene','data/route','data/robot')
        for p in sorted((package/directory).rglob('*')) if p.is_file()}
    # Material target expansion is applied to clean as well as attack runs.
    material_targets=sorted({p for profile in (package/'data/attack/profiles').glob('*.json')
        for a in read(profile).get('attack',{}).get('attacks',[])
        if a.get('handler')=='install_visual_material_override'
        for p in a['parameters'].get('expected_affected_mesh_prims',[])})
    manifest=read(root/'manifests/questions.json');scene=next(r['scene'] for r in manifest if r['question']==question)
    contract_path=root/'benchmark/contracts'/(question+'.json')
    contract=read(contract_path) if contract_path.exists() else {}
    identity={'schema':'paper200-resident-physical-group/v1','scene':scene,'physical_task':physical,
        'scene_adapter_and_route_files':assets,'material_target_expansion':material_targets,
        'runtime_implementation':contract.get('implementation'),
        'scene_revision':contract.get('scene_revisions'),
        'seed_scope':'seed is reset per episode; same-seed parity is verified separately',
        'reuse_mode':'resident_kit_fresh_stage_and_physx'}
    return {'key':digest(identity),'identity':identity}


def inventory(root):
    root=Path(root).resolve();groups={};scenes={}
    for row in read(root/'manifests/questions.json'):
        value=physical_group(root,row['question']);groups.setdefault(value['key'],{'identity':value['identity'],'questions':[]})['questions'].append(row['question'])
        scenes.setdefault(row['scene'],[]).append(row['question'])
    result={'schema':'paper200-resident-group-inventory/v1','created':now(),'nominal_scenes':scenes,
        'nominal_scene_count':len(scenes),'physical_group_count':len(groups),'groups':groups,
        'policy':'same scene label alone never authorizes mutable scene reuse'}
    write(root/'reports/RESIDENT_PHYSICAL_GROUPS.json',result);return result


class _AppProxy:
    def __init__(self,actual):self.actual=actual
    def __getattr__(self,name):return getattr(self.actual,name)
    def close(self,*args,**kwargs):
        # Collector's per-job finally must not terminate the resident Kit.
        return None


def execute_collector(source,source_path):
    """runtime_entry's sole integration hook; ordinary execution unchanged."""
    namespace={'__file__':str(source_path),'__name__':'__main__'}
    if _SESSION is None:
        if os.environ.get('PAPER200_RESIDENT_ACTIVE')=='1':raise RuntimeError('resident session module identity lost')
        exec(compile(source,str(source_path),'exec'),namespace);return
    import importlib
    from isaaclab.app import AppLauncher as original
    launcher_module=importlib.import_module('isaaclab.app')
    had_launcher_attribute=hasattr(launcher_module,'AppLauncher')
    session=_SESSION
    class Launcher:
        add_app_launcher_args=staticmethod(original.add_app_launcher_args)
        def __init__(self,args):
            options={k:getattr(args,k,None) for k in ('gpu','device','enable_cameras','headless')}
            if session.get('app') is None:
                session['app']=original(args).app;session['app_options']=options
            elif options!=session['app_options']:
                raise ValueError('resident Kit options changed: '+str(options))
            self.app=_AppProxy(session['app'])
    launcher_module.AppLauncher=Launcher
    try:exec(compile(source,str(source_path),'exec'),namespace)
    finally:
        if had_launcher_attribute:launcher_module.AppLauncher=original
        else:del launcher_module.AppLauncher
        session['namespace']=namespace


def _cleanup(session):
    """Destroy mutable state. Any cleanup failure poisons the whole worker."""
    namespace=session.pop('namespace',{});collector=namespace.get('ACTIVE_COLLECTOR')
    world=namespace.get('ACTIVE_WORLD');errors=[];closed=[];stage_measurements={}
    if world is not None:
        for name in ('stop','clear_all_callbacks'):
            try:getattr(world,name)();closed.append('World.'+name)
            except Exception as error:errors.append(name+':'+repr(error))
    if collector is not None:
        for name in ('paper_inventory_writer','paper_geometry_writer','rubric_evidence_writer',
                     'rubric_evidence_runtime','fixed_evidence_sidecar','fixed_evidence_runtime','safety_contact_tracker'):
            value=getattr(collector,name,None)
            if value is None:continue
            file=getattr(value,'file',None)
            already_closed=file is not None and hasattr(file,'id') and not file.id.valid
            try:
                if not already_closed and hasattr(value,'close'):value.close()
                closed.append(name)
            except Exception as error:
                # Closed streams are harmless; all other cleanup failures fail
                # the worker instead of contaminating the following episode.
                if 'closed' not in str(error).lower() and 'invalid identifier' not in str(error).lower():errors.append(name+':'+repr(error))
            setattr(collector,name,None)
        for name in ('paper_backend','paper_physical_observer','paper_discrete_observer'):
            if hasattr(collector,name):setattr(collector,name,None)
    from . import attack_runtime
    attack_runtime.CURRENT_COLLECTOR=None
    if session.get('app') is not None:
        import omni.usd
        from isaacsim.core.api import World
        World.clear_instance()
        context=omni.usd.get_context()
        if context.get_stage() is not None:
            stage_measurements['root_layer_identifier']=context.get_stage().GetRootLayer().identifier
            stage_measurements['session_layer_identifier']=context.get_stage().GetSessionLayer().identifier
            if not context.close_stage():errors.append('USD_STAGE_CLOSE_FAILED')
        namespace.clear();collector=None;world=None;gc.collect()
        for _ in range(4):session['app'].update()
        stage_measurements['stage_after_close_is_none']=context.get_stage() is None
        if context.get_stage() is not None:errors.append('USD_STAGE_REMAINED_AFTER_CLOSE')
    gc.collect()
    return {'closed':closed,'errors':errors,'passed':not errors,'stage_measurements':stage_measurements,'mode':'full_stage_physx_collector_recreation',
        'state_reset_by_recreation':['poses','velocities','joint_drives','constraints','materials','contacts',
            'collector_history','Agent_instance','attack_runtime','clock','RNG','writers']}


def run_request(request,session):
    global _SESSION
    config=Path(request['config_path']).resolve(strict=True);cfg=read(config)
    out=Path(cfg['output']).resolve();root=Path(cfg['collection_root']).resolve()
    if 'collection_20260924_v2' in str(root):raise ValueError('frozen v2 cannot use new resident runtime')
    group=physical_group(root,request['question'])
    if group['key']!=request['physical_group_key']:raise ValueError('resident physical group no longer matches source')
    if session.get('scene') not in (None,group['identity']['scene']):raise ValueError('worker nominal scene mismatch')
    session['scene']=group['identity']['scene']
    if session.get('poisoned'):raise ValueError('resident worker cleanup failed; restart required')
    session['group_key']=group['key'];session['episodes']=session.get('episodes',0)+1
    original_env=dict(os.environ);original_argv=sys.argv[:];original_path=sys.path[:];cwd=Path.cwd()
    started=time.monotonic();exit_code=0;error=None;_SESSION=session
    environment=request['environment']
    if environment.get('THREE_SCENES_RUN_CONFIG')!=str(config):raise ValueError('resident config environment mismatch')
    log=out/'resident_runtime.log';out.mkdir(parents=True,exist_ok=True)
    completed=threading.Event()
    cancel=out/'RESIDENT_CANCEL.json'
    def watch_cancel():
        while not completed.wait(.2):
            from .resident_pool import process_token
            orphan=request.get('client_pid') is not None and process_token(request['client_pid'])!=request['client_token']
            if cancel.exists() or orphan:
                write(out/'RESIDENT_ABORTED.json',{'worker_pid':os.getpid(),'cancel':str(cancel),'time':now(),
                    'reason':'owned episode canceled or client exited; terminate whole resident worker to prevent state carryover'})
                os._exit(124)
    watchdog=threading.Thread(target=watch_cancel,daemon=True);watchdog.start()
    try:
        os.environ.clear();os.environ.update(environment);os.environ['PAPER200_RESIDENT_ACTIVE']='1'
        os.chdir(request.get('cwd',str(root)));sys.argv=[request['runtime_script'],*request['argv']]
        sys.path[:]=environment.get('PYTHONPATH','').split(os.pathsep)+original_path
        import random,numpy as np
        seed=int(cfg['experiment_seed']);random.seed(seed);np.random.seed(seed)
        if 'torch' in sys.modules:
            sys.modules['torch'].manual_seed(seed)
            if sys.modules['torch'].cuda.is_initialized():sys.modules['torch'].cuda.manual_seed_all(seed)
        with log.open('w',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            runpy.run_path(request['runtime_script'],run_name='__main__')
    except SystemExit as exc:exit_code=int(exc.code or 0)
    except BaseException as exc:exit_code=1;error={'error':repr(exc),'traceback':traceback.format_exc()}
    finally:
        cleanup=_cleanup(session);session['poisoned']=not cleanup['passed'];_SESSION=None
        completed.set();watchdog.join(timeout=1)
        os.environ.clear();os.environ.update(original_env);sys.argv=original_argv;sys.path[:]=original_path;os.chdir(cwd)
    result={'schema':'paper200-resident-episode/v1','worker_pid':os.getpid(),'episode_index':session['episodes'],
        'exit_code':exit_code if cleanup['passed'] else 1,'error':error,'cleanup':cleanup,
        'physical_group_key':group['key'],'reuse_mode':'resident_kit_fresh_stage_and_physx',
        'in_place_physics_reset':False,'wall_seconds':time.monotonic()-started,'log':str(log)}
    write(out/'RESIDENT_EPISODE_RESULT.json',result);return result


def serve(socket_path):
    socket_path=Path(socket_path);socket_path.parent.mkdir(parents=True,exist_ok=True)
    if socket_path.exists():raise ValueError('resident socket already exists; inspect owner before replacing')
    session={};server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(socket_path));os.chmod(socket_path,0o600);server.listen(8)
    canonical=sys.modules.get('paper200.resident_scene') is sys.modules[__name__]
    if not canonical:raise RuntimeError('resident server canonical module identity mismatch')
    write(socket_path.with_suffix('.ready.json'),{'pid':os.getpid(),'socket':str(socket_path),'created':now(),
        'canonical_module_identity_verified':canonical})
    try:
        while True:
            connection,_=server.accept()
            with connection:
                with connection.makefile('r',encoding='utf-8') as stream:request=json.loads(stream.readline())
                if request.get('operation')=='shutdown':connection.sendall(b'{"stopped":true}\n');break
                try:result=run_request(request,session)
                except BaseException as error:result={'exit_code':1,'error':repr(error),'traceback':traceback.format_exc()}
                connection.sendall((json.dumps(result)+'\n').encode())
                if session.get('poisoned'):break
    finally:
        server.close();socket_path.unlink(missing_ok=True)
        if session.get('app') is not None:session['app'].close()


def submit(socket_path,request):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.connect(str(socket_path));client.sendall((json.dumps(request)+'\n').encode())
        with client.makefile('r',encoding='utf-8') as stream:
            line=stream.readline()
        if not line:raise RuntimeError('resident worker disconnected; episode failed without a completed receipt')
        return json.loads(line)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve');s.add_argument('--socket',required=True,type=Path)
    c=sub.add_parser('client');c.add_argument('--socket',required=True,type=Path);c.add_argument('--question',required=True);c.add_argument('runtime',nargs=argparse.REMAINDER)
    i=sub.add_parser('inventory');i.add_argument('root',type=Path)
    a=p.parse_args()
    if a.command=='serve':serve(a.socket)
    elif a.command=='inventory':print(json.dumps({k:v for k,v in inventory(a.root).items() if k.endswith('count')}))
    else:
        argv=a.runtime[1:] if a.runtime and a.runtime[0]=='--' else a.runtime
        cfg_path=Path(os.environ['THREE_SCENES_RUN_CONFIG']).resolve();cfg=read(cfg_path)
        request={'config_path':str(cfg_path),'question':a.question,'physical_group_key':physical_group(cfg['collection_root'],a.question)['key'],
            'runtime_script':argv[0],'argv':argv[1:],'environment':dict(os.environ),'cwd':str(Path.cwd())}
        result=submit(a.socket,request);print(json.dumps(result));raise SystemExit(result['exit_code'])


if __name__=='__main__':main()
