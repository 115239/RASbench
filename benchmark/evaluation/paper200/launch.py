from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import signal
import socket
import shutil
import subprocess
import time
import traceback
import urllib.request
from .common import BASE, NATIVE, PROJECT, read, write, environment, now, sha, digest
from . import queue
from .bindings import goals
from .wallclock_budget import enforce_wallclock, EXPORT_RESERVE_S


def interpreter_environment(env, kind):
    """Native python.sh selects its ABI from this variable, not its filename."""
    if kind not in ('model', 'isaac'):
        raise ValueError('unsupported native interpreter kind')
    return dict(env, BASELINE_PYTHON_KIND=kind)


def runner_environment(root,runtime_root=None):
    runtime_root=Path(runtime_root) if runtime_root else root/'runtime'
    env=interpreter_environment(environment(root),'isaac')
    env['PAPER200_PROJECT_ROOT']=str(PROJECT)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    paths=[PROJECT/'experiment',runtime_root/'baseline_runner',runtime_root/'agent',
           runtime_root/'runner',runtime_root/'runner/m20_runtime',
           runtime_root/'baseline_code/grasp',runtime_root/'baseline_code/navigation',
           runtime_root/'baseline_code/navigation/nav2_outbound',
           runtime_root/'baseline_code/runner/vendor',runtime_root/'baseline_code/runner/vendor/_vendor_v072']
    env['PYTHONPATH']=os.pathsep.join(map(str,paths))
    isolated=runtime_root/'baseline'
    isolated.mkdir(exist_ok=True)
    for name,target in [('code',runtime_root/'baseline_code'),('models',BASE/'baseline/models'),
                        ('assets',BASE/'baseline/assets')]:
        try:(isolated/name).symlink_to(target,target_is_directory=True)
        except FileExistsError:pass
    for name in ['config.json','ROBOT_MANIFEST.json']:
        if not (isolated/name).exists():shutil.copy2(BASE/'baseline'/name,isolated/name)
    env['PAPER_BASELINE_ROOT']=str(isolated)
    env['PAPER_RUNNER_ROOT']=str(runtime_root/'runner')
    env['PATH']=str(NATIVE)+':'+env.get('PATH','/usr/bin:/bin')
    return env


def runner_command(root,q,out,system,variant,seed,gpu,port,preflight_only=False):
    row=next(r for r in read(root/'manifests/questions.json') if r['question']==q)
    command=[str(NATIVE/'isaac_python.sh'),'-m','robolab_runner.m20_question_runner',
       '--package-root',str(root/'benchmark/task'/q),'--output-dir',str(out),
       '--policy-endpoint','ws://127.0.0.1:%d'%port,
       '--runtime-script',str(out.parent/'code/paper200/runtime_entry.py' if (out.parent/'code/paper200/runtime_entry.py').exists() else PROJECT/'experiment/paper200/runtime_entry.py'),
       '--runtime-root',str((out.parent/'runtime_snapshot' if (out.parent/'runtime_snapshot').is_dir() else root/'runtime')/'runner/m20_runtime'),'--isaac-python',str(NATIVE/'isaac_python.sh'),
       '--locomotion-policy',str(BASE/'baseline/models/locomotion/policy.onnx'),
       '--gpu',str(gpu),'--capture-hz','10','--video-fps','10','--vla-action-hz','2',
       '--maximum-vla-actions',str(row['action_budget']),'--maximum-duration-seconds',str(row['duration_s']),
       '--attack-mode','off' if variant=='clean' else 'on','--experiment-seed',str(seed),
       '--navigation-mode','vla']
    if variant!='clean':command+=['--attack-variant',variant]
    if preflight_only:command+=['--preflight-only']
    return command


def preflight(root,questions='all'):
    from .provenance import package_identity
    items=read(root/'manifests/questions.json')
    if questions!='all':
        selected=set(questions.split(','));items=[q for q in items if q['question'] in selected]
    batch=root/'pilot'/('preflight_%d'%time.time_ns())
    batch.mkdir(parents=True)
    def check(q):
        out=batch/q['question'];log=batch/(q['question']+'.log')
        command=runner_command(root,q['question'],out,'groot_agent_off','clean',72026001,1,18999,True)
        with log.open('w') as f:
            result=subprocess.run(command,env=runner_environment(root),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        return {'question':q['question'],'passed':result.returncode==0,'log':str(log),'output':str(out),
                'package_sha256':package_identity(root,q['question'])['sha256']}
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(check,items))
    summary={'checked':len(results),'passed':sum(r['passed'] for r in results),'results':results}
    write(root/'reports/preflight.json',summary)
    return summary


def live_group(pgid):
    result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            raw=(entry/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if int(fields[2])==pgid and fields[0]!='Z':result.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result


def stop_group(process,grace_seconds=20):
    # A wrapper may exit before its simulator child. Waiting only for the
    # wrapper would incorrectly certify cleanup and leak GPU allocations.
    process.poll()
    if not live_group(process.pid):return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+grace_seconds
    while live_group(process.pid) and time.monotonic()<deadline:
        process.poll();time.sleep(.2)
    if live_group(process.pid):
        os.killpg(process.pid,signal.SIGKILL)
        deadline=time.monotonic()+grace_seconds
        while live_group(process.pid) and time.monotonic()<deadline:
            process.poll();time.sleep(.2)
    process.wait(timeout=20)
    if live_group(process.pid):raise RuntimeError('owned process group did not exit')


@enforce_wallclock
def run_job(root,job,gpu):
    from .provenance import package_identity,runtime_identity,verify_frozen,scene_revision_identity
    if job['purpose'] in ('main','assisted') and (root/'REPAIR_IN_PROGRESS.json').exists():
        gate=read(root/'reports/SCORING_READINESS.json')
        from .resume_repaired import collection_release_allowed
        if not collection_release_allowed(gate):
            raise ValueError('v2 formal dispatch blocked: Runner/Judge collection release incomplete')
    frozen=verify_frozen(root) if (root/'manifests/FROZEN.json').exists() else None
    if job['purpose'] in ('main','assisted') and frozen is None:raise ValueError('formal collections require frozen inputs')
    out=Path(job['output']);row=next(q for q in read(root/'manifests/questions.json') if q['question']==job['question'])
    prelaunch=None
    if (root/'REPAIR_IN_PROGRESS.json').exists() or (root/'benchmark/contracts').is_dir():
        from .prelaunch_identity import verify as verify_prelaunch
        prelaunch=verify_prelaunch(root,job)
        write(out/'PRELAUNCH_IDENTITY.json',prelaunch)
    from .runtime_cache import finish as finish_cache
    source=root/'runtime/collection_code' if (root/'manifests/FROZEN.json').exists() else PROJECT/'experiment'
    shutil.copytree(source/'paper200',out/'code/paper200',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['live_models.py','common.py']:
        text=(source/name).read_text(encoding='utf-8')
        if name=='common.py':text=text.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path('+repr(str(PROJECT/'experiment'))+')')
        (out/'code'/name).write_text(text,encoding='utf-8')
    write(out/'code/manifest.json',{str(p.relative_to(out/'code')):sha(p) for p in (out/'code').rglob('*.py')})
    runtime_root=root/'runtime'
    if (root/'REPAIR_IN_PROGRESS.json').exists() and job['purpose'] not in ('main','assisted'):
        from .development_snapshot import snapshot
        runtime_root=snapshot(root,out)
    task=read(root/'benchmark/task'/job['question']/'data/task/task.json')
    settings=read(root/'runtime/baseline/config.json')
    models=read(root/'manifests/models.json')
    cfg={**settings,'collection_root':str(root),'output':str(out),'backend':'pi05' if job['system'].startswith('pi05') else 'groot',
         'experiment_seed':job['seed'],'nav_sampling_seed':job['seed'],'model_bindings':models,
         'nav_model':models['navigation']['path'],'grasp_model':models['manipulation']['path'],
         'task':{'question':'Q%03d'%int(job['question'][1:]),'task_id':task['task_id'],'instruction':task['instruction'],
                 'scene':'kitchen' if row['scene']=='kitchen_livingroom' else ('floor8_'+row['scene'] if int(job['question'][1:])>150 else row['scene']),
                 'stages':row['stages'],'duration_s':row['duration_s'],'maximum_actions':row['action_budget'],
                 'benchmark_id':root.name,'task_identity':root.name+':'+job['question']},
         'goals':goals(root),'visit_distance_trigger':False,'action_hz':2,'paper200':True}
    cfg['runtime_snapshot']=str(runtime_root)
    cfg['wall_clock_started_unix_s']=job['_wall_clock_started_unix_s']
    cfg['wall_clock_deadline_unix_s']=job['_wall_clock_deadline_unix_s']
    cfg['wall_clock_export_reserve_s']=EXPORT_RESERVE_S
    cfg['purpose']=job['purpose'];cfg['assistance']=None;cfg['scene_key']=row['scene']
    performance_path=root/'runtime/baseline/performance.json'
    performance=read(performance_path) if performance_path.exists() else {}
    cfg['execution_profile']=performance
    from .evidence_contract import condition_id
    selected=[]
    if job['variant']!='clean':
        profile=read(root/'benchmark/task'/job['question']/'data/attack/profiles'/
                     ('q%d_%s.json'%(int(job['question'][1:]),job['variant'])))
        selected=profile['attack']['attacks']
    attack_seeds={a.get('parameters',{}).get('seed') for a in selected
                  if a.get('parameters',{}).get('seed') is not None}
    if len(attack_seeds)>1:raise ValueError('multiple attack seeds require per-attack identity')
    identity={'schema':'paper200-runner-judge/v2', 'episode_id':job['run_id']+':'+str(job['attempts']),
        'run_id':job['run_id'],'attempt_id':str(job['attempts']), 'task_id':task['task_id'],
        'system_id':job['system'],'system_version':root.name,
        'condition_id':condition_id([a['id'] for a in selected]),
        'selected_attack_ids':[a['id'] for a in selected], 'seed':job['seed'],'policy_seed':job['seed'],
        'attack_seed':next(iter(attack_seeds)) if attack_seeds else None,
        'block_id':job['question']+':'+str(job['seed']), 'repeat_index':0, 'purpose':job['purpose'],
        'field_sources':{'condition_id':'queue variant validated against frozen profile',
            'seed':'queue paired initial state seed','policy_seed':'model service configured RNG seed',
            'attack_seed':'selected profile parameters.seed; null when no explicit RNG seed',
            'repeat_index':'one planned repetition per matrix condition; retries have distinct attempt_id'}}
    write(out/'identity.json',identity)
    if cfg['backend']=='pi05':cfg['transfer_adapter']=read(root/'manifests/pi05_transfer.json')
    native_binding=root/'manifests/native_runtime.json'
    cfg['native_runtime']=read(native_binding) if native_binding.exists() else None
    if job['purpose'] in ('assisted','assisted_pilot'):
        manifest=read(root/'manifests/assistance.json')
        item=next(x for x in manifest['questions'] if x['question']==job['question'])
        if cfg['backend']!='groot' or not manifest['frozen']:raise ValueError('invalid assisted run')
        cfg['assistance']={**item,'manifest_sha256':sha(root/'manifests/assistance.json')}
        cfg['auxiliary_destination_world_xyz']=task.get('manipulation',{}).get('destination_world_center_xyz')
        if 'PUSH' in row['stages']:
            cfg['auxiliary_push_reference_prim']=task.get('tracked_objects',{}).get('reference_1')
        # Keep the original mission stages. A pure manipulation task starts at
        # its validated operation pose and needs no added navigation prerequisite.
        cfg['goals']['ASSIST_PUBLIC_TARGET']={'prim':task['tracked_objects']['target_0'],'distance_m':.5,
            'task_questions':[cfg['task']['question']],'via_regions':[],'source':'explicit auxiliary public target binding'}
        destination=task.get('manipulation',{}).get('destination_prim')
        if destination:
            cfg['goals']['ASSIST_PUBLIC_DESTINATION']={'prim':destination,'distance_m':.5,
                'task_questions':[cfg['task']['question']],'via_regions':[],'source':'explicit auxiliary public destination binding'}
    write(out/'config.json',cfg)
    contract_path=root/'benchmark/contracts'/(job['question']+'.json')
    if (root/'REPAIR_IN_PROGRESS.json').exists():
        from .evidence_contract import canonical_digest
        contract=read(contract_path)
        if contract['question']!=job['question']:raise ValueError('contract question mismatch')
        write(out/'runner_judge_contract.json',contract)
    write(out/'input_identity.json',{'model_bindings_sha256':sha(root/'manifests/models.json'),
        'task_sha256':sha(root/'benchmark/task'/job['question']/'data/task/task.json'),
        'frozen_version':frozen,'package':package_identity(root,job['question']),
        'runtime':runtime_identity(root),
        'scene_revisions':scene_revision_identity(root,job['question']),
        'rubric_sha256':sha(root/'benchmark/task'/job['question']/'data/task/rubric.json'),
        'environment_sha256':sha(root/'benchmark/task'/job['question']/'data/env/environment.json'),
        'public_task_sha256':sha(root/'manifests/public'/job['question']/'task.json'),
        'pi05_transfer_sha256':sha(root/'manifests/pi05_transfer.json') if cfg['backend']=='pi05' else None,
        'agent_mode':'on' if job['system']=='groot_agent_on' else 'off',
        'runtime_attack_client_sha256':sha(root/'runtime/runner/m20_runtime/attack_tools/attackable_vla_client.py')})
    env=runner_environment(root,runtime_root)
    env['PYTHONPATH']=str(out/'code')+':'+env['PYTHONPATH']
    env['PAPER200_PROJECT_ROOT']=str(PROJECT)
    private=out/'private_runtime'
    for name in ['cache','config','tmp']:(private/name).mkdir(parents=True,exist_ok=True)
    env.update(XDG_CACHE_HOME=str(private/'cache'),XDG_CONFIG_HOME=str(private/'config'),
               XDG_RUNTIME_DIR=str(private/'tmp'),TMPDIR=str(private/'tmp'),
               OMNICLIENT_HUB_MODE='disabled',CUDA_CACHE_PATH=str(private/'cache/cuda'))
    mode='on' if job['system']=='groot_agent_on' else 'off'
    env.update(THREE_SCENES_RUN_CONFIG=str(out/'config.json'),SAFE_AGENT_MODE=mode,
        PAPER200_SYSTEM_ID=job['system'],PAPER200_SYSTEM_VERSION=root.name,
        SAFE_AGENT_TASK_SPEC=str(root/'manifests/public'/job['question']/'task.json'),
        SAFE_AGENT_EPISODE_ID=job['run_id']+':'+str(job['attempts']),
        SAFE_AGENT_ACTION_BUDGET=str(row['action_budget']),SAFE_AGENT_DURATION_S=str(row['duration_s']))
    if mode=='on':
        from .vision_pool import choose
        endpoint=choose(root,out)
        env['SAFE_AGENT_VISION_ENDPOINT']=endpoint
    else:env.pop('SAFE_AGENT_VISION_ENDPOINT',None)
    model_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
    if cfg['backend']=='groot':
        model_env=interpreter_environment(model_env,'model')
        model_python=str(NATIVE/'python.sh')
        model_env['PYTHONPATH']=':'.join([str(out/'code'),str(runtime_root/'baseline_runner'),
            str(runtime_root/'baseline_code/Isaac-GR00T'),str(runtime_root/'baseline_code/pylibs'),model_env['PYTHONPATH']])
    else:
        openpi_source=runtime_root/'openpi'
        if not openpi_source.exists():raise ValueError('isolated openpi source snapshot missing')
        model_python=os.environ['RASBENCH_OPENPI_PYTHON']
        model_env['PYTHONPATH']=':'.join(map(str,[out/'code',openpi_source/'src',openpi_source/'packages/openpi-client/src']))
        model_env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',
                         JAX_COMPILATION_CACHE_DIR=str(private/'cache/jax'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[]
    resident_model=None
    available=sorted(os.sched_getaffinity(0))
    width=min(32,len(available));offset=(gpu*width)%len(available)
    affinity=available[offset:offset+width]
    if len(affinity)<width:affinity+=available[:width-len(affinity)]
    cpu_prefix=['taskset','-c',','.join(map(str,affinity))]
    write(out/'cpu_affinity.json',{'cpus':affinity,'reason':'bound per-process native/JAX thread pools; GPU task count remains unrestricted'})
    previous_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(signum,frame):
        raise RuntimeError('WORKER_INTERRUPTED_SIGNAL_%d'%signum)
    for signum in previous_handlers:signal.signal(signum,interrupted)
    try:
        if prelaunch is not None:
            # Copy/configuration preparation may overlap a candidate revision.
            # Refuse it before the first model process is created.
            verify_prelaunch(root,job,expected_digest=prelaunch['contract_sha256'])
        if performance.get('resident_models'):
            from .resident_model_pool import acquire
            resident_model=acquire(root,cfg,out,gpu,model_python=model_python,model_env=model_env,
                cpu_prefix=cpu_prefix,timeout_s=max(0,job['_wall_clock_deadline_unix_s']-time.time()),
                code_root=source,service_gpu=performance['model_gpus'][cfg['backend']])
            model=resident_model.proxy;port=resident_model.port
            model_command=read(out/'RESIDENT_MODEL_BINDING.json')['proxy_command']
        else:
            model_command=cpu_prefix+[model_python,'-m','paper200.model_service','--config',str(out/'config.json'),
                           '--out',str(out/'model'),'--port',str(port)]
            with (out/'model.log').open('w') as log:
                model=subprocess.Popen(model_command,env=model_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(model)
        persistent={'resident_model_pid':resident_model.service_pid,'resident_model_gpu':resident_model.service['gpu']} if resident_model else {}
        write(out/'processes.json',{'launcher':os.getpid(),'model':model.pid,'gpu':gpu,'port':port,'created':now(),**persistent})
        deadline=time.monotonic()+max(0,job['_wall_clock_deadline_unix_s']-time.time())
        while not (out/'model/READY.json').exists():
            if model.poll() is not None:raise RuntimeError('MODEL_START_FAILED; see model.log')
            if time.monotonic()>deadline:raise RuntimeError('WALLCLOCK_LIMIT_600S; model initialization')
            time.sleep(1)
        if prelaunch is not None:
            # Model loading can take minutes; the Runner still resolves the
            # package/public TaskSpec from root, so check again before physics.
            verify_prelaunch(root,job,expected_digest=prelaunch['contract_sha256'])
        command=cpu_prefix+runner_command(root,job['question'],out/'episode',job['system'],job['variant'],job['seed'],gpu,port)
        runner_env=env.copy()
        if performance.get('resident_scenes'):
            shim=root/'runtime/runner/resident_isaac.sh'
            if not shim.is_file():raise RuntimeError('resident scene shim missing')
            command[command.index('--isaac-python')+1]=str(shim)
            runner_env['PAPER200_NATIVE_ISAAC_PYTHON']=str(NATIVE/'isaac_python.sh')
        write(out/'launch.json',{'runner':command,'model':model_command,'gpu':gpu,**persistent})
        native_trace=PROJECT/'experiment/diagnostics/trace_system_error.so'
        if job['purpose']=='pilot' and native_trace.exists():
            runner_env['LD_PRELOAD']=str(native_trace)
            write(out/'NATIVE_DIAGNOSTICS.json',{'preload':str(native_trace),'sha256':sha(native_trace),
                  'scope':'development exception stack logging only; exceptions remain uncaught'})
        with (out/'runner.log').open('w') as log:
            runner=subprocess.Popen(command,env=runner_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(runner)
        write(out/'processes.json',{'launcher':os.getpid(),'model':model.pid,'runner':runner.pid,'gpu':gpu,'port':port,'created':now(),**persistent})
        deadline=time.monotonic()+max(0,job['_wall_clock_deadline_unix_s']-time.time())
        startup_deadline=time.monotonic()+900
        while runner.poll() is None:
            if model.poll() is not None:raise RuntimeError('MODEL_EXITED')
            if time.monotonic()>deadline:raise RuntimeError('WALLCLOCK_LIMIT_600S; physics')
            if time.monotonic()>startup_deadline and (not (out/'episode/PHYSICAL_TRACE.jsonl').exists() or (out/'episode/PHYSICAL_TRACE.jsonl').stat().st_size==0):
                raise RuntimeError('RUNNER_INITIALIZATION_TIMEOUT; no physical trace after 900 seconds')
            with queue.connect(root/'queue.sqlite3') as db:
                db.execute('UPDATE runs SET heartbeat=? WHERE run_id=? AND owner=?',(time.time(),job['run_id'],job['owner']))
            time.sleep(3)
        if runner.returncode:raise RuntimeError('RUNNER_EXIT_%d; see runner.log'%runner.returncode)
        return validate_existing(root,out)
    finally:
        if performance.get('resident_scenes') and not (out/'RESIDENT_EPISODE_RESULT.json').exists():
            from .resident_pool import abort
            abort(out,reason='episode launcher exited before resident job completed')
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
        for process in reversed(processes):stop_group(process)
        write(out/'PROCESS_CLEANUP.json',{'completed':True,'pids':[p.pid for p in processes],'time':now()})
        if (out/'COLLECTION_VALIDATION.json').exists():finish_cache(root,out,row['scene'])


def validate_existing(root,out):
    # The simulator environment owns h5py and the protocol dependencies.
    cfg=read(out/'config.json')
    env=runner_environment(root,cfg.get('runtime_snapshot'))
    if (out/'code').is_dir():env['PYTHONPATH']=str(out/'code')+':'+env['PYTHONPATH']
    with (out/'collection_validation.log').open('w') as log:
        result=subprocess.run([str(NATIVE/'isaac_python.sh'),'-m','paper200.validate',
            '--root',str(root),'--attempt',str(out)],env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError('COLLECTION_VALIDATION_FAILED; see collection_validation.log')
    return read(out/'COLLECTION_VALIDATION.json')


def worker(root,gpu,run_id=None,system=None,purpose='main'):
    if purpose in ('main','assisted'):
        if (root/'STOP').exists():
            return [{'status':'dispatch_stopped','reason':'STOP is present; no new claim made'}]
        gate=read(root/'reports/MAIN_RELEASE_GATE.json')
        if not gate.get('passed'):raise RuntimeError('main release gate has not passed')
    owner='%d:%d:%d'%(os.getpid(),gpu,time.time_ns());results=[]
    while True:
        if purpose in ('main','assisted') and (root/'STOP').exists():break
        job=queue.claim(root,owner,run_id=run_id,system=system,purpose=purpose)
        if job is None:break
        try:
            result=run_job(root,job,gpu);queue.finish(root,job,result)
        except Exception as error:
            result={'error':str(error),'traceback':traceback.format_exc()}
            write(Path(job['output'])/'FAILURE.json',result)
            queue.finish(root,job,result,error=str(error))
        results.append({'run_id':job['run_id'],**result})
        queue.report(root)
        if run_id:break
    return results
