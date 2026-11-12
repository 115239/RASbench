"""Shared paper200 task adapter and episode launcher, independent of Agent mode.

This is runner infrastructure, not an Agent input module. Release paths and seeds
are consumed here; only TaskSpec and public sensor data enter RunnerBridge.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from .task_spec import TaskSpec, sha256, write_json
from .owned_processes import OwnedProcesses

SCENES = {'工厂':'factory','厨房':'kitchen','商超':'market','超市':'market',
          '卧室':'floor8_bedroom','餐厅':'floor8_dining','客厅':'floor8_living','书房':'floor8_study'}


def scene_key(spec):
    if int(spec.question[1:]) >= 151:
        return {15:'floor8_bedroom',16:'floor8_dining',17:'floor8_kitchen',
                18:'floor8_living',19:'floor8_study'}[(int(spec.question[1:])-1)//10]
    for name, key in SCENES.items():
        if name in spec.scene:
            return key
    raise ValueError('unrecognized public scene: '+spec.scene)


def runner_task(spec, binding, *, seed, action_budget, duration_s):
    if binding.get('task_identity') != spec.identity:
        raise ValueError('runtime binding belongs to a different benchmark/question')
    allowed = {'task_identity','task_id','stages','goals','model_goal_ids','stage_reviewed'}
    if set(binding) != allowed or binding['stage_reviewed'] is not True:
        raise ValueError('reviewed public runtime binding required')
    stages = binding['stages']
    accepted = {'NAV_OUTBOUND','NAV_RETURN','NAV_DELIVERY','PICK','PLACE','PUSH','BUTTON','OPEN','HANDOFF'}
    if not stages or any(s not in accepted for s in stages):
        raise ValueError('unsupported public task stage; do not silently skip it')
    spec.bind(action_budget=action_budget,duration_s=duration_s)
    return {'benchmark_id':spec.benchmark_id,'question':spec.question,'task_identity':spec.identity,
            'task_id':binding['task_id'],'instruction':spec.instruction,'scene':scene_key(spec),
            'seed':seed,'stages':stages,'maximum_actions':action_budget,'duration_s':duration_s}


def vocabulary_audit(task, binding, vocabulary):
    available = set(vocabulary['goal_ids'])
    required = set(binding['model_goal_ids'])
    outside = sorted(required-available)
    unknown_scene = task['scene'] not in {'factory','kitchen','market'}
    return {'task_identity':task['task_identity'],'model_goal_count':len(available),
            'required_goal_ids':sorted(required),'outside_vocabulary':outside,
            'scene_outside_navigation_training':unknown_scene,
            'mapping_status':'DECLARED_PUBLIC_BINDING','actual_task_score':None,
            'weights_changed':False,'navigation_fallback':'ABSTAIN_WITH_REASON' if outside or unknown_scene else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    job=json.loads(a.job.read_text());release=job['release'];out=Path(job['output'])
    spec=TaskSpec.load(job['spec'])
    binding=json.loads(Path(release['public_runtime_binding']).read_text())
    settings=json.loads(Path(release['runner_settings']).read_text())
    baseline=Path(release['baseline_root']).resolve()
    overlay=Path(__file__).resolve().parents[1]/'integrations/safety_embodiment'
    code=overlay/'baseline/code/runner'
    os.environ['PAPER_BASELINE_ROOT']=str(baseline)
    sys.path.insert(0,str(code))
    from launch import model_binding
    from common import read
    package=Path(release['package_root']).resolve()
    task=runner_task(spec,binding,seed=job['seed'],action_budget=release['action_budget'],duration_s=release['duration_s'])
    package_task=read(package/'data/task/task.json')
    if package_task['task_id'] != task['task_id'] or package_task['instruction'].strip() != task['instruction'].strip():
        raise ValueError('published package and public task contract disagree')
    models=release['models']
    bindings={name:model_binding(models[name]) for name in ('navigation','manipulation')}
    bindings['locomotion']={'path':models['locomotion'],'sha256':sha256(models['locomotion'])}
    cfg={**settings,'task':task,'goals':binding['goals'],'output':str(out),'model_bindings':bindings,
         'nav_model':models['navigation'],'grasp_model':models['manipulation'],'nav_sampling_seed':job['seed'],
         'experiment_seed':job['seed'],'visit_distance_trigger':False,'paper200':True,
         'attack_mode':'off' if job['variant']=='clean' else 'on'}
    write_json(out/'VOCABULARY_AUDIT.json',vocabulary_audit(task,binding,read(Path(models['navigation'])/'GOALS.json')))
    write_json(out/'RUN_CONFIG.json',cfg)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=dict(os.environ)
    runner=overlay/'competition_runner/runner'
    env.update(THREE_SCENES_RUN_CONFIG=str(out/'RUN_CONFIG.json'),PAPER_RUNNER_ROOT=str(runner),
               PYTHONHASHSEED=str(job['seed']))
    env['PYTHONPATH']=os.pathsep.join(map(str,[code,baseline/'code/grasp',baseline/'code/navigation',
        baseline/'code/navigation/nav2_outbound',baseline/'code',code/'vendor',code/'vendor/_vendor_v072',runner,
        Path(__file__).resolve().parents[1]]))+os.pathsep+env.get('PYTHONPATH','')
    with OwnedProcesses(out, 'paper_runner') as owned:
        with (out/'model_server.log').open('xb') as log:
            server=owned.start([release['model_python'],str(code/'model_server.py'),'--config',str(out/'RUN_CONFIG.json'),
                    '--out',str(out/'model'),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+900
        while not (out/'model/READY.json').exists():
            if server.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('model server did not become ready; see model_server.log')
            time.sleep(1)
        command=[release['runtime_python'],str(runner/'robolab_runner/m20_question_runner.py'),
            '--package-root',str(package),'--output-dir',str(out/'episode'),'--policy-endpoint',f'ws://127.0.0.1:{port}',
            '--runtime-script',str(code/'runtime_entry.py'),'--runtime-root',str(runner/'m20_runtime'),
            '--locomotion-policy',models['locomotion'],'--gpu',str(release.get('gpu',0)),
            '--capture-hz','10','--video-fps','10','--vla-action-hz',str(settings['action_hz']),
            '--maximum-vla-actions',str(task['maximum_actions']),'--maximum-duration-seconds',str(task['duration_s']),
            '--attack-mode',cfg['attack_mode'],'--experiment-seed',str(job['seed']),'--navigation-mode','hybrid']
        if job['variant']!='clean': command += ['--attack-variant',job['variant']]
        write_json(out/'LAUNCH.json',{'argv':command,'task_identity':spec.identity,'seed':job['seed']})
        with (out/'physical_runner.log').open('xb') as log:
            process=owned.start(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+release.get('wall_timeout_s',7200)
            while process.poll() is None:
                if server.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('model server exited or episode wall timeout')
                time.sleep(1)
        if process.returncode: raise RuntimeError('physical runner failed; preserve output for inspection')
        if not (out/'episode/submission.hdf5').is_file(): raise RuntimeError('missing formal HDF5')


if __name__=='__main__': main()
