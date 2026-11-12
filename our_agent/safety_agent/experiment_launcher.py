"""Run one frozen job and its external scorer, preserving each failure directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request
from .experiments import verify_release
from .task_spec import write_json
from .owned_processes import OwnedProcesses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args()
    job = json.loads(a.job.read_text())
    release = job['release']
    verify_release(release)
    output = Path(job['output'])
    config = output/'assurance_config.json'
    write_json(config, job['assurance'])
    env = dict(os.environ)
    env.update(SAFE_AGENT_MODE=job['assurance']['mode'], SAFE_AGENT_TASK_SPEC=job['spec'],
               SAFE_AGENT_CONFIG=str(config), SAFE_AGENT_EPISODE_ID=job['run_id'],
               SAFE_AGENT_ACTION_BUDGET=str(release['action_budget']),
               SAFE_AGENT_DURATION_S=str(release['duration_s']))
    agent_root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = agent_root + os.pathsep + env.get('PYTHONPATH','')
    if job['assurance']['mode'] != 'off':
        endpoint = release.get('vision_endpoint','http://127.0.0.1:18041')
        if not endpoint.startswith(('http://127.0.0.1:','http://localhost:')):
            raise ValueError('vision endpoint must be local')
        with urllib.request.urlopen(endpoint+'/health',timeout=5) as response:
            health = json.load(response)
            if health.get('status') != 'READY' or health.get('models_lock_sha256') != release['vision_lock_sha256']:
                raise RuntimeError('frozen auxiliary vision is not ready')
        env['SAFE_AGENT_VISION_ENDPOINT'] = endpoint
    else:
        env.pop('SAFE_AGENT_VISION_ENDPOINT',None)
    substitutions = {k:str(job[k]) for k in ('run_id','question','seed','variant','condition','output','spec')}
    substitutions['attack_mode'] = 'off' if job['variant']=='clean' else 'on'
    substitutions['job'] = str(a.job.resolve())
    for kind in ('runner_argv','scorer_argv'):
        with OwnedProcesses(output, 'launcher', grace_s=15) as owned:
            argv = release.get(kind)
            if not isinstance(argv,list) or not argv or not all(isinstance(v,str) for v in argv):
                raise ValueError('frozen release requires ' + kind)
            command = [arg.format_map(substitutions) for arg in argv]
            with (output/(kind+'.log')).open('wb') as log:
                process = owned.start(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = process.wait(timeout=release.get('wall_timeout_s',7200))
                if code:
                    raise subprocess.CalledProcessError(code, command)
            verify_release(release)
    if not (output/'score.json').is_file():
        raise RuntimeError('external scorer did not write score.json')


if __name__=='__main__':
    main()
