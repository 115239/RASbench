"""Self-contained contestant run, resume, validation and submission packaging."""
import argparse
import csv
import ctypes
import fcntl
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile
import h5py
import numpy as np
from common import *
from launch import model_binding, make_goals
from catalog import build


def integrated_registry():
    return read(BASE / 'INTEGRATED_PROFILES.json')


def selected_profile(q):
    registry = integrated_registry()
    name = registry['question_profiles'][q]
    return {'name': name, **registry['profiles'][name]}


def verify_integrated_profiles():
    registry = integrated_registry()
    if set(registry['question_profiles']) != {f'Q{i:02}' for i in range(1,25)}:
        raise RuntimeError('INCOMPLETE_PROFILE_REGISTRY')
    for profile in registry['profiles'].values():
        for rel, expected in profile['files'].items():
            if sha(BASE / profile['directory'] / rel) != expected:
                raise RuntimeError('PROFILE_HASH_MISMATCH: ' + profile['directory'] + '/' + rel)
    return registry


def hash_contract(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_hdf5(path):
    with h5py.File(path, 'r') as f:
        if 'data/demo_0' not in f:
            raise ValueError('MISSING_DEMO_GROUP')
        d = f['data/demo_0']
        if 'actions' not in d or len(d['actions']) == 0:
            raise ValueError('MISSING_ACTIONS')
        a = d['actions'][:]
        if a.ndim != 2 or a.shape[1] != 10 or not np.isfinite(a).all():
            raise ValueError('INVALID_ACTIONS')
        # The official exporter's conformance report is also required by run_one.
        metadata = f.get('metadata')
        complete = None
        if metadata is not None:
            if 'complete' in metadata.attrs: complete = bool(metadata.attrs['complete'])
            elif 'complete' in metadata: complete = bool(metadata['complete'][()])
        if complete is not True:
            raise ValueError('INCOMPLETE_HDF5')
        times = np.asarray(d['time/action_time_s'][:]).reshape(-1)
        if len(times) != len(a) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('INVALID_TIMELINE')
        return {'frames': len(a), 'action_dim': 10, 'metadata_complete': complete, 'sha256': sha(path)}


def validate_video(path):
    raw=subprocess.check_output(['/opt/ffmpeg7/bin/ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(path)],text=True,timeout=300)
    streams=json.loads(raw)['streams']
    if len(streams)!=1 or int(streams[0].get('nb_read_frames',0))<=0:
        raise ValueError('EMPTY_RECORDING_VIDEO')
    subprocess.run(['/opt/ffmpeg7/bin/ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True,timeout=300)
    return {'path':str(path),'full_decode':True,**streams[0]}


def stop_process(process):
    if process is None or process.poll() is not None: return
    os.killpg(process.pid, signal.SIGTERM)
    try: process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=15)


def resources(gpu, out, required_gib=50):
    try:
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True,stderr=subprocess.PIPE)
    except (FileNotFoundError,subprocess.CalledProcessError) as e:
        raise RuntimeError('GPU_UNAVAILABLE: start a GPU container with --runtime=nvidia --gpus device=0') from e
    try:
        ctypes.CDLL('libGLX_nvidia.so.0')
    except OSError as e:
        raise RuntimeError('NVIDIA_GRAPHICS_DRIVER_UNAVAILABLE: start Docker with --runtime=nvidia and NVIDIA_DRIVER_CAPABILITIES=all') from e
    rows = list(csv.reader(text.strip().splitlines()))
    # CUDA ordinals are container-local. UUID masking is set before model startup.
    if gpu < 0 or gpu >= len(rows): raise ValueError('GPU_NOT_VISIBLE')
    row = rows[gpu]
    if required_gib not in (30,50): raise ValueError('INVALID_SCENE_VRAM_RESERVATION')
    if int(row[2]) < required_gib * 1024: raise RuntimeError('INSUFFICIENT_FREE_VRAM: scene requires '+str(required_gib)+' GiB free')
    if shutil.disk_usage(out).free < 50 * 1024**3: raise RuntimeError('INSUFFICIENT_OUTPUT_SPACE: requires 50 GiB free for render cache and recordings')
    return {'visible_gpu': gpu, 'uuid': row[1].strip(), 'free_mib': int(row[2]), 'required_free_gib': required_gib, 'disk_free_bytes': shutil.disk_usage(out).free}


def run_one(args, q, task, goals, bindings, out):
    out.mkdir(parents=True, exist_ok=False)
    profile = selected_profile(q)
    worker_root = BASE / profile['directory'] / 'runner'
    settings = read(BASE / profile['directory'] / 'config.json')
    cfg = {**settings, 'task': task, 'goals': goals, 'output': str(out), 'nav_model': str(args.nav_model),
           'grasp_model': str(args.vla_model), 'model_bindings': bindings, 'attack_mode': 'on',
           'robot_contact_profile': settings.get('robot_contact_profile', 'official'), 'candidate_evidence_profile': 'official',
           'nav_stall_recoveries': settings.get('nav_stall_recoveries', 0), 'dock_target_mode': settings.get('dock_target_mode', 'shelf'), 'visit_distance_trigger': bool(settings.get('visit_distance_trigger', False))}
    cfg['integrated_profile'] = profile
    write(out/'RUN_CONFIG.json', cfg)
    status = {'question': q, 'status': 'PREPARING', 'started': time.time()}
    server = runner = None
    try:
        resource = resources(args.gpu, out, 50 if task["scene"] == "factory" else 30); write(out/'RESOURCES.json', resource)
        with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
        runtime_home=out.parents[2]/'.runtime_home'
        for name in ['.cache','tmp','kit','warp']:(runtime_home/name).mkdir(parents=True,exist_ok=True)
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=resource['uuid']
        env.update(HOME=str(runtime_home),XDG_CACHE_HOME=str(runtime_home/'.cache'),TMPDIR=str(runtime_home/'tmp'),OMNI_KIT_CACHE_DIR=str(runtime_home/'kit'),WARP_CACHE_PATH=str(runtime_home/'warp'))
        with (out/'model_server.log').open('x') as log:
            server=subprocess.Popen([str(MODEL_PYTHON),str(worker_root/'model_server.py'),'--config',str(out/'RUN_CONFIG.json'),'--out',str(out/'model'),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='LOADING_MODELS',model_server_pid=server.pid);write(out/'STATUS.json',status)
        deadline=time.monotonic()+900
        while not (out/'model/READY.json').exists():
            if server.poll() is not None: raise RuntimeError('MODEL_SERVER_EXIT: see model_server.log')
            if time.monotonic()>deadline: raise RuntimeError('MODEL_LOAD_TIMEOUT')
            time.sleep(1)
        env.update(THREE_SCENES_RUN_CONFIG=str(out/'RUN_CONFIG.json'), ACCEPT_EULA='Y', OMNI_KIT_ACCEPT_EULA='Y', PRIVACY_CONSENT='Y')
        env['PYTHONPATH']=':'.join([str(worker_root),str(BASE/'code/grasp'),str(VLN),str(VLN/'nav2_outbound'),str(BASE/'code'),str(worker_root/'vendor'),str(worker_root/'vendor/_vendor_v072'),IMAGE_RUNNER])
        # Isaac retains its own runtime dependencies and Python environment.
        env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib64'
        command=['/opt/robolab-env/bin/python',IMAGE_RUNNER+'/robolab_runner/m20_question_runner.py',
                 '--package-root',IMAGE_PACKAGE+'/task/'+q,'--output-dir',str(out/'episode'),
                 '--policy-endpoint',f'ws://127.0.0.1:{port}','--runtime-script',str(worker_root/'runtime_entry.py'),
                 '--runtime-root',IMAGE_RUNNER+'/m20_runtime','--locomotion-policy',str(args.locomotion_model),
                 '--gpu',str(args.gpu),'--capture-hz','10','--video-fps','10','--vla-action-hz',str(settings['action_hz']),
                 '--maximum-vla-actions',str(task['maximum_actions']),'--maximum-duration-seconds',str(task['duration_s']),
                 '--attack-mode','on','--navigation-mode','hybrid']
        write(out/'LAUNCH.json',{'command':command,'gpu_uuid':resource['uuid'],'runner_sha256':sha(Path(IMAGE_RUNNER)/'m20_runtime/runners/m20_fourview_runner.py')})
        with (out/'runner.log').open('x') as log:
            runner=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='RUNNING',runner_pid=runner.pid);write(out/'STATUS.json',status)
        deadline=time.monotonic()+args.wall_timeout
        while runner.poll() is None:
            if time.monotonic()>deadline: raise RuntimeError('RUNNER_WALL_TIMEOUT')
            if server.poll() is not None: raise RuntimeError('MODEL_SERVER_EXIT_DURING_RUN')
            time.sleep(2)
        if runner.returncode: raise RuntimeError('RUNNER_FAILED: see runner.log')
        submission=out/'episode/submission.hdf5'
        submission.chmod(0o644)
        verification=validate_hdf5(submission)
        verification['video']=validate_video(out/'episode/episode.mp4')
        flow=read(out/'episode/FULL_CHAIN_RESULT.json')
        formal=read(out/'episode/record.json')['protocol_v072_validation']
        if formal['status']!='VALID' or formal['delivery_validation']['status']!='VALID':
            raise RuntimeError('OFFICIAL_HDF5_VALIDATION_FAILED')
        actual_robot=read(out/'episode/runner_preflight.json')['robot']
        expected_robot=next(r['sha256'] for r in bindings['robot']['files'] if r['path']==bindings['robot']['entrypoint'])
        if actual_robot['sha256']!=expected_robot or formal['delivery_validation']['robot']['asset_sha256']!=expected_robot:
            raise RuntimeError('EXECUTED_ROBOT_MISMATCH')
        if not all(flow[k] is True for k in ['continuous_world','no_intermediate_reset','attack_boundary_preserved']):
            raise RuntimeError('CONTINUOUS_RECORDING_CONTRACT_FAILED')
        verification.update(robot=actual_robot,official_protocol_status=formal['status'],continuous_world=True)
        # A normal official runner exit follows its atomic export and validator.
        verification.update(official_runner_exit=runner.returncode,physical_task_completed=flow['flow_completed'])
        write(out/'VERIFICATION.json',verification)
        status.update(status='RECORDED',flow_completed=flow['flow_completed'],submission=str(submission),sha256=verification['sha256'])
    except BaseException as e:
        status.update(status='FAILED',error=str(e));write(out/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
        if isinstance(e,(KeyboardInterrupt,SystemExit)): raise
    finally:
        stop_process(runner);stop_process(server)
        status['finished']=time.time();write(out/'STATUS.json',status)
    return status


def next_attempt(root):
    numbers = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()] if root.exists() else []
    return root / f'{max(numbers, default=0) + 1:03}'


def main():
    p=argparse.ArgumentParser(description='Run the packaged Q01–Q24 baseline or export the supplied example.')
    p.add_argument('--example',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--questions',nargs='+',type=question,default=[f'Q{i:02}' for i in range(1,25)])
    p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--nav-model',type=Path,default=BASE/'models/vln_0915');p.add_argument('--vla-model',type=Path,default=BASE/'models/vla')
    p.add_argument('--locomotion-model',type=Path,default=LOCOMOTION);p.add_argument('--wall-timeout',type=float,default=7200)
    args=p.parse_args()
    if len(set(args.questions))!=len(args.questions):p.error('duplicate questions')
    if args.resume and args.output is None:p.error('--resume requires --output')
    out=(args.output or Path('/opt/safety_embodiment/output/baseline')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists() and not args.resume:p.error('output exists; use --resume or a new directory')
    out.mkdir(parents=True,exist_ok=True)
    run_lock=(out/'.run.lock').open('a')
    try: fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('another process is using this output directory')
    if args.example:
        source=BASE/'examples/HDF5.zip';target=out/'HDF5.zip'
        if target.exists():p.error('example output already exists')
        example=read(BASE/'examples/EXAMPLE_MANIFEST.json')
        expected_names={f'Q{i:02}/submission.hdf5' for i in range(1,25)}
        if set(example['member_paths'])!=expected_names or len(example['member_paths'])!=24:
            raise RuntimeError('INVALID_EXAMPLE_MANIFEST')
        source_sha=sha(source)
        if source.stat().st_size!=example['size_bytes'] or source_sha!=example['sha256']:
            raise RuntimeError('EXAMPLE_SOURCE_HASH_MISMATCH')
        with zipfile.ZipFile(source) as z:
            names=[n for n in z.namelist() if not n.endswith('/')]
            if len(names)!=24 or set(names)!=expected_names or z.testzip() is not None:
                raise RuntimeError('INVALID_EXAMPLE_ARCHIVE')
        temporary=out/'HDF5.zip.partial'
        shutil.copyfile(source,temporary)
        if sha(temporary)!=source_sha:raise RuntimeError('EXAMPLE_COPY_HASH_MISMATCH')
        os.replace(temporary,target)
        write(out/'EXAMPLE_PROVENANCE.json',example)
        print(target,flush=True);return 0
    for attr in ['nav_model','vla_model','locomotion_model']:setattr(args,attr,getattr(args,attr).resolve(strict=True))
    bindings={'nav_model':model_binding(args.nav_model),'grasp_model':model_binding(args.vla_model),'locomotion':binding(args.locomotion_model)}
    robot_manifest=read(BASE/'ROBOT_MANIFEST.json')
    for item in robot_manifest['files']:
        if sha(BASE/item['path']) != item['sha256']:
            raise RuntimeError('ROBOT_ASSET_HASH_MISMATCH: '+item['path'])
    bindings['robot']=robot_manifest
    pre=out/'preflight';pre.mkdir(exist_ok=True);catalog=build(pre);goals=make_goals(catalog)
    profiles = verify_integrated_profiles()
    contract={'version':'0915-final-integrated-20260917','profiles':profiles,'scene_changes':read(BASE/'RAMP_DYNAMIC_CHANGE.json'),'questions':args.questions,'models':bindings,'settings':read(BASE/'config.json'),
              'task_hashes':{q:t['task']['sha256'] for q,t in catalog['questions'].items()},
              'code':{x.name:sha(x) for x in ROOT.glob('*.py')}}
    identity=hash_contract(contract)
    if args.resume:
        previous=read(out/'RUN_MANIFEST.json')
        if previous['contract_sha256']!=identity:raise RuntimeError('RESUME_CONTRACT_MISMATCH')
    else:write(out/'RUN_MANIFEST.json',{'contract':contract,'contract_sha256':identity})
    write(pre/'RESOURCES.json',resources(args.gpu,out,50 if any(catalog['questions'][q]['scene']=='factory' for q in args.questions) else 30))
    if args.preflight:print(pre,flush=True);return 0
    results=[]
    try:
        for q in args.questions:
            prior=list((out/'attempts'/q).glob('*/STATUS.json')) if (out/'attempts'/q).exists() else []
            done=None
            for entry in sorted(prior):
                r=read(entry)
                if r['status']=='RECORDED' and Path(r['submission']).exists() and sha(Path(r['submission']))==r['sha256']:done=r
            if done:results.append(done);continue
            attempt=next_attempt(out/'attempts'/q)
            print(f'{q}: starting',flush=True)
            row=run_one(args,q,catalog['questions'][q],goals,bindings,attempt);results.append(row)
            write(out/'BATCH_RESULT.json',{'questions':results,'requested':len(args.questions),'complete':False})
            print(f"{q}: {row['status']}",flush=True)
        if any(r['status']!='RECORDED' for r in results):
            print('Some recordings failed. See BATCH_RESULT.json; rerun with --resume.',file=sys.stderr);return 1
        folder=out/'HDF5';folder.mkdir(exist_ok=True)
        for row in results:
            src=Path(row['submission']);validate_hdf5(src);shutil.copyfile(src,folder/(row['question']+'.hdf5'))
        temporary=out/'HDF5.zip.partial'
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for row in results:z.write(folder/(row['question']+'.hdf5'),row['question']+'/submission.hdf5')
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise RuntimeError('ZIP_CRC_ERROR')
        os.replace(temporary,out/'HDF5.zip')
        video_temp=out/'MP4.zip.partial'
        with zipfile.ZipFile(video_temp,'w',compression=zipfile.ZIP_STORED) as z:
            for row in results:
                z.write(Path(row['submission']).with_name('episode.mp4'),row['question']+'/episode.mp4')
        with zipfile.ZipFile(video_temp) as z:
            if z.testzip() is not None:raise RuntimeError('VIDEO_ZIP_CRC_ERROR')
        os.replace(video_temp,out/'MP4.zip')
        write(out/'BATCH_RESULT.json',{'questions':results,'requested':len(args.questions),'complete':True,'full_24':len(results)==24,'physical_task_completed_count':sum(r['flow_completed'] for r in results),'zip_sha256':sha(out/'HDF5.zip')})
        print(out/'HDF5.zip',flush=True);return 0
    except KeyboardInterrupt:
        print('Interrupted; recordings retained. Use --resume.',file=sys.stderr);return 130

if __name__=='__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
