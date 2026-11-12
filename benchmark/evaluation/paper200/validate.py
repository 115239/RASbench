"""Collection acceptance is independent of task success and future scoring."""
from pathlib import Path
import json
import subprocess
import sys
import h5py
import numpy as np
from .common import NATIVE, read, write, sha


def validate_episode(root,job):
    out=Path(job['output']);episode=out/'episode'
    protocol=read(episode/'protocol_v072_validation.json')
    if protocol.get('status')!='VALID':raise ValueError('formal HDF5 protocol is invalid')
    result=read(episode/'FULL_CHAIN_RESULT.json')
    if result.get('collection_infrastructure_error'):raise ValueError('runtime infrastructure exception: '+str(result.get('failure')))
    record=read(episode/'record.json') if (episode/'record.json').exists() else {}
    hdf=episode/'submission.hdf5';video=episode/'episode.mp4'
    if not hdf.is_file() or not video.is_file():raise ValueError('required HDF5 or MP4 missing')
    with (out/'video_validation.log').open('w') as log:
        decoded=subprocess.run([str(NATIVE/'ffmpeg'),'-v','error','-xerror','-i',str(video),
                               '-progress','pipe:1','-nostats','-f','null','-'],
                               stdout=subprocess.PIPE,stderr=log,encoding='utf-8',timeout=300)
        log.write(decoded.stdout)
    if decoded.returncode:raise ValueError('MP4 decoding failed')
    video_progress=dict(line.split('=',1) for line in decoded.stdout.splitlines() if '=' in line)
    task=read(root/'benchmark/task'/job['question']/'data/task/task.json')
    runtime_root=Path(read(out/'config.json').get('runtime_snapshot',str(root/'runtime')))
    sys.path.insert(0,str(runtime_root/'runner/m20_runtime'))
    from rubric_evidence_io import validate as validate_rubric
    with h5py.File(hdf,'r') as f:
        if job['purpose'] in ('main','assisted'):
            if f['metadata'].attrs.get('system_id')!=job['system'] or f['metadata'].attrs.get('system_version')!=root.name:
                raise ValueError('base HDF5 system identity does not match actual experiment')
        times=f['data/demo_0/time/state_time_s'][...]
        if len(times)<1 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
            raise ValueError('missing/nonmonotonic physical time')
        physical_times=f['extensions/bboard_rubric_v04/sim_time_s'][...]
        rubric=validate_rubric(f,expected_task_id=task['task_id'],expected_plan=task['rubric_evidence'])
        if len(physical_times)<1 or not np.allclose(np.diff(physical_times),.005,rtol=0,atol=1e-7):
            raise ValueError('physical recording is not at 200 Hz')
        if int(video_progress.get('frame',-1))!=len(times):raise ValueError('video/state frame counts differ')
        if abs(int(video_progress.get('out_time_us',0))/1e6-len(times)/10.)>1e-5:
            raise ValueError('video does not use the frozen 10 FPS clock')
        if not np.allclose(np.diff(times),.1,rtol=0,atol=1e-6):raise ValueError('base camera clock is not 10 Hz')
    events_path=episode/'assurance_v1/events.jsonl'
    if not events_path.is_file():raise ValueError('execution receipt log missing')
    events=[json.loads(line) for line in events_path.read_text().splitlines()]
    from safety_agent.evidence import verify_chain
    from .execution_evidence import inspect as inspect_execution
    execution_chain=verify_chain(events_path)
    execution_chain.update(inspect_execution(events,job['run_id']+':'+str(job['attempts'])))
    receipts=[e for e in events if e.get('type')=='execution_receipt']
    if not receipts:raise ValueError('no physical execution receipts')
    identities=[e for e in events if e.get('type')=='identity']
    expected_mode='on' if job['system']=='groot_agent_on' else 'off'
    if len(identities)!=1 or identities[0]['config']['mode']!=expected_mode:
        raise ValueError('Agent switch evidence mismatch')
    if identities[0]['episode_id']!=job['run_id']+':'+str(job['attempts']):
        raise ValueError('episode identity mismatch')
    if expected_mode=='on' and any(e.get('failed_requests',0)>0 for e in events if e.get('type')=='vision_latency_summary'):
        raise ValueError('vision infrastructure request failed; retain and retry episode')
    from .chain_evidence import inspect
    attack_evidence=inspect(episode,job)
    probe=episode/'HANDLER_PROBE.json'
    if probe.exists()!=(job['purpose']=='probe'):raise ValueError('diagnostic/original-trigger experiment isolation mismatch')
    inputs=list((out/'model').glob('input_*.msgpack'));outputs=list((out/'model').glob('output_*.msgpack'))
    if len(inputs)!=len(outputs) or len(inputs)<result['inference_count']:
        raise ValueError('model input/output evidence incomplete')
    inventory_path=episode/'physics_inventory.hdf5'
    if job['purpose']=='main' and not inventory_path.exists():raise ValueError('main collection requires full physical inventory')
    inventory_count=0
    if inventory_path.exists():
        with h5py.File(inventory_path,'r') as inv:
            inventory_count=len(inv['time_s'])
            if not inv.attrs.get('complete') or not np.array_equal(inv['time_s'][...],physical_times):
                raise ValueError('physical inventory/rubric clocks incomplete or different')
            for name in ['body_pose','body_velocity','joint_position','joint_velocity']:
                d=inv[name]
                if len(d)!=inventory_count:raise ValueError('physical inventory length mismatch')
                for start in range(0,len(d),256):
                    if not np.isfinite(d[start:start+256]).all():raise ValueError('nonfinite physical inventory')
            initial_keys=['initial_body_pose','initial_body_velocity','initial_joint_position']
            if all(k in inv for k in initial_keys):
                paths=json.loads(bytes(inv['body_paths_json'][...]).decode());order=np.argsort(paths)
                joint_names=json.loads(bytes(inv['joint_names_json'][...]).decode());joint_order=np.argsort(joint_names)
                np.savez_compressed(out/'initial_state.npz',body_paths=np.asarray(paths)[order],joint_names=np.asarray(joint_names)[joint_order],
                    body_pose=inv['initial_body_pose'][...][order],body_velocity=inv['initial_body_velocity'][...][order],
                    joint_position=inv['initial_joint_position'][...][joint_order])
            elif job['purpose']=='main':raise ValueError('paired initial-state evidence missing')
    formal_initial=episode/'formal_initial_state.npz'
    if result.get('task_started'):
        if not formal_initial.is_file():raise ValueError('formal initial state missing')
        with np.load(formal_initial,allow_pickle=False) as initial:
            order=np.argsort(initial['body_paths']);joint_order=np.argsort(initial['joint_names'])
            from .formal_reference_pairing import FIELDS
            if any(key not in initial for key in FIELDS):raise ValueError('formal authoritative reference fields missing')
            authoritative={key:initial[key] for key in FIELDS}
            np.savez_compressed(out/'initial_state.npz',body_paths=initial['body_paths'][order],
                joint_names=initial['joint_names'][joint_order],body_pose=initial['body_pose'][order],
                body_velocity=initial['body_velocity'][order],joint_position=initial['joint_position'][joint_order],
                physics_step=initial['physics_step'],sim_time_s=initial['sim_time_s'],
                reference=np.asarray('TASK_RUNNING'),**authoritative)
    if job['purpose'] in ('main','assisted','assisted_pilot'):
        if not (episode/'RUBRIC_CONTACT_POINTS.jsonl').is_file():raise ValueError('raw physical contact evidence missing')
        enabled=result.get('assistance',{}).get('enabled')
        if enabled!=(job['purpose'] in ('assisted','assisted_pilot')):raise ValueError('auxiliary/main isolation mismatch')
    # The legacy exporter is left intact; provenance/receipts use a new extension.
    with h5py.File(hdf,'a') as f:
        g=f.require_group('extensions/paper200_collection')
        identity=read(out/'identity.json')
        for key in ('episode_id','run_id','attempt_id','task_id','system_id','system_version',
                    'condition_id','seed','policy_seed','attack_seed','block_id','repeat_index','purpose'):
            value=identity.get(key)
            if value is None:continue
            if key in f['metadata'].attrs and f['metadata'].attrs[key]!=value:
                raise ValueError('standard identity conflict: '+key)
            f['metadata'].attrs[key]=value
        f['metadata'].attrs['selected_attack_ids']=json.dumps(identity['selected_attack_ids'])
        g.attrs.update(schema='paper200-collection/v2',system=job['system'],question=job['question'],
                       variant=job['variant'],seed=job['seed'],purpose=job['purpose'],score_status='NOT_SCORED')
        for key,value in [('identity_json',identity),('lifecycle_json',read(episode/'TASK_LIFECYCLE.json')),
                          ('provenance_json',job),('input_identity_json',read(out/'input_identity.json')),
                          ('model_identity_json',read(out/'model/READY.json')),
                          ('result_json',result),('execution_events_json',events),('attack_evidence_json',attack_evidence)]:
            if key in g:raise ValueError('refusing to overwrite finalized collection metadata')
            encoded=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            g.create_dataset(key,data=np.frombuffer(encoded,dtype=np.uint8),compression='gzip')
        if probe.exists():
            g.create_dataset('handler_probe_json',data=np.frombuffer(probe.read_bytes(),np.uint8),compression='gzip')
        if formal_initial.is_file():
            initial_group=g.create_group('formal_initial_state')
            with np.load(formal_initial,allow_pickle=False) as initial:
                for key,value in initial.items():
                    if value.dtype.kind=='U':
                        initial_group.create_dataset(key,data=np.asarray(value,dtype=h5py.string_dtype('utf-8')))
                    else:initial_group.create_dataset(key,data=value)
        contract_path=out/'runner_judge_contract.json'
        if contract_path.exists():
            from .evidence_contract import canonical_digest
            contract=read(contract_path)
            g.create_dataset('runner_judge_contract_json',data=np.frombuffer(
                json.dumps(contract,ensure_ascii=False,allow_nan=False).encode(),np.uint8),compression='gzip')
            f['metadata'].attrs['runner_judge_contract_sha256']=canonical_digest(contract)
        for file_name in ('VISUAL_INSTANCE_BINDING.json','DISCRETE_STATE_BINDINGS.json','REFERENCE_STABILITY.json',
                          'REFERENCE_STATE_BINDINGS.json','SAFETY_PHYSICS_PRESERVATION.json'):
            source=episode/file_name
            if source.exists():g.create_dataset(file_name.removesuffix('.json').lower()+'_json',
                data=np.frombuffer(source.read_bytes(),np.uint8),compression='gzip')
        from .geometry_evidence import validate_and_embed as embed_geometry
        embed_geometry(g,episode,physical_times)
        from .export_model_boundaries import embed
        embed(g,episode)
        consumers=episode/'attack_evidence/raw_payloads/consumer_receipts.jsonl'
        if consumers.exists():
            d=g.create_dataset('consumer_receipts_jsonl',shape=(0,),maxshape=(None,),dtype=np.uint8,
                chunks=(1048576,),compression='gzip')
            with consumers.open('rb') as stream:
                for block in iter(lambda:stream.read(1048576),b''):
                    start=len(d);d.resize(start+len(block),axis=0);d[start:]=np.frombuffer(block,np.uint8)
        for file_name,dataset_name in [('EE_COMMAND_CHAIN.jsonl','ee_command_chain_jsonl'),
                                       ('PHYSICAL_OPPORTUNITIES.jsonl','physical_opportunities_jsonl'),
                                       ('DISCRETE_STATE_EVENTS.jsonl','discrete_state_events_jsonl'),
                                       ('REFERENCE_STATE_SOURCE_EVENTS.jsonl','ref_source_events_jsonl')]:
            source=episode/file_name
            d=g.create_dataset(dataset_name,shape=(0,),maxshape=(None,),dtype=np.uint8,
                chunks=(1048576,),compression='gzip')
            d.attrs['source_exists']=source.exists()
            if source.exists():
                with source.open('rb') as stream:
                    for block in iter(lambda:stream.read(1048576),b''):
                        start=len(d);d.resize(start+len(block),axis=0);d[start:]=np.frombuffer(block,np.uint8)
        if inventory_path.exists():
            inventory_group=f.create_group('extensions/paper200_physics')
            with h5py.File(inventory_path,'r') as inv:
                inventory_group.attrs.update(dict(inv.attrs))
                for name in inv:inv.copy(inv[name],inventory_group,name=name)
        contact_points=episode/'RUBRIC_CONTACT_POINTS.jsonl'
        if contact_points.exists():
            d=g.create_dataset('contact_points_jsonl',shape=(0,),maxshape=(None,),dtype=np.uint8,
                chunks=(1048576,),compression='gzip')
            with contact_points.open('rb') as stream:
                for block in iter(lambda:stream.read(1048576),b''):
                    start=len(d);d.resize(start+len(block),axis=0);d[start:]=np.frombuffer(block,np.uint8)
    status={'collection_status':'VALID','score_status':'NOT_SCORED','question':job['question'],
            'system':job['system'],'variant':job['variant'],'simulation_samples':len(physical_times),
            'camera_state_samples':len(times),'simulation_duration_s':float(physical_times[-1]-physical_times[0]),'inference_count':result['inference_count'],
            'termination':result.get('failure'),'task_score':None,'rubric_validation':rubric,
            'receipts':len(receipts),'hdf5':str(hdf),'mp4':str(video),'hdf5_sha256':sha(hdf),'mp4_sha256':sha(video)}
    status['physical_inventory_samples']=inventory_count
    status['execution_chain']=execution_chain
    status['attack_evidence']=attack_evidence
    status['model_wire_calls']=len(inputs)
    status['navigation_network_score_files']=len(list((out/'model').glob('nav_raw_*.npz')))
    write(out/'COLLECTION_VALIDATION.json',status)
    return status


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate_episode(args.root,read(args.attempt/'job.json')),ensure_ascii=False,indent=2))
