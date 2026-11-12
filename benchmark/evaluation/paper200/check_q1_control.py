"""Verify the Q1 original-trigger controlled run from real immutable evidence."""
import argparse
from pathlib import Path
import numpy as np
from .common import read,write,sha,now,digest


def rows(path):
    import json
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def check(out):
    ep=out/'episode';root=Path(read(out/'config.json')['collection_root']);issues=[]
    source=root/'benchmark/task/Q1/data/attack/profiles/q1_state_feedback_01.json'
    contract=read(out/'runner_judge_contract.json')
    original_profile=next(r['profile'] for r in contract['attacks'] if r['path'].endswith('/q1_state_feedback_01.json'))
    original=original_profile['attack']['attacks'][0]
    resolved=ep/'resolved_attack_profile/all_attack_definitions.json'
    selected=[r for r in read(resolved)['attack']['attacks'] if r.get('enabled') and r.get('effect_enabled',True)]
    if len(selected)!=1 or any(selected[0].get(k)!=original.get(k) for k in ('id','stage','handler','trigger','parameters','duration_s','once')):
        issues.append('ORIGINAL_TRIGGER_OR_PAYLOAD_CHANGED')
    audit=rows(ep/'attack_evidence/unified_attack_audit.jsonl')
    triggers=[r for r in audit if r['event']=='condition_triggered' and r.get('attack_id')==original['id']]
    if len(triggers)!=1:issues.append('EXPECTED_ONE_ORIGINAL_TRIGGER')
    trigger=triggers[0] if triggers else None
    if trigger:
        snap=trigger['evidence']['observed_snapshot']
        actual_distance=float(np.linalg.norm(np.asarray(snap['robot']['position_xyz'])[:2]-np.asarray(snap['objects']['target_0']['position_xyz'])[:2]))
        if actual_distance>1. or not snap['locomotion']['walking'] or snap['locomotion']['horizontal_speed_mps']<=.01:
            issues.append('ACTUAL_WALKING_DISTANCE_PREREQUISITES_NOT_MET')
    else:actual_distance=None
    payload=rows(ep/'attack_evidence/raw_payloads/sensor_joint_encoder.jsonl')
    all_consumers=rows(ep/'attack_evidence/raw_payloads/consumer_receipts.jsonl')
    consumers={r['physics_step']:r for r in all_consumers
        if r.get('consumer')=='locomotion_onnx' and r.get('receipt_status')=='inference_returned'}
    applied={r['command_id']:r for r in all_consumers if r.get('consumer')=='articulation_controller.apply_action'
        and r.get('receipt_status')=='apply_action_returned'}
    execution={r['command_id']:r for r in rows(ep/'assurance_v1/events.jsonl') if r.get('type')=='execution_receipt'}
    changed=[];held=[];restored=[];joined=[];window=[];reference=None
    if trigger:
        t0=float(trigger['time_s']);t1=t0+float(original['duration_s'])
        for row in payload:
            stamp=row['runtime_context']['time_s'];step=row['runtime_context']['physics_step']
            names=row['original']['joint_names'];index=names.index(original['parameters']['joint_name'])
            before=np.asarray(row['original']['position_rad']);after=np.asarray(row['attacked']['position_rad'])
            if t0-1e-8<=stamp<t1-1e-8:
                if reference is None:reference=float(before[index])
                if abs(after[index]-reference)>1e-8:issues.append('HOLD_TRIGGER_VALUE_DRIFT')
                other=np.arange(len(names))!=index
                if not np.array_equal(before[other],after[other]):issues.append('NONSELECTED_ENCODERS_CHANGED')
                held.append(float(after[index]));window.append({'step':step,'time_s':stamp})
                if not np.array_equal(before,after):changed.append(step)
                c=consumers.get(step)
                if not c:issues.append('CONSUMER_RECEIPT_MISSING');continue
                obs=np.asarray(c['observation']).reshape(-1)
                consumed=np.asarray(c['consumed_position_rad'],np.float32);defaults=np.asarray(c['position_default_rad'],np.float32)
                if (c['joint_names']!=names or obs.shape!=(57,) or not np.array_equal(before,np.asarray(c['measured_position_rad']))
                    or not np.array_equal(after,np.asarray(c['attacked_position_rad']))
                    or not np.array_equal(after,np.asarray(c['consumed_position_rad']))
                    or not np.allclose(obs[9:21],consumed-defaults,rtol=0,atol=1e-7)
                    or not np.isfinite(c['output']).all()):issues.append('ENCODER_ONNX_OBSERVATION_JOIN_FAILED')
                physical=applied.get(c.get('command_id'));matches=[]
                if not physical or physical['physics_step']!=step:
                    issues.append('CONSUMER_ACTUATOR_RECEIPT_LINK_MISSING')
                else:
                    # The SDK transaction ID and Agent receipt ID are separate
                    # namespaces. Join SDK return by its exact ID, then require
                    # one same-time Agent receipt with identical named targets.
                    names_to_indices=dict(zip(physical['joint_names'],physical['joint_indices']))
                    for receipt in execution.values():
                        if abs(receipt['start_s']-physical['sim_time_s'])>1e-8:continue
                        good=True
                        for field,key in (('positions','applied_joint_positions'),('wheel_velocities','applied_joint_velocities')):
                            for name,value in physical['consumed'][field].items():
                                actual=receipt[key][names_to_indices[name]]
                                if actual is None or abs(actual-value)>1e-7:good=False
                        if good:matches.append(receipt['command_id'])
                    if len(matches)!=1:issues.append('SDK_AGENT_PHYSICAL_RECEIPT_VALUE_JOIN_FAILED')
                joined.append({'physics_step':step,'command_id':c.get('command_id'),'actual_output':c['output'],
                    'sdk_apply_returned':physical is not None,'Agent_receipt_id':matches[0] if len(matches)==1 else None,
                    'Agent_join':'unique equal start-time plus all consumed named positions/wheel velocities'})
            elif t1-1e-8<=stamp<=t1+.1:
                if not np.array_equal(before,after):issues.append('ENCODER_NOT_RESTORED_AFTER_HALF_SECOND')
                restored.append(step)
        if len(window)!=25 or (window and abs(window[-1]['time_s']-window[0]['time_s']-.48)>1e-7):
            issues.append('FULL_0_5_SECOND_50HZ_HOLD_NOT_PROVEN')
    if not changed:issues.append('NO_ACTUAL_ENCODER_MUTATION')
    if not restored:issues.append('NO_POST_WINDOW_RESTORATION_RECEIPT')
    geometry=read(ep/'Q1_PREMOVEMENT_GEOMETRY.json')
    if geometry['world_aabb_overlap']:issues.append('PREMOVEMENT_TARGET_OVERLAP')
    result={'schema':'paper200-q1-controlled-acceptance/v1','checked':now(),'passed':not issues,'issues':sorted(set(issues)),
        'scope':'Full Q1 real scene, real model/ONNX/control receipts, changed initial pose and explicit movement proposal; not original-condition task success',
        'original_trigger':trigger,'actual_trigger_distance_m':actual_distance,'measured_trigger_joint_reference_rad':reference,
        'hold_consumers':joined,'hold_window':window,'changed_encoder_steps':changed,'post_window_restored_steps':restored,
        'geometry':geometry,'formal_task_success':None,'current_source_profile_sha256':sha(source),
        'frozen_attempt_contract_sha256':sha(out/'runner_judge_contract.json'),
        'frozen_original_profile_digest':digest(original_profile),'resolved_profile_sha256':sha(resolved),
        'source_evidence_sha256':{name:sha(ep/name) for name in ('Q1_CONTROL_PROPOSALS.jsonl',
            'attack_evidence/raw_payloads/sensor_joint_encoder.jsonl','attack_evidence/raw_payloads/consumer_receipts.jsonl',
            'attack_evidence/unified_attack_audit.jsonl','assurance_v1/events.jsonl')}}
    write(out/'Q1_CONTROL_ACCEPTANCE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    result=check(a.attempt);print('Q1_CONTROL',result['passed'],result['issues'])
