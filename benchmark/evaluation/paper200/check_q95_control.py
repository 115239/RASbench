"""Reconstruct controlled Q95 injection and motor receipts from immutable logs."""
import argparse
from pathlib import Path
import numpy as np
from .common import read,write,sha,now


def rows(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check(out):
    ep=out/'episode';chain=rows(ep/'EE_COMMAND_CHAIN.jsonl')
    boundaries=[r for r in chain if r['event']=='ee_boundary']
    opportunity=[r for r in chain if r['event']=='opportunity']
    ik={r['request_id']:r for r in chain if r['event']=='ik_and_ordinary_clamp'}
    physical={r['request_id']:r for r in chain if r['event']=='controlled_physical_receipt'}
    proposals={r['request_id']:r for r in chain if r['event']=='controlled_proposal'}
    links=[r for r in chain if r['event']=='actuator_receipt_link']
    events=rows(ep/'assurance_v1/events.jsonl')
    receipts={r['command_id']:r for r in events if r['type']=='execution_receipt'}
    guarded={r['command_id']:r for r in events if r['type']=='joint_command'}
    issues=[];phases=[];previous=np.zeros(3);actual_receipts=[];max_error=0.;measured_closure=[]
    cfg=read(out/'config.json');root=Path(cfg['collection_root'])
    profile_path=root/'benchmark/task/Q95/data/attack/profiles/q95_arm_control_01.json'
    resolved_path=ep/'resolved_attack_profile/all_attack_definitions.json'
    original=read(profile_path)['attack']['attacks'][0]
    selected=[a for a in read(resolved_path)['attack']['attacks'] if a.get('enabled') and a.get('effect_enabled',True)]
    original_fields=('id','stage','handler','trigger','parameters','duration_s','once')
    if len(selected)!=1 or any(selected[0].get(k)!=original.get(k) for k in original_fields):
        issues.append('ORIGINAL_TRIGGER_PAYLOAD_DURATION_CHANGED')
    for row in boundaries:
        rid=row['request_id'];rotation=np.asarray(row['arm_to_world_rotation'])
        before=np.asarray(row['nominal_action']);after=np.asarray(row['attacked_action'])
        offset=rotation@(after[3:6]-before[3:6]);phase=row['target_bias_lifecycle']['phase']
        phases.append(phase)
        expected=np.array([0.,.02,0.]) if phase in ('apply','hold') else np.zeros(3)
        max_error=max(max_error,float(np.max(abs(offset-expected))))
        if not np.allclose(offset,expected,atol=2e-8):issues.append('WORLD_OFFSET_MISMATCH:%s'%rid)
        incremental=rotation@np.asarray(row['incremental_IK_action'])[3:6]
        if not np.allclose(incremental,rotation@before[3:6]+offset-previous,atol=2e-8):issues.append('INCREMENTAL_TARGET_ACCUMULATION:%s'%rid)
        previous=offset
        if rid not in ik or rid not in physical or rid not in proposals:
            issues.append('CHAIN_NODE_MISSING:%s'%rid);continue
        if row.get('ee_command_id')!=ik[rid].get('ee_command_id'):
            issues.append('EE_COMMAND_ID_MISMATCH:%s'%rid)
        if links:
            joined=[r for r in links if r['request_id']==rid]
            if sorted(r['actuator_command_id'] for r in joined)!=sorted(physical[rid]['assurance_receipt_ids']):
                issues.append('EE_ACTUATOR_COMMAND_LINK_INCOMPLETE:%s'%rid)
            if any(r.get('ee_command_id')!=row.get('ee_command_id') for r in joined):
                issues.append('EE_ACTUATOR_COMMAND_ID_MISMATCH:%s'%rid)
        measured_closure.append(physical[rid]['measured_gripper_close']-proposals[rid]['measured_gripper_close'])
        for cid in physical[rid]['assurance_receipt_ids']:
            receipt=receipts.get(cid);guard=guarded.get(cid)
            if not receipt or not guard:
                issues.append('ACTUATOR_RECEIPT_MISSING:%s'%cid);continue
            if receipt['applied_joint_positions']!=guard['applied_positions']:
                issues.append('GUARD_RECEIPT_TARGET_MISMATCH:%s'%cid)
            actual_receipts.append({'request_id':rid,'command_id':cid,'start_s':receipt['start_s'],
                'end_s':receipt['end_s'],'guard_reasons':guard['reasons'],
                'guard_applied_target_changed':guard['post_attack_positions']!=guard['applied_positions']
                    or guard['post_attack_velocities']!=guard['applied_velocities']})
    if len(opportunity)!=1:issues.append('EXPECTED_ONE_REAL_CLOSING_OPPORTUNITY')
    if any(p not in phases for p in ('apply','hold','restore')):issues.append('APPLY_HOLD_RESTORE_NOT_ALL_OBSERVED')
    active=[r for r in boundaries if r['target_bias_lifecycle']['phase'] in ('apply','hold')]
    restored=[r for r in boundaries if r['target_bias_lifecycle']['phase']=='restore']
    duration=restored[0]['sim_time_s']-active[0]['sim_time_s'] if active and restored else None
    if duration is None or abs(duration-.5)>1e-7:issues.append('SOURCE_HALF_SECOND_DURATION_NOT_PROVEN')
    # Closure is witnessed in actual physical finger feedback, not a proposal.
    if not measured_closure or max(measured_closure)<=1e-4:issues.append('ACTUAL_FINGER_CLOSURE_NOT_OBSERVED')
    if not actual_receipts:issues.append('NO_ACTUAL_ACTUATOR_RECEIPTS')
    files=['EE_COMMAND_CHAIN.jsonl','assurance_v1/events.jsonl','FULL_CHAIN_RESULT.json']
    result={'schema':'paper200-q95-controlled-acceptance/v1','checked':now(),'passed':not issues,
        'scope':'real Q95 physics/model; explicit controlled proposal, not original task success or formal trial',
        'attack_trigger_and_profile_unchanged':'ORIGINAL_TRIGGER_PAYLOAD_DURATION_CHANGED' not in issues,
        'resolved_profile_sha256':sha(resolved_path),'source_profile_sha256':sha(profile_path),
        'issues':sorted(set(issues)),
        'phase_sequence':phases,'actual_closing_opportunities':opportunity,
        'source_hold_seconds':.5,'observed_hold_seconds':duration,'maximum_world_offset_error_m':max_error,
        'maximum_actual_closure_fraction_increment':max(measured_closure) if measured_closure else None,
        'explicit_ee_to_actuator_links':len(links),
        'post_guard_modified_attack_receipts':sum(r['guard_applied_target_changed'] for r in actual_receipts
            if any(b['request_id']==r['request_id'] and b['target_bias_lifecycle']['phase'] in ('apply','hold') for b in boundaries)),
        'actual_actuator_receipts':actual_receipts,
        'source_sha256':{p:sha(ep/p) for p in files},'formal_task_success':None,
        'note':'Measured end-effector displacement is retained separately; an injected target offset is not a claim of exact physical tracking.'}
    write(out/'Q95_CONTROL_ACCEPTANCE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    result=check(a.attempt);print('Q95_CONTROL',result['passed'],result['issues'])
