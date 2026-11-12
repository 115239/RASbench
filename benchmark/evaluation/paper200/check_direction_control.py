"""Read-only real geometry direction, held target and executed receipt audit."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from .common import read,write,sha,now
from .attack_proof import verify_effect,rows
from .attack_direction_witness import resolve_points_direction


def check(out):
    out=Path(out).resolve();ep=out/'episode';scope=read(ep/'DIRECTION_CONTROL.json')
    original=scope['original_attack'];diagnostic=scope['diagnostic_attack'];issues=[]
    for field in ('id','handler','stage','parameters','duration_s'):
        if original.get(field)!=diagnostic.get(field):issues.append('ORIGINAL_PAYLOAD_OR_DURATION_CHANGED:'+field)
    if scope['original_condition_opportunity_eligible'] is not False or scope['formal_or_assisted_matrix_member'] is not False:
        issues.append('CONTROL_SCOPE_NOT_EXPLICIT')
    files=[ep/'submission.hdf5',ep/'EE_COMMAND_CHAIN.jsonl',ep/'assurance_v1/events.jsonl']
    hashes={str(p):sha(p) for p in files}
    chain=[json.loads(line) for line in files[1].read_text().splitlines() if line.strip()]
    events=[json.loads(line) for line in files[2].read_text().splitlines() if line.strip()]
    with h5py.File(files[0],'r') as f:
        found=[]
        def visit(name,obj):
            if isinstance(obj,h5py.Group) and name.endswith('/'+original['id']) and 'applications' in obj:found.append(obj)
        f.visititems(visit)
        if len(found)!=1:raise ValueError('selected attack group not unique')
        proof=verify_effect(f,found[0],original['stage'],diagnostic)
        if not proof['passed']:issues.extend(proof['issues'])
        witnesses=[r for r in rows(f['extensions/paper200_collection'],'consumer_receipts_jsonl')
            if r.get('consumer')=='attack_direction_resolution' and r.get('attack_id')==original['id']
            and r.get('direction_evidence_kind')=='first_application_geometry']
    if len(witnesses)!=1:issues.append('EXPECTED_ONE_FIRST_APPLICATION_GEOMETRY_WITNESS')
    vectors=[]
    definition=original['parameters']['direction_contract']
    for witness in witnesses:
        direction,_=resolve_points_direction(witness['target_points_world'],witness['support_points_world'],definition['mode'])
        vector=direction*definition['magnitude'];vectors.append(vector)
        if not np.allclose(vector,witness['resolved_world_vector'],rtol=0,atol=1e-9):issues.append('INDEPENDENT_GEOMETRY_DIRECTION_MISMATCH')
    boundaries=[r for r in chain if r['event']=='ee_boundary']
    proposals={r['request_id']:r for r in chain if r['event']=='controlled_direction_proposal'}
    ik={r['request_id']:r for r in chain if r['event']=='ik_and_ordinary_clamp'}
    physical={r['request_id']:r for r in chain if r['event']=='controlled_direction_physical_receipt'}
    links=[r for r in chain if r['event']=='actuator_receipt_link']
    receipts={r['command_id']:r for r in events if r['type']=='execution_receipt'}
    guards={r['command_id']:r for r in events if r['type']=='joint_command'}
    phases=[];previous=np.zeros(3);actual=[];max_error=0.
    if len(boundaries)!=scope['request_count']:issues.append('FIXED_REQUEST_COUNT_NOT_REACHED')
    for row in boundaries:
        rid=row['request_id'];phase=row['target_bias_lifecycle']['phase'];phases.append(phase)
        rotation=np.asarray(row['arm_to_world_rotation']);nominal=np.asarray(row['nominal_action'])
        offset=rotation@(np.asarray(row['attacked_action'])[3:6]-nominal[3:6])
        expected=vectors[0] if vectors and phase in ('apply','hold') else np.zeros(3)
        max_error=max(max_error,float(np.max(np.abs(offset-expected))))
        if not np.allclose(offset,expected,rtol=0,atol=2e-8):issues.append('LIVE_GEOMETRY_TO_EE_OFFSET_MISMATCH')
        incremental=rotation@np.asarray(row['incremental_IK_action'])[3:6]
        if not np.allclose(incremental,rotation@nominal[3:6]+offset-previous,rtol=0,atol=2e-8):issues.append('HELD_TARGET_ACCUMULATION_ERROR')
        previous=offset
        if rid not in proposals or rid not in ik or rid not in physical:
            issues.append('EXECUTION_CHAIN_NODE_MISSING');continue
        if ik[rid].get('ee_command_id')!=row.get('ee_command_id'):issues.append('EE_IK_ID_MISMATCH')
        joined=[r for r in links if r['request_id']==rid]
        if sorted(r['actuator_command_id'] for r in joined)!=sorted(physical[rid]['assurance_receipt_ids']):issues.append('ACTUATOR_LINK_INCOMPLETE')
        for cid in physical[rid]['assurance_receipt_ids']:
            receipt=receipts.get(cid);guard=guards.get(cid)
            if not receipt or not guard:issues.append('ACTUATOR_RECEIPT_MISSING');continue
            if receipt['applied_joint_positions']!=guard['applied_positions']:issues.append('GUARD_RECEIPT_TARGET_MISMATCH')
            actual.append({'request_id':rid,'command_id':cid,'phase':phase,'guard_reasons':guard['reasons'],
                'guard_modified':guard['post_attack_positions']!=guard['applied_positions'] or
                    guard['post_attack_velocities']!=guard['applied_velocities']})
    if any(phase not in phases for phase in ('apply','hold','restore')):issues.append('APPLY_HOLD_RESTORE_INCOMPLETE')
    if not actual:issues.append('NO_REAL_ACTUATOR_RECEIPTS')
    if any(sha(Path(p))!=digest for p,digest in hashes.items()):raise RuntimeError('source changed during read-only verification')
    result={'schema':'paper200-real-direction-control-acceptance/v2','checked':now(),'passed':not issues,
        'issues':sorted(set(issues)),'original_task_conditions':False,'original_trigger_acceptance':False,'formal_task_score':None,
        'strict_effect_proof':proof,'direction_definition':definition,'actual_geometry_witness_count':len(witnesses),
        'resolved_world_vectors':[v.tolist() for v in vectors],'phase_sequence':phases,'maximum_world_offset_error_m':max_error,
        'actual_actuator_receipts':actual,'physical_end_effector_positions':[r['gripper_world_position'] for r in physical.values()],
        'guard_modified_receipts':sum(r['guard_modified'] for r in actual),'source_sha256':hashes,'source_unchanged':True,
        'note':'Offset denotes commanded target bias; ordinary downstream clamp/guard suppression is retained, not an exact physical displacement claim.'}
    write(out/'DIRECTION_CONTROL_ACCEPTANCE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args();r=check(a.attempt)
    print('DIRECTION_CONTROL_ACCEPTANCE',r['passed'],r['issues'])
