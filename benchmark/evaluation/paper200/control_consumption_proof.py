"""Value-level joins for control, LiDAR and persistent material evidence.

No physical-success claim is inferred from apply_action returning. That receipt
establishes delivery; subsequent safety/trajectory evidence remains independent.
"""
import hashlib
import json
import numpy as np


def finite(value,shape=None):
    value=np.asarray(value,float)
    return (shape is None or value.shape==shape) and np.isfinite(value).all()


def verify(collection,stage,steps,exact,rows,text):
    evidence=[];issues=[];changed=False;consumed=False
    records=rows(collection,'consumer_receipts_jsonl')
    def identity_join(row):
        command_id=row.get('command_id')
        if not command_id:return False
        try:
            step=int(row['physics_step']);dt=float(row['physics_dt_s']);time=float(row['sim_time_s'])
            if dt<=0 or not np.isclose(time,step*dt,rtol=0,atol=1e-9):return False
            return any(r.get('runtime_context',{}).get('command_id')==command_id and
                r['runtime_context'].get('physics_step')==step and
                np.isclose(float(r['runtime_context']['time_s']),time,rtol=0,atol=1e-9)
                for r in rows(collection.get('interception_payloads'),stage))
        except (KeyError,ValueError,TypeError):return False

    if stage=='control_base':
        for row in records:
            if row.get('consumer')!='locomotion_onnx' or row.get('receipt_status')!='inference_returned':continue
            step=row.get('physics_step')
            try:
                before=np.asarray(row['original_base_command'],np.float32)
                after=np.asarray(row['attacked_base_command'],np.float32)
                actual=np.asarray(row['consumed_base_command'],np.float32)
                obs=np.asarray(row['observation'],np.float32).reshape(-1)
                joined=any(s==step and np.array_equal(np.asarray(b,np.float32),before) and
                    np.array_equal(np.asarray(a,np.float32),after) for s,b,a in exact)
                valid=(identity_join(row) and joined and finite(before,(3,)) and finite(after,(3,)) and finite(actual,(3,)) and
                    not np.array_equal(before,after) and finite(obs,(57,)) and
                    row['base_command_frame']=='base_link' and row['base_command_units']==['m/s','m/s','rad/s'] and
                    row['base_command_observation_slice']==[6,9] and
                    np.array_equal(obs[6:9],actual*np.asarray(row['base_command_scale'],np.float32)) and
                    np.array_equal(obs[6:9],np.asarray(row['policy_base_command'],np.float32)) and finite(row['output']))
            except (KeyError,ValueError,TypeError):valid=False
            if valid:
                changed=consumed=True;evidence.append({'physics_step':step,'consumer':'locomotion_onnx',
                    'actual_observation_slice':[6,9],'post_Agent_differs':not np.array_equal(actual,after)})
    elif stage=='control_joint':
        for row in records:
            if row.get('consumer')!='articulation_controller.apply_action' or row.get('receipt_status')!='apply_action_returned':continue
            step=row.get('physics_step')
            try:
                before=row['original'];after=row['attacked'];clamped=row['ordinary_clamped'];actual=row['consumed']
                valid=identity_join(row) and any(s==step and b==before and a==after for s,b,a in exact) and before!=after
                names=row['joint_names'];indices=row['joint_indices']
                valid &= len(set(names))==len(names)==len(set(indices))==len(indices)
                for kind in ('positions','wheel_velocities'):
                    valid &= set(before[kind])==set(after[kind])==set(clamped[kind])==set(actual[kind])
                    valid &= set(before[kind]).issubset(names)
                    valid &= all(finite(list(v[kind].values())) for v in (before,after,clamped,actual))
                valid &= row['coordinate_frame']=='articulation_joint_local' and row['velocity_units']=='rad/s'
                valid &= set(row['position_units'])==set(before['positions'])
                # A complete M20+Piper transaction must include all six arm DOFs
                # and both fingers, not just an isolated attacked submapping.
                valid &= all('joint%d'%i in before['positions'] for i in range(1,9))
            except (KeyError,ValueError,TypeError):valid=False
            if valid:
                changed=consumed=True;evidence.append({'physics_step':step,'consumer':row['consumer'],
                    'post_Agent_differs':actual!=clamped,'ordinary_clamp_differs':clamped!=after})
    elif stage=='sensor_lidar' and collection is not None:
        payloads=collection.get('lidar_payloads');consumers=collection.get('lidar_consumers')
        if payloads is not None and consumers is not None:
            for sample in payloads.values():
                try:
                    step=int(sample['physics_step'][()]);before=sample['original_points'][...];after=sample['attacked_points'][...]
                    if step not in steps or np.array_equal(before,after):continue
                    if not(finite(before) and finite(after) and before.ndim==after.ndim==2 and before.shape[1]==after.shape[1]==3):continue
                    changed=True
                    for receipt in consumers.values():
                        if int(receipt['frame_physics_step'][()])!=step:continue
                        actual=receipt['points'][...]
                        valid=(np.array_equal(after.astype(actual.dtype),actual) and finite(receipt['command'][...],(3,)) and
                            int(receipt['physics_step'][()])>=step and
                            text(receipt['frame_id'][()])==text(sample['attacked_frame_id'][()])=='m20_rtx_lidar' and
                            np.array_equal(receipt['sensor_to_world'][...],sample['attacked_sensor_to_world'][...]) and
                            finite(receipt['sensor_to_world'][...],(4,4)) and
                            float(receipt['timestamp_s'][()])==float(sample['attacked_timestamp_s'][()]))
                        if valid:
                            consumed=True;evidence.append({'path':receipt.name,'payload_path':sample.name,
                                'consumer':'M20LidarLocalPlanner.command','frame_physics_step':step,
                                'consumed_physics_step':int(receipt['physics_step'][()])})
                except (KeyError,ValueError,TypeError):continue
    elif stage=='immediate_visual_material' and collection is not None:
        boundaries=collection.get('model_boundaries')
        terminal=None
        if 'result_json' in collection:
            terminal=json.loads(text(collection['result_json'][()])).get('termination_physics_step')
        for row in records:
            if row.get('consumer')!='rendered_policy_observation' or row.get('receipt_status')!='pixels_read':continue
            try:
                step=row['physics_step'];before=row['original_materials'];actual=row['actual_materials'];override=row['override_material']
                if before==actual or set(before)!=set(actual) or not actual or any(v!=override for v in actual.values()):continue
                if not any(s<=step and b==before and a==actual for s,b,a in exact):continue
                changed=True
                bindings={r['physics_step']:r for r in records if r.get('consumer')=='usd_renderer_binding' and
                          r.get('attack_id')==row['attack_id']}
                begin=min((s for s in steps if s<=step),default=None)
                end=min(int(row['until_physics_step']),int(terminal)) if terminal is not None else None
                if begin is None or end is None:continue
                # Every physics step while active is read from actual computed
                # USD bindings. Sparse all-clear markers cannot prove a hold.
                covered=all(s in bindings and bindings[s]['actual_materials']==actual for s in range(begin+1,end+1))
                if not covered:
                    issues.append('MATERIAL_HOLD_COVERAGE_INCOMPLETE');continue
                if boundaries is None or not boundaries.attrs.get('complete'):continue
                for request in boundaries.values():
                    if int(request.attrs.get('physics_step',-1))!=step or request.attrs.get('receipt_status')!='inference_returned':continue
                    valid=True
                    for name,description in row['images'].items():
                        image=request[name]['original'][...]
                        valid &= list(image.shape)==description['shape'] and str(image.dtype)==description['dtype']
                        valid &= hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()==description['sha256']
                    if valid and row['images']:
                        from .neural_consumption import model_use
                        neural=[model_use(boundaries,request,name,text) for name in row['images']]
                        if not any(x.get('neural_verified') for x in neural):continue
                        consumed=True;evidence.append({'path':request.name,'consumer':'rendered_policy_observation',
                            'binding_hold_interval':[begin,end],'rendered_physics_step':step,
                            'scope':'actual material binding and rendered policy pixels; target visibility is separate'})
            except (KeyError,ValueError,TypeError):continue
    return changed,consumed,evidence,list(dict.fromkeys(issues))
