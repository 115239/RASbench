"""Verify persisted payload values and their consumer, never status counters."""
import json
import numpy as np


def text(value):
    if isinstance(value,bytes):return value.decode('utf-8')
    if isinstance(value,np.ndarray) and value.dtype==np.uint8:return value.tobytes().decode('utf-8')
    return str(value)


def semantic(value):
    if isinstance(value,dict):
        # Diagnostic annotations/digests cannot establish a physical mutation.
        if value.get('type') in ('ndarray','array') and 'values' not in value:return None
        return {k:semantic(v) for k,v in value.items() if k not in
                ('attack_metadata','sha256','dtype','shape','type','changed','timestamp_s')}
    if isinstance(value,list):return [semantic(x) for x in value]
    return value


def rows(group,name):
    if group is None or name not in group:return []
    return [json.loads(line) for line in text(group[name][()]).splitlines() if line.strip()]


def verify_effect(hdf5,attack,stage,expected_attack=None):
    evidence=[];changed=False;consumed=False;issues=[]
    from .payload_consistency import matches
    application_values={}
    applications=attack.get('applications')
    times=[];steps=[];declared_changed_steps=set();declared_noop_steps=set()
    if applications is not None and all(k in applications for k in ('original_json','attacked_json','physics_step','timestamp_s')):
        count=len(applications['original_json'])
        if any(len(applications[k])!=count for k in ('attacked_json','physics_step','timestamp_s')):
            issues.append('APPLICATION_FIELD_LENGTH_MISMATCH')
        else:
            for index in range(count):
                original_record=json.loads(text(applications['original_json'][index]))
                attacked_record=json.loads(text(applications['attacked_json'][index]))
                before=semantic(original_record)
                after=semantic(attacked_record)
                step=int(applications['physics_step'][index]);time=float(applications['timestamp_s'][index])
                if step>=0 and np.isfinite(time):steps.append(step);times.append(time)
                if original_record==attacked_record:declared_noop_steps.add(step)
                application_values.setdefault(step,[]).append((original_record,attacked_record))
                if before is not None and after is not None and before!=after and step>=0 and np.isfinite(time):
                    changed=True;declared_changed_steps.add(step)
                    evidence.append({'path':applications.name,'row':index,'physics_step':step,'time_s':time})
    from .neural_consumption import model_use
    neural_evidence=[]
    collection=hdf5.get('extensions/paper200_collection')
    exact=[]
    if stage=='immediate_visual_material' and collection is not None:
        # The base protocol stores immediate USD installations as lifecycle
        # events, not per-interception applications. Read the actual captured
        # installation and join its complete material maps to a binding read.
        immediate=attack.get('immediate_effect')
        if immediate is not None and 'lifecycle_event_json' in immediate:
            expected_id=(expected_attack or {}).get('id')
            if expected_id is None and 'metadata/attack_id' in attack:
                expected_id=text(attack['metadata/attack_id'][()])
            for value in immediate['lifecycle_event_json'][...].reshape(-1):
                event=json.loads(text(value))
                if event.get('event')!='visual_material_override_installed':continue
                try:
                    start=int(event['activated_physics_step']);until=int(event['until_physics_step'])
                    before=event['original_materials'];meshes=event['affected_mesh_prims']
                    after={path:event['override_material'] for path in meshes}
                    valid=(expected_id is not None and event['attack_id']==expected_id and start>=0 and until>start and
                           bool(before) and set(before)==set(meshes) and before!=after and
                           isinstance(event['texture_sha256'],str) and len(event['texture_sha256'])==64)
                    matched=any(r.get('consumer')=='usd_renderer_binding' and r.get('receipt_status')=='bindings_read' and
                        r.get('attack_id')==expected_id and r.get('physics_step')==start and
                        r.get('until_physics_step')==until and r.get('original_materials')==before and
                        r.get('actual_materials')==after and r.get('override_material')==event['override_material'] and
                        r.get('texture_sha256')==event['texture_sha256']
                        for r in rows(collection,'consumer_receipts_jsonl'))
                    if not valid or not matched:raise ValueError('installation does not match actual initial binding')
                    steps.append(start);declared_changed_steps.add(start);changed=True
                    exact.append((start,before,after))
                    evidence.append({'path':immediate.name+'/lifecycle_event_json','physics_step':start,
                                     'source':'actual installation maps joined to computed USD binding'})
                except (KeyError,ValueError,TypeError):issues.append('MATERIAL_INSTALL_EVENT_BINDING_MISMATCH')
    if collection is not None:
        for row in rows(collection.get('interception_payloads'),stage):
            context=row.get('runtime_context',{});step=context.get('physics_step')
            if step not in steps:continue
            before=semantic(row.get('original'));after=semantic(row.get('attacked'))
            if before is not None and after is not None and before!=after:
                changed=True;exact.append((int(step),before,after))
                if not any(matches(b,row['original']) and matches(a,row['attacked']) for b,a in application_values.get(int(step),[])):
                    issues.append('APPLICATION_VALUES_DIFFER_FROM_EXACT_INTERCEPTION')
                evidence.append({'path':collection.name+'/interception_payloads/'+stage,'physics_step':step})

    def sensor_join(step,field,before,after,names=None):
        for app_step,original,attacked in exact:
            if app_step!=step or not isinstance(original,dict) or not isinstance(attacked,dict):continue
            try:
                if names is not None and (original.get('joint_names')!=names or attacked.get('joint_names')!=names):continue
                if np.array_equal(np.asarray(original[field],float),before) and np.array_equal(np.asarray(attacked[field],float),after):return True
            except (KeyError,ValueError,TypeError):continue
        return False

    payload_evidence_count=len(evidence)
    if stage in ('control_base','control_joint','sensor_lidar','immediate_visual_material'):
        from .control_consumption_proof import verify
        actual_change,consumed,additional,additional_issues=verify(collection,stage,steps,exact,rows,text)
        changed |= actual_change;evidence.extend(additional);issues.extend(additional_issues)
    elif stage=='sensor_joint_encoder':
        for row in rows(collection,'consumer_receipts_jsonl'):
            if row.get('consumer')!='locomotion_onnx' or row.get('receipt_status')!='inference_returned':continue
            if row.get('physics_step') not in steps:continue
            try:
                original=np.asarray(row['measured_position_rad'],float)
                attacked=np.asarray(row['attacked_position_rad'],float)
                actual=np.asarray(row['consumed_position_rad'],float)
                valid=(original.shape==attacked.shape==actual.shape and original.size>0 and
                    np.isfinite(original).all() and np.isfinite(attacked).all() and
                    np.isfinite(actual).all() and not np.array_equal(original,attacked) and
                    np.isfinite(np.asarray(row['output'],float)).all())
                names=row['joint_names'];indices=row['joint_indices']
                obs=np.asarray(row['observation'],float).reshape(-1)
                start,end=row['position_observation_slice']
                defaults=np.asarray(row['position_default_rad'],np.float32)
                valid &= sensor_join(row['physics_step'],'position_rad',original,attacked,names)
                valid &= (len(names)==len(set(names))==len(indices)==len(original) and
                    names==row['observation_joint_names'][:len(names)] and
                    row['position_observation_slice']==[9,21] and obs.shape==(57,) and
                    np.allclose(obs[start:end],actual.astype(np.float32)-defaults,rtol=0,atol=1e-7))
            except (KeyError,ValueError,TypeError):valid=False
            if valid:
                consumed=True;evidence.append({'path':collection.name+'/consumer_receipts_jsonl',
                    'physics_step':row['physics_step'],'post_Agent_differs':not np.array_equal(actual,attacked)})
    elif stage in ('sensor_imu','sensor_wheel_encoder'):
        for row in rows(collection,'consumer_receipts_jsonl'):
            if row.get('consumer')!='locomotion_onnx' or row.get('receipt_status')!='inference_returned':continue
            if row.get('physics_step') not in steps:continue
            valid=False
            try:
                observation=np.asarray(row['observation'],float).reshape(-1)
                if observation.shape!=(57,) or not np.isfinite(observation).all():continue
                if stage=='sensor_imu':
                    if row['coordinate_frame']!='base_link' or row['angular_velocity_unit']!='rad/s':continue
                    if np.asarray(row['base_to_world_rotation']).shape!=(3,3):continue
                    for field in ('angular_velocity','projected_gravity'):
                        original=np.asarray(row['measured_imu'][field],float)
                        attacked=np.asarray(row['attacked_imu'][field],float)
                        actual=np.asarray(row['consumed_imu'][field],float)
                        start,end=row['imu_observation_slices'][field]
                        scale=row['angular_velocity_observation_scale'] if field=='angular_velocity' else 1.
                        if (sensor_join(row['physics_step'],field,original,attacked) and
                            original.shape==attacked.shape==actual.shape==(3,) and
                            np.isfinite(original).all() and np.isfinite(attacked).all() and
                            not np.array_equal(original,attacked) and
                            np.allclose(observation[start:end],actual.astype(np.float32)*scale,rtol=0,atol=1e-7)):
                            valid=True
                            evidence.append({'physics_step':row['physics_step'],'field':field,
                                'post_Agent_differs':not np.array_equal(actual,attacked),
                                'actual_observation_slice':[start,end]})
                else:
                    before=np.asarray(row['measured_wheel_velocity'],float)
                    after=np.asarray(row['attacked_wheel_velocity'],float)
                    actual=np.asarray(row['consumed_wheel_velocity'],float)
                    valid=(sensor_join(row['physics_step'],'angular_velocity_rad_s',before,after,row['wheel_joint_names']) and
                        before.shape==after.shape==actual.shape==(4,) and
                        len(set(row['wheel_joint_names']))==4 and len(row['wheel_joint_indices'])==4 and
                        np.isfinite(before).all() and np.isfinite(after).all() and
                        not np.array_equal(before,after) and row['wheel_observation_slice']==[37,41] and
                        np.allclose(observation[37:41],actual.astype(np.float32)*.05,rtol=0,atol=1e-7))
                valid &= np.isfinite(np.asarray(row['output'],float)).all()
            except (KeyError,TypeError,ValueError,IndexError):valid=False
            if valid:
                consumed=True;evidence.append({'path':collection.name+'/consumer_receipts_jsonl','physics_step':row['physics_step']})
    elif stage=='state_feedback':
        boundaries=collection.get('model_boundaries') if collection is not None else None
        if boundaries is not None and boundaries.attrs.get('complete'):
            for request in boundaries.values():
                if request.attrs.get('receipt_status')!='inference_returned' or 'raw_actions' not in request:continue
                step=request.attrs.get('physics_step')
                for app_step,before,after in exact:
                    if app_step!=step:continue
                    try:
                        attacked=np.asarray(after,float)
                        delivered=request['state/original'][...]
                        valid=(attacked.shape==delivered.shape==(25,) and np.isfinite(attacked).all() and
                               np.array_equal(attacked.astype(delivered.dtype),delivered))
                    except (ValueError,TypeError,KeyError):valid=False
                    if valid:
                        use=model_use(boundaries,request,'state',text,changed_indices=np.flatnonzero(np.asarray(before)!=np.asarray(after)))
                        neural_evidence.append(use)
                        if not use['service_verified']:continue
                        consumed=True;evidence.append({'path':request.name+'/state','request_id':int(request.attrs['request_id']),'physics_step':int(request.attrs['physics_step']),
                            'post_Agent_differs':not np.array_equal(request['state/consumed'][...],delivered)})
    elif stage=='response_action':
        boundaries=collection.get('model_boundaries') if collection is not None else None
        if boundaries is not None and boundaries.attrs.get('complete'):
            for request in boundaries.values():
                for app_step,before,after in exact:
                    if app_step!=request.attrs.get('physics_step'):continue
                    try:
                        attacked=np.asarray(after.get('actions') if isinstance(after,dict) else after,float)
                        delivered=request['returned_actions'][...]
                        valid=attacked.size>0 and attacked.shape==delivered.shape and np.array_equal(attacked.astype(delivered.dtype),delivered)
                    except (KeyError,TypeError,ValueError):valid=False
                    if valid:
                        consumed=True;evidence.append({'path':request.name+'/returned_actions','request_id':int(request.attrs['request_id']),'physics_step':int(request.attrs['physics_step']),
                            'consumer':'ordinary response adapter input before Agent finish_response'})
    elif stage in ('request_prompt','request_state','request_image'):
        boundaries=collection.get('model_boundaries') if collection is not None else None
        names={'request_prompt':('prompt',),'request_state':('state',),'request_image':('front','rear','wrist')}[stage]
        if boundaries is not None and boundaries.attrs.get('complete'):
            for request in boundaries.values():
                if request.attrs.get('physics_step') not in steps:continue
                if request.attrs.get('receipt_status')!='inference_returned' or 'raw_actions' not in request:continue
                for name in names:
                    if name not in request or any(k not in request[name] for k in ('original','attacked','consumed')):continue
                    values=request[name];before=values['original'][()];after=values['attacked'][()]
                    if np.shape(before)!=np.shape(after) or np.array_equal(before,after):continue
                    def plain(v):return text(v) if isinstance(v,bytes) else v
                    if not any(matches(b,plain(before)) and matches(a,plain(after))
                        for b,a in application_values.get(int(request.attrs['physics_step']),[])):
                        issues.append('APPLICATION_VALUES_DIFFER_FROM_MODEL_BOUNDARY')
                    use=model_use(boundaries,request,name,text)
                    neural_evidence.append(use)
                    if not use['service_verified']:continue
                    consumed=True;changed=True
                    evidence.append({'path':values.name,'request_id':int(request.attrs['request_id']),'physics_step':int(request.attrs['physics_step']),
                        'post_Agent_differs':not np.array_equal(values['consumed'][()],after)})
    elif stage=='control_ee_target':
        records=rows(collection,'ee_command_chain_jsonl')
        ik={r['request_id']:r for r in records if r.get('event')=='ik_and_ordinary_clamp'}
        for row in records:
            if row.get('event')!='ee_boundary' or row.get('physics_step') not in steps:continue
            if row.get('request_id') not in ik:continue
            before=np.asarray(row['nominal_action']);after=np.asarray(row['attacked_action'])
            if before.shape==after.shape==(10,) and not np.array_equal(before,after):
                # Proof of reaching the ordinary adapter; final actuator receipt
                # remains separately required for the whole execution chain.
                consumed=True;evidence.append({'path':collection.name+'/ee_command_chain_jsonl','request_id':row['request_id'],'physics_step':row['physics_step']})
    # Unsupported/missing consumers remain unavailable, including legacy files.
    # Persistent material effects need their actual material/render bindings;
    # an immediate_executed marker alone deliberately cannot certify them.
    verified_steps={int(e.get('frame_physics_step',e.get('physics_step',-1)))
                    for e in evidence[payload_evidence_count:]}
    if (verified_steps & declared_noop_steps) - declared_changed_steps:
        issues.append('APPLICATION_PAYLOAD_CONTRADICTS_CAPTURED_MUTATION')
    if stage!='immediate_visual_material' and consumed:
        missing=sorted((declared_changed_steps|{step for step,_,_ in exact})-verified_steps)
        if missing:issues.append('CHANGED_APPLICATION_CONSUMER_COVERAGE_INCOMPLETE')
    else:missing=[]
    if not changed:issues.append('PHYSICAL_PAYLOAD_CHANGE_NOT_PROVEN')
    if changed and not consumed:
        issues.append('ACTUAL_CONSUMER_RECEIPT_NOT_PROVEN')
        if stage=='sensor_imu' and any(isinstance(before,dict) and isinstance(after,dict) and
            before.get('linear_acceleration')!=after.get('linear_acceleration') and
            before.get('angular_velocity')==after.get('angular_velocity') and
            before.get('projected_gravity')==after.get('projected_gravity') for _,before,after in exact):
            issues.append('LINEAR_ACCELERATION_NOT_CONSUMED_BY_LOCOMOTION_ONNX')
    if stage in ('request_prompt','request_state','request_image','state_feedback') and changed:
        if not any(v.get('neural_verified') for v in neural_evidence):
            issues.append('ACTUAL_NEURAL_MODALITY_CONSUMPTION_NOT_PROVEN')
    parameter_proof={'passed':None,'status':'not_requested'}
    if expected_attack is not None:
        from .attack_parameter_proof import verify as verify_parameters
        parameter_proof=verify_parameters(collection,attack,expected_attack,stage,exact,rows,text)
        if not parameter_proof['passed']:issues.extend(parameter_proof['issues'])
    return {'passed':changed and consumed and not issues,'payload_changed':changed,
            'parameter_proof':parameter_proof,'neural_consumption':neural_evidence,
            'consumer_verified':consumed and not missing,'issues':issues,'evidence':evidence,
            'changed_application_steps':sorted(declared_changed_steps),
            'verified_consumer_steps':sorted(verified_steps-{ -1 }),
            'missing_consumer_steps':missing,
            'parameter_and_hold_semantics':'consumer proof only; frozen handler parameter review and scheduled opportunity checks remain separate'}
