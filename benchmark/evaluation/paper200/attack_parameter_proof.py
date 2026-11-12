"""Independent numeric parameter checks; unsupported semantics stay unknown."""
import json
import numpy as np


def equal(a,b):
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    try:
        x=np.asarray(a);y=np.asarray(b)
        if x.shape!=y.shape:return False
        if x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf':return np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,rtol=0,atol=2e-6)
        return np.array_equal(x,y)
    except (ValueError,TypeError):return a==b


def verify(collection,attack,spec,stage,exact,rows,text):
    if stage in ('sensor_lidar','request_image','immediate_visual_material'):
        from .attack_parameter_extra import verify as geometric_parameters
        return geometric_parameters(collection,attack,spec,stage,exact,rows,text)
    handler=spec.get('handler');p=spec.get('parameters',{});issues=[];checked=0
    if spec.get('stage')!=stage:return {'passed':False,'issues':['EXPECTED_ATTACK_STAGE_MISMATCH']}
    timing=attack.get('timing')
    activation=float(timing['activation_time_s'][()]) if timing is not None and 'activation_time_s' in timing else None
    records=rows(collection,'consumer_receipts_jsonl')
    boundaries=collection.get('model_boundaries') if collection is not None else None
    if handler in ('replace_prompt','replace_prompt_sequence'):
        if boundaries is not None:
            for request in boundaries.values():
                if 'prompt' not in request or np.array_equal(request['prompt/original'][()],request['prompt/attacked'][()]):continue
                actual=text(request['prompt/attacked'][()]);time=float(request.attrs['sim_time_s'])
                if handler=='replace_prompt':wanted=p['replacement']
                elif activation is None:issues.append('PARAMETER_ACTIVATION_CLOCK_MISSING');continue
                else:
                    sequence=p['steps'];elapsed=time-activation
                    candidates=[x for x in sequence if elapsed+1e-7>=float(x['at_s'])]
                    if not candidates:issues.append('PROMPT_BEFORE_SCHEDULE');continue
                    wanted=candidates[-1]['replacement']
                    if p.get('delivery')=='once_per_scheduled_request' and abs(elapsed-float(candidates[-1]['at_s']))>1e-6:issues.append('PROMPT_SCHEDULE_TIME_MISMATCH')
                checked+=1
                if actual!=wanted:issues.append('PROMPT_REPLACEMENT_PARAMETER_MISMATCH')
            if handler=='replace_prompt_sequence' and p.get('delivery')=='once_per_scheduled_request' and checked!=len(p['steps']):
                issues.append('PROMPT_SCHEDULE_ROUND_COUNT_MISMATCH')
    elif handler=='paper200_world_action_offset':
        try:expected=direction_vector(spec,records) if p.get('direction_contract') else np.asarray(p['world_offset_m'],float)
        except (KeyError,ValueError,TypeError,IndexError):return {'passed':False,'checked_rows':0,'handler':handler,'issues':['DIRECTION_GEOMETRY_PARAMETER_EVIDENCE_INVALID']}
        previous=np.zeros(3)
        chain=rows(collection,'ee_command_chain_jsonl');active=[];restored=False
        for row in chain:
            if row.get('event')!='ee_boundary':continue
            try:
                nominal=np.asarray(row['nominal_action'],float);after=np.asarray(row['attacked_action'],float)
                rotation=np.asarray(row['arm_to_world_rotation'],float);offset=rotation@(after[3:6]-nominal[3:6])
                phase=row['target_bias_lifecycle']['phase']
                if phase in ('apply','hold'):
                    checked+=1;active.append(float(row['sim_time_s']))
                    if not equal(offset,expected):issues.append('WORLD_EE_OFFSET_PARAMETER_MISMATCH')
                    if not equal(np.delete(after,[3,4,5]),np.delete(nominal,[3,4,5])):issues.append('WORLD_EE_OFFSET_MODIFIED_OTHER_AXES')
                if phase=='restore':restored=True
                incremental=np.asarray(row['incremental_IK_action'],float)
                if not equal(rotation@incremental[3:6],rotation@nominal[3:6]+offset-previous):issues.append('WORLD_EE_BIAS_ACCUMULATED_OR_RESTORE_MISMATCH')
                if not equal(row['target_bias_lifecycle']['commanded_world_target_bias_m'],offset):issues.append('WORLD_EE_HELD_TARGET_MISMATCH')
                previous=offset
            except (KeyError,ValueError,TypeError):issues.append('WORLD_EE_PARAMETER_EVIDENCE_MISSING')
        if active and not restored:
            # Early formal stop cancels future motion; it is not a fabricated
            # restoration and does not invalidate a correctly injected prefix.
            result=json.loads(text(collection['result_json'][()])) if 'result_json' in collection else {}
            stop=result.get('termination_time_s');duration=spec.get('duration_s')
            if stop is None or activation is None or duration is None or float(stop)>=activation+float(duration)+1e-6:
                issues.append('WORLD_EE_RESTORATION_NOT_PROVEN')
    else:
        invocations={r.get('physics_step'):r for r in records if r.get('consumer')=='attack_handler_boundary' and
            r.get('attack_id')==spec.get('id') and r.get('stage')==stage}
        unsupported=False
        for step,before,after in exact:
            context=invocations.get(step,{});snapshot=context.get('snapshot',{});runtime=context.get('runtime',{})
            try:
                if handler=='affine_actions':
                    wanted=np.asarray(before,float).copy();indices=p.get('indices',list(range(wanted.shape[-1])))
                    wanted[...,indices]=wanted[...,indices]*np.asarray(p.get('scale',1.))+np.asarray(p.get('offset',0.))
                elif handler in ('state_offset','state_set_values'):
                    wanted=np.asarray(before,float).copy();indices=p['indices']
                    if handler=='state_offset':wanted[indices]+=p['offsets']
                    else:wanted[indices]=p['values']
                elif handler in ('gripper_opening_command_offset','gripper_opening_feedback_offset','gripper_directional_scale','translation_direction_offset','world_frame_planar_velocity_offset'):
                    body=before.get('actions') if isinstance(before,dict) else before;wanted=np.asarray(body,float).copy()
                    if handler.startswith('gripper_opening_'):
                        wanted[...,int(p['index'])]=np.clip(wanted[...,int(p['index'])]-float(p['opening_offset_m'])/float(p['calibrated_maximum_opening_m']),0,1)
                    elif handler=='gripper_directional_scale':
                        i=int(p['index']);close=p['direction']=='close';v=wanted[...,i];mask=v>.5 if close else v<.5;neutral=0. if close else 1.
                        wanted[...,i]=np.where(mask,neutral+(v-neutral)*float(p['scale']),v)
                    elif handler=='translation_direction_offset':
                        indices=p.get('translation_indices',[3,4,5]);v=wanted[...,indices];norm=np.linalg.norm(v,axis=-1,keepdims=True)
                        wanted[...,indices]+=np.divide(v,norm,out=np.zeros_like(v),where=norm>1e-9)*float(p['offset_m'])
                    else:
                        yaw=float(runtime['base_yaw_rad']);offset=np.asarray(p['offset_world_xy_mps'],float)
                        wanted[...,:2]+=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@offset
                    if isinstance(before,dict):wanted=dict(before,actions=wanted)
                elif handler=='base_velocity_toward_object':
                    wanted=np.asarray(before,float).copy();robot=snapshot['robot'];target=snapshot['objects'][p['object_id']]
                    direction=np.asarray(target['position_xyz'],float)[:2]-np.asarray(robot['position_xyz'],float)[:2]
                    norm=np.linalg.norm(direction);world=direction*float(p['speed_mps'])/norm if norm>1e-9 else np.zeros(2)
                    yaw=float(robot['rpy'][2]);wanted[...,:2]=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@world
                elif handler=='state_hold_trigger_arm_positions':
                    wanted=np.asarray(before,float).copy()
                    wanted[...,p['indices']]=[snapshot['arm']['measured_joint_positions_rad'][n] for n in p['joint_names']]
                elif handler=='joint_encoder_position_feedback':
                    wanted=dict(before);q=np.asarray(before['position_rad'],float).copy();names=before['joint_names']
                    scope=p.get('joint_names') or ([p['joint_name']] if p.get('joint_name') else names)
                    for name in scope:
                        index=names.index(name)
                        if p['operation']=='offset':q[index]+=float(p['offset_rad'])
                        elif p['operation']=='hold_trigger':q[index]=snapshot['locomotion']['measured_joint_positions_rad'][name]
                        else:unsupported=True
                    wanted['position_rad']=q
                elif handler in ('arm_joint_hold_measured','arm_joint_hold_trigger_command'):
                    wanted=dict(before);wanted['positions']=dict(before['positions'])
                    source='measured_joint_positions_rad' if handler.endswith('measured') else 'commanded_joint_positions_rad'
                    scope=p.get('joint_names') or (['joint7','joint8'] if p.get('joint_scope')=='gripper' else ['joint%d'%i for i in range(1,7)])
                    for name in scope:wanted['positions'][name]=snapshot['arm'][source][name]
                elif handler=='wheel_encoder_velocity_bias':
                    wanted=dict(before);wanted['angular_velocity_rad_s']=np.asarray(before['angular_velocity_rad_s'],float)+float(p['linear_velocity_bias_mps'])/float(before['wheel_radius_m'])
                elif handler in ('imu_oscillating_bias','paper200_world_imu_bias'):
                    wanted={k:np.asarray(v,float).copy() for k,v in before.items()}
                    if handler=='imu_oscillating_bias':
                        wanted[p['key']][int(p['axis'])]+=float(p['amplitude'])*np.sin(2*np.pi*float(p['frequency_hz'])*float(runtime['time_s'])+float(p.get('phase_rad',0)))
                    else:
                        receipt=next(r for r in records if r.get('consumer')=='locomotion_onnx' and r.get('physics_step')==step)
                        rotation=np.asarray(receipt['base_to_world_rotation'],float)
                        vector=direction_vector(spec,records) if p.get('direction_contract') else np.asarray(p['world_bias_mps2'],float)
                        wanted['linear_acceleration']+=rotation.T@vector
                elif handler=='imu_sample_transform':
                    wanted={k:np.asarray(v,float).copy() for k,v in before.items()}
                    for key in ('angular_velocity','linear_acceleration','projected_gravity'):
                        if p['attack']=='bias':wanted[key]+=p.get(key+'_bias',[0,0,0])
                        elif p['attack']=='scale':wanted[key]*=p.get(key+'_scale',[1,1,1])
                        else:unsupported=True
                else:unsupported=True;continue
                if context and (context.get('handler')!=handler or not equal(context.get('parameters'),p)):
                    issues.append('ACTUAL_HANDLER_ARGUMENTS_DIFFER_FROM_FROZEN_SPEC')
                checked+=1
                if not equal(wanted,after):issues.append('ATTACK_NUMERIC_PARAMETER_MISMATCH')
            except (KeyError,ValueError,TypeError,IndexError,StopIteration):issues.append('PARAMETER_SOURCE_EVIDENCE_MISSING')
        if unsupported:issues.append('PARAMETER_SEMANTICS_NOT_VERIFIED:'+str(handler))
    # A bounded hold must visit every real interception opportunity while
    # active. No inference is fabricated when the model/controller made no call.
    if stage in ('sensor_joint_encoder','sensor_imu','sensor_wheel_encoder','control_base','control_joint','state_feedback','response_action'):
        if activation is None:
            issues.append('PARAMETER_ACTIVATION_CLOCK_MISSING')
        elif spec.get('duration_s') is not None and collection is not None and 'result_json' in collection:
            result=json.loads(text(collection['result_json'][()]))
            stop=result.get('termination_time_s')
            if stop is None:issues.append('HOLD_TERMINATION_CLOCK_MISSING')
            else:
                end=min(activation+float(spec['duration_s']),float(stop))
                if timing is not None and 'deactivation_time_s' in timing:
                    deactivated=float(timing['deactivation_time_s'][()])
                    if np.isfinite(deactivated) and deactivated>=activation:end=min(end,deactivated)
                expected_steps=set()
                for boundary in rows(collection.get('interception_payloads'),stage):
                    c=boundary.get('runtime_context',{});time=c.get('time_s')
                    if time is not None and activation-1e-8<=float(time)<end-1e-8:
                        expected_steps.add(int(c['physics_step']))
                observed={int(r['physics_step']) for r in records if r.get('consumer')=='attack_handler_boundary' and
                    r.get('attack_id')==spec.get('id') and r.get('stage')==stage and r.get('physics_step') is not None}
                if not expected_steps:issues.append('HOLD_REAL_BOUNDARY_COVERAGE_NOT_PROVEN')
                elif not expected_steps.issubset(observed):issues.append('HOLD_SKIPPED_REAL_INTERCEPTION_BOUNDARIES')
    if checked==0:issues.append('NO_NUMERIC_PARAMETER_EVIDENCE_VERIFIED')
    return {'passed':checked>0 and not issues,'checked_rows':checked,'handler':handler,'issues':list(dict.fromkeys(issues)),
            'scope':'numeric parameter and available hold checks; opportunity semantics verified separately'}


def direction_vector(spec,records):
    """Independently reconstruct a frozen object-relative direction witness."""
    definition=spec['parameters']['direction_contract']
    witnesses=[r for r in records if r.get('consumer')=='attack_direction_resolution' and
        r.get('attack_id')==spec['id'] and r.get('direction_evidence_kind')=='first_application_geometry']
    if len(witnesses)!=1:raise ValueError('direction geometry witness missing or ambiguous')
    witness=witnesses[0]
    if not equal(witness['definition'],definition):raise ValueError('direction definition mismatch')
    vertices=np.asarray(witness['target_points_world'],float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all():raise ValueError('invalid target vertices')
    for binding in witness['target_binding']:
        path=binding['collider'];root=definition['target_prim'].rstrip('/')
        if path!=root and not path.startswith(root+'/'):raise ValueError('direction target collider scope mismatch')
    mode=definition['mode']
    if mode in ('principal_long_axis','principal_short_axis'):
        centered=vertices-vertices.mean(axis=0)
        values,vectors=np.linalg.eigh(centered.T@centered)
        indices=np.argsort(values)[::-1];index=0 if mode=='principal_long_axis' else 1
        direction=vectors[:,indices[index]]
        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    elif mode=='nearest_support_edge_outward':
        from scipy.spatial import ConvexHull
        support=np.asarray(witness['support_points_world'],float)
        if support.ndim!=2 or support.shape[1]!=3 or not np.isfinite(support).all():raise ValueError('invalid support')
        points=np.unique(support[:,:2],axis=0);hull=points[ConvexHull(points).vertices];center=vertices.mean(axis=0)[:2]
        candidates=[]
        for i,start in enumerate(hull):
            delta=hull[(i+1)%len(hull)]-start;length=np.linalg.norm(delta)
            t=np.clip(np.dot(center-start,delta)/length**2,0,1);distance=np.linalg.norm(start+t*delta-center)
            candidates.append((distance,i,np.array([delta[1],-delta[0],0.])/length))
        direction=min(candidates,key=lambda r:(r[0],r[1]))[2]
    else:raise ValueError('unsupported direction contract')
    vector=direction*float(definition['magnitude'])
    if not equal(vector,witness['resolved_world_vector']):raise ValueError('resolved direction mismatch')
    for row in records:
        if row.get('consumer')=='attack_direction_resolution' and row.get('attack_id')==spec['id']:
            if row.get('frozen_activation_key')!=witness.get('frozen_activation_key') or not equal(row['resolved_world_vector'],vector):raise ValueError('held direction drift')
    return vector
