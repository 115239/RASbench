"""Typed trigger distances and geometric attack context, isolated from policy."""
import numpy as np

def physical_payload_digest(value):
    """Audit annotations must never turn an unchanged sensor frame into an effect."""
    import hashlib,json
    from collections.abc import Mapping
    from attack_tools.audit import summarize
    def physical(x):
        if isinstance(x,Mapping):return {k:physical(v) for k,v in x.items() if k!='attack_metadata'}
        if isinstance(x,(list,tuple)):return [physical(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(summarize(physical(value)),ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def distance(condition,snapshot):
    metric=condition.get('metric','prim_origin');subject=condition.get('subject','robot_base');oid=str(condition['object_id'])
    if metric=='world_aabb_xy':return snapshot.get('boundary_distances',{}).get(oid)
    if metric in ('world_aabb_surface_xy','world_aabb_surface_3d'):
        a=snapshot.get('subjects',{}).get(subject,{}).get('world_aabb')
        b=snapshot.get('objects',{}).get(oid,{}).get('world_aabb')
        if a is None or b is None:return None
        a,b=np.asarray(a,float),np.asarray(b,float);axes=2 if metric.endswith('_xy') else 3
        delta=np.maximum(0,np.maximum(a[0,:axes]-b[1,:axes],b[0,:axes]-a[1,:axes]))
        return float(np.linalg.norm(delta))
    if metric not in ('prim_origin','prim_origin_3d','prim_origin_xy'):raise ValueError('unsupported distance metric '+metric)
    a=snapshot.get('subjects',{}).get(subject,{}).get('position_xyz')
    if a is None and subject=='robot_base':a=snapshot.get('robot',{}).get('position_xyz')
    if a is None and subject=='gripper':a=snapshot.get('grasp',{}).get('tcp_xyz')
    b=snapshot.get('objects',{}).get(oid,{}).get('position_xyz')
    if a is None or b is None:return None
    axes=2 if metric.endswith('_xy') or condition.get('axes')=='xy' else 3
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not np.isfinite([a,b]).all():raise ValueError('nonfinite trigger coordinates')
    return float(np.linalg.norm(a[:axes]-b[:axes]))

def install():
    from attack_tools.triggers import TriggerEvaluator
    from attack_tools.attack_orchestrator import AttackOrchestrator
    AttackOrchestrator._payload_digest=staticmethod(physical_payload_digest)
    TriggerEvaluator.distance=lambda self,c,s:distance(c,s)
    if not hasattr(TriggerEvaluator,'_paper200_original_observable'):
        old=TriggerEvaluator.observable
        TriggerEvaluator._paper200_original_observable=old
        def observable(self,c,s):
            if c.get('type')=='distance_to_object':return distance(c,s) is not None
            return old(self,c,s)
        TriggerEvaluator.observable=observable

def augment_snapshot(c,snapshot):
    evidence=c.rubric_evidence_runtime
    rows=evidence._kinematics()
    states={o['prim']:r for o,r in zip(evidence.objects,rows)}
    root='/M20PiperVLA/M20_Piper'
    gripper=[states[root+'/Piper/'+k] for k in ('link7','link8','gripper_base')]
    boxes=[s['world_aabb'] for s in gripper if 'world_aabb' in s]
    gb=np.stack([np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0)])
    snapshot['subjects']={'robot_base':{'position_xyz':snapshot['robot']['position_xyz'],'world_aabb':states[root]['world_aabb'].tolist()},
                          'gripper':{'position_xyz':snapshot['grasp']['tcp_xyz'],'world_aabb':gb.tolist()}}
    for oid,path in c.tracked_object_prims.items():
        if path in states:
            row=states[path];snapshot.setdefault('objects',{}).setdefault(oid,{})['position_xyz']=row['root_pose'][:3].tolist()
            if 'world_aabb' in row:snapshot['objects'][oid]['world_aabb']=row['world_aabb'].tolist()
    return snapshot

CURRENT_COLLECTOR=None

def rotation_wxyz(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:raise ValueError('invalid orientation')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def roi_dropout(frame,parameters,context):
    points=np.asarray(frame['points']);transform=np.asarray(frame['sensor_to_world'])
    world=points@transform[:3,:3].T+transform[:3,3]
    bounds=np.asarray(parameters['world_aabb'],float);remove=np.all((world>=bounds[0])&(world<=bounds[1]),axis=1)
    attacked={}
    for key,value in frame.items():
        if key in ('stable_id_map','sensor_to_world'):attacked[key]=value;continue
        a=np.asarray(value)
        attacked[key]=a[~remove].copy() if a.ndim>0 and len(a)==len(points) else value
    attacked['attack_metadata']={'selection':'frozen_registered_world_roi','removed_points':int(remove.sum()),'world_aabb':bounds.tolist()}
    return attacked

def world_action_offset(action,parameters,context):
    result=np.asarray(action).copy();c=CURRENT_COLLECTOR
    if c is None:raise ValueError('no physical attack adapter')
    _,q=c.link_transform(c.arm_body)
    rot=rotation_wxyz(q)
    from .attack_direction_witness import resolve_vector
    result[...,3:6]+=rot.T@resolve_vector(c,parameters,context,'world_offset_m')
    return result

def projected_patch(image,parameters,context):
    c=CURRENT_COLLECTOR
    if c is None:raise ValueError('no camera attack adapter')
    from PIL import Image
    segmentation=c.cameras[0].get_current_frame().get('instance_id_segmentation')
    if not isinstance(segmentation,dict) or 'data' not in segmentation:
        raise ValueError('visual attack instance mask is unavailable')
    labels=segmentation.get('info',{}).get('idToLabels',{})
    target=parameters['target_prim'];ids=[]
    for key,label in labels.items():
        if isinstance(label,dict):label=label.get('primPath','')
        if label==target or str(label).startswith(target+'/'):ids.append(int(key))
    mask=np.isin(np.asarray(segmentation['data']).squeeze(),ids)
    result=np.asarray(image).copy()
    if mask.shape!=result.shape[:2]:mask=np.asarray(Image.fromarray(mask).resize((result.shape[1],result.shape[0]),resample=Image.Resampling.NEAREST))
    from attack_tools.standard_attack_handlers import picture_region_patch
    patch=picture_region_patch(image,{**parameters,'bbox_norm':[0.,0.,1.,1.]},context)
    result[mask]=patch[mask]
    from .common import write
    out=getattr(c,'paper_visual_evidence_dir',None)
    if out is not None:
        out.mkdir(exist_ok=True)
        request_id=context.get('runtime',{}).get('request_id',0)
        stem='%s_%09d_%06d'%(context['attack_id'],c.physics_step,request_id)
        write(out/(stem+'.json'),{'attack_id':context['attack_id'],'target_prim':target,'idToLabels':labels,
            'request_id':request_id,'matched_ids':ids,'physics_step':c.physics_step,'visible_target_pixels':int(mask.sum())})
        np.savez_compressed(out/(stem+'.npz'),image_before=np.asarray(image),
            image_after=result,mask=mask,instance_ids=np.asarray(segmentation['data']))
    c.attack_boundary.recorder.append({'event':'visual_attack_instance_mask','attack_id':context['attack_id'],
        'target_prim':target,'visible_target_pixels':int(mask.sum()),'mask_source':'rendered instance IDs; occlusion respected',
        'physics_step':c.physics_step})
    return result

_old_install=install

def install():
    _old_install()
    from attack_tools.standard_attack_handlers import STANDARD_HANDLERS
    STANDARD_HANDLERS.update(paper200_roi_dropout=roi_dropout,paper200_world_action_offset=world_action_offset,
                             paper200_projected_patch=projected_patch)
    STANDARD_HANDLERS['lidar_phantom_add_frame']=anchored_phantom
    STANDARD_HANDLERS['paper200_world_imu_bias']=world_imu_bias
    from .repaired_visual_handlers import install as install_visual_textures
    install_visual_textures(STANDARD_HANDLERS)


def anchored_phantom(frame,parameters,context):
    from attack_tools.standard_attack_handlers import lidar_phantom_add_frame,_append_lidar_points
    if parameters.get('anchor_frame')!='world_at_trigger':return lidar_phantom_add_frame(frame,parameters,context)
    c=CURRENT_COLLECTOR
    if c is None:raise ValueError('world anchored point cloud requires a physical collector')
    snapshot=context['snapshot'];robot=snapshot['robot'];position=np.asarray(robot['position_xyz'],float)
    key=(context['attack_id'],snapshot.get('_attack_runtime',{}).get('activated_s'))
    cache=c.__dict__.setdefault('_paper200_phantom_world',{})
    if key not in cache:
        direction=np.array([np.cos(robot['rpy'][2]),np.sin(robot['rpy'][2]),0.])
        if parameters.get('direction_binding')=='return_destination':
            direction[:2]=np.asarray(snapshot['mission']['return_destination_xy'])-position[:2]
            length=np.linalg.norm(direction)
            if length<1e-8:raise ValueError('return direction is undefined at spawn')
            direction/=length
        side=np.array([-direction[1],direction[0],0.]);center=position+direction*parameters['forward_distance_m']
        from omni.physx import get_physx_scene_query_interface
        hit=get_physx_scene_query_interface().raycast_closest(tuple(position-[0,0,.25]),(0.,0.,-1.),5.)
        if not hit.get('hit'):raise ValueError('cannot bind phantom bottom to actual ground')
        ground=float(hit['position'][2])
        x,y,z=np.meshgrid(np.linspace(-parameters['thickness_m']/2,parameters['thickness_m']/2,parameters['layers']),
            np.linspace(-parameters['width_m']/2,parameters['width_m']/2,parameters['columns']),
            np.linspace(ground,ground+parameters['height_m'],parameters['rows']),indexing='ij')
        points=center+x.reshape(-1,1)*direction+y.reshape(-1,1)*side
        points[:,2]=z.ravel();cache[key]=points
        c.attack_boundary.recorder.capture_consumption({'consumer':'phantom_anchor','receipt_status':'physical_ground_raycast_returned',
            'attack_id':context['attack_id'],'physics_step':int(c.physics_step),
            'trigger_position':position,'trigger_yaw':float(robot['rpy'][2]),
            'return_destination_xy':snapshot.get('mission',{}).get('return_destination_xy'),
            'ground_hit_position':np.asarray(hit['position']),
            'ground_hit_collider':str(hit.get('collision','')),
            'world_points':points,'anchor_key':list(key)})
    transform=np.asarray(frame['sensor_to_world']);world=cache[key]
    phantom=((world-transform[:3,3])@transform[:3,:3]).astype(np.float32)
    result=_append_lidar_points(frame,np.asarray(frame['points']),phantom,parameters)
    result['attack_metadata']={'kind':'phantom_wall','anchor_frame':'world_at_trigger',
        'added_point_count':len(phantom),'world_aabb':[world.min(axis=0).tolist(),world.max(axis=0).tolist()]}
    return result

def world_imu_bias(sample,parameters,context):
    c=CURRENT_COLLECTOR
    if c is None:raise ValueError('no physical IMU adapter')
    _,q=c.robot.get_world_pose();rot=rotation_wxyz(q)
    result={k:np.asarray(v).copy() for k,v in sample.items()}
    from .attack_direction_witness import resolve_vector
    result['linear_acceleration']+=rot.T@resolve_vector(c,parameters,context,'world_bias_mps2')
    return result
