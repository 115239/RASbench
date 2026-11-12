"""Bind object-relative adversarial directions from live physical geometry."""
import numpy as np


def resolve_points_direction(target_points,support_points,mode):
    target=np.asarray(target_points,float)
    if target.ndim!=2 or target.shape[1]!=3 or len(target)<3 or not np.isfinite(target).all():
        raise ValueError('target geometry is missing or invalid')
    if mode in ('principal_long_axis','principal_short_axis'):
        center=target.mean(axis=0);cov=(target-center).T@(target-center)/len(target)
        eigenvalues,eigenvectors=np.linalg.eigh(cov);order=np.argsort(eigenvalues)[::-1]
        axis=0 if mode=='principal_long_axis' else 1;direction=eigenvectors[:,order[axis]]
        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
        return direction,{'method':'PCA of actual world collider vertices',
            'centroid_world':center.tolist(),'eigenvalues_desc':eigenvalues[order].tolist(),
            'eigenvectors_desc_columns':eigenvectors[:,order].tolist(),
            'selected_eigenvector':axis,'sign_registration':'largest absolute world component positive'}
    if mode!='nearest_support_edge_outward':raise ValueError('unknown direction mode '+mode)
    from scipy.spatial import ConvexHull
    support=np.asarray(support_points,float)
    if support.ndim!=2 or support.shape[1]!=3 or len(support)<3 or not np.isfinite(support).all():
        raise ValueError('support geometry is missing or invalid')
    xy=np.unique(support[:,:2],axis=0);hull=xy[ConvexHull(xy).vertices]
    center=target.mean(axis=0)[:2];candidates=[]
    for index,start in enumerate(hull):
        end=hull[(index+1)%len(hull)];edge=end-start;length=np.linalg.norm(edge)
        if length<1e-10:continue
        fraction=float(np.clip(np.dot(center-start,edge)/(length*length),0,1))
        point=start+fraction*edge
        candidates.append((float(np.linalg.norm(point-center)),index,point,np.array([edge[1],-edge[0]])/length))
    distance,index,point,normal=min(candidates,key=lambda x:(x[0],x[1]))
    return np.r_[normal,0.],{'method':'nearest segment of CCW support collider XY convex hull',
        'target_centroid_world_xy':center.tolist(),'support_hull_world_xy':hull.tolist(),
        'edge_index':index,'closest_edge_point_world_xy':point.tolist(),'distance_m':distance,
        'outward_unit_normal_world_xy':normal.tolist(),'tie_break':'ascending hull edge index'}


def actual_world_vertices(collector,semantic_prim):
    runtime=collector.rubric_evidence_runtime
    from rubric_evidence import array,matrix_from_pose
    runtime.cache.Clear()
    obj=next((o for o in runtime.objects if o['prim']==semantic_prim),None)
    if obj is None:raise ValueError('direction target has no runtime geometry binding: '+semantic_prim)
    all_points=[];records=[]
    for path in obj['colliders']:
        binding=runtime.geometry[path]
        if binding['analytic']:raise ValueError('direction PCA requires actual mesh vertices, not an AABB approximation')
        if binding['owner']:
            pose=array(runtime.views[binding['owner']].get_transforms())[0]
            matrix=matrix_from_pose(pose)@binding['local'];pose_source='actual PhysX owner with fixed authored local transform'
        else:
            matrix=runtime._world(binding['sample_prim']);pose_source='live static USD transform'
        points=np.asarray(binding['points'],float)@matrix[:3,:3].T+matrix[:3,3]
        all_points.append(points);records.append({'collider':path,'owner':binding['owner'],
            'world_matrix_column_vector':matrix.tolist(),'pose_source':pose_source,'vertex_count':len(points)})
    if not all_points:raise ValueError('direction geometry has no colliders: '+semantic_prim)
    return np.unique(np.concatenate(all_points),axis=0),records


def resolve_vector(collector,parameters,context,legacy_key):
    definition=parameters.get('direction_contract')
    if not definition:return np.asarray(parameters[legacy_key],float)
    from .common import write
    from pathlib import Path
    key=(context['attack_id'],context['snapshot'].get('_attack_runtime',{}).get('activated_s'))
    cache=collector.__dict__.setdefault('_paper_attack_direction_witness',{})
    if key not in cache:
        target,target_binding=actual_world_vertices(collector,definition['target_prim'])
        support=np.empty((0,3));support_binding=[]
        if definition.get('support_prim'):
            support,support_binding=actual_world_vertices(collector,definition['support_prim'])
        direction,detail=resolve_points_direction(target,support,definition['mode'])
        magnitude=float(definition['magnitude']);vector=direction*magnitude
        record={'schema':'paper200-attack-direction/v2','event':'attack_direction_bound',
            'attack_id':context['attack_id'],'physics_step':int(collector.physics_step),
            'frame':'world','binding_time':'first actual payload application; fixed during this activation',
            'definition':definition,'direction_world_unit':direction.tolist(),
            'vector_world':vector.tolist(),'detail':detail,
            'target_binding':target_binding,'support_binding':support_binding}
        out=getattr(collector,'paper_visual_evidence_dir',None)
        if out is not None:
            out=Path(out)/'directions';out.mkdir(parents=True,exist_ok=True)
            stem='%s_%09d'%(context['attack_id'],collector.physics_step)
            write(out/(stem+'.json'),record)
            np.savez_compressed(out/(stem+'.npz'),target_points_world=target,support_points_world=support)
            record['evidence_stem']='directions/'+stem
        collector.attack_boundary.recorder.append(record)
        collector.attack_boundary.recorder.capture_consumption({**record,
            'consumer':'attack_direction_resolution','mode':definition['mode'],
            'frozen_activation_key':list(key),'resolved_world_vector':vector.tolist(),
            'target_points_world':target.tolist(),'support_points_world':support.tolist(),
            'direction_evidence_kind':'first_application_geometry'})
        cache[key]=(vector,record)
    vector,record=cache[key]
    collector.attack_boundary.recorder.append({'event':'attack_direction_consumed',
        'attack_id':context['attack_id'],'physics_step':int(collector.physics_step),
        'bound_physics_step':record['physics_step'],'vector_world':vector.tolist(),
        'frame':'world','binding_schema':record['schema']})
    collector.attack_boundary.recorder.capture_consumption({
        'consumer':'attack_direction_resolution','attack_id':context['attack_id'],
        'physics_step':int(collector.physics_step),'mode':definition['mode'],
        'frozen_activation_key':list(key),'bound_physics_step':record['physics_step'],
        'resolved_world_vector':vector.tolist(),'direction_evidence_kind':'held_vector_consumption'})
    return vector.copy()
