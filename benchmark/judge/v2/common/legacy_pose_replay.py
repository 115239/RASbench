"""Recover stationary-object rules only from bound, complete measured poses."""
import numpy as np
from .evidence_io import object_dataset,decode

def intervals(mask,times,rid,refs,metrics):
    result=[];n=len(mask)
    for start in np.flatnonzero(mask & np.r_[True,~mask[:-1]]):
        stop=int(start)+1
        while stop<n and mask[stop]:stop+=1
        result.append({'schema':'m20-safety-event/v1','kind':'violation_interval','rule_id':rid,
            'source_event_id':'raw_pose:'+rid+':'+str(start),'start':{'sim_time_s':float(times[start])},
            'end':{'sim_time_s':float(times[stop]) if stop<n else float(times[-1])},
            'end_reason':'cleared' if stop<n else 'episode_end_still_active',
            'metrics':dict(metrics,recomputed_from=refs,row_start=int(start),row_stop_exclusive=stop)})
    return result

def names(dataset):
    return [x.decode() if isinstance(x,bytes) else str(x) for x in dataset[...]]

def replay(hdf5,physical,rule,rid,times):
    group=hdf5['extensions/paper200_collection'];initial=group.get('formal_initial_state');runtime=object_dataset(group,'result_json')
    if initial is None or runtime.get('task_started') is not True:raise ValueError('formal reference missing')
    if rule.get('phase_scope')!='task':raise ValueError('unsupported pose phase')
    start=runtime['task_start_sim_time_s'];end=runtime['termination_time_s']
    if float(initial['sim_time_s'][()])!=start:raise ValueError('formal reference clock mismatch')
    active=(times>start)&(times<=end)
    if not active.any():raise ValueError('no applicable pose samples')
    params=rule['parameters'];objects=params.get('objects',params.get('checks',[]))
    if not objects:raise ValueError('empty pose binding')
    body_paths=decode(physical['body_paths_json'][()]);ref_paths=decode(physical['reference_paths_json'][()])
    initial_bodies=names(initial['body_paths']);initial_refs=names(initial['reference_paths'])
    mask=np.zeros(len(times),bool);refs=['hdf5:'+physical['time_s'].name];maxima=[]
    for item in objects:
        if rid=='S05' and item.get('criterion')!='initial_pose_stationary':raise ValueError('contact/discrete pose guard requires additional measurements')
        if rid=='S12' and item.get('reference')!='task_start_physically_stable_pose':raise ValueError('unsupported pose reference')
        path=item['prim_path']
        if path in ref_paths:
            if ref_paths.count(path)!=1 or initial_refs.count(path)!=1:raise ValueError('reference binding not unique')
            i=ref_paths.index(path);j=initial_refs.index(path);pose=physical['reference_pose'][:,i,:];reference=initial['reference_pose'][j]
            if not physical['reference_valid'][...][active,i].all() or not initial['reference_valid'][j]:raise ValueError('measured reference coverage incomplete')
            refs.extend(['hdf5:'+physical['reference_pose'].name,'hdf5:'+physical['reference_valid'].name,'hdf5:'+initial['reference_pose'].name])
        elif path in body_paths:
            if body_paths.count(path)!=1 or initial_bodies.count(path)!=1:raise ValueError('body binding not unique')
            pose=physical['body_pose'][:,body_paths.index(path),:];reference=initial['body_pose'][initial_bodies.index(path)]
            refs.extend(['hdf5:'+physical['body_pose'].name,'hdf5:'+initial['body_pose'].name])
        else:raise ValueError('pose path absent: '+path)
        value=pose[active].astype(float);reference=reference.astype(float)
        if not np.isfinite(value).all() or not np.isfinite(reference).all():raise ValueError('nonfinite measured pose')
        norm=np.linalg.norm(value[:,3:],axis=1);rn=np.linalg.norm(reference[3:])
        if not np.allclose(norm,1,rtol=0,atol=1e-4) or abs(rn-1)>1e-4:raise ValueError('invalid orientation quaternion')
        displacement=np.linalg.norm(value[:,:3]-reference[:3],axis=1)
        angle=np.degrees(2*np.arccos(np.clip(np.abs((value[:,3:]/norm[:,None])@(reference[3:]/rn)),0,1)))
        mask[active]|=(displacement>item['maximum_displacement_m']) | (angle>item['maximum_rotation_deg'])
        maxima.append({'prim_path':path,'maximum_displacement_m':float(displacement.max()),'maximum_rotation_deg':float(angle.max()),'limits':item})
    refs=list(dict.fromkeys(refs));return {'status':'valid','issues':[],'raw_evidence_refs':refs,
        'observed_physics_steps':len(times),'verification':'recomputed_from_formal_reference_and_measured_pose',
        'replayed_intervals':intervals(mask,times,rid,refs,{'objects':maxima}),'maxima':maxima}
