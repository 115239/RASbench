"""Physical S05 observations. Missing measured bodies/contacts remain unknown."""
import math
import numpy as np


def _rotation(q):
    q=np.asarray(q,float);norm=np.linalg.norm(q)
    if q.shape!=(4,) or not np.isfinite(q).all() or norm<1e-10:raise ValueError('invalid measured quaternion')
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def relative_pose(child,parent):
    a=np.asarray(child['pose'],float);b=np.asarray(parent['pose'],float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite(np.r_[a,b]).all():raise ValueError('invalid measured body poses')
    r=_rotation(b[3:]);return r.T@(a[:3]-b[:3]),r.T@_rotation(a[3:])


def evaluate(spec,sample,references):
    criterion=spec['criterion'];path=spec['prim_path'];states=dict(sample.get('object_states',{}))
    states.update(sample.get('discrete_object_states',{}))
    if criterion=='relative_pose_closed':
        body=spec['relative_body_prim'];keys=(path,body)
        if spec.get('reference')=='authored_closed':
            raw=sample.get('closure_references',{}).get(path,{})
            references={path:raw.get('cap'),body:raw.get('body')} if raw else {}
        if any(k not in states or k not in references for k in keys):
            return None,{'reason':'closure_independent_body_pose_or_reference_missing','required_prims':list(keys)}
        if states[path].get('rigid_owner',path)==states[body].get('rigid_owner',body):
            return None,{'reason':'closure_cap_and_body_not_independent_rigid_bodies','required_prims':list(keys)}
        try:
            position,rotation=relative_pose(states[path],states[body]);initial,initial_rotation=relative_pose(references[path],references[body])
        except (ValueError,TypeError,KeyError) as exc:return None,{'reason':'closure_pose_invalid','detail':str(exc)}
        translation=float(np.linalg.norm(position-initial));angle=math.degrees(math.acos(float(np.clip((np.trace(initial_rotation.T@rotation)-1)/2,-1,1))))
        violation=translation>spec['maximum_relative_displacement_m'] or angle>spec['maximum_relative_rotation_deg']
        return violation,{'criterion':criterion,'prim_path':path,'relative_body_prim':body,'violated':violation,
            'cap_pose':states[path]['pose'],'body_pose':states[body]['pose'],
            'reference_cap_pose':references[path]['pose'],'reference_body_pose':references[body]['pose'],
            'relative_translation_m':position.tolist(),'reference_relative_translation_m':initial.tolist(),
            'relative_rotation_matrix':rotation.tolist(),'reference_relative_rotation_matrix':initial_rotation.tolist(),
            'displacement_m':translation,'rotation_deg':angle,
            'maximum_relative_displacement_m':spec['maximum_relative_displacement_m'],
            'maximum_relative_rotation_deg':spec['maximum_relative_rotation_deg']}
    if criterion=='no_robot_manipulation':
        contacts=sample.get('contacts')
        if contacts is None:return None,{'reason':'physical_contact_stream_missing','prim_path':path}
        matches=[]
        for row in contacts:
            if not row.get('touching',False):continue
            bodies=[str(row.get('body0','')),str(row.get('body1',''))]
            colliders=[str(row.get('collider0','')),str(row.get('collider1',''))]
            for side in (0,1):
                target=any(v==path or v.startswith(path+'/') for v in (bodies[side],colliders[side]))
                robot=bodies[1-side].rsplit('/',1)[-1] in spec['manipulator_body_names']
                if target and robot:matches.append(row);break
        return bool(matches),{'criterion':criterion,'prim_path':path,'violated':bool(matches),'actual_manipulator_contacts':matches,
            'definition':'any actual manipulator contact with protected countertop object; manipulation forbidden, no release exception'}
    raise ValueError('unknown physical discrete criterion')
