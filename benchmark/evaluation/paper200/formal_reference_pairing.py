"""Compare authoritative task/protected references omitted from rigid-body views."""
import json
import numpy as np
from .evidence_contract import canonical_digest

FIELDS=('reference_paths','reference_pose','reference_valid','reference_bindings_json')


def required_reference_paths(contract):
    task=contract.get('task',{});profile=contract.get('safety_profile',{});paths=set()
    target=task.get('manipulation',{}).get('scene_object_prim')
    if target:paths.add(target)
    rules=profile.get('rules',{})
    if rules.get('S12',{}).get('enabled'):
        paths.update(row['prim_path'] for row in rules['S12']['parameters']['objects'])
    if rules.get('S05',{}).get('enabled'):
        for row in rules['S05']['parameters']['checks']:
            if row.get('criterion') in ('relative_pose_closed','no_robot_manipulation'):
                paths.add(row['prim_path'])
                if row.get('relative_body_prim'):paths.add(row['relative_body_prim'])
    return sorted(paths)


def _read(state,expected):
    if any(k not in state for k in FIELDS):raise ValueError('FORMAL_AUTHORITATIVE_REFERENCE_FIELDS_MISSING')
    paths=np.asarray(state['reference_paths']).astype(str);pose=np.asarray(state['reference_pose']);valid=np.asarray(state['reference_valid'])
    if paths.ndim!=1 or len(set(paths))!=len(paths) or any(not p.startswith('/') for p in paths):raise ValueError('FORMAL_REFERENCE_PATHS_INVALID')
    if expected is not None and sorted(paths)!=sorted(expected):raise ValueError('FORMAL_REFERENCE_COVERAGE_MISMATCH')
    n=len(paths)
    if pose.shape!=(n,7) or valid.shape!=(n,) or valid.dtype.kind!='b' or not valid.all() or not np.isfinite(pose).all():raise ValueError('FORMAL_REFERENCE_POSE_OR_VALIDITY_INVALID')
    if n and not np.allclose(np.linalg.norm(pose[:,3:],axis=1),1.,atol=1e-4,rtol=0):raise ValueError('FORMAL_REFERENCE_ORIENTATION_INVALID')
    raw=np.asarray(state['reference_bindings_json'])
    if raw.ndim!=0:raise ValueError('FORMAL_REFERENCE_BINDING_SCHEMA_INVALID')
    raw=raw.item();raw=raw.decode('utf-8') if isinstance(raw,bytes) else raw
    bindings=json.loads(raw)
    if not isinstance(bindings,list) or len(bindings)!=n:raise ValueError('FORMAL_REFERENCE_BINDING_COUNT_MISMATCH')
    by_path={row.get('prim'):row for row in bindings if isinstance(row,dict)}
    if len(by_path)!=n or set(by_path)!=set(paths):raise ValueError('FORMAL_REFERENCE_BINDING_PATH_MISMATCH')
    bodies=set(np.asarray(state['body_paths']).astype(str))
    for path,row in by_path.items():
        source=row.get('source');owner=row.get('rigid_owner');anc=row.get('enabled_ancestor_owners');desc=row.get('enabled_descendant_owners')
        if row.get('pose_order')!='xyz_xyzw' or not isinstance(anc,list) or not isinstance(desc,list):raise ValueError('FORMAL_REFERENCE_AUTHORITY_SCHEMA_INVALID')
        if source=='physx_rigid_body':
            if owner!=path or owner not in bodies or owner not in anc:raise ValueError('FORMAL_REFERENCE_PHYSX_OWNER_INVALID')
        elif source=='physx_owner_fixed_descendant':
            if owner not in bodies or not path.startswith(owner+'/') or not anc or anc[0]!=owner:raise ValueError('FORMAL_REFERENCE_DESCENDANT_OWNER_INVALID')
            matrix=np.asarray(row.get('fixed_local_matrix_row_vector'),float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('FORMAL_REFERENCE_LOCAL_TRANSFORM_INVALID')
        elif source=='physx_articulation_reference_frame':
            from .reference_states import ARTICULATION_REFERENCE_BODIES
            if ARTICULATION_REFERENCE_BODIES.get(path)!=owner or owner not in bodies or owner not in desc or anc or not set(desc)<=bodies:
                raise ValueError('FORMAL_REFERENCE_ARTICULATION_OWNER_INVALID')
            if row.get('reference_frame_semantics')!='authored_object_frame_rigidly_attached_to_declared_body':
                raise ValueError('FORMAL_REFERENCE_ARTICULATION_FRAME_INVALID')
            matrix=np.asarray(row.get('fixed_local_matrix_row_vector'),float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('FORMAL_REFERENCE_LOCAL_TRANSFORM_INVALID')
            edges=row.get('articulation_joint_topology',[])
            parents={}
            for edge in edges:
                if not isinstance(edge,list) or len(edge)!=4:raise ValueError('FORMAL_REFERENCE_ARTICULATION_TOPOLOGY_INVALID')
                joint,a,b,kind=edge
                if not isinstance(joint,str) or not joint.startswith(path+'/') or a not in desc or b not in desc or b in parents:
                    raise ValueError('FORMAL_REFERENCE_ARTICULATION_TOPOLOGY_INVALID')
                parents[b]=a
            if set(desc)-set(parents)!={owner} or len(edges)!=len(desc)-1:
                raise ValueError('FORMAL_REFERENCE_ARTICULATION_TOPOLOGY_INVALID')
            for body in desc:
                seen=set();node=body
                while node!=owner:
                    if node in seen or node not in parents:raise ValueError('FORMAL_REFERENCE_ARTICULATION_TOPOLOGY_INVALID')
                    seen.add(node);node=parents[node]
        elif source=='usd_authoritative_static':
            if owner is not None or anc or desc:raise ValueError('FORMAL_REFERENCE_STATIC_HAS_DYNAMIC_OWNER')
            matrix=np.asarray(row.get('initial_world_matrix'),float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('FORMAL_REFERENCE_STATIC_TRANSFORM_INVALID')
        else:raise ValueError('FORMAL_REFERENCE_AUTHORITY_UNAVAILABLE')
    order=np.argsort(paths)
    return paths[order],pose[order].astype(float),[by_path[p] for p in paths[order]]


def compare_references(a,b,tolerance=1e-5,expected_paths=None):
    issues=[];maximum=None
    try:
        paths_a,pose_a,bindings_a=_read(a,expected_paths);paths_b,pose_b,bindings_b=_read(b,expected_paths)
        if not np.array_equal(paths_a,paths_b):issues.append('FORMAL_REFERENCE_PATH_BINDING_MISMATCH')
        if canonical_digest(bindings_a)!=canonical_digest(bindings_b):issues.append('FORMAL_REFERENCE_AUTHORITY_BINDING_MISMATCH')
        if pose_a.shape!=pose_b.shape:issues.append('FORMAL_REFERENCE_POSE_SHAPE_MISMATCH')
        else:
            pose_b[:,3:]*=np.where(np.sum(pose_a[:,3:]*pose_b[:,3:],axis=1)<0,-1.,1.)[:,None]
            maximum=float(np.max(np.abs(pose_a-pose_b),initial=0.))
            if maximum>tolerance:issues.append('FORMAL_REFERENCE_POSE_MISMATCH')
    except (ValueError,KeyError,TypeError) as exc:issues.append(str(exc))
    return {'matched':not issues,'issues':issues,'maximum_absolute_differences':{'reference_pose':maximum},
            'absolute_tolerance':tolerance,'pose_order':'xyz_xyzw','authority_policy':'PhysX rigid, PhysX owner fixed descendant, explicit validated articulation reference frame, or verified static USD only; missing references never inherit warmup poses'}
