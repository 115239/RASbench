"""Common scene bindings applied identically before physics in all conditions."""
import os
from pathlib import Path
from .common import read,write


def materialize_visual_targets(stage):
    package=Path(os.environ['M20_QUESTION_PACKAGE_ROOT']);paths=set()
    task=read(package/'data/task/task.json')
    target=task.get('manipulation',{}).get('scene_object_prim')
    if target:paths.add(target)
    for file in (package/'data/attack/profiles').glob('*.json'):
        for attack in read(file).get('attack',{}).get('attacks',[]):
            if attack.get('handler')=='install_visual_material_override':
                paths.update(attack['parameters'].get('expected_affected_mesh_prims',[]))
    changed=[];previous=stage.GetEditTarget();stage.SetEditTarget(stage.GetSessionLayer())
    try:
        for path in sorted(paths):
            parts=path.strip('/').split('/')
            for depth in range(1,len(parts)+1):
                prim=stage.GetPrimAtPath('/'+'/'.join(parts[:depth]))
                if prim and prim.IsInstance():
                    changed.append(str(prim.GetPath()));prim.SetInstanceable(False)
        rigid_binding=normalize_target_rigid_owner(stage,target) if target else None
        parity=canonicalize_pose_parity(stage,target) if target else None
        from .cap_scene import apply_scene_state_repairs
        discrete_repairs=apply_scene_state_repairs(stage,package)
    finally:stage.SetEditTarget(previous)
    cfg=read(os.environ['THREE_SCENES_RUN_CONFIG'])
    write(Path(cfg['output'])/'episode/VISUAL_INSTANCE_BINDING.json',{
        'expanded_instances':sorted(set(changed)),'applied_to_clean_and_all_variants':True,
        'pose_frame_parity_correction':parity,
        'rigid_body_binding':rigid_binding,'discrete_physics_revision':discrete_repairs,
        'world_geometry_changed':False,'physics_topology_changed':bool(discrete_repairs and discrete_repairs['changes']),'local_pose_frame_changed':parity is not None,
        'reason':'enable authored material and rigid-body bindings on instance descendants'})


def normalize_target_rigid_owner(stage,path):
    """Bind a single imported body to its configured wrapper, without moving meshes.

    The common Runner enables a body on the task's configured wrapper. Some
    imported assets already enable one on their only mesh child. Leaving both
    enabled creates an invalid nested PhysX body. Never collapse articulations,
    independently transformed bodies, or collections of several bodies.
    """
    from pxr import Usd,UsdGeom,UsdPhysics
    import numpy as np
    root=stage.GetPrimAtPath(path)
    if not root:raise ValueError('missing manipulation target: '+path)
    children=list(Usd.PrimRange(root,Usd.TraverseInstanceProxies()))[1:]
    bodies=[p for p in children if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]
    if not bodies:return None
    if len(bodies)!=1 or any(p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.Joint)
                           for p in [root]+children):
        raise ValueError('cannot collapse compound/articulated task body: '+path)
    body=bodies[0];ancestor=body
    while ancestor!=root:
        if UsdGeom.Xformable(ancestor).GetResetXformStack():
            raise ValueError('independent task child rigid frame: '+str(ancestor.GetPath()))
        ancestor=ancestor.GetParent()
    # Reject joints elsewhere in the scene that refer to this child body.
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            if body.GetPath() in joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets():
                raise ValueError('task child body is referenced by joint: '+str(prim.GetPath()))
    before={str(p.GetPath()):np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            for p in children if p.IsA(UsdGeom.Mesh)}
    authored={a.GetName():str(a.Get()) for a in body.GetAttributes()
              if a.GetName().startswith(('physics:','physxRigidBody:'))}
    if not body.RemoveAPI(UsdPhysics.RigidBodyAPI):raise RuntimeError('cannot remove redundant rigid schema')
    after={str(p.GetPath()):np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
           for p in children if p.IsA(UsdGeom.Mesh)}
    error=max((float(np.max(np.abs(before[k]-after[k]))) for k in before),default=0.)
    if error!=0:raise RuntimeError('rigid owner correction moved geometry')
    return {'configured_owner':path,'removed_redundant_child_body':str(body.GetPath()),
            'previous_child_physics_attributes':authored,'mesh_world_transform_max_change':error,
            'mass_source':'task manipulation scene_object_mass_kg / object_mass_kg, common Runner',
            'semantic_child_prims_and_colliders_preserved':True}


def canonicalize_pose_parity(stage,path):
    """Move reflection from a rigid pose into local geometry, preserving world vertices."""
    import numpy as np
    from pxr import Usd,UsdGeom,Gf,Vt
    prim=stage.GetPrimAtPath(path);x=UsdGeom.Xformable(prim)
    world=np.asarray(x.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    if np.linalg.det(world[:3,:3])>=0:return None
    def vertices():
        cache=UsdGeom.XformCache();values={}
        for item in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if item.IsA(UsdGeom.Mesh):
                points=np.asarray(UsdGeom.Mesh(item).GetPointsAttr().Get(),float)
                matrix=np.asarray(cache.GetLocalToWorldTransform(item))
                values[str(item.GetPath())]=points@matrix[:3,:3]+matrix[3,:3]
        return values
    before=vertices();reflection=np.diag([-1.,1.,1.,1.])
    def transform(node,matrix):
        xf=UsdGeom.Xformable(node);reset=xf.GetResetXformStack();xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='paper200Parity').Set(Gf.Matrix4d(matrix.tolist()));xf.SetResetXformStack(reset)
    local=np.asarray(x.GetLocalTransformation());transform(prim,reflection@local)
    for child in prim.GetChildren():
        xf=UsdGeom.Xformable(child)
        if xf and not xf.GetResetXformStack():transform(child,np.asarray(xf.GetLocalTransformation())@reflection)
    if prim.IsA(UsdGeom.Mesh):
        mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get(),float);points[:,0]*=-1
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        normals=mesh.GetNormalsAttr().Get()
        if normals:
            values=np.asarray(normals,float);values[:,0]*=-1
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(values.astype(np.float32)))
    after=vertices();errors=[np.max(np.abs(before[k]-after[k])) for k in before if len(before[k])]
    maximum=float(max(errors,default=0.))
    if maximum>1e-6:raise ValueError('pose parity adapter changed world geometry')
    return {'prim':path,'reason':'SingleRigidPrim requires a right-handed orientation',
        'maximum_world_vertex_change_m':maximum,'local_frame_reflection':[[-1,0,0],[0,1,0],[0,0,1]],
        'world_geometry_preserved':True}


class RecordedSceneTarget:
    """Read-only legacy object evidence adapter for a navigation landmark."""
    def __init__(self,collector,path):self.c=collector;self.path=path;self.cached_step=None;self.cached_row=None
    def row(self):
        runtime=self.c.rubric_evidence_runtime
        if self.cached_step!=self.c.physics_step:
            self.cached_row=next(row for obj,row in zip(runtime.objects,runtime._kinematics()) if obj['prim']==self.path)
            self.cached_step=self.c.physics_step
        return self.cached_row
    def get_world_pose(self):
        pose=self.row()['root_pose'];return pose[:3],pose[3:]
    def get_linear_velocity(self):return self.row()['root_velocity'][:3]
    def get_angular_velocity(self):return self.row()['root_velocity'][3:]


def bind_navigation_evidence(c,cfg):
    package=Path(os.environ['M20_QUESTION_PACKAGE_ROOT']);task=read(package/'data/task/task.json')
    if task['execution_mode']!='navigation_only':return
    path=task['tracked_objects']['target_0'];c.cube=RecordedSceneTarget(c,path)
    write(Path(cfg['output'])/'episode/LEGACY_OBJECT_BINDING.json',{
        'recorded_object_prim':path,'policy_observation_changed':False,
        'reason':'legacy object fields reference the actual navigation landmark, not the unused training prop'})
