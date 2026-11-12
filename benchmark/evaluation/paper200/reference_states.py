"""Observer-only authoritative pose registry for task/protected references.

Dynamic bodies always use PhysX. A fixed child uses its measured rigid owner's
pose and a verified fixed local transform. Only truly static geometry uses USD.
"""
import json
import numpy as np
from .common import write


def reference_paths(task,profile):
    paths=set();target=task.get('manipulation',{}).get('scene_object_prim')
    if target:paths.add(target)
    rules=(profile or {}).get('rules',{})
    if rules.get('S12',{}).get('enabled'):
        paths.update(x['prim_path'] for x in rules['S12']['parameters']['objects'])
    if rules.get('S05',{}).get('enabled'):
        for check in rules['S05']['parameters']['checks']:
            if check['criterion'] in ('relative_pose_closed','no_robot_manipulation'):
                paths.add(check['prim_path'])
                if check.get('relative_body_prim'):paths.add(check['relative_body_prim'])
    return sorted(paths)


def preserve_safety_rigid_owners(stage,profile,protected,owner_resolver,output):
    """Preserve original physics of safety objects; never activate authored static bodies."""
    from pxr import Usd,UsdPhysics
    before=set(protected);records=[]
    for path in reference_paths({},profile):
        prim=stage.GetPrimAtPath(path);owners=set()
        if prim and prim.IsValid():
            owner=owner_resolver(stage,path)
            if owner:owners.add(owner)
            owners.update(str(p.GetPath()) for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.RigidBodyAPI))
        details=[]
        for owner in sorted(owners):
            api=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(owner));enabled=api.GetRigidBodyEnabledAttr().Get() is not False
            if enabled:protected.add(owner)
            details.append({'owner':owner,'original_rigid_enabled':enabled,
                'original_kinematic_enabled':bool(api.GetKinematicEnabledAttr().Get()),
                'added_to_freeze_exemption':enabled and owner not in before})
        records.append({'configured_prim':path,'found':bool(prim),'owners':details})
    write(output/'SAFETY_PHYSICS_PRESERVATION.json',{'schema':'paper200-safety-physics-preservation/v1',
        'records':records,'newly_preserved_owners':sorted(set(protected)-before),
        'source':'actual composed scene before ordinary-furniture freezing',
        'semantics':'retain existing enabled S05/S12 body physics for all systems/conditions; no body enabled artificially',
        'policy_or_agent_receives_registry':False})


# Explicit observer-only object-frame bindings, validated against authored joint
# topology below. These are not inferred from child names, mass, or list order.
ARTICULATION_REFERENCE_BODIES = {
    '/World/Market/Articulations/Z02/cart_02':
        '/World/Market/Articulations/Z02/cart_02/Chassis',
}


class ReferenceStateRegistry:
    def __init__(self,c,rigid_paths,output):
        from pxr import Usd,UsdGeom,UsdPhysics,Tf
        self.c=c;self.stage=c.final_stage;self.Usd=Usd;self.UsdGeom=UsdGeom;self.UsdPhysics=UsdPhysics
        self.rigid_index={p:i for i,p in enumerate(rigid_paths)}
        self.paths=reference_paths(getattr(c,'paper_task_config',{}),c.safety_profile)
        self.bindings=[];self.revision=0;self.classification_revision={}
        self.last_states={};self.last_valid=[];self.last_poses=None
        for path in self.paths:self.bindings.append(self.bind(path))
        self.log=(output/'REFERENCE_STATE_SOURCE_EVENTS.jsonl').open('x',buffering=1)
        self.last_errors={}
        def changed(notice,sender):
            # Transforms are sampled afresh. Any physics/topology change forces
            # a fresh authority check so a formerly static body cannot silently
            # acquire a stale USD fallback after becoming dynamic.
            paths=list(notice.GetResyncedPaths())+list(notice.GetChangedInfoOnlyPaths())
            if any(p.IsPrimPath() or str(p).endswith('.apiSchemas') or '.physics:' in str(p) or '.physx' in str(p) for p in paths):
                self.revision+=1
        self._notice_callback=changed
        self.notice=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._notice_callback,self.stage)
        write(output/'REFERENCE_STATE_BINDINGS.json',{'schema':'paper200-reference-state/v1','pose_order':'xyz_xyzw',
            'paths':self.paths,'bindings':self.bindings,'sampling_frequency_hz':200,'not_policy_input':True})

    def enabled_owners(self,path):
        prim=self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():return [],[]
        ancestor=[];p=prim
        while p and not p.IsPseudoRoot():
            if p.HasAPI(self.UsdPhysics.RigidBodyAPI) and self.UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False:
                ancestor.append(str(p.GetPath()))
            p=p.GetParent()
        descendants=[str(p.GetPath()) for p in self.Usd.PrimRange(prim,self.Usd.TraverseInstanceProxies())
            if p!=prim and p.HasAPI(self.UsdPhysics.RigidBodyAPI)
            and self.UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]
        return ancestor,descendants

    def usd_matrix(self,path):
        prim=self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not self.UsdGeom.Xformable(prim):raise ValueError('missing reference Xform')
        return self.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(self.Usd.TimeCode.Default())

    @staticmethod
    def matrix_pose(matrix):
        from pxr import Gf
        transform=Gf.Transform(matrix);q=transform.GetRotation().GetQuat();v=np.r_[matrix.ExtractTranslation(),q.GetImaginary(),q.GetReal()].astype(float)
        if not np.isfinite(v).all() or np.linalg.norm(v[3:])<1e-12:raise ValueError('invalid reference transform')
        v[3:]/=np.linalg.norm(v[3:]);return v

    @staticmethod
    def pose_matrix(pose):
        from pxr import Gf
        pose=np.asarray(pose,float);t=Gf.Transform();t.SetTranslation(Gf.Vec3d(*pose[:3]));t.SetRotation(Gf.Rotation(Gf.Quatd(float(pose[6]),Gf.Vec3d(*pose[3:6]))))
        return t.GetMatrix()

    def local_matrix(self,path,owner):
        # Remove only the owner's rigid pose; all authored scale and child
        # geometry offset remain in this fixed relative transform.
        p=self.stage.GetPrimAtPath(path)
        while str(p.GetPath())!=owner:
            if self.UsdGeom.Xformable(p).GetResetXformStack():raise ValueError('dynamic descendant resets transform stack')
            p=p.GetParent()
        owner_pose=self.matrix_pose(self.usd_matrix(owner))
        return self.usd_matrix(path)*self.pose_matrix(owner_pose).GetInverse()

    def articulation_binding(self,path,row):
        """A declared body-attached object frame; never a measured USD parent."""
        owner=ARTICULATION_REFERENCE_BODIES[path]
        prim=self.stage.GetPrimAtPath(path)
        bodies=set(row['enabled_descendant_owners'])
        if not prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
            raise ValueError('reference_container_not_articulation')
        if owner not in bodies or bodies-set(self.rigid_index):
            raise ValueError('articulation_reference_body_missing_physx_view')
        edges=[];parents={}
        for p in self.Usd.PrimRange(prim,self.Usd.TraverseInstanceProxies()):
            if p!=prim and p.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                raise ValueError('nested_reference_articulation')
            if not p.IsA(self.UsdPhysics.Joint):continue
            joint=self.UsdPhysics.Joint(p)
            if joint.GetJointEnabledAttr().Get() is False or joint.GetExcludeFromArticulationAttr().Get():
                raise ValueError('reference_articulation_joint_disabled_or_excluded')
            a=list(joint.GetBody0Rel().GetTargets());b=list(joint.GetBody1Rel().GetTargets())
            if len(a)!=1 or len(b)!=1 or str(a[0]) not in bodies or str(b[0]) not in bodies:
                raise ValueError('reference_articulation_external_or_unbound_joint')
            a,b=str(a[0]),str(b[0])
            if b in parents:raise ValueError('reference_articulation_multiple_parents')
            parents[b]=a;edges.append([str(p.GetPath()),a,b,p.GetTypeName()])
        if bodies-set(parents)!={owner} or len(edges)!=len(bodies)-1:
            raise ValueError('reference_articulation_root_not_declared_body')
        for body in bodies:
            seen=set();node=body
            while node!=owner:
                if node in seen or node not in parents:raise ValueError('reference_articulation_disconnected_or_cyclic')
                seen.add(node);node=parents[node]
        # The object reference frame has the authored container origin at bind
        # time, then moves with the declared chassis. It is not USD ancestry.
        local=self.usd_matrix(path)*self.pose_matrix(self.matrix_pose(self.usd_matrix(owner))).GetInverse()
        self.matrix_pose(local)
        return dict(row,source='physx_articulation_reference_frame',rigid_owner=owner,
                    reference_frame_semantics='authored_object_frame_rigidly_attached_to_declared_body',
                    fixed_local_matrix_row_vector=np.asarray(local).tolist(),
                    articulation_joint_topology=sorted(edges))

    def bind(self,path):
        prim=self.stage.GetPrimAtPath(path);anc,desc=self.enabled_owners(path)
        row={'prim':path,'pose_order':'xyz_xyzw','enabled_ancestor_owners':anc,'enabled_descendant_owners':desc}
        if not prim or not prim.IsValid():return dict(row,source='unavailable',reason='missing_reference_prim')
        if path in self.rigid_index:
            if path not in anc:return dict(row,source='unavailable',reason='measured_rigid_body_disabled_after_binding')
            return dict(row,source='physx_rigid_body',rigid_owner=path)
        if anc:
            owner=anc[0]
            if owner not in self.rigid_index:return dict(row,source='unavailable',reason='dynamic_owner_missing_physx_view')
            try:local=self.local_matrix(path,owner)
            except ValueError as error:return dict(row,source='unavailable',reason=str(error))
            return dict(row,source='physx_owner_fixed_descendant',rigid_owner=owner,
                        fixed_local_matrix_row_vector=np.asarray(local).tolist())
        if path in ARTICULATION_REFERENCE_BODIES:
            try:return self.articulation_binding(path,row)
            except ValueError as error:return dict(row,source='unavailable',reason=str(error))
        if desc:return dict(row,source='unavailable',reason='ambiguous_dynamic_descendants_require_individual_bindings')
        try:self.matrix_pose(self.usd_matrix(path))
        except ValueError as error:return dict(row,source='unavailable',reason=str(error))
        return dict(row,source='usd_authoritative_static',rigid_owner=None,
            initial_world_matrix=np.asarray(self.usd_matrix(path)).tolist())

    def sample(self,rigid_poses,time_s=None):
        from pxr import Gf
        poses=np.full((len(self.paths),7),np.nan,np.float64);valid=np.zeros(len(self.paths),np.bool_);states={}
        for index,binding in enumerate(self.bindings):
            path=binding['prim'];source=binding['source'];error=None
            try:
                if self.classification_revision.get(path)!=self.revision:
                    current=self.bind(path)
                    if current['source']!=source or current.get('rigid_owner')!=binding.get('rigid_owner') or current.get('articulation_joint_topology')!=binding.get('articulation_joint_topology'):
                        raise ValueError('reference_authority_changed_after_binding')
                    self.classification_revision[path]=self.revision
                if source=='physx_rigid_body':pose=np.asarray(rigid_poses[self.rigid_index[path]],float)
                elif source=='physx_owner_fixed_descendant':
                    owner=binding['rigid_owner'];local=np.asarray(self.local_matrix(path,owner))
                    if not np.allclose(local,binding['fixed_local_matrix_row_vector'],atol=1e-7,rtol=0):
                        raise ValueError('rigid_descendant_local_transform_changed')
                    world=Gf.Matrix4d(binding['fixed_local_matrix_row_vector'])*self.pose_matrix(rigid_poses[self.rigid_index[owner]])
                    pose=self.matrix_pose(world)
                elif source=='physx_articulation_reference_frame':
                    owner=binding['rigid_owner']
                    world=Gf.Matrix4d(binding['fixed_local_matrix_row_vector'])*self.pose_matrix(rigid_poses[self.rigid_index[owner]])
                    pose=self.matrix_pose(world)
                elif source=='usd_authoritative_static':pose=self.matrix_pose(self.usd_matrix(path))
                else:raise ValueError(binding.get('reason','reference_source_unavailable'))
                if not np.isfinite(pose).all():raise ValueError('nonfinite_reference_pose')
                poses[index]=pose;valid[index]=True
                states[path]={'pose':[*pose[:3],pose[6],*pose[3:6]],'source':source,'rigid_owner':binding.get('rigid_owner'),
                    'reference_binding_index':index,'pose_order':'xyz_wxyz'}
            except ValueError as exc:error=str(exc)
            if self.last_errors.get(path)!=error:
                self.log.write(json.dumps({'prim':path,'time_s':time_s,'source':source,'error':error})+'\n')
                self.last_errors[path]=error
        self.last_states=states;self.last_poses=poses;self.last_valid=valid
        return poses,valid,states

    def close(self):self.notice.Revoke();self.log.close()
