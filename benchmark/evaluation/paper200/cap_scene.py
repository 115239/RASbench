"""Versioned detachable closures made from existing asset triangles.

This is a breakaway closure approximation, not a simulated screw thread.
All authoring lives in the per-episode session layer; original scenes remain intact.
"""
import hashlib
import json
import numpy as np


def _select_faces(mesh, selected):
    from pxr import UsdGeom,Vt
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
    offsets=np.r_[0,np.cumsum(counts)]
    corners=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in selected])
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        interpolation=pv.GetInterpolation()
        if interpolation not in ('faceVarying','uniform'):continue
        selection=corners if interpolation=='faceVarying' else selected
        if pv.IsIndexed():
            values=pv.GetIndices();pv.SetIndices(type(values)([values[int(i)] for i in selection]))
        else:
            values=pv.Get()
            if values is not None:pv.Set(type(values)([values[int(i)] for i in selection]))
    normals=mesh.GetNormalsAttr().Get()
    if normals and mesh.GetNormalsInterpolation() in ('faceVarying','uniform'):
        selection=corners if mesh.GetNormalsInterpolation()=='faceVarying' else selected
        mesh.GetNormalsAttr().Set(type(normals)([normals[int(i)] for i in selection]))
    chosen=indices[corners]
    used=np.unique(chosen);remap=np.full(len(mesh.GetPointsAttr().Get()),-1,dtype=int);remap[used]=np.arange(len(used))
    # Convex cooking may consume every authored point, not just referenced faces.
    # Remove unused vertices from both body and cap so their hulls cannot overlap
    # because a discarded triangle left a remote vertex behind.
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if pv.GetInterpolation() not in ('vertex','varying'):continue
        if pv.IsIndexed():
            values=pv.GetIndices();pv.SetIndices(type(values)([values[int(i)] for i in used]))
        else:
            values=pv.Get()
            if values is not None:pv.Set(type(values)([values[int(i)] for i in used]))
    normals=mesh.GetNormalsAttr().Get()
    if normals and mesh.GetNormalsInterpolation() in ('vertex','varying'):
        mesh.GetNormalsAttr().Set(type(normals)([normals[int(i)] for i in used]))
    points=np.asarray(mesh.GetPointsAttr().Get(),np.float32)[used]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0),points.max(0)])))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts[selected].tolist()))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(remap[chosen].tolist()))


def install_cap(stage,spec):
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf,Vt
    target=stage.GetPrimAtPath(spec['body_prim']);source=stage.GetPrimAtPath(spec['source_mesh'])
    if not target or not source or not source.IsA(UsdGeom.Mesh):raise ValueError('closure source mesh missing')
    original=UsdGeom.Mesh(source);points=np.asarray(original.GetPointsAttr().Get(),float)
    matrix=np.asarray(UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world=np.c_[points,np.ones(len(points))]@matrix;world=world[:,:3]
    counts=np.asarray(original.GetFaceVertexCountsAttr().Get(),int);indices=np.asarray(original.GetFaceVertexIndicesAttr().Get(),int)
    if spec['selection']=='whole_mesh':selected=np.arange(len(counts))
    elif spec['selection']=='tube_existing_bottom_ring':
        # Frozen measured ring: 20 mm above the bottom. No face is clipped or invented.
        end=world[:,2].min()+spec['ring_height_m'];offsets=np.r_[0,np.cumsum(counts)]
        selected=np.array([i for i in range(len(counts)) if np.max(world[indices[offsets[i]:offsets[i+1]],2])<=end+1e-6])
        if len(selected)==0 or len(selected)==len(counts):raise ValueError('invalid existing cap ring selection')
        # The remaining body must start at this same original ring, with no crossing cap faces.
        remaining=np.setdiff1d(np.arange(len(counts)),selected)
        body_vertices=np.unique(np.concatenate([indices[offsets[i]:offsets[i+1]] for i in remaining]))
        if abs(float(world[body_vertices,2].min())-end)>1e-6:raise ValueError('tube split does not follow frozen mesh ring')
    else:raise ValueError('unknown closure selection')
    offsets=np.r_[0,np.cumsum(counts)];used=np.unique(np.concatenate([indices[offsets[i]:offsets[i+1]] for i in selected]))
    center=(world[used].min(0)+world[used].max(0))/2
    cap=UsdGeom.Xform.Define(stage,spec['cap_prim']);cap.AddTranslateOp().Set(Gf.Vec3d(*center));cap.SetResetXformStack(True)
    capmesh=spec['cap_prim']+'/Geometry';flat=stage.Flatten()
    Sdf.CopySpec(flat,source.GetPath(),stage.GetEditTarget().GetLayer(),Sdf.Path(capmesh))
    clone=UsdGeom.Mesh(stage.GetPrimAtPath(capmesh));UsdGeom.Xformable(clone).ClearXformOpOrder()
    clone.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((world-center).astype(np.float32)))
    if clone.GetNormalsAttr().Get():
        normals=np.asarray(clone.GetNormalsAttr().Get(),float)@np.linalg.inv(matrix[:3,:3]).T
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
        clone.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    _select_faces(clone,selected)
    recovered=np.asarray(clone.GetPointsAttr().Get(),float)+center
    geometry_error=float(np.max(np.abs(recovered-world[used])))
    if geometry_error>1e-6:raise ValueError('closure adapter moved visible vertices')
    if spec['selection']=='whole_mesh':source.SetActive(False)
    else:_select_faces(original,np.setdiff1d(np.arange(len(counts)),selected))
    for item in (cap.GetPrim(),clone.GetPrim()):
        if item.HasAPI(UsdPhysics.RigidBodyAPI):item.RemoveAPI(UsdPhysics.RigidBodyAPI)
    UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(cap.GetPrim()).CreateMassAttr(spec['mass_kg'])
    UsdPhysics.CollisionAPI.Apply(clone.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(clone.GetPrim()).CreateApproximationAttr('convexHull')
    UsdPhysics.RigidBodyAPI.Apply(target).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(target).CreateMassAttr(spec['body_mass_kg'])
    bodymatrix=UsdGeom.Xformable(target).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inverse=bodymatrix.GetInverse();local=inverse.Transform(Gf.Vec3d(*center));rotation=bodymatrix.ExtractRotationQuat().GetInverse()
    joint=UsdPhysics.FixedJoint.Define(stage,spec['joint_prim'])
    joint.CreateBody0Rel().SetTargets([target.GetPath()]);joint.CreateBody1Rel().SetTargets([cap.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(local));joint.CreateLocalRot0Attr(Gf.Quatf(rotation))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateBreakForceAttr(spec['break_force_n']);joint.CreateBreakTorqueAttr(spec['break_torque_nm']);joint.CreateCollisionEnabledAttr(False)
    rotation=bodymatrix.ExtractRotationQuat();imag=rotation.GetImaginary();origin=bodymatrix.ExtractTranslation()
    authored={'cap':{'pose':[*center,1.,0.,0.,0.]},
              'body':{'pose':[*origin,float(rotation.GetReal()),*imag]}}
    cap.GetPrim().SetCustomDataByKey('paper200:authoredClosureReference',json.dumps(authored))
    # The hidden closure observer reads these actual independent body poses, not a constant state flag.
    return {'spec':spec,'cap_world_center':center.tolist(),'authored_closure_reference':authored,'selected_face_indices':selected.tolist(),
        'original_faces':len(counts),'cap_vertex_count':len(clone.GetPointsAttr().Get()),'cap_face_point_reference_count':len(np.unique(np.asarray(clone.GetFaceVertexIndicesAttr().Get()))),'source_geometry_sha256':hashlib.sha256(points.tobytes()+counts.tobytes()+indices.tobytes()).hexdigest(),
        'source_world_bounds':[world[used].min(0).tolist(),world[used].max(0).tolist()],
        'original_visible_geometry_preserved':True,'maximum_world_vertex_error_m':geometry_error,'thread_mechanics_simulated':False,
        'mechanics':'independent rigid cap, breakaway fixed joint; closure evaluated from actual relative body poses'}


def apply_scene_state_repairs(stage,package):
    from .common import read
    path=package/'data/scene/discrete_state_adapter.json'
    if not path.exists():return None
    config=read(path)
    return {'schema':'paper200-physical-closure/v2','changes':[install_cap(stage,x) for x in config.get('closures',[])],
            'all_systems_and_conditions_identical':True,'adapter_contract':config}
