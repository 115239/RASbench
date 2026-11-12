"""Native USD/PhysX diagnostics of the actual missing car drive colliders."""
import argparse
from pathlib import Path
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--name',default='collision_sdf_query_001')
    a=p.parse_args();root=a.root.resolve()
    out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools
        from isaacsim.core.api import World
        import omni.physx as physx
        import numpy as np
        from safety_tools.physx_geometry import SafetyGeometryTracker
        scene=read(root/'manifests/geometry.json')['questions']['Q1']['scene']
        source=Usd.Stage.Open(scene);rows=[]
        for side in ('FL','RL','CL','CR','FR','RR'):
            for suffix in ('','_01','_02'):
                path='/World/Cars/c_01/%s_Drive/Mesh%s'%(side,suffix);prim=source.GetPrimAtPath(path)
                attrs={attr.GetName():str(attr.Get())[:3000] for attr in prim.GetAttributes()
                    if 'phys' in attr.GetName().lower() or attr.GetName() in ('points','extent','faceVertexCounts','faceVertexIndices')}
                rows.append({'path':path,'type':prim.GetTypeName(),'schemas':list(prim.GetAppliedSchemas()),'attrs':attrs})
        write(out/'ATTRIBUTES.json',{'scene':scene,'rows':rows})
        world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.)
        stage=world.stage;path='/World/Cars/c_01';stage.DefinePrim(path,'Xform').GetReferences().AddReference(scene,path)
        for _ in range(4):app.update()
        world.reset()
        for _ in range(2):world.step(render=False)
        cooking=physx.get_physx_cooking_interface();cache=UsdGeom.XformCache()
        query=physx.get_physx_property_query_interface();queried=[];finished=[]
        for side in ('FL','RL','CL','CR','FR','RR'):
            owner='/World/Cars/c_01/%s_Drive'%side
            def collider(info,owner=owner):
                queried.append({'query_owner':owner,'path':str(PhysicsSchemaTools.intToSdfPath(info.path_id)),
                    'result':str(info.result),'aabb_local_min':list(info.aabb_local_min),'aabb_local_max':list(info.aabb_local_max),
                    'local_pos':list(info.local_pos),'local_rot_xyzw':list(info.local_rot),'volume':float(info.volume)})
            query.query_prim(stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                prim_id=PhysicsSchemaTools.sdfPathToInt(owner),collider_fn=collider,
                finished_fn=lambda owner=owner:finished.append(owner),timeout_ms=1000)
        immediate={'callbacks':len(queried),'finished':list(finished)}
        for _ in range(10):app.update()
        write(out/'PHYSX_PROPERTY_QUERY.json',{'immediate':immediate,'finished':finished,'colliders':queried})
        results=[]
        for item in rows:
            prim=stage.GetPrimAtPath(item['path']);result=dict(path=item['path'])
            def cb(status,meshes):
                result['cooking_status']=str(status)
                result['cooked_parts']=[{'vertices':[[float(v.x),float(v.y),float(v.z)] for v in mesh.vertices],
                    'fields':dir(mesh)} for mesh in meshes]
            try:
                cooking.request_convex_collision_representation(stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(item['path']),run_asynchronously=False,on_result=cb)
                result['extracted_parts']=len(SafetyGeometryTracker._mesh_parts(prim,cache,np.eye(4)))
            except Exception as error:result['error']=repr(error)
            results.append(result)
        write(out/'NATIVE_COOKING.json',{'scene':scene,'actual_car_reference':path,
            'cooking_interface_methods':{n:str(getattr(cooking,n).__doc__) for n in dir(cooking) if not n.startswith('_')},
            'physx_interface_methods':{n:str(getattr(physx.get_physx_interface(),n).__doc__) for n in dir(physx.get_physx_interface()) if 'mesh' in n or 'shape' in n},
            'results':results})
        print('CAR_DRIVE_NATIVE_DIAGNOSTICS_COMPLETE',flush=True)
    finally:app.close()


if __name__=='__main__':main()
