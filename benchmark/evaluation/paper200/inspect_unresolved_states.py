"""Read physical cap meshes/material subsets and island-top object candidates."""
import argparse
from pathlib import Path
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();root=a.root.resolve()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        import numpy as np
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
        result={};geometry=read(root/'manifests/geometry.json')['questions']
        for q in ('Q128','Q130','Q136','Q92'):
            stage=Usd.Stage.Open(geometry[q]['scene']);bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
            if q=='Q92':
                target=read(root/'benchmark/task/Q92/data/task/task.json')['tracked_objects']['target_0']
                bound=bbox.ComputeWorldBound(stage.GetPrimAtPath(target)).ComputeAlignedRange()
                bounds=np.array([bound.GetMin(),bound.GetMax()]);rows=[]
                for prim in stage.Traverse():
                    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
                    b=bbox.ComputeWorldBound(prim).ComputeAlignedRange();v=np.array([b.GetMin(),b.GetMax()])
                    if not np.isfinite(v).all():continue
                    center=v.mean(0)
                    if np.all(center[:2]>=bounds[0,:2]-.1) and np.all(center[:2]<=bounds[1,:2]+.1) and bounds[1,2]-.1<=center[2]<=bounds[1,2]+1:
                        rows.append({'prim':str(prim.GetPath()),'bounds':v.tolist()})
                result[q]={'target':target,'target_bounds':bounds.tolist(),'top_rigid_candidates':rows};continue
            target=read(root/'benchmark/task'/q/'data/safety/safety_profile.json')['rules']['S05']['parameters']['checks'][0]['prim_path'];rows=[]
            for prim in Usd.PrimRange(stage.GetPrimAtPath(target),Usd.TraverseInstanceProxies()):
                if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):continue
                mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial();b=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                row={'prim':str(prim.GetPath()),'type':prim.GetTypeName(),'material':str(mat.GetPath()) if mat else None,
                     'bounds':[list(b.GetMin()),list(b.GetMax())],'attributes':{}}
                for name in ('faceVertexCounts','indices','familyName','elementType'):
                    attr=prim.GetAttribute(name)
                    if attr and attr.Get() is not None:row['attributes'][name]=str(attr.Get())[:1200]
                if q=='Q130' and prim.IsA(UsdGeom.Mesh):
                    from scipy.sparse import coo_matrix
                    from scipy.sparse.csgraph import connected_components
                    mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get())
                    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                    unique,inverse=np.unique(np.round(points,5),axis=0,return_inverse=True)
                    starts=np.r_[0,np.cumsum(counts)];edges=[]
                    for i in range(len(counts)):
                        face=inverse[indices[starts[i]:starts[i+1]]]
                        edges.extend(zip(face,np.roll(face,-1)))
                    edges=np.array(edges)
                    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(unique),len(unique)))
                    total,labels=connected_components(graph,directed=False)
                    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
                    world=np.c_[unique,np.ones(len(unique))]@matrix
                    components=[]
                    for label in range(total):
                        pts=world[labels==label,:3]
                        components.append({'id':label,'vertices':len(pts),'bounds':[pts.min(0).tolist(),pts.max(0).tolist()]})
                    row['connected_components_world']=components
                rows.append(row)
            result[q]={'target':target,'meshes':rows}
        write(root/'reports/UNRESOLVED_STATE_COMPONENTS.json',result);print('inspection complete',flush=True)
    finally:app.close()


if __name__=='__main__':main()
