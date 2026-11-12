"""Inspect composed scene geometry; no policy or scoring decisions are made."""
import argparse
from pathlib import Path
from .common import read,write,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);a=p.parse_args();root=a.root.resolve()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'extra_args':['--/app/extensions/fsWatcherEnabled=false']})
    from pxr import Usd,UsdGeom,UsdPhysics
    import numpy as np
    scenes={};records={};issues=[]
    for row in read(root/'manifests/questions.json'):
        q=row['question'];package=root/'benchmark/task'/q
        env=read(package/'data/env/environment.json');task=read(package/'data/task/task.json')
        scene=(package/env['scene_usd']).resolve()
        if str(scene) not in scenes:
            stage=Usd.Stage.Open(str(scene));scenes[str(scene)]=(stage,UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']),UsdGeom.XformCache())
        stage,bbox,xform=scenes[str(scene)];objects={}
        for spec in task['rubric_evidence']['objects']:
            path=spec['prim'];prim=stage.GetPrimAtPath(path);item={'valid':bool(prim),'scene':str(scene)}
            if path.startswith('/M20PiperVLA'):continue
            if prim:
                colliders=[str(p.GetPath()) for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
                meshes=[str(p.GetPath()) for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
                bound=bbox.ComputeWorldBound(prim).ComputeAlignedRange();lo=np.array(bound.GetMin());hi=np.array(bound.GetMax())
                item.update(colliders=colliders,meshes=meshes,translation=list(xform.GetLocalToWorldTransform(prim).ExtractTranslation()),matrix=np.asarray(xform.GetLocalToWorldTransform(prim)).tolist(),aabb=[lo.tolist(),hi.tolist()] if np.isfinite([lo,hi]).all() else None)
                if q in ['Q44','Q45'] and path==task['tracked_objects']['target_0']:
                    parts=[]
                    for meshpath in meshes:
                        mesh=stage.GetPrimAtPath(meshpath);points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),float)
                        m=np.asarray(xform.GetLocalToWorldTransform(mesh)).T
                        parts.append(points@m[:3,:3].T+m[:3,3])
                    points=np.concatenate(parts);centered=points-points.mean(axis=0)
                    values,axes=np.linalg.eigh(centered.T@centered)
                    item['principal_axes_w']=axes[:,np.argsort(values)[::-1]].T.tolist()
                if not colliders:
                    parent=prim.GetParent()
                    while parent and str(parent.GetPath()) not in ['/','/World','/World_01']:
                        physical=[str(x.GetPath()) for x in Usd.PrimRange(parent,Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(x).GetCollisionEnabledAttr().Get() is not False]
                        if physical:
                            item['nearest_collision_ancestor']=str(parent.GetPath());item['ancestor_colliders']=physical;break
                        parent=parent.GetParent()
            else:issues.append({'question':q,'prim':path,'reason':'missing composed prim'})
            objects[path]=item
        records[q]={'scene':str(scene),'objects':objects}
        write(root/'reports/geometry_audit.partial.json',{'completed':len(records),'issues':issues})
    write(root/'manifests/geometry.json',{'created':now(),'questions':records,'issues':issues})
    print('GEOMETRY_AUDIT',len(records),'questions;',len(issues),'issues',flush=True)
    app.close()
if __name__=='__main__':main()
