"""Read composed protected-object geometry and button state attributes."""
import argparse
from pathlib import Path
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);args=p.parse_args();root=args.root.resolve()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        from pxr import Usd,UsdGeom,UsdPhysics
        result={}
        inventory=read(root/'manifests/geometry.json')['questions']
        for q,prefix in [('Q96','/World_01/kitchen_chopping_board_with_food'),
                         ('Q28','/World/Power_supply_area')]:
            stage=Usd.Stage.Open(inventory[q]['scene']);rows=[]
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix),Usd.TraverseInstanceProxies()):
                path=str(prim.GetPath());attrs={}
                for attr in prim.GetAttributes():
                    name=attr.GetName();typ=str(attr.GetTypeName())
                    if typ=='bool' or any(x in name.lower() for x in ('button','state','trigger','script','filename')):
                        attrs[name]={'type':typ,'value':str(attr.Get()),'connections':[str(x) for x in attr.GetConnections()]}
                item={'prim':path,'type':prim.GetTypeName(),'rigid_body':prim.HasAPI(UsdPhysics.RigidBodyAPI),
                      'collision':prim.HasAPI(UsdPhysics.CollisionAPI),'attributes':attrs}
                if q=='Q96' and prim.IsA(UsdGeom.Xform):
                    bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    item['aabb']=[list(bounds.GetMin()),list(bounds.GetMax())]
                if attrs or q=='Q96':rows.append(item)
            result[q]=rows
        write(root/'reports/REPAIR_ASSET_INSPECTION.json',result)
        print('REPAIR_ASSET_INSPECTION', {q:len(v) for q,v in result.items()},flush=True)
    finally:app.close()


if __name__=='__main__':main()
