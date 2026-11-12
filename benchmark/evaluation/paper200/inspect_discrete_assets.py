"""Inspect authoritative graph variables and unresolved cap geometry, read-only."""
import argparse
from pathlib import Path
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args();r=a.root.resolve()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(r/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        from pxr import Usd,UsdPhysics
        geometry=read(r/'manifests/geometry.json')['questions'];result={}
        for q in ('Q28','Q128','Q130','Q136','Q92'):
            stage=Usd.Stage.Open(geometry[q]['scene'])
            if q=='Q28':prefix='/World/Power_supply_area/Xform/ActionGraph'
            elif q=='Q92':prefix='/World_01/kitchen_chopping_board_with_food'
            else:
                profile=read(r/'benchmark/task'/q/'data/safety/safety_profile.json')
                prefix=profile['rules']['S05']['parameters']['checks'][0]['prim_path']
            rows=[]
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix),Usd.TraverseInstanceProxies()):
                attrs={}
                for attr in prim.GetAttributes():
                    if q=='Q28' or 'state' in attr.GetName().lower() or 'joint' in attr.GetName().lower():
                        attrs[attr.GetName()]={'type':str(attr.GetTypeName()),'value':str(attr.Get()),'connections':[str(x) for x in attr.GetConnections()]}
                rows.append({'prim':str(prim.GetPath()),'type':prim.GetTypeName(),
                    'rigid_body':prim.HasAPI(UsdPhysics.RigidBodyAPI),'collision':prim.HasAPI(UsdPhysics.CollisionAPI),
                    'attributes':attrs})
            result[q]={'prefix':prefix,'prims':rows}
        write(r/'reports/DISCRETE_ASSET_INSPECTION.json',result)
        print({k:len(v['prims']) for k,v in result.items()},flush=True)
    finally:app.close()


if __name__=='__main__':main()
