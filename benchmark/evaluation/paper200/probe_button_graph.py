"""Developer-only native graph diagnostics; never a model experiment."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .common import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--name',default='button_graph_002');a=p.parse_args();root=a.root.resolve();out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        import omni.kit.app,omni.graph.core as og
        from isaacsim.core.api import World
        from pxr import UsdGeom,Usd,UsdPhysics
        import numpy as np
        from isaacsim.core.api.objects import DynamicSphere
        manager=omni.kit.app.get_app().get_extension_manager()
        names=['isaacsim.core.nodes','omni.physx.graph','omni.graph.action','omni.graph.nodes','omni.graph.scriptnode']
        before={n:manager.is_extension_enabled(n) for n in names}
        for name in names:manager.set_extension_enabled_immediate(name,True)
        world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.0)
        stage=world.stage;cabinet='/World/Power_supply_area/Xform'
        prim=stage.DefinePrim(cabinet,'Xform')
        scene=read(root/'manifests/geometry.json')['questions']['Q28']['scene']
        prim.GetReferences().AddReference(scene,cabinet)
        for _ in range(5):app.update()
        probe=world.scene.add(DynamicSphere(prim_path='/ProbeStimulus',name='probe',position=np.array([100.,100.,100.]),radius=.003))
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath('/ProbeStimulus')).CreateKinematicEnabledAttr(True)
        bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
        trigger=cabinet+'/Triggers/YellowPressTrigger'
        bounds=bbox.ComputeWorldBound(stage.GetPrimAtPath(trigger)).ComputeAlignedRange()
        center=np.array(bounds.GetMidpoint())
        world.reset()
        c=SimpleNamespace(final_stage=stage,physics_step=0,paper_lifecycle=SimpleNamespace(phase='TASK_RUNNING'),
            safety_profile=read(root/'benchmark/task/Q28/data/safety/safety_profile.json'))
        from .discrete_state import DiscreteObserver
        observer=DiscreteObserver(c,out,.005)
        nodes=[]
        for prim in stage.Traverse():
            if prim.GetTypeName()=='OmniGraphNode':
                node=og.Controller.node(str(prim.GetPath()))
                nodes.append({'path':str(prim.GetPath()),'type':node.get_type_name(),'valid':node.is_valid(),'disabled':node.is_disabled(),'relationships':{r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}})
        samples=[]
        for step in range(40):
            if step==15:probe.set_world_pose(position=center)
            if step==16:probe.set_world_pose(position=np.array([100.,100.,100.]))
            world.step(render=False)
            samples.append(observer.sample(step+1));c.physics_step=step+1
        write(out/'GRAPH_DIAGNOSTICS.json',{'before_extensions':before,'nodes':nodes,'samples':samples,'source_scene':scene,'controlled_stimulus':{'type':'kinematic collider entering actual trigger region for one physics step','target':trigger,'center':center.tolist(),'enter_step':16,'exit_step':17,'not_a_model_episode':True}})
        observer.close();print('GRAPH_DIAGNOSTICS',out,flush=True)
    finally:app.close()


if __name__=='__main__':main()
