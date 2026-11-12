"""Native PhysX authority controls, isolated from task/model experiments."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .common import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--name',default='reference_authority_001');a=p.parse_args();root=a.root.resolve()
    out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/plugins/carb.tasking.plugin/threadCount=8','--/app/extensions/fsWatcherEnabled=false']})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from pxr import UsdGeom,UsdPhysics,Gf
        import omni.physics.tensors
        import numpy as np
        from .reference_states import ReferenceStateRegistry
        world=World(physics_dt=.005,rendering_dt=.1);stage=world.stage
        body=world.scene.add(DynamicCuboid(prim_path='/Owner',name='owner',position=np.array([2.,0.,1.]),size=.1))
        child=UsdGeom.Xform.Define(stage,'/Owner/Child');child.AddTranslateOp().Set(Gf.Vec3d(.2,0,0))
        static=UsdGeom.Xform.Define(stage,'/Static');static_op=static.AddTranslateOp();static_op.Set(Gf.Vec3d(4.,0,1.))
        world.reset();sim=omni.physics.tensors.create_simulation_view('numpy');sim.set_subspace_roots('/')
        view=sim.create_rigid_body_view(['/Owner'])
        c=SimpleNamespace(final_stage=stage,paper_task_config={},safety_profile={'rules':{'S12':{'enabled':True,
            'parameters':{'objects':[{'prim_path':p} for p in ['/Owner','/Owner/Child','/Static']]}}}})
        registry=ReferenceStateRegistry(c,list(view.prim_paths),out)
        try:
            world.step(render=False);actual=view.get_transforms().copy();poses,valid,_=registry.sample(actual,.005)
            indices={p:i for i,p in enumerate(registry.paths)}
            base=poses[indices['/Owner']];desc=poses[indices['/Owner/Child']]
            checks={'actual_body_and_child_valid':bool(valid.all()),
                'fixed_child_offset':bool(np.allclose(desc[:3]-base[:3],[.2,0,0],atol=1e-6))}
            # Deliberately perturb authored world transform between physics
            # steps: dynamic child must still follow the unchanged PhysX pose.
            owner_op=next(op for op in UsdGeom.Xformable(stage.GetPrimAtPath('/Owner')).GetOrderedXformOps()
                          if op.GetOpType()==UsdGeom.XformOp.TypeTranslate)
            owner_op.Set(Gf.Vec3d(900.,0.,1.))
            poses2,valid2,_=registry.sample(actual,.005)
            checks['dynamic_child_ignores_stale_usd_world_pose']=bool(valid2.all() and np.allclose(poses2[indices['/Owner/Child']],desc,atol=1e-6))
            static_op.Set(Gf.Vec3d(4.1,0,1.));poses3,valid3,_=registry.sample(actual,.005)
            checks['static_live_usd_changes_observed']=bool(valid3[indices['/Static']] and abs(poses3[indices['/Static'],0]-4.1)<1e-8)
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/Static')).CreateRigidBodyEnabledAttr(True)
            _,valid4,_=registry.sample(actual,.005)
            checks['static_became_dynamic_does_not_fallback']=not bool(valid4[indices['/Static']])
            result={'passed':all(checks.values()),'checks':checks,'bindings':registry.bindings,
                'scope':'independent native PhysX/Usd authority controls, not task success'}
            write(out/'REFERENCE_AUTHORITY_CHECK.json',result);print('REFERENCE_AUTHORITY_CHECK',result['passed'],checks,flush=True)
        finally:registry.close()
    finally:app.close()


if __name__=='__main__':main()
