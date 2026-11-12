"""Actual asset cap/body PhysX positive and negative controls, no model success labels."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .common import read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--name',default='cap_physics_001');p.add_argument('--gpu',type=int,default=0);p.add_argument('--stimulus',choices=['release','force'],default='release');a=p.parse_args();root=a.root.resolve();out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1','--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=8']})
 try:
  import numpy as np
  from pxr import Usd,UsdGeom,UsdPhysics,Gf
  from isaacsim.core.api import World
  from isaacsim.core.prims import SingleRigidPrim
  from .cap_scene import apply_scene_state_repairs
  from safety_tools.discrete_physics import evaluate
  world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.0);stage=world.stage
  world.get_physics_context().set_gravity(0.)
  geometry=read(root/'manifests/geometry.json')['questions'];items=[]
  for q in ('Q128','Q130','Q136'):
   package=root/'benchmark/task'/q;spec=read(package/'data/scene/discrete_state_adapter.json')['closures'][0];scene=geometry[q]['scene']
   source=Usd.Stage.Open(scene);target=source.GetPrimAtPath(spec['body_prim']);matrix=UsdGeom.Xformable(target).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
   prim=stage.DefinePrim(spec['body_prim'],'Xform');prim.GetReferences().AddReference(scene,spec['body_prim']);prim.SetInstanceable(False)
   xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp().Set(matrix);xf.SetResetXformStack(True)
   # Source may put body on its sole mesh. Normalize as common Runner does.
   from .scene_adapters import normalize_target_rigid_owner,canonicalize_pose_parity
   normalize_target_rigid_owner(stage,spec['body_prim']);canonicalize_pose_parity(stage,spec['body_prim'])
   revision=apply_scene_state_repairs(stage,package);UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
   body=SingleRigidPrim(prim_path=spec['body_prim'],name=q+'_body',reset_xform_properties=False)
   cap=SingleRigidPrim(prim_path=spec['cap_prim'],name=q+'_cap',reset_xform_properties=False)
   check=read(package/'data/safety/safety_profile.json')['rules']['S05']['parameters']['checks'][0]
   items.append((q,spec,check,body,cap,revision))
  world.reset();rows=[]
  for q,spec,check,body,cap,revision in items:body.initialize();cap.initialize()
  def sample(step):
   for q,spec,check,body,cap,revision in items:
    bp,bq=body.get_world_pose();cp,cq=cap.get_world_pose()
    refs=revision['changes'][0]['authored_closure_reference']
    states={spec['body_prim']:{'pose':np.r_[bp,bq].tolist(),'rigid_owner':spec['body_prim']},spec['cap_prim']:{'pose':np.r_[cp,cq].tolist(),'rigid_owner':spec['cap_prim']}}
    value,metrics=evaluate(check,{'object_states':states,'closure_references':{spec['cap_prim']:refs}}, {})
    rows.append({'question':q,'physics_step':step,'sim_time_s':step*.005,'violation':value,'metrics':metrics})
  for step in range(1,81):
   if step==41 and a.stimulus=='release':
    # Explicit opening control: release actual fixed joint, then apply upward velocity.
    # This is diagnostic stimulus only, never credited as policy action or attack.
    for q,spec,check,body,cap,revision in items:
     UsdPhysics.FixedJoint(stage.GetPrimAtPath(spec['joint_prim'])).CreateJointEnabledAttr(False)
     cap.set_linear_velocity(np.array([0.,0.,.5]))
   if a.stimulus=='force' and 41<=step<=43:
    for q,spec,check,body,cap,revision in items:cap._rigid_prim_view.apply_forces(np.array([[0.,0.,25.]],np.float32),is_global=True)
   world.step(render=False);sample(step)
  verdict={q:{'closed_negative':all(r['violation'] is False for r in rows if r['question']==q and r['physics_step']<=40),
               'opened_positive':any(r['violation'] is True for r in rows if r['question']==q and r['physics_step']>41)} for q,*_ in items}
  write(out/'CAP_PHYSICS_CONTROL.json',{'scope':'isolated original asset PhysX bodies; explicit '+a.stimulus+' opening stimulus, not model task success',
    'stimulus':a.stimulus,'force_stimulus':{'world_force_n':[0,0,25],'physics_steps':[41,42,43],'joint_enabled_unchanged':True} if a.stimulus=='force' else None,'dt_s':.005,'initial_gravity':0.,'rows':rows,'revisions':[x[-1] for x in items],'verdict':verdict,
    'passed':all(all(v.values()) for v in verdict.values())});print('CAP_CONTROL',verdict,flush=True)
 except Exception:
  import traceback
  error=traceback.format_exc();(out/'ERROR.txt').write_text(error);print(error,flush=True);raise
 finally:app.close()

if __name__=='__main__':main()
