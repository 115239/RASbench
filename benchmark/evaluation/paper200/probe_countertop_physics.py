"""Measured original kitchen assets: protected-state positive/negative controls."""
from pathlib import Path
import argparse
from .common import read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--name',default='countertop_physics_001');p.add_argument('--gpu',type=int,default=0);a=p.parse_args();root=a.root.resolve();out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1','--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=8']})
 try:
  from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade
  import numpy as np
  from isaacsim.core.api import World
  from isaacsim.core.api.objects import DynamicSphere
  from isaacsim.core.prims import SingleRigidPrim
  from safety_tools.discrete_physics import evaluate
  from safety_tools.physx_contacts import SafetyContactTracker
  source_path=read(root/'manifests/geometry.json')['questions']['Q92']['scene'];source=Usd.Stage.Open(source_path)
  checks=read(root/'benchmark/task/Q92/data/safety/safety_profile.json')['rules']['S05']['parameters']['checks']
  world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.0);world.get_physics_context().set_gravity(0.);stage=world.stage
  bodies={};paths=sorted(set(x['prim_path'] for x in checks));source_bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
  for i,path in enumerate(paths):
   origin=source.GetPrimAtPath(path);prim=stage.DefinePrim(path,'Xform');prim.GetReferences().AddReference(source_path,path);prim.SetInstanceable(False)
   xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp().Set(UsdGeom.Xformable(origin).ComputeLocalToWorldTransform(Usd.TimeCode.Default()));xf.SetResetXformStack(True)
   UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
   PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
   bodies[path]=None
  diagnostic_disabled_joints=[]
  for prim in stage.Traverse():
   if prim.IsA(UsdPhysics.Joint):
    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False);diagnostic_disabled_joints.append(str(prim.GetPath()))
  stimulus=world.scene.add(DynamicSphere(prim_path='/DiagnosticManipulator/link7',name='stimulus',position=np.array([20.,20.,20.]),radius=.015,mass=.05))
  PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath('/DiagnosticManipulator/link7')).CreateThresholdAttr(0.)
  tracker=SafetyContactTracker(stage,robot_root_prim='/DiagnosticManipulator',physics_dt=.005);world.reset()
  import omni.physics.tensors
  physics=omni.physics.tensors.create_simulation_view('numpy');physics.set_subspace_roots('/')
  for path in bodies:bodies[path]=physics.create_rigid_body_view(path)
  def states():
   result={}
   for path,obj in bodies.items():
    pose=obj.get_transforms()[0];result[path]={'pose':np.r_[pose[:3],pose[6],pose[3:6]].tolist(),'rigid_owner':path,'source':'actual_physx_rigid_body'}
   return result
  references=states();rows=[]
  board='/World_01/kitchen_chopping_board_with_food/E_Component96_1';bb=source_bbox.ComputeWorldBound(source.GetPrimAtPath(board)).ComputeAlignedRange();center=np.asarray(bb.GetMidpoint());center[2]=bb.GetMax()[2]+.010
  relative_checks=[x for x in checks if x['criterion']=='relative_pose_closed']
  for step in range(1,81):
   if step==41:
    stimulus.set_world_pose(position=center);stimulus.set_linear_velocity(np.zeros(3))
    for spec in relative_checks:
     obj=bodies[spec['prim_path']];pose=obj.get_transforms().copy();pose[0,2]+=.03;obj.set_transforms(pose,np.array([0],np.uint32))
   tracker.begin_step();world.step(render=False);sample={'object_states':states(),'contacts':tracker.snapshot()}
   outcomes=[]
   for spec in checks:
    if spec['criterion'] in ('relative_pose_closed','no_robot_manipulation'):
     value,metrics=evaluate(spec,sample,references);outcomes.append({'criterion':spec['criterion'],'prim_path':spec['prim_path'],'violation':value,'metrics':metrics})
   rows.append({'physics_step':step,'sim_time_s':step*.005,'outcomes':outcomes,'contacts':sample['contacts'],'object_states':sample['object_states']})
  negative=all(x['violation'] is False for r in rows[:40] for x in r['outcomes'])
  positive_relative=all(any(x['violation'] is True for r in rows[40:] for x in r['outcomes'] if x['prim_path']==s['prim_path'] and x['criterion']=='relative_pose_closed') for s in relative_checks)
  positive_contact=any(x['violation'] is True for r in rows[40:] for x in r['outcomes'] if x['criterion']=='no_robot_manipulation' and x['prim_path']==board)
  verdict={'negative_all_states':negative,'positive_lid_and_drawer':positive_relative,'positive_actual_manipulator_contact':positive_contact}
  write(out/'COUNTERTOP_PHYSICS_CONTROL.json',{'scope':'isolated original 15 countertop assets, controlled kinematic lid/drawer motion and actual dynamic diagnostic gripper collider contacts; not a model episode','source_scene':source_path,'isolated_control_only_disabled_joints':diagnostic_disabled_joints,'rows':rows,'verdict':verdict,'passed':all(verdict.values())})
  print('COUNTERTOP_CONTROL',verdict,flush=True)
  # Read-only materials inventory requested by attack binding review.
  materials=[]
  for prim in source.Traverse():
   path=str(prim.GetPath())
   if prim.IsA(UsdGeom.Mesh) and any(k in path.lower() for k in ('floor','mesa','cabinet','door')):
    mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial();bound=source_bbox.ComputeWorldBound(prim).ComputeAlignedRange()
    materials.append({'prim':path,'bounds':[list(bound.GetMin()),list(bound.GetMax())],'material':str(mat.GetPath()) if mat else None})
  write(root/'reports/Q92_SURFACE_MATERIALS.json',materials)
 except Exception:
  import traceback
  error=traceback.format_exc();(out/'ERROR.txt').write_text(error);print(error,flush=True);raise
 finally:app.close()

if __name__=='__main__':main()
