"""Isolated real PhysX controls; not model episodes or benchmark scores.

Grasp/drop controls use a documented calibration cube and velocity-controlled
dynamic fingers. S12 controls use the actual Q84 basket and Q95 platter meshes.
No contact, velocity or pose observation is fabricated.
"""
import argparse
import copy
import json
from pathlib import Path
from .common import read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--name',default='physical_controls_001');a=p.parse_args();root=a.root.resolve()
    out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from safety_tools.physx_contacts import SafetyContactTracker
        from safety_tools.drop_state import DropState
        from safety_tools.monitor import EpisodeSafetyMonitor
        from .physical_opportunities import LiftOpportunity
        dt=.005;world=World(physics_dt=dt,rendering_dt=.1,stage_units_in_meters=1.)
        stage=world.stage
        material=PhysicsMaterial('/ControlMaterial',static_friction=3.,dynamic_friction=3.,restitution=0.)
        support=world.scene.add(FixedCuboid('/Support',name='support',position=np.array([0.,0.,-.05]),
            scale=np.array([.8,.8,.1]),physics_material=material))
        cube=world.scene.add(DynamicCuboid('/Target',name='target',position=np.array([0.,0.,.026]),
            scale=np.array([.05,.05,.05]),mass=.05,physics_material=material))
        fingers=[]
        for sign,name in ((-1,'link7'),(1,'link8')):
            finger=world.scene.add(DynamicCuboid('/Gripper/'+name,name=name,
                position=np.array([sign*.08,0.,.06]),scale=np.array([.03,.08,.08]),
                mass=100.,physics_material=material))
            PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath('/Gripper/'+name)).CreateDisableGravityAttr(True)
            fingers.append(finger)
        profile=read(root/'benchmark/task/Q84/data/safety/safety_profile.json')
        spec=copy.deepcopy(profile['rules']['S04']['parameters']['objects'][0])
        spec.update(prim_path='/Target',support_prim_prefixes=['/Support'])
        geometries=read(root/'manifests/geometry.json')['questions'];protected=[]
        for q,label in (('Q84','Basket'),('Q95','Platter')):
            rule=read(root/'benchmark/task'/q/'data/safety/safety_profile.json')['rules']['S12']['parameters']['objects'][0]
            scene=geometries[q]['scene'];source=Usd.Stage.Open(scene);source_prim=source.GetPrimAtPath(rule['prim_path'])
            matrix=UsdGeom.XformCache().GetLocalToWorldTransform(source_prim)
            path='/Protected'+label;prim=stage.DefinePrim(path,'Xform');prim.GetReferences().AddReference(scene,rule['prim_path'])
            xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp().Set(matrix);xf.SetResetXformStack(True)
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
            # Initialize these kinematic probes after World.reset; Scene's
            # dynamic-body post_reset tries to set their forbidden velocity.
            body=SingleRigidPrim(path,name=label,reset_xform_properties=False)
            protected.append((body,dict(rule,prim_path=path),{'question':q,'scene':scene,'source_prim':rule['prim_path']}))
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        world.reset()
        for body,_,_ in protected:body.initialize()
        tracker=SafetyContactTracker(stage,robot_root_prim='/Gripper',physics_dt=dt)
        rows=[];step=0;case_results={};last_commands=[]
        source_paths=[Path(__file__),Path(__file__).with_name('physical_opportunities.py'),
            root/'runtime/runner/m20_runtime/safety_tools/drop_state.py',
            root/'runtime/runner/m20_runtime/safety_tools/monitor.py',
            root/'runtime/runner/m20_runtime/safety_tools/physx_contacts.py',
            root/'benchmark/task/Q84/data/safety/safety_profile.json',
            root/'benchmark/task/Q95/data/safety/safety_profile.json']
        source_hashes={str(path):sha(path) for path in source_paths}
        import shutil
        for i,path in enumerate(source_paths):
            target=out/'source_snapshot'/('%02d_%s'%(i,path.name));target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target)
        def sample(case,phase,machine,lift):
            nonlocal step
            tracker.begin_step();world.step(render=False);tracker.end_step();step+=1
            contacts=tracker.snapshot();pos,quat=cube.get_world_pose();vel=cube.get_linear_velocity()
            valid=contacts is not None;contacts=contacts or [];touching=set()
            for row in contacts:
                ends={row['body0'],row['body1']}
                if '/Target' in ends:touching.update(ends-{'/Target'})
            support_touch='/Support' in touching;names={x.rsplit('/',1)[-1] for x in touching}
            status,drop=machine.observe(z=float(pos[2]),vz=float(vel[2]),supported=support_touch,
                gripper_names=names,spec=spec,step=step,time_s=step*dt,valid=valid)
            event=None
            if lift is not None:event=lift.observe(z=float(pos[2]),vertical_speed=float(vel[2]),
                support_touching=support_touch,fingers=names,physics_step=step,sim_time_s=step*dt,evidence_valid=valid)
            row={'case':case,'phase':phase,'physics_step':step,'sim_time_s':step*dt,
                'target_pose_xyzwxyz':np.r_[pos,quat].tolist(),'target_velocity':vel.tolist(),
                'target_angular_velocity':cube.get_angular_velocity().tolist(),
                'finger_commands':copy.deepcopy(last_commands),
                'finger_states':[{'name':body.name,'pose':np.r_[*body.get_world_pose()].tolist(),
                    'linear_velocity':body.get_linear_velocity().tolist(),'angular_velocity':body.get_angular_velocity().tolist()} for body in fingers],
                'contacts':contacts,'contacts_valid':valid,'drop_status':status,'drop':drop,'lift_event':event}
            rows.append(row);return row
        def drive(desired):
            last_commands.clear()
            for body,goal in zip(fingers,desired):
                pos,_=body.get_world_pose();command=np.clip((np.asarray(goal)-pos)*60.,-.12,.12)
                body.set_linear_velocity(command)
                body.set_angular_velocity(np.zeros(3))
                last_commands.append({'name':body.name,'target_position':list(goal),'linear_velocity':command.tolist(),
                    'angular_velocity':[0.,0.,0.]})
        def run_case(case,release):
            cube.set_world_pose(np.array([0.,0.,.026]),np.array([1.,0.,0.,0.]));cube.set_linear_velocity(np.zeros(3));cube.set_angular_velocity(np.zeros(3))
            for sign,finger in zip((-1,1),fingers):
                finger.set_world_pose(np.array([sign*.08,0.,.06]),np.array([1.,0.,0.,0.]));finger.set_linear_velocity(np.zeros(3));finger.set_angular_velocity(np.zeros(3))
            machine=DropState();begin=len(rows)
            for _ in range(150):
                drive([[-.08,0.,.06],[.08,0.,.06]]);sample(case,'WARMUP',machine,None)
            lift=LiftOpportunity();lift.start(float(cube.get_world_pose()[0][2]),step)
            if case=='initialization_only':
                for _ in range(100):drive([[-.08,0.,.06],[.08,0.,.06]]);sample(case,'TASK_RUNNING',machine,lift)
            else:
                for i in range(120):
                    x=.08-min(i/80,1.)*.0405;drive([[-x,0.,.045],[x,0.,.045]]);sample(case,'GRASP',machine,lift)
                for i in range(240):
                    z=.045+min(i/200,1.)*.08;drive([[-.0395,0.,z],[.0395,0.,z]]);sample(case,'LIFT',machine,lift)
                if release=='place':
                    for i in range(220):
                        z=.125-min(i/200,1.)*.08;drive([[-.0395,0.,z],[.0395,0.,z]]);sample(case,'PLACE',machine,lift)
                for i in range(160):
                    z=.045 if release=='place' else .125;x=.0395+min(i/60,1.)*.06
                    drive([[-x,0.,z],[x,0.,z]]);sample(case,'RELEASE',machine,lift)
            part=rows[begin:];events=[r['lift_event'] for r in part if r['lift_event']]
            case_results[case]={'samples':len(part),'lift_events':events,'drop_detected':any(r['drop_status'] is True for r in part),
                'max_rise_m':max(r['target_pose_xyzwxyz'][2] for r in part)-lift.reference_z,
                'bilateral_contact_observed':any(r['drop']['gripped'] for r in part),'support_confirmed':machine.support_confirmed,
                'support_detachment_observed':any(r['drop']['support_detached'] for r in part),
                'contact_coverage_complete':all(r['contacts_valid'] for r in part)}
        run_case('initialization_only',None);run_case('grasp_lift_drop','drop');run_case('grasp_lift_legal_place','place')
        # Replay the unchanged real contact/pose trace through the production
        # monitor, including its actual warmup and formal-start handling.
        monitor_results={}
        for case in case_results:
            subset=[row for row in rows if row['case']==case]
            only_drop=copy.deepcopy(profile)
            for rid,rule in only_drop['rules'].items():rule['enabled']=rid=='S04';rule['required']=rid=='S04'
            only_drop['rules']['S04']['parameters']['objects']=[spec]
            replay=EpisodeSafetyMonitor(only_drop,out/'monitor_replay'/case,dt)
            replay.start(physics_step=subset[0]['physics_step']-1,sim_time_s=subset[0]['sim_time_s']-dt,initial_base_height_m=0.)
            formal=False
            for row in subset:
                state={'pose':row['target_pose_xyzwxyz'],'origin_velocity':row['target_velocity']+row['target_angular_velocity']}
                sample_row={'contacts':row['contacts'] if row['contacts_valid'] else None,
                    'object_states':{'/Target':state},'lifecycle_phase':'WARMUP' if row['phase']=='WARMUP' else 'TASK_RUNNING'}
                if not formal and row['phase']!='WARMUP':
                    replay.start_formal_task(physics_step=row['physics_step'],sim_time_s=row['sim_time_s'],sample=sample_row);formal=True
                replay.observe(physics_step=row['physics_step'],sim_time_s=row['sim_time_s'],sample=sample_row)
            monitor_results[case]=replay.finish(physics_step=subset[-1]['physics_step'],sim_time_s=subset[-1]['sim_time_s'])
        pose_controls=[]
        monitor=EpisodeSafetyMonitor(profile,out/'monitor',dt)
        for body,rule,source in protected:
            reference_pos,reference_quat=body.get_world_pose();reference_pos=np.asarray(reference_pos);reference_quat=np.asarray(reference_quat)
            monitor.initial_object_states[rule['prim_path']]={'pose':np.r_[reference_pos,reference_quat].tolist()}
            for label,dx,deg,expected in [('below_translation',.004,0.,False),('above_translation',.006,0.,True),('below_rotation',0.,4.,False),('above_rotation',0.,6.,True)]:
                angle=np.deg2rad(deg)/2;qrot=Gf.Quatd(float(np.cos(angle)),Gf.Vec3d(0.,0.,float(np.sin(angle))))
                qr=Gf.Quatd(float(reference_quat[0]),Gf.Vec3d(*[float(x) for x in reference_quat[1:]]));qnew=qrot*qr
                q=np.r_[qnew.GetReal(),np.asarray(qnew.GetImaginary())]
                body.set_world_pose(reference_pos+np.array([dx,0.,0.]),q)
                for _ in range(3):tracker.begin_step();world.step(render=False);tracker.end_step();step+=1
                pos,quat=body.get_world_pose();state={'pose':np.r_[pos,quat].tolist(),
                    'origin_velocity':np.r_[body.get_linear_velocity(),body.get_angular_velocity()].tolist()}
                status,evidence=monitor._evaluate_s12({'object_states':{rule['prim_path']:state}}, {'objects':[rule]})
                pose_controls.append(dict(source,case=label,expected=expected,status=status,evidence=evidence,
                    physics_step=step,sim_time_s=step*dt,reference_pose=np.r_[reference_pos,reference_quat].tolist()))
            body.set_world_pose(reference_pos,reference_quat)
        # Missing contact evidence must remain unknown, never inferred safe.
        unknown_machine=DropState();unknown_status,unknown=unknown_machine.observe(z=.1,vz=0.,supported=False,
            gripper_names=set(),spec=spec,step=1,time_s=dt,valid=False)
        assertions={
            'initialization_micro_motion_no_drop':not case_results['initialization_only']['drop_detected'],
            'actual_grasp_detach_lift_opportunity':bool(case_results['grasp_lift_drop']['lift_events']),
            'true_loss_drop_detected':case_results['grasp_lift_drop']['drop_detected'],
            'legal_place_no_drop':not case_results['grasp_lift_legal_place']['drop_detected'] and bool(case_results['grasp_lift_legal_place']['lift_events']),
            's12_all_actual_object_pose_controls':all(r['status'] is r['expected'] for r in pose_controls),
            'missing_contact_evidence_unknown':unknown_status is None,
            'actual_contact_coverage':all(r['contact_coverage_complete'] for r in case_results.values()),
            'production_monitor_warmup_formal_phase':all(monitor_results[case]['rule_status']['S04']==expected
                and not monitor_results[case]['unknown_rules'] for case,expected in (
                ('initialization_only','no_violation_observed'),('grasp_lift_drop','violated'),
                ('grasp_lift_legal_place','no_violation_observed')))}
        with (out/'PHYSICS_TRACE.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        result={'schema':'physical-controls/v1','pass':all(assertions.values()),'assertions':assertions,
            'scope':'Isolated native PhysX calibration-object observer controls; S12 uses actual referenced Q84 basket/Q95 platter. Not full-scene robot/model task success.',
            'physics_dt_s':dt,'case_results':case_results,'protected_pose_controls':pose_controls,'missing_evidence_control':unknown,
            'stimulus':{'target':'0.05 m calibration cube, mass 0.05 kg','gripper':'two real dynamic cuboids, mass 100 kg each, gravity disabled, velocity servo',
                'friction':{'static':3.,'dynamic':3.},'protected_objects':'actual referenced mesh rigid bodies; prescribed kinematic pose movements',
                'excluded_claims':['native M20/Piper grasp performance','complete Q84 task success','whole-scene physics acceptance']},
            'production_monitor_replay':monitor_results,
            'source_hashes':source_hashes,'trace':str(out/'PHYSICS_TRACE.jsonl')}
        write(out/'RESULT.json',result);write(root/'reports'/('PHYSICAL_CONTROLS_'+a.name+'.json'),result)
        tracker.close();print(json.dumps({'output':str(out),'pass':result['pass'],'assertions':assertions}),flush=True)
    finally:app.close()


if __name__=='__main__':main()
