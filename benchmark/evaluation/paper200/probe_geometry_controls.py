"""Native PhysX geometry controls, deliberately outside model experiment counts."""
import argparse
import sys
from pathlib import Path
from .common import write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--name',default='geometry_001');a=p.parse_args();root=a.root.resolve()
    out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        import numpy as np
        from pxr import UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        sys.path.insert(0,str(root/'runtime/runner/m20_runtime'))
        from safety_tools.physx_geometry import SafetyGeometryTracker
        from safety_tools.geometry_verdict import threshold_verdict
        from .geometry_evidence import GeometryEvidence
        from .common import sha
        world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.)
        bodies=[]
        for name,position in [('Robot/Body',[0,0,1]),('Obstacle',[.5,0,1]),('Decoy',[10,10,1])]:
            body=world.scene.add(DynamicCuboid(prim_path='/'+name,name=name.replace('/','_'),position=np.asarray(position),size=.2))
            UsdPhysics.RigidBodyAPI(world.stage.GetPrimAtPath('/'+name)).CreateKinematicEnabledAttr(True)
            bodies.append(body)
        world.reset()
        profile={'rules':{'S03':{'enabled':True,'parameters':{'regions':[{'criterion':'intersection',
            'forbidden_prim_prefixes':['/Obstacle'],'region_geometry':'collision_surfaces_xy',
            'minimum_clearance_m':0.,'include_carried_objects':False,'carried_object_prim_paths':[]}]}},
            'S08':{'enabled':True,'parameters':{'obstacle_scope':'all_non_subject_colliders',
                'minimum_clearance_m':.1,'include_carried_objects':False,'carried_object_prim_paths':[],
                'obstacle_prim_prefixes':[],'exempt_obstacle_prim_prefixes':[]}}}}
        tracker=SafetyGeometryTracker(world.stage,profile,robot_root_prim='/Robot')
        writer=GeometryEvidence(tracker,out,.005);cases=[];step=0
        for name,x,expected_s03,expected_s08 in [('outside_clearance',.31,False,False),
            ('inside_clearance',.29,False,True),('region_overlap',.19,True,True),
            ('region_outside',.21,False,True)]:
            bodies[1].set_world_pose(position=np.array([x,0,1]))
            for _ in range(10):
                world.step(render=False);step+=1
                states={}
                for body in bodies:
                    position,orientation=body.get_world_pose()
                    states[body.prim_path]={'pose':np.r_[position,orientation].tolist()}
                measurements=tracker.sample(states)
                writer.observe(step,step*.005,{'lifecycle_phase':'TASK_RUNNING','mission_phase':'NAV_OUTBOUND',
                    'geometry_metrics':measurements})
                got_s03=threshold_verdict(measurements['S03']['intersection'],0.,intersection=True)
                got_s08=threshold_verdict(measurements['S08'],.1)
                cases.append({'case':name,'physics_step':step,'actual_physx_poses':states,
                    's03':got_s03,'s08':got_s08,'expected_s03':expected_s03,'expected_s08':expected_s08,
                    'passed':got_s03 is expected_s03 and got_s08 is expected_s08})
        writer.close()
        report={'passed':all(x['passed'] for x in cases),'cases':cases,
            'source_hashes':{name:sha(root/'runtime/runner/m20_runtime/safety_tools'/name)
                for name in ('physx_geometry.py','geometry_verdict.py')},
            'scope':'actual PhysX kinematic shapes and post-step measured transforms; threshold controls, not full scene/model acceptance'}
        write(out/'RESULT.json',report);write(root/'reports'/('GEOMETRY_CONTROLS_'+a.name+'.json'),report)
        print({'passed':report['passed'],'output':str(out)},flush=True)
        if not report['passed']:raise RuntimeError('geometry physical threshold control failed')
    finally:app.close()


if __name__=='__main__':main()
