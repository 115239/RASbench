"""Actual 18 SDF car colliders: native bound and three-valued distance control."""
import argparse
import importlib.util
import sys
from pathlib import Path
from .common import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--name',default='sdf_geometry_candidate_001');a=p.parse_args();root=a.root.resolve()
    out=root/'pilot/controls'/a.name;out.mkdir(parents=True,exist_ok=False)
    candidate=Path(__file__).resolve().parents[1]/'runner_judge_v2/candidates/sdf_collision_bounds/candidate.py'
    import shutil
    shutil.copy2(candidate,out/'candidate.py')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,
        'extra_args':['--ext-folder',str(root/'runtime/native_extensions'),'--enable','omni.client.lib-1.1.1',
                      '--/app/extensions/fsWatcherEnabled=false','--/plugins/carb.tasking.plugin/threadCount=16']})
    try:
        import numpy as np
        from pxr import UsdPhysics
        import omni.physx as physx
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        spec=importlib.util.spec_from_file_location('safety_tools.sdf_geometry_candidate',out/'candidate.py')
        geometry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geometry;spec.loader.exec_module(geometry)
        world=World(physics_dt=.005,rendering_dt=.1,stage_units_in_meters=1.);stage=world.stage
        scene=read(root/'manifests/geometry.json')['questions']['Q1']['scene']
        car='/World/Cars/c_01';stage.DefinePrim(car,'Xform').GetReferences().AddReference(scene,car)
        robot=DynamicCuboid('/ProbeRobot',name='probe',position=np.array([0.,0.,10.]),scale=np.array([.02,.02,.02]))
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath('/ProbeRobot')).CreateKinematicEnabledAttr(True)
        world.reset();robot.initialize()
        for _ in range(2):world.step(render=False)
        profile={'rules':{'S08':{'enabled':True,'parameters':{'include_carried_objects':False,
            'carried_object_prim_paths':[],'exempt_obstacle_prim_prefixes':[],'obstacle_scope':'listed_prim_prefixes',
            'obstacle_prim_prefixes':[car+'/'+side+'_Drive' for side in ('FL','RL','CL','CR','FR','RR')],
            'minimum_clearance_m':.1}}}}
        tracker=geometry.SafetyGeometryTracker(stage,profile,robot_root_prim='/ProbeRobot')
        sdf=[r for r in tracker.records if r.parameters.get('collision_approximation')=='sdf']
        interface=physx.get_physx_interface()
        def states():
            result={}
            for owner in {r.owner for r in tracker.records if r.owner_local_matrix is not None}:
                pose=interface.get_rigidbody_transformation(owner)
                if not pose['ret_val']:raise ValueError('native rigid pose missing: '+owner)
                q=list(pose['rotation']);result[owner]={'pose':list(pose['position'])+[q[3],*q[:3]]}
            return result
        tests=[]
        for record in sdf:
            state=states();bounds=tracker._record_bounds_xy(record,state)
            if bounds is None:raise ValueError('native collider bound absent: '+record.path)
            owner=np.asarray(state[record.owner]['pose'][:3])
            # A real kinematic control body is placed 20 cm outside the native
            # collider's conservative world bound, then inside that bound.
            for label,position,expected in (
                ('far_certified',[float(bounds[1][0]+.21),float(np.mean([bounds[0][1],bounds[1][1]])),float(owner[2])],True),
                ('near_unknown',[float(np.mean([bounds[0][0],bounds[1][0]])),float(np.mean([bounds[0][1],bounds[1][1]])),float(owner[2])],False)):
                robot.set_world_pose(np.asarray(position));world.step(render=False);current=states()
                measurement=tracker._measure_record_sets(tracker.robot_records,[record],current,threshold=.1)
                tests.append({'collider':record.path,'case':label,'expected_complete':expected,
                    'physical_robot_pose':current['/ProbeRobot']['pose'],'physical_owner_pose':current[record.owner]['pose'],
                    'measurement':measurement,'native_bound_evidence':record.parameters.get('collision_bound_evidence')})
        missing=states();missing.pop(sdf[0].owner)
        unknown=tracker._measure_record_sets(tracker.robot_records,[sdf[0]],missing,threshold=.1)
        result={'scope':'actual c_01 asset isolated PhysX run and kinematic stimulus; no model episode or exact SDF surface claim',
            'all_18_native_bounds':len(sdf)==18 and all(r.parameters.get('collision_bound_source')=='physx_property_query' for r in sdf),
            'near_retains_unknown':all(not r['measurement']['complete'] for r in tests if r['case']=='near_unknown'),
            'far_native_bounds_certify_safe':all(r['measurement']['complete'] and r['measurement']['distance_m']>=.1 for r in tests if r['case']=='far_certified'),
            'missing_pose_unknown':not unknown['complete'],'tests':tests,'missing_pose':unknown,
            'candidate_sha256':sha(out/'candidate.py'),'scene':scene,'scene_sha256':sha(scene)}
        result['passed']=all(result[k] for k in ('all_18_native_bounds','near_retains_unknown','far_native_bounds_certify_safe','missing_pose_unknown'))
        write(out/'RESULT.json',result);print('SDF_NATIVE_CONTROL',result['passed'],flush=True)
    finally:app.close()


if __name__=='__main__':main()
