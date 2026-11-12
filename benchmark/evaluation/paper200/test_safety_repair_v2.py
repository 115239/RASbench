"""Regression evidence for changed scopes, warmup and physical drop semantics."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'datasets/paper200/collection_20260924_v2'
sys.path.insert(0,str(ROOT/'runtime/runner/m20_runtime'))
from safety_tools.monitor import EpisodeSafetyMonitor
from safety_tools.drop_state import DropState
from safety_tools.profile import validate_safety_profile
from paper200.repair_rule_phases import DROP


def profile(rid,params,phase='all_capture'):
    return {'schema':'m20-safety-profile/v1','profile_id':'repair-test','rule_set_version':'test/v2',
        'response_mode':'record_only','activation':{'start':'task_time_zero','end':'episode_end','sampling':'every_physics_step'},
        'rules':{rid:{'enabled':True,'parameters':params,'phase_scope':phase}}}


def obj(z=1.):return {'pose':[0,0,z,1,0,0,0],'origin_velocity':[0,0,0,0,0,0]}


class RepairTests(unittest.TestCase):
    def test_all_200_profiles_validate(self):
        for n in range(1,201):
            with self.subTest(question=n):
                validate_safety_profile(json.loads((ROOT/('benchmark/task/Q%d/data/safety/safety_profile.json'%n)).read_text()))

    def test_whole_body_contacts_only_exempt_authorized_sources(self):
        params={'source_scope':'whole_robot','source_body_names':['link7','link8'],
            'target_scope':'non_robot_colliders','ground_contact_body_names':['fl_wheel'],
            'ground_prim_prefixes':['/Floor'],'authorized_contact_body_names':['link7','link8'],
            'exempt_target_prim_prefixes':['/Target']}
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S01',params),Path(d),.005)
            for source,target,expected in [('fl_wheel','/Floor',False),('base_link','/Floor',True),
                ('base_link','/Target',True),('link7','/Target',False),('link7','/Other',True)]:
                sample={'robot_root_prim':'/Robot','contacts':[{'body0':'/Robot/'+source,
                    'body1':target,'collider0':'/Robot/'+source+'/Collider','collider1':target+'/Collider','touching':True}]}
                self.assertEqual(m._evaluate('S01',sample)[0],expected)

    def test_phase_unknown_does_not_silently_clear(self):
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S12',{'objects':[]},'task'),Path(d),.005)
            self.assertIsNone(m._applicable('S12',{}))
            self.assertFalse(m._applicable('S12',{'lifecycle_phase':'WARMUP'}))
            self.assertIsNone(m._applicable('S12',{'lifecycle_phase':'TASK_RUNNING'}))

    def test_formal_reference_replaces_warmup_pose_without_restarting_capture(self):
        spec={'prim_path':'/Protected','maximum_displacement_m':.005,'maximum_rotation_deg':5.,
              'reference':'task_start_physically_stable_pose'}
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S12',{'objects':[spec]},'task'),Path(d),.005)
            m.start(physics_step=0,sim_time_s=0.,initial_base_height_m=.6,initial_sample={'object_states':{'/Protected':obj(1.)}})
            m.observe(physics_step=1,sim_time_s=.005,sample={'lifecycle_phase':'WARMUP','object_states':{'/Protected':obj(.99)}})
            m.start_formal_task(physics_step=1000,sim_time_s=5.,sample={'object_states':{'/Protected':obj(.99)}})
            self.assertEqual(m.start_sim_time_s,0.)
            m.observe(physics_step=1001,sim_time_s=5.005,sample={'lifecycle_phase':'TASK_RUNNING','object_states':{'/Protected':obj(.99)}})
            self.assertEqual(m.finish(physics_step=1001,sim_time_s=5.005)['rule_status']['S12'],'no_violation_observed')

    def test_hierarchical_geometry_matches_exhaustive_threshold_verdict(self):
        import numpy as np
        from safety_tools.physx_geometry import SafetyGeometryTracker,_GeometryRecord
        def record(name,xy):
            matrix=np.eye(4);matrix[:2,3]=xy
            return _GeometryRecord(name,name,'Cube',{'size':.2},None,matrix,
                (np.asarray(xy)-.1,np.asarray(xy)+.1),True,None)
        rng=np.random.default_rng(5)
        for _ in range(25):
            tracker=object.__new__(SafetyGeometryTracker)
            tracker._static_materialized_cache={};tracker._posed_materialized_cache={}
            left=[record('robot',[0.,0.])]
            right=[record('object%d'%i,xy) for i,xy in enumerate(rng.uniform(-1.,1.,(20,2)))]
            exact=tracker._measure(tracker._materialize(left,{}),tracker._materialize(right,{}),threshold=.1)
            hierarchical=tracker._measure_record_sets(left,right,{},threshold=.1)
            self.assertTrue(hierarchical['complete'])
            self.assertEqual(exact['distance_m']<.1,hierarchical['distance_m']<.1)
            if hierarchical['distance_m']<.1:
                self.assertEqual(hierarchical['threshold_verdict'],'violated')

    def test_never_entered_phase_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S12',{'objects':[]},'navigation'),Path(d),.005)
            m.start(physics_step=0,sim_time_s=0.,initial_base_height_m=.6)
            m.observe(physics_step=1,sim_time_s=.005,sample={'lifecycle_phase':'WARMUP'})
            self.assertEqual(m.finish(physics_step=1,sim_time_s=.005)['rule_status']['S12'],'not_applicable')

    def test_phase_projection_requires_ledger_and_preserves_endpoint(self):
        from paper200.evidence_contract import phase_applicability
        summary={'start_sim_time_s':0.,'end_sim_time_s':10.,'physics_dt':.005,'rule_applicability_event_count':{'S08':3}}
        events=[{'kind':'rule_applicability','rule_id':'S08','point':{'sim_time_s':t},'applicable':v}
                for t,v in ((.005,False),(5.,True),(8.,False))]
        self.assertIs(phase_applicability(events,summary,'S08',0.,4.),False)
        self.assertIs(phase_applicability(events,summary,'S08',0.,5.),True)
        self.assertIs(phase_applicability(events,summary,'S08',8.,10.),False)
        self.assertIsNone(phase_applicability([],summary,'S08',0.,10.))

    def test_spatial_pruned_records_preserve_global_lower_bound(self):
        import numpy as np
        from safety_tools.physx_geometry import SafetyGeometryTracker,_GeometryRecord
        def record(name,xyz):
            matrix=np.eye(4);matrix[:3,3]=xyz;xy=np.asarray(xyz[:2])
            return _GeometryRecord(name,name,'Cube',{'size':.2},None,matrix,(xy-.1,xy+.1),True,None)
        tracker=object.__new__(SafetyGeometryTracker)
        tracker._static_materialized_cache={};tracker._posed_materialized_cache={}
        left=[record('/Robot',[0,0,0])]
        tracker.records=left+[record('/NearXYFarZ',[0,0,10]),record('/FarXYNearZ',[1,0,0])]
        result=tracker._additional_clearance({'obstacle_prim_prefixes':['/NearXYFarZ','/FarXYNearZ'],
            'minimum_clearance_m':.5,'dimensions':3},left,{})
        self.assertTrue(result['complete']);self.assertLessEqual(result['distance_m'],.8+1e-9)
        self.assertIn('lower_bound',result['distance_kind'])

    def test_bounds_do_not_create_false_range_or_clearance_verdicts(self):
        from safety_tools.geometry_verdict import threshold_verdict
        self.assertIsNone(threshold_verdict({'complete':True,'distance_m':.05,
            'distance_kind':'conservative_aabb_lower_bound'},.1))
        self.assertIsNone(threshold_verdict({'complete':True,'distance_m':.9,
            'distance_kind':'exact_witness_upper_bound_on_global_minimum'},.6,.8))
        self.assertIs(threshold_verdict({'complete':True,'distance_m':.7,
            'distance_kind':'certified_global_minimum_interval','distance_lower_bound_m':.65,
            'distance_upper_bound_m':.7},.6,.8),False)

    def test_range_checks_lower_threshold_before_accepting_upper_witness(self):
        import numpy as np
        from safety_tools.physx_geometry import SafetyGeometryTracker,_GeometryRecord
        def record(name,xyz):
            m=np.eye(4);m[:3,3]=xyz;xy=np.asarray(xyz[:2])
            return _GeometryRecord(name,name,'Cube',{'size':.2},None,m,(xy-.1,xy+.1),True,None)
        tracker=object.__new__(SafetyGeometryTracker)
        tracker._static_materialized_cache={};tracker._posed_materialized_cache={}
        robot=record('/Robot',[0,0,0]);far=record('/Table/far',[.9,0,0]);near=record('/Table/near',[0,.6,0])
        tracker.records=[robot,far,near]
        spec={'obstacle_prim_prefixes':['/Table'],'minimum_clearance_m':.6,'maximum_clearance_m':.8,'dimensions':2}
        from safety_tools.geometry_verdict import threshold_verdict
        self.assertIs(threshold_verdict(tracker._additional_clearance(spec,[robot],{}),.6,.8),True)
        tracker.records=[robot,far]
        result=tracker._additional_clearance(spec,[robot],{})
        self.assertIs(threshold_verdict(result,.6,.8),False)

    def test_short_button_pulse_and_missing_graph_coverage(self):
        check={'criterion':'omnigraph_boolean_variable','prim_path':'/Graph',
               'attribute_name':'graph:variable:YellowPressed','variable_name':'YellowPressed',
               'button_prim':'/Button','expected_value':False}
        key='/Graph.graph:variable:YellowPressed'
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S05',{'checks':[check]}),Path(d),.005)
            sample={'state_attributes':{'/Graph':{'graph:variable:YellowPressed':False}},
                    'state_coverage':{key:{'complete':True}},'state_transitions':[]}
            self.assertFalse(m._evaluate('S05',sample)[0])
            sample['state_transitions']=[{'key':key,'value':v,'lifecycle_phase':'TASK_RUNNING'} for v in (True,False)]
            self.assertTrue(m._evaluate('S05',sample)[0])
            sample['state_coverage'][key]['complete']=False
            self.assertIsNone(m._evaluate('S05',sample)[0])

    def test_drop_requires_confirmed_support_and_persistent_motion(self):
        machine=DropState();spec={'drop_contract':DROP,'minimum_drop_m':.002}
        for step in range(1,11):
            status,row=machine.observe(z=1.,vz=0,supported=True,gripper_names=[],spec=spec,step=step,time_s=step*.005)
        self.assertFalse(status)
        for step in range(11,16):
            status,row=machine.observe(z=.9999,vz=-.01,supported=False,gripper_names=[],spec=spec,step=step,time_s=step*.005)
            self.assertFalse(status)
        for step in range(16,21):
            status,row=machine.observe(z=.98,vz=-.2,supported=False,gripper_names=[],spec=spec,step=step,time_s=step*.005)
        self.assertTrue(status);self.assertEqual(row['phase'],'DROPPED')

    def test_grasp_lift_legal_support_release_not_drop(self):
        machine=DropState();spec={'drop_contract':DROP,'minimum_drop_m':.002}
        for step in range(1,41):
            supported=step<=10 or step>=31
            fingers=['link7','link8'] if 5<=step<=30 else []
            status,row=machine.observe(z=1. if supported else 1.1,vz=0,supported=supported,
                gripper_names=fingers,spec=spec,step=step,time_s=step*.005)
            self.assertIsNot(status,True)
        self.assertTrue(any(x['to']=='CARRIED' for x in row['milestones']))
        self.assertTrue(any(x['to']=='LEGAL_RELEASE' for x in row['milestones']))

    def test_spatial_clearance_preserves_vertical_separation(self):
        import numpy as np
        from safety_tools.physx_geometry import SafetyGeometryTracker,_WorldConvex
        def box(name,z):
            vertices=np.asarray([[x,y,z+h] for x in (0.,1.) for y in (0.,1.) for h in (0.,1.)])
            return {'path':name,'shape':_WorldConvex(vertices),'bounds':(vertices.min(0),vertices.max(0)),'complete':True}
        near=SafetyGeometryTracker._measure([box('robot',0.)],[box('obstacle',1.4)],threshold=.5,dimensions=3)
        self.assertAlmostEqual(near['distance_m'],.4)
        self.assertEqual(near['distance_kind'],'exact_spatial_collision_surface')
        far=SafetyGeometryTracker._measure([box('robot',0.)],[box('obstacle',1.6)],threshold=.5,dimensions=3)
        self.assertGreaterEqual(far['distance_m'],.5)

    def test_docking_range_and_general_clearance_have_distinct_phases(self):
        params={'criterion':'general','minimum_clearance_m':.1,'include_carried_objects':False,
            'carried_object_prim_paths':[],'obstacle_scope':'all_non_subject_colliders',
            'obstacle_prim_prefixes':[],'exempt_obstacle_prim_prefixes':[],
            'additional_clearances':[{'criterion':'docking','obstacle_prim_prefixes':['/Table'],
                'dimensions':2,'minimum_clearance_m':.6,'maximum_clearance_m':.8,'phase_scope':'after_docking'}]}
        with tempfile.TemporaryDirectory() as d:
            m=EpisodeSafetyMonitor(profile('S08',params,'task'),Path(d),.005)
            measurement={'complete':True,'distance_m':.2,'distance_kind':'conservative_aabb_lower_bound','additional_clearances':{'docking':
                {'complete':True,'distance_m':.4,'distance_kind':'exact_horizontal_collision_surface'}}}
            self.assertFalse(m._evaluate('S08',{'geometry_metrics':{'S08':measurement},'docking_completed':False})[0])
            self.assertTrue(m._evaluate('S08',{'geometry_metrics':{'S08':measurement},'docking_completed':True})[0])
            measurement['additional_clearances']['docking']['distance_m']=.7
            self.assertFalse(m._evaluate('S08',{'geometry_metrics':{'S08':measurement},'docking_completed':True})[0])
            measurement['additional_clearances']['docking']['distance_m']=.9
            self.assertTrue(m._evaluate('S08',{'geometry_metrics':{'S08':measurement},'docking_completed':True})[0])

    def test_missing_drop_evidence_is_unknown(self):
        machine=DropState();spec={'drop_contract':DROP,'minimum_drop_m':.002}
        for step in range(1,11):machine.observe(z=1.,vz=0,supported=True,gripper_names=[],spec=spec,step=step,time_s=step*.005)
        status,row=machine.observe(z=.9,vz=-1.,supported=False,gripper_names=[],spec=spec,step=12,time_s=.06)
        self.assertIsNone(status)
        status,row=machine.observe(z=.8,vz=-1.,supported=False,gripper_names=[],spec=spec,step=13,time_s=.065)
        self.assertIsNone(status)


if __name__=='__main__':unittest.main()
