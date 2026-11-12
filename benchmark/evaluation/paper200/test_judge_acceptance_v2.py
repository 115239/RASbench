"""J3-J8 rejection controls; synthetic fixtures never count as physical runs."""
import copy,json,tempfile,unittest,zipfile
from pathlib import Path
import h5py,numpy as np
from paper200.evidence_contract import canonical_digest,phase_applicability
from paper200.contract_verification import compare_pair
from paper200.common import PROJECT

ROOT=PROJECT/'datasets/paper200/collection_20260924_v2'
PILOT=ROOT/'pilot/groot_agent_off/Q084/prompt_01/seed_72026001/attempt_002'


def blob(group,key,value):
    group.create_dataset(key,data=np.frombuffer(json.dumps(value).encode(),np.uint8))


def paired_fixture(f):
    g=f.create_group('extensions/paper200_collection')
    blob(g,'runner_judge_contract_json',{'safety_profile':{'rules':{}}})
    blob(g,'model_identity_json',{'backend':'groot','models':{'grasp':{'files':{'weights':'hash'}}}})
    blob(g,'input_identity_json',{k:'same' for k in ('runtime','environment_sha256','scene_revisions','pi05_transfer_sha256','model_bindings_sha256')})
    blob(g,'result_json',dict(task_started=True,task_start_physics_step=1000,task_start_sim_time_s=5.))
    state=g.create_group('formal_initial_state')
    for k,v in dict(body_paths=np.asarray(['/robot'],dtype=h5py.string_dtype()),joint_names=np.asarray(['j0'],dtype=h5py.string_dtype()),body_pose=[[0,0,0,0,0,0,1]],body_velocity=[[0]*6],joint_position=[0.],physics_step=1000,sim_time_s=5.).items():state.create_dataset(k,data=v)
    state.create_dataset('reference_paths',data=np.asarray([],dtype=h5py.string_dtype()))
    state.create_dataset('reference_pose',data=np.empty((0,7)));state.create_dataset('reference_valid',data=np.empty((0,),bool))
    state.create_dataset('reference_bindings_json',data='[]')
    return g,state


class PairAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.a=h5py.File(Path(self.temp.name)/'a.h5','w');self.b=h5py.File(Path(self.temp.name)/'b.h5','w')
        self.ga,self.sa=paired_fixture(self.a);self.gb,self.sb=paired_fixture(self.b)
    def tearDown(self):self.a.close();self.b.close();self.temp.cleanup()
    def test_measured_pair_and_quaternion_sign(self):
        self.assertTrue(compare_pair(self.a,self.b)['passed']);self.sb['body_pose'][0,-1]=-1
        self.assertTrue(compare_pair(self.a,self.b)['passed'])
    def test_unphysical_empty_reference_is_not_pair(self):
        del self.sb['body_paths'];self.sb.create_dataset('body_paths',data=np.asarray([],dtype=h5py.string_dtype()))
        self.assertIn('PAIR_BODY_BINDING_INVALID:body_paths',compare_pair(self.a,self.b)['issues'])
    def test_actual_loaded_model_differs_despite_same_global_model_registry(self):
        del self.gb['model_identity_json'];blob(self.gb,'model_identity_json',{'backend':'pi05','models':{'grasp':{'files':{'weights':'different'}}}})
        self.assertIn('PAIR_ACTUAL_MODEL_IDENTITY_MISMATCH',compare_pair(self.a,self.b)['issues'])
    def test_formal_origin_must_match_measured_reference(self):
        self.sb['physics_step'][()]=0
        self.assertIn('PAIR_FORMAL_REFERENCE_CLOCK_INVALID:physics_step',compare_pair(self.a,self.b)['issues'])
    def test_invalid_orientation_is_not_pair(self):
        self.sa['body_pose'][0,3:]=0;self.sb['body_pose'][0,3:]=0
        self.assertIn('PAIR_INITIAL_ORIENTATION_INVALID',compare_pair(self.a,self.b)['issues'])


@unittest.skipUnless((PILOT/'episode/submission.hdf5').exists(),'preserved real pilot not available')
class ActualEvidenceRejectionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from v2.safety.judge_result.episode_delivery import build_episode_delivery
        cls.build=staticmethod(build_episode_delivery)
        with zipfile.ZipFile(ROOT/'runtime/judge/v2/rubric_q1_q200.zip') as z:
            cls.rubric=json.loads(z.read('tasks/Q84/rubric.json'));cls.package=json.loads(z.read('tasks/Q84/rule_package.json'));cls.info=json.loads(z.read('tasks/Q84/task_info.json'))
        cls.contract=json.loads((PILOT/'runner_judge_contract.json').read_text())
    def build_result(self,contract=None,**kw):
        return self.build(question_id='Q84',hdf5_path=PILOT/'episode/submission.hdf5',rubric=self.rubric,rule_package=self.package,registered_attacks=self.info['registered_attacks'],expected_runner_judge_contract=self.contract if contract is None else contract,**kw)[1]
    def test_wrong_frozen_contents_excludes_attack_and_opportunity_without_mutating_source(self):
        changed=copy.deepcopy(self.contract);changed['injected_rejection_control']='different'
        result=self.build_result(changed);elig=result['validation']['eligibility']
        self.assertFalse(elig['conditional_attack_effect_eligible']);self.assertFalse(elig['opportunity_eligible'])
        self.assertIn('FROZEN_RUNNER_JUDGE_CONTRACT_MISMATCH',elig['reasons']['conditional_attack'])
    def test_summary_preserves_development_purpose_seed_and_selection(self):
        result=self.build_result()
        self.assertEqual(result['purpose'],'pilot');self.assertEqual(result['data_role'],'development')
        self.assertEqual(result['seed'],72026001);self.assertEqual(result['policy_seed'],72026001)
        self.assertIn('attack_seed',result);self.assertEqual(result['selected_attack_ids'],['q84_prompt_01'])
        self.assertTrue(result['opportunities']['q84_prompt_01']['injection_configured'])
        self.assertFalse(result['opportunities']['q84_imu_01']['injection_configured'])

    def test_real_control_is_explicitly_excluded_from_formal_data_role(self):
        path=ROOT/'pilot/controls/Q95/groot_agent_off/control_002'
        if not (path/'episode/submission.hdf5').exists():self.skipTest('real controlled Q95 diagnostic unavailable')
        with zipfile.ZipFile(ROOT/'runtime/judge/v2/rubric_q1_q200.zip') as z:
            rubric=json.loads(z.read('tasks/Q95/rubric.json'));package=json.loads(z.read('tasks/Q95/rule_package.json'))
        result=self.build(question_id='Q95',hdf5_path=path/'episode/submission.hdf5',rubric=rubric,rule_package=package,
                          expected_runner_judge_contract=json.loads((path/'runner_judge_contract.json').read_text()))[1]
        self.assertEqual(result['purpose'],'control_q95');self.assertEqual(result['data_role'],'diagnostic_control')

    def test_contradictory_application_payload_does_not_count_as_effect(self):
        path=ROOT/'pilot/judge_schema_controls/q84_declared_attack_payload_noop/episode/submission.hdf5'
        if not path.exists():self.skipTest('explicit synthetic rejection fixture not available')
        result=self.build(question_id='Q84',hdf5_path=path,rubric=self.rubric,rule_package=self.package,
                          registered_attacks=self.info['registered_attacks'],expected_runner_judge_contract=self.contract)[1]
        self.assertEqual(result['condition_id'],'attack__q84_prompt_01')
        self.assertFalse(result['validation']['eligibility']['conditional_attack_effect_eligible'])

    def test_frozen_registry_replaces_legacy_compiled_handler(self):
        from v2.safety.judge_result.episode_delivery import frozen_attack_registry
        rows=frozen_attack_registry(self.contract)
        self.assertTrue(rows)
        self.assertTrue(all(r['source'].startswith('trusted_expected_runner_judge_contract:') for r in rows))
        legacy=[dict(r,handler='incorrect_legacy_handler') for r in rows]
        result=self.build(question_id='Q84',hdf5_path=PILOT/'episode/submission.hdf5',rubric=self.rubric,rule_package=self.package,
            registered_attacks=legacy,expected_runner_judge_contract=self.contract)[1]
        self.assertNotIn('DECLARED_ATTACK_REGISTRY_MISMATCH',result['validation']['issues'])

    def test_clean_episode_id_without_physical_file_does_not_claim_verified_pair(self):
        result=self.build_result(matched_clean_episode_id='unverified-clean-id')
        self.assertFalse(result['validation']['eligibility']['pairing_eligible'])
        self.assertFalse(result['validation']['pairing_ready'])


class ScopeAcceptanceTests(unittest.TestCase):
    def test_phase_truncated_transition_ledger_is_unknown(self):
        summary={'start_sim_time_s':0.,'end_sim_time_s':10.,'physics_dt':.005,
                 'rule_applicability_event_count':{'S08':2}}
        rows=[{'kind':'rule_applicability','rule_id':'S08','point':{'sim_time_s':.005},'applicable':False}]
        self.assertIsNone(phase_applicability(rows,summary,'S08',0.,4.))
        rows.append({'kind':'rule_applicability','rule_id':'S08','point':{'sim_time_s':5.},'applicable':True})
        self.assertIs(phase_applicability(rows,summary,'S08',0.,4.),False)
        self.assertIs(phase_applicability(rows,summary,'S08',0.,5.),True)

    def safety(self,events,start,end):
        from v2.safety.judge_result.episode_delivery import read_embedded_safety
        from v2.safety.judge_result.windows import ScopeWindow
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'events.h5'
            with h5py.File(path,'w') as f:
                g=f.create_group('extensions/m20_safety_v1');g.attrs['complete']=True
                blob(g,'profile_json',{'rules':{'S11':{'enabled':True}}})
                blob(g,'summary_json',{'complete':True,'start_sim_time_s':0.,'end_sim_time_s':10.,'unknown_rules':[],'violation_interval_count':{'S11':len({e.get('source_event_id',str(i)) for i,e in enumerate(events) if e.get('kind')=='violation_interval'})}})
                g.create_dataset('events_jsonl',data=np.frombuffer(('\n'.join(json.dumps(e) for e in events)).encode(),np.uint8))
            return read_embedded_safety(path,{'test':ScopeWindow('test','available',start,end,None,None,None,None,None)})['test']

    def test_overlap_duration_and_source_dedup(self):
        row={'kind':'violation_interval','rule_id':'S11','source_event_id':'one',
             'start':{'sim_time_s':1.},'end':{'sim_time_s':9.},'end_reason':'cleared'}
        result=self.safety([row,row],4.,6.)
        self.assertEqual(result['violation_event_count_by_type'],{'S11':1})
        self.assertEqual(result['violation_windows']['S11'][0]['scope_duration_s'],2.)
        self.assertEqual(result['violation_windows']['S11'][0]['original_interval']['end']['sim_time_s'],9.)
        self.assertEqual(self.safety([row],9.,10.)['per_type']['S11'],'clear')
        row['end_reason']='episode_end_still_active'
        self.assertEqual(self.safety([row],9.,9.)['per_type']['S11'],'violated')

    def test_malformed_interval_not_silently_clear(self):
        row={'kind':'violation_interval','rule_id':'S11','start':{'sim_time_s':6.},'end':{'sim_time_s':2.}}
        result=self.safety([row],0.,10.)
        self.assertEqual(result['per_type']['S11'],'unknown')
        self.assertIn('SAFETY_VIOLATION_LEDGER_INVALID',result['unknown_reason_codes_by_type']['S11'])


class GeometryLedgerAcceptanceTests(unittest.TestCase):
    def test_clock_and_shapes_required(self):
        from v2.common.geometry_contract import verify_geometry_ledger
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'g.h5','w') as f:
            summary={'start_sim_time_s':0.,'end_sim_time_s':.01}
            self.assertFalse(verify_geometry_ledger(f,summary)['passed'])
            root=f.create_group('extensions/paper200_collection');g=root.create_group('geometry_observations')
            g.attrs.update(complete=True,physics_dt=.005)
            rows=[json.dumps({'physics_step':i,'sim_time_s':i*.005}).encode() for i in (1,2)]
            g.create_dataset('physics_step',data=[1,2]);g.create_dataset('time_s',data=[.005,.01])
            g.create_dataset('offsets',data=[0,len(rows[0]),sum(map(len,rows))]);g.create_dataset('json_bytes',data=np.frombuffer(b''.join(rows),np.uint8))
            shapes=root.create_group('geometry_shapes');shapes.attrs['complete']=True
            blob(shapes,'records_json',[{'path':'/Robot'}]);blob(shapes,'scope_manifest_json',{'robot_records':['/Robot'],'S08_obstacles':[],'S03_forbidden':{}})
            shapes.create_dataset('vertices',data=np.zeros((0,3)));shapes.create_dataset('part_offsets',data=[0]);shapes.create_dataset('record_part_ranges',data=[[0,0]])
            self.assertTrue(verify_geometry_ledger(f,summary)['passed'])
            g['physics_step'][1]=3
            self.assertIn('GEOMETRY_OBSERVATION_CLOCK_INVALID',verify_geometry_ledger(f,summary)['issues'])


if __name__=='__main__':unittest.main()
