import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from paper200.paired_initial import compare_formal_arrays,load_formal,evaluate
from paper200.common import write


def state():
    return dict(body_paths=np.array(['/b','/a']),joint_names=np.array(['j1','j0']),
        body_pose=np.array([[1,2,3,0,0,0,1],[4,5,6,0,0,0,1.]]),body_velocity=np.zeros((2,6)),
        joint_position=np.array([.1,.2]),physics_step=np.array(1000),sim_time_s=np.array(5.),
        reference_paths=np.array(['/protected']),reference_pose=np.array([[0.,0.,0.,0.,0.,0.,1.]]),reference_valid=np.array([True]),
        reference_bindings_json=np.asarray(json.dumps([dict(prim='/protected',pose_order='xyz_xyzw',source='usd_authoritative_static',rigid_owner=None,enabled_ancestor_owners=[],enabled_descendant_owners=[],initial_world_matrix=np.eye(4).tolist())])))


def attempt(root,system,variant='clean'):
    path=root/system/variant;path.mkdir(parents=True);(path/'episode').mkdir()
    job=dict(run_id=system+variant,question='Q1',system=system,variant=variant,seed=82026001,purpose='main',status='collected',output=str(path))
    write(path/'job.json',job)
    write(path/'identity.json',dict(schema='paper200-runner-judge/v2',task_id='task1',system_id=system,system_version='test-v2',seed=82026001,policy_seed=82026001,repeat_index=0,block_id='q1seed',purpose='main',condition_id='clean' if variant=='clean' else 'attack__q1_'+variant))
    write(path/'episode/result.json',dict(task_started=True,task_start_physics_step=1000,task_start_sim_time_s=5.))
    write(path/'input_identity.json',dict(task_sha256='task',environment_sha256='scene',scene_revisions={},public_task_sha256='public',runtime={'files':{'runtime/runner/physics.py':'same'}},selected_model=system))
    write(path/'runner_judge_contract.json',dict(schema='paper200-runner-judge/v2',all_models=['groot','pi05'],task={'manipulation':{'scene_object_prim':'/protected'}}))
    np.savez(path/'episode/formal_initial_state.npz',**state())
    warmup=state();warmup['body_pose'][:,:3]+=100.;np.savez(path/'initial_state.npz',**warmup)
    return job


class FormalPairingTests(unittest.TestCase):
    def test_binding_order_and_quaternion_sign_do_not_change_physical_state(self):
        a=state();b={k:v.copy() for k,v in a.items()}
        for key in ('body_paths','body_pose','body_velocity','joint_names','joint_position'):b[key]=b[key][::-1]
        b['body_pose'][:,3:]*=-1
        self.assertTrue(compare_formal_arrays(a,b)['matched'])
        b['body_pose'][0,0]+=.1
        self.assertFalse(compare_formal_arrays(a,b)['matched'])
    def test_legacy_reference_fields_missing_is_not_current_formal_pair(self):
        a=state();b=state();del b['reference_pose']
        self.assertIn('FORMAL_AUTHORITATIVE_REFERENCE_FIELDS_MISSING',compare_formal_arrays(a,b)['issues'])
    def test_protected_static_motion_rejected_when_all_rigid_bodies_match(self):
        a=state();b=state();b['reference_pose'][0,0]=.006
        self.assertIn('FORMAL_REFERENCE_POSE_MISMATCH',compare_formal_arrays(a,b)['issues'])
    def test_static_and_dynamic_descendant_authority_cannot_match_by_pose_alone(self):
        a=state();b=state()
        for value in (a,b):
            value['reference_paths']=np.array(['/a/child'])
            row=json.loads(value['reference_bindings_json'].item())[0];row['prim']='/a/child'
            value['reference_bindings_json']=np.asarray(json.dumps([row]))
        row=json.loads(b['reference_bindings_json'].item())[0]
        row.update(source='physx_owner_fixed_descendant',rigid_owner='/a',enabled_ancestor_owners=['/a'],fixed_local_matrix_row_vector=np.eye(4).tolist())
        row.pop('initial_world_matrix');b['reference_bindings_json']=np.asarray(json.dumps([row]))
        self.assertIn('FORMAL_REFERENCE_AUTHORITY_BINDING_MISMATCH',compare_formal_arrays(a,b)['issues'])
        b['reference_pose'][0,1]=.02
        self.assertIn('FORMAL_REFERENCE_POSE_MISMATCH',compare_formal_arrays(b,a)['issues'])
    def test_invalid_reference_mask_does_not_become_equal_unknown_pair(self):
        a=state();b=state();a['reference_valid'][0]=False;b['reference_valid'][0]=False
        self.assertIn('FORMAL_REFERENCE_POSE_OR_VALIDITY_INVALID',compare_formal_arrays(a,b)['issues'])

    def test_formal_file_selected_even_when_warmup_sidecar_differs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=attempt(root,'groot_agent_off');value=load_formal(root,job)
            self.assertEqual(value['state']['body_pose'][0,0],1.)
            (Path(job['output'])/'episode/formal_initial_state.npz').unlink()
            with self.assertRaises(FileNotFoundError):load_formal(root,job)
    def test_actual_clock_and_seed_purpose_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=attempt(root,'groot_agent_off')
            wrong=copy.deepcopy(job);wrong['seed']=7
            with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):load_formal(root,wrong)
            wrong=copy.deepcopy(job);wrong['purpose']='assisted'
            with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):load_formal(root,wrong)
            write(Path(job['output'])/'episode/result.json',dict(task_started=True,task_start_physics_step=0,task_start_sim_time_s=0.))
            with self.assertRaisesRegex(ValueError,'CLOCK_MISMATCH'):load_formal(root,job)
    def test_cross_system_different_models_can_match_physics_without_clean_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=[attempt(root,s,'prompt_01') for s in ('groot_agent_on','groot_agent_off','pi05_agent_off')]
            result=evaluate(root,jobs)
            self.assertEqual(result['compared_pairs'],2);self.assertTrue(all(p['matched'] for p in result['pairs']))
            self.assertFalse(result['pairing_ready']);self.assertEqual(len(result['pairing_pending']),3)
            self.assertEqual(result['issues'],[]);self.assertTrue(result['collection_completeness_independent'])
    def test_parked_prior_release_never_supplies_clean(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'current';old=Path(d)/'prior';root.mkdir();old.mkdir()
            clean=attempt(old,'groot_agent_off');attack=attempt(root,'groot_agent_off','prompt_01')
            result=evaluate(root,[clean,attack])
            self.assertFalse(result['pairing_ready']);self.assertEqual(result['compared_pairs'],0)
            self.assertIn('PARKED_OR_EXTERNAL_RELEASE_NOT_PAIR_CANDIDATE',[p.get('reason') for p in result['pairing_pending']])
    def test_same_system_pair_requires_strict_hdf5_model_runtime_check(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=[attempt(root,'groot_agent_off',v) for v in ('clean','prompt_01')]
            with patch('paper200.paired_initial.compare_hdf5',return_value={'passed':False,'issues':['PAIR_ACTUAL_MODEL_IDENTITY_MISMATCH']}) as compare:
                result=evaluate(root,jobs)
            compare.assert_called_once();self.assertFalse(result['pairs'][0]['matched'])
            self.assertIn('PAIR_ACTUAL_MODEL_IDENTITY_MISMATCH',result['pairs'][0]['issues'])


if __name__=='__main__':unittest.main()
