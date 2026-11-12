import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from paper200.evidence_contract import (Lifecycle, canonical_digest, resolve_identity,
                                       resolve_termination, project_observation_gaps)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = h5py.File(Path(self.temp.name) / 'e.hdf5', 'w')
        self.file.create_group('metadata').attrs['task_id'] = 'task-q84'

    def tearDown(self):
        self.file.close()
        self.temp.cleanup()

    def legacy(self, variant='prompt_01'):
        g = self.file.create_group('extensions/paper200_collection')
        row = dict(question='Q84', system='groot_agent_off', variant=variant,
                   seed=82026001, purpose='main', run_id='example-run')
        g.attrs.update(row)
        g.create_dataset('provenance_json', data=np.frombuffer(json.dumps(row).encode(), dtype=np.uint8))
        return g

    def test_untriggered_attack_does_not_become_clean(self):
        self.legacy()
        result = resolve_identity(self.file, question_id='Q084')
        self.assertEqual(result['condition_id'], 'attack__q84_prompt_01')
        self.assertEqual(result['seed'], 82026001)
        self.assertIsNone(result['policy_seed'])

    def test_explicit_clean(self):
        self.legacy('clean')
        self.assertEqual(resolve_identity(self.file, question_id='Q84')['condition_id'], 'clean')

    def test_missing_does_not_invent_clean_or_seed(self):
        result = resolve_identity(self.file, question_id='Q84')
        self.assertIsNone(result['condition_id'])
        self.assertIsNone(result['seed'])

    def test_conflicting_metadata_fails(self):
        self.legacy()
        self.file['metadata'].attrs['condition_id'] = 'clean'
        with self.assertRaisesRegex(ValueError, 'SOURCE_MISMATCH'):
            resolve_identity(self.file, question_id='Q84')

    def test_wrong_question_fails(self):
        self.legacy()
        with self.assertRaisesRegex(ValueError, 'QUESTION_SOURCE_MISMATCH'):
            resolve_identity(self.file, question_id='Q85')

    def test_legacy_conflicting_seed_fails(self):
        self.legacy().attrs['seed'] = 0
        with self.assertRaisesRegex(ValueError, 'SOURCE_MISMATCH:seed'):
            resolve_identity(self.file, question_id='Q84')

    def test_manifest_conflict_fails_instead_of_overriding(self):
        self.legacy()
        with self.assertRaisesRegex(ValueError, 'SOURCE_MISMATCH:condition_id'):
            resolve_identity(self.file, question_id='Q84', trusted={'condition_id': 'clean'})

    def test_parameter_changes_contract_digest(self):
        a = {'rule': 'S12', 'threshold': .005, 'objects': ['/fruit']}
        self.assertNotEqual(canonical_digest(a), canonical_digest(dict(a, threshold=.05)))
        self.assertEqual(canonical_digest(a), canonical_digest(dict(reversed(list(a.items())))))

    def test_legacy_false_acceptance_and_completed_flow_are_unscored(self):
        g=self.legacy('clean')
        g.create_dataset('result_json',data=np.frombuffer(json.dumps({
            'task_accepted':False,'flow_completed':True,'failure':None}).encode(),dtype=np.uint8))
        result=resolve_termination(self.file)
        self.assertIsNone(result['task_success'])
        self.assertIsNone(result['task_failure'])
        self.assertIsNone(result['termination_time_s'])
        self.assertEqual(result['task_evaluation_status'],'not_evaluated')

    def test_stop_decision_and_parking_tail_stay_distinct(self):
        g=self.legacy('clean')
        g.create_dataset('result_json',data=np.frombuffer(json.dumps({
            'termination_time_s':10.005,'termination_physics_step':2001,
            'record_end_sim_time_s':10.105,'record_end_physics_step':2021,
            'task_success':None,'task_failure':None,'truncated':False}).encode(),dtype=np.uint8))
        result=resolve_termination(self.file)
        self.assertEqual(result['termination_time_s'],10.005)
        self.assertEqual(result['record_end_sim_time_s'],10.105)
        self.assertFalse(result['truncated'])


class GapTests(unittest.TestCase):
    def setUp(self):
        self.events=[{'kind':'observation_gap','rule_id':'S08',
                      'start':{'sim_time_s':1.,'physics_step':200,
                               'metrics':{'reason':'geometry_missing','missing_field':'colliders'}},
                      'end':{'sim_time_s':2.,'physics_step':400},'end_reason':'observation_resumed'}]
        self.summary={'unknown_rules':['S08'],'unknown_interval_count':{'S08':1}}

    def project(self,start,end,complete=True):
        return project_observation_gaps(self.events,self.summary,start_time_s=start,end_time_s=end,
                                        coverage_complete=complete)

    def test_gap_projects_evidence_and_clock(self):
        gap=self.project(1.5,3.)
        self.assertEqual(gap['unknown_rules'],['S08'])
        row=gap['windows']['S08'][0]
        self.assertEqual(row['window_projection']['start_time_s'],1.5)
        self.assertEqual(row['metrics']['missing_field'],'colliders')

    def test_resumed_sample_is_observed_only_with_complete_ledger(self):
        self.assertEqual(self.project(2.,3.)['unknown_rules'],[])
        self.assertEqual(self.project(2.,3.,False)['unknown_rules'],['S08'])
        self.summary['unknown_interval_count']['S08']=2
        self.assertEqual(self.project(2.,3.)['unknown_rules'],['S08'])

    def test_small_window_does_not_clear_unexplained_unknown(self):
        self.events=[]
        self.assertEqual(self.project(4.,4.01)['unknown_rules'],['S08'])

    def test_gap_rows_without_matching_summary_cannot_clear_other_windows(self):
        self.summary={'unknown_rules':[],'unknown_interval_count':{}}
        self.assertEqual(self.project(4.,5.)['unknown_rules'],['S08'])

    def test_open_gap_at_episode_end_includes_last_sample(self):
        self.events[0]['end_reason']='episode_end_unknown'
        self.assertEqual(self.project(2.,2.)['unknown_rules'],['S08'])


class ClockTests(unittest.TestCase):
    def test_warmup_is_not_task_time_and_origin_is_measured(self):
        clock = Lifecycle()
        clock.transition('WARMUP', 1., 200)
        self.assertIsNone(clock.clock(7., 1400)['task_time_s'])
        clock.transition('READY', 7., 1400)
        ready = dict(initialization=True, stable_reference=True, policy_channel=True)
        clock.transition('TASK_RUNNING', 7.125, 1425, readiness=ready)
        self.assertAlmostEqual(clock.clock(8.125, 1625)['task_time_s'], 1.)
        clock.transition('STOPPING', 8.125, 1625)
        clock.transition('FINALIZED', 8.225, 1645)
        self.assertEqual(clock.task_start_physics_step, 1425)

    def test_not_ready_and_failed_init_cannot_start_task(self):
        clock = Lifecycle()
        clock.transition('WARMUP', 0., 0)
        clock.transition('READY', 5., 1000)
        with self.assertRaises(ValueError):
            clock.transition('TASK_RUNNING', 5., 1000, readiness={'initialization': True})
        clock.transition('STOPPING', 5., 1000)
        clock.transition('FINALIZED', 5.1, 1020)
        self.assertFalse(clock.clock(5.1, 1020)['task_started'])


if __name__ == '__main__':
    unittest.main()
