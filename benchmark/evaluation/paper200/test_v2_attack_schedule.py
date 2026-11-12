"""Run against the isolated v2 Runner via PYTHONPATH, without Isaac startup."""
import unittest
from attack_tools.attack_orchestrator import AttackOrchestrator
from attack_tools.standard_attack_handlers import replace_prompt_sequence
from robolab_runner.attack_hdf5_enricher import STAGE_DIMENSION, _write_attack_instance


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.engine=AttackOrchestrator({'attacks':[{
            'id':'q84_prompt_01','mode':'intercept','stage':'request_prompt',
            'handler':'replace_prompt_sequence','max_applications':2,'duration_s':5.,
            'trigger':{'all':[{'type':'field','field':'simulation.task_time_s','op':'gte','value':0.}]},
            'parameters':{'delivery':'once_per_scheduled_request','after_sequence':'restore_original',
                'steps':[{'at_s':0.,'replacement':'round zero'},{'at_s':1.,'replacement':'round one'}]}}]},
            {'replace_prompt_sequence':replace_prompt_sequence},audit=self.events.append)

    def snapshot(self, elapsed, phase='TASK_RUNNING'):
        self.engine.observe({'simulation':{'time_s':7.+elapsed,'physics_step':1400+round(elapsed*200),
            'task_time_s':elapsed,'task_start_sim_time_s':7.,'task_start_physics_step':1400,
            'lifecycle_phase':phase}})

    def request(self, elapsed, request_id):
        return self.engine.process_vla_request({'prompt':'original'},
            {'time_s':7.+elapsed,'physics_step':1400+round(elapsed*200),'request_id':request_id})['prompt']

    def test_two_real_distinct_requests_and_restore(self):
        self.snapshot(0.)
        self.assertEqual(self.request(0.,10),'round zero')
        self.snapshot(.5)
        self.assertEqual(self.request(.5,11),'original')
        self.snapshot(1.)
        self.assertEqual(self.request(1.,12),'round one')
        self.assertEqual(self.request(1.5,13),'original')
        rows=[x for x in self.events if x['event']=='payload_intercepted']
        self.assertEqual([x['runtime_context']['request_id'] for x in rows],[10,12])
        self.assertEqual([x['prompt_round_index'] for x in rows],[0,1])
        self.assertEqual(self.engine.application_counts['q84_prompt_01'],2)
        marker=next(x for x in self.events if x['event']=='condition_triggered')
        self.assertEqual((marker['time_s'],marker['physics_step']),(7.,1400))

    def test_warmup_never_arms_even_if_legacy_task_clock_is_zero(self):
        self.snapshot(0.,'WARMUP')
        self.assertEqual(self.request(0.,1),'original')
        self.assertFalse(self.events)

    def test_late_first_request_is_incomplete_not_skipped_to_second(self):
        self.snapshot(0.)
        self.assertEqual(self.request(1.,1),'original')
        self.assertFalse(self.engine.application_counts)
        self.assertEqual(self.engine.stop_reasons['q84_prompt_01'],'prompt_schedule_incomplete')

    def test_early_stop_does_not_claim_second_round(self):
        self.snapshot(0.)
        self.request(0.,1)
        self.assertEqual(self.engine.application_counts['q84_prompt_01'],1)

    def test_joint_encoder_is_feedback(self):
        self.assertEqual(STAGE_DIMENSION['sensor_joint_encoder'],'state_feedback')

    def test_unknown_stage_fails_before_writing_a_misleading_group(self):
        with self.assertRaisesRegex(ValueError,'Unregistered attack evidence stage'):
            _write_attack_instance(None,None,{}, {'id':'bad','stage':'unknown'},[],None,[],[])


if __name__=='__main__':unittest.main()
