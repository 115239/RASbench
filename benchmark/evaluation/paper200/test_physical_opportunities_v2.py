import unittest
from paper200.physical_opportunities import LiftOpportunity
from paper200.execution_boundary import closing_opportunity


class LiftTests(unittest.TestCase):
    def setUp(self):
        self.detector=LiftOpportunity();self.detector.start(.5,1000)

    def sample(self,step,z=.5,support=True,fingers=(),speed=0.,valid=True):
        return self.detector.observe(z=z,vertical_speed=speed,support_touching=support,
            fingers=fingers,physics_step=1000+step,sim_time_s=5.+step*.005,evidence_valid=valid)

    def test_empty_gripper_and_supported_target_never_emit(self):
        for step in range(1,30):self.assertIsNone(self.sample(step,z=.51,support=True))

    def test_ungrasped_object_rising_never_emits(self):
        self.sample(1)
        for step in range(2,30):self.assertIsNone(self.sample(step,z=.52,support=False,speed=.1))

    def test_millimeter_noise_never_emits(self):
        for step in range(1,30):
            self.assertIsNone(self.sample(step,z=.501,support=step<10,fingers=('link7','link8'),speed=.01))

    def test_real_grasp_detach_rise_emits_once_with_witness(self):
        for step in range(1,11):self.assertIsNone(self.sample(step,fingers=('link7','link8')))
        event=None
        for step in range(11,16):
            event=self.sample(step,z=.504,support=False,fingers=('link7','link8'),speed=.04)
        self.assertEqual(event['type'],'target_object_lift_started')
        self.assertEqual(event['first_rise']['physics_step'],1011)
        self.assertEqual(event['physics_step'],1015)
        self.assertFalse(event['support_touching'])
        self.assertIsNone(self.sample(16,z=.506,support=False,fingers=('link7','link8'),speed=.04))

    def test_missing_contacts_reset_confirmation(self):
        for step in range(1,11):self.sample(step,fingers=('link7','link8'))
        self.sample(11,valid=False)
        for step in range(12,20):
            self.assertIsNone(self.sample(step,z=.51,support=False,fingers=('link7','link8'),speed=.1))

    def test_world_target_offset_holds_and_restores_without_accumulation(self):
        import numpy as np
        from paper200.execution_boundary import WorldTargetOffset
        tracker=WorldTargetOffset();seen=[]
        for offset in ([0,0,0],[0,.02,0],[0,.02,0],[0,.02,0],[0,0,0]):
            delta,row=tracker.adapt([.001,0,0],offset);seen.append(delta)
            np.testing.assert_allclose(row['commanded_world_target_bias_m'],offset,atol=1e-12)
        self.assertAlmostEqual(seen[1][1],.02)
        self.assertAlmostEqual(seen[2][1],0.)
        self.assertAlmostEqual(seen[-1][1],-.02)
        self.assertEqual(row['phase'],'restore')

    def test_closing_requires_executable_motion(self):
        self.assertFalse(closing_opportunity(nominal_close=1.,measured_close=0.,maximum_finger_opening=.035,enabled=False))
        self.assertFalse(closing_opportunity(nominal_close=1.,measured_close=1.,maximum_finger_opening=.035,enabled=True))
        self.assertTrue(closing_opportunity(nominal_close=.8,measured_close=.2,maximum_finger_opening=.035,enabled=True))


if __name__=='__main__':unittest.main()
