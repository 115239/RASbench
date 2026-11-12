import unittest
from .interaction_events import derived_events

class PhysicalEventTests(unittest.TestCase):
    def kinds(self,**updates):
        state=dict(phase='PICK',model_phase='ARM_APPROACH',world_delta=[.01,0,0],tcp=[0,0,1],target=[1,0,1],initial_target=[1,0,1],touching_target=False,support_touching=True,support_seen=True,question='Q79')
        state.update(updates)
        return {r['type'] for r in derived_events(**state)}
    def test_approach_requires_motion_toward_target(self):
        self.assertIn('arm_approach_started',self.kinds())
        self.assertNotIn('arm_approach_started',self.kinds(world_delta=[-.01,0,0]))
        self.assertNotIn('arm_approach_started',self.kinds(world_delta=[0,0,0]))
    def test_contact_is_actual(self):
        self.assertNotIn('physical_contact',self.kinds())
        self.assertIn('physical_contact',self.kinds(touching_target=True))
    def test_support_departure_requires_history_separation_and_rise(self):
        self.assertNotIn('object_left_support',self.kinds(support_touching=False))
        self.assertNotIn('object_left_support',self.kinds(support_touching=False,target=[1,0,1.05],support_seen=False))
        self.assertIn('object_left_support',self.kinds(support_touching=False,target=[1,0,1.05]))
    def test_hold_phase_alone_cannot_fake_requested_height(self):
        self.assertNotIn('hold_started',self.kinds(model_phase='HOLD'))
        self.assertIn('hold_started',self.kinds(model_phase='HOLD',target=[1,0,1.03]))
        self.assertNotIn('hold_started',self.kinds(model_phase='HOLD',target=[1,0,1.03],target_speed_mps=.08))
        self.assertIn('hold_started',self.kinds(model_phase='ARM_APPROACH',target=[1,0,1.03]))

if __name__=='__main__':unittest.main()
