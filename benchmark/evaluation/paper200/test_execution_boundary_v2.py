import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from paper200.execution_boundary import WorldTargetOffset,cancel_at_stop,record_actuator_link,apply_ordinary_ik


class StopAndReceiptTests(unittest.TestCase):
    def test_ordinary_clamp_rebases_incremental_target_accumulator(self):
        c=SimpleNamespace(robot=SimpleNamespace(get_joint_positions=lambda:np.zeros(8)),
            piper_command_targets=np.zeros(7),last_ik_debug={})
        def ik(delta,grip):
            c.piper_command_targets=c.piper_command_targets+np.r_[np.repeat(delta[0],6),0.]
            return c.piper_command_targets.copy()
        c.differential_ik_targets=ik
        b=SimpleNamespace(c=c,arm_indices=np.arange(6),safe=lambda:np.zeros(25))
        action=np.zeros(10);action[3]=.2
        apply_ordinary_ik(b,action)
        np.testing.assert_allclose(c.piper_command_targets[:6],.05)
        action[3]=0.;apply_ordinary_ik(b,action)
        np.testing.assert_allclose(c.last_ik_debug['targets_before_ordinary_clamp'][:6],.05)
        np.testing.assert_array_equal(b.arm,c.piper_command_targets)

    def test_moving_base_refused_before_ik_except_actual_retreat_phase(self):
        calls=[];state=np.zeros(25);state[6]=.031
        c=SimpleNamespace(robot=SimpleNamespace(get_joint_positions=lambda:np.zeros(8)),last_ik_debug={})
        def ik(delta,grip):calls.append(True);return np.zeros(7)
        c.differential_ik_targets=ik
        b=SimpleNamespace(c=c,arm_indices=np.arange(6),safe=lambda:state,_paper_model_phase='ARM_APPROACH')
        with self.assertRaisesRegex(RuntimeError,'BASE_NOT_STABLE_FOR_MANIPULATION'):
            apply_ordinary_ik(b,np.zeros(10))
        self.assertEqual(calls,[])
        b._paper_model_phase='BASE_RETREAT';apply_ordinary_ik(b,np.zeros(10))
        self.assertEqual(calls,[True])
        b._paper_model_phase='CONTACT_CLOSE';state[6]=0.;state[11]=.051
        with self.assertRaisesRegex(RuntimeError,'BASE_NOT_STABLE_FOR_MANIPULATION'):
            apply_ordinary_ik(b,np.zeros(10))

    def test_stop_cancels_without_fabricating_restore_or_motion(self):
        with tempfile.TemporaryDirectory() as d:
            tracker=WorldTargetOffset();tracker.adapt([0,0,0],[0,.02,0])
            b=SimpleNamespace(out=Path(d),_paper_world_target_offset=tracker,
                c=SimpleNamespace(physics_step=123),r={'PHYSICS_DT':.005})
            cancel_at_stop(b);cancel_at_stop(b)
            records=[json.loads(s) for s in (Path(d)/'EE_COMMAND_CHAIN.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),1)
            self.assertFalse(records[0]['physical_restoration_executed'])
            self.assertEqual(records[0]['cancelled_future_bias_world_m'],[0,.02,0])
            np.testing.assert_array_equal(tracker.previous,[0,0,0])

    def test_receipt_link_cannot_reuse_previous_request(self):
        with tempfile.TemporaryDirectory() as d:
            b=SimpleNamespace(out=Path(d),_paper_ee_request_id=4,_paper_ee_command_id='ee:4:100',
                c=SimpleNamespace(physics_step=104),r={'PHYSICS_DT':.005})
            record_actuator_link(b,3,{'command_id':42})
            self.assertFalse((Path(d)/'EE_COMMAND_CHAIN.jsonl').exists())
            record_actuator_link(b,4,{'command_id':43})
            row=json.loads((Path(d)/'EE_COMMAND_CHAIN.jsonl').read_text())
            self.assertEqual(row['actuator_command_id'],43)
            self.assertEqual(row['ee_command_id'],'ee:4:100')


if __name__=='__main__':unittest.main()
