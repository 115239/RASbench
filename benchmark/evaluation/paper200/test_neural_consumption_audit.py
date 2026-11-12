import tempfile
from pathlib import Path
import unittest
import numpy as np
from paper200.model_service import NeuralAudit,flatten_raw_arrays


def observation(time_s):
    image=np.full((2,2,3),int(time_s*10),np.uint8)
    return {'mission':{'phase':'NAV_OUTBOUND','epoch':1,'time_s':time_s},
        'prompt':'public instruction','observation/state':np.zeros(25,np.float32),
        'observation/m20_camera_0_rgb':image,'observation/m20_camera_1_rgb':image,
        'observation/piper_wrist_rgb':image}


class NeuralAuditTests(unittest.TestCase):
    def test_wait_receipt_does_not_claim_a_forward(self):
        with tempfile.TemporaryDirectory() as d:
            a=NeuralAudit('groot',Path(d));a.begin(observation(0),1,b'wire')
            row=a.finish({'selection':{'decision':'WAIT_FOR_FRESH_FRAMES'},'actions':np.zeros((1,10))})
            self.assertFalse(row['forward_executed'])
            self.assertTrue(all(not ids for ids in row['consumed_request_ids'].values()))
            self.assertIsNone(row['raw_output_path'])

    def test_actual_navigation_joins_cached_front_not_unused_state(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=NeuralAudit('groot',d);history=[]
            for rid,t in enumerate((0.,.5,1.),1):
                obs=observation(t);a.begin(obs,rid,('wire%d'%rid).encode())
                history.append((t,obs['observation/m20_camera_0_rgb'],obs['prompt']))
            raw=d/'raw.npz';np.savez(raw,goals=np.zeros((3,3,6)))
            a.record_navigation(history,raw,3);row=a.finish({'selection':{'decision':'ACCEPT'}})
            self.assertTrue(row['forward_executed'])
            self.assertEqual(row['consumed_request_ids']['front'],[1,2,3])
            self.assertEqual(row['consumed_request_ids']['prompt'],[3])
            self.assertEqual(row['consumed_request_ids']['state'],[])
            self.assertEqual(row['state_indices'],[])

    def test_pi_state_subset_and_unavailable_camera_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=NeuralAudit('pi05',d);a.begin(observation(0),1,b'wire')
            raw=d/'raw.npz';np.savez(raw,actions=np.zeros((10,32)))
            a.record_direct(raw,('front','wrist','state','prompt'),[12,13,14,15,16,17,24])
            row=a.finish({});self.assertEqual(row['consumed_request_ids']['rear'],[])
            self.assertEqual(row['state_indices'],[12,13,14,15,16,17,24])

    def test_raw_model_outputs_saved_without_object_pickle(self):
        raw=flatten_raw_arrays({'action':{'arm':np.zeros((2,6),np.float32)}})
        self.assertEqual(raw['action/arm'].shape,(2,6))
        with self.assertRaises(ValueError):flatten_raw_arrays({'unsupported':object()})


if __name__=='__main__':unittest.main()
