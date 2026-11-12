import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from paper200.geometry_evidence import GeometryEvidence, validate_and_embed


class GeometryEvidenceTests(unittest.TestCase):
    def test_lossless_step_ledger_and_truncated_clock_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);writer=GeometryEvidence(None,out,.005)
            metric={'complete':True,'distance_m':.12,'distance_kind':'conservative_collision_lower_bound'}
            writer.observe(1,.005,{'lifecycle_phase':'WARMUP','geometry_metrics':{}})
            writer.observe(2,.01,{'lifecycle_phase':'TASK_RUNNING','geometry_metrics':{'S08':metric}})
            writer.close()
            with h5py.File(out/'test.h5','w') as f:
                self.assertTrue(validate_and_embed(f,out,np.array([.005,.01])))
                g=f['geometry_observations'];start,end=g['offsets'][1:3]
                row=json.loads(g['json_bytes'][start:end].tobytes())
                self.assertEqual(row['geometry_metrics']['S08'],metric)
                self.assertEqual(row['physics_step'],2)
            with h5py.File(out/'bad.h5','w') as f:
                with self.assertRaisesRegex(ValueError,'clock mismatch'):
                    validate_and_embed(f,out,np.array([.005]))

    def test_physics_gap_rejected_before_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=GeometryEvidence(None,Path(directory),.005)
            try:
                with self.assertRaisesRegex(ValueError,'clock gap'):
                    writer.observe(2,.01,{})
            finally:writer.file.close()


if __name__=='__main__':unittest.main()
