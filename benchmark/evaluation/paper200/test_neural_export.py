"""Synthetic export checks prevent unverified neural receipt claims."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from .export_model_boundaries import embed

class NeuralExportTests(unittest.TestCase):
    def test_requires_both_wire_and_real_raw_artifact_hashes(self):
        for tamper in (False,True):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory() as d:
                root=Path(d);episode=root/'episode';directory=episode/'attack_evidence/raw_payloads/vla_requests';directory.mkdir(parents=True)
                model=root/'model';model.mkdir();wire=b'SYNTHETIC_NOT_MODEL_WIRE';(model/'input_000001.msgpack').write_bytes(wire)
                raw=model/'raw.npz';np.savez(raw,synthetic_output=np.zeros((1,10)))
                receipt={'call_id':1,'forward_executed':True,'input_wire_sha256':hashlib.sha256(wire).hexdigest(),
                    'raw_output_path':str(raw),'raw_output_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
                record={'request_id':1,'physics_step':100,'time_s':.5,'original_prompt':'a','attacked_prompt':'b',
                    'original_state':np.zeros(25),'attacked_state':np.zeros(25),'vla_raw_actions':np.zeros((1,10)),
                    'vla_attacked_actions':np.zeros((1,10)),
                    'raw_response_json':json.dumps({'_paper_neural_receipt':receipt})}
                for prefix in ('original','forwarded'):
                    for camera in ('m20_camera_0_rgb','m20_camera_1_rgb','piper_wrist_rgb'):
                        record[prefix+'_'+camera]=np.zeros((2,2,3),np.uint8)
                np.savez(directory/'request_000001.npz',**record)
                if tamper:(model/'input_000001.msgpack').write_bytes(b'wrong wire')
                with h5py.File(root/'test.h5','w') as f:
                    c=f.create_group('collection');embed(c,episode)
                    self.assertEqual(bool(c['model_boundaries/1'].attrs['neural_evidence_verified']),not tamper)
                    self.assertIn('neural_raw_output_npz',c['model_boundaries/1'])

if __name__=='__main__':unittest.main()
