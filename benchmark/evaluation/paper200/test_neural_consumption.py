"""Explicit synthetic receipts test joins, never actual model execution claims."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from .neural_consumption import model_use
from .attack_proof import text


def synthetic_receipt(request,*,sources=None,forward=True,state_indices=None):
    sources=[request] if sources is None else sources
    inputs={name:[] for name in ('front','rear','wrist','state','prompt')}
    for source in sources:
        for name in inputs:
            if name not in source or 'consumed' not in source[name]:continue
            value=source[name+'/consumed'][()]
            payload=text(value).encode() if name=='prompt' else np.ascontiguousarray(value).tobytes()
            field={'front':'observation/m20_camera_0_rgb','rear':'observation/m20_camera_1_rgb',
                'wrist':'observation/piper_wrist_rgb','state':'observation/state','prompt':'prompt'}[name]
            inputs[name].append({'request_id':int(source.attrs['request_id']),'value_sha256':hashlib.sha256(payload).hexdigest(),
                'source_field':field,'dtype':'utf8' if name=='prompt' else str(np.asarray(value).dtype),
                'shape':[] if name=='prompt' else list(np.asarray(value).shape)})
    receipt={'forward_executed':forward,'call_id':int(request.attrs['request_id']),'consumed_sources':inputs if forward else {},
        'state_indices':list(range(25)) if state_indices is None else state_indices,'scope':'SYNTHETIC_UNIT_TEST_ONLY'}
    request.attrs['neural_evidence_verified']=True
    if 'neural_receipt_json' in request:del request['neural_receipt_json']
    request['neural_receipt_json']=np.frombuffer(json.dumps(receipt).encode(),np.uint8)


class NeuralConsumptionTests(unittest.TestCase):
    def test_wait_frame_can_be_used_later_but_unused_state_cannot(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            root=f.create_group('boundaries');first=root.create_group('1');first.attrs['request_id']=1
            first['front/consumed']=np.ones((2,2,3),np.uint8)
            first['state/original']=np.zeros(25);first['state/attacked']=np.ones(25);first['state/consumed']=np.ones(25)
            synthetic_receipt(first,forward=False)
            self.assertFalse(model_use(root,first,'front',text)['neural_verified'])
            later=root.create_group('3');later.attrs['request_id']=3
            synthetic_receipt(later,sources=[first],state_indices=[])
            self.assertTrue(model_use(root,first,'front',text)['neural_verified'])
            self.assertFalse(model_use(root,first,'state',text)['neural_verified'])
            first['front/consumed'][0,0,0]=2
            self.assertFalse(model_use(root,first,'front',text)['neural_verified'])
    def test_only_used_state_indices_can_prove_state_attack(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            root=f.create_group('boundaries');request=root.create_group('1');request.attrs['request_id']=1
            request['state/original']=np.zeros(25);request['state/attacked']=np.eye(1,25,5).ravel();request['state/consumed']=np.zeros(25)
            synthetic_receipt(request,state_indices=[12,13,14,15,16,17,24])
            self.assertFalse(model_use(root,request,'state',text)['neural_verified'])
            request['state/attacked'][14]=.5
            self.assertTrue(model_use(root,request,'state',text)['neural_verified'])

if __name__=='__main__':unittest.main()
