"""Synthetic rejection controls for independently reconstructed parameters."""
import json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from .attack_parameter_proof import verify
from .attack_proof import rows,text,verify_effect
from .test_judge_contract_v2 import blob
from .test_neural_consumption import synthetic_receipt

class ParameterProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.f=h5py.File(Path(self.tmp.name)/'a.h5','w')
        self.c=self.f.create_group('extensions/paper200_collection');self.a=self.f.create_group('attack');self.a['timing/activation_time_s']=0.
    def tearDown(self):self.f.close();self.tmp.cleanup()
    def receipts(self,records):self.c['consumer_receipts_jsonl']=np.frombuffer('\n'.join(json.dumps(r) for r in records).encode(),np.uint8)
    def check(self,spec,exact):return verify(self.c,self.a,spec,spec['stage'],exact,rows,text)
    def test_affine_checks_parameter_and_preserves_other_channels(self):
        spec={'id':'a','stage':'control_base','handler':'affine_actions','parameters':{'indices':[2],'scale':-1.,'offset':0.}}
        self.assertTrue(self.check(spec,[(1,[.1,.2,.3],[.1,.2,-.3])])['passed'])
        self.assertFalse(self.check(spec,[(1,[.1,.2,.3],[.1,.8,-.3])])['passed'])
    def test_hold_uses_trigger_measurement_not_current_value(self):
        spec={'id':'a','stage':'sensor_joint_encoder','handler':'joint_encoder_position_feedback','parameters':{'operation':'hold_trigger','joint_name':'fl_hipy_joint'}}
        before={'joint_names':['fl_hipy_joint','fr_hipy_joint'],'position_rad':[.1,.2]};after=dict(before,position_rad=[.3,.2])
        self.assertFalse(self.check(spec,[(1,before,after)])['passed'])
        self.receipts([{'consumer':'attack_handler_boundary','attack_id':'a','stage':spec['stage'],'physics_step':1,
            'handler':spec['handler'],'parameters':spec['parameters'],'snapshot':{'locomotion':{'measured_joint_positions_rad':{'fl_hipy_joint':.3}}},'runtime':{}}])
        self.assertTrue(self.check(spec,[(1,before,after)])['passed'])
        after['position_rad'][1]=9.;self.assertFalse(self.check(spec,[(1,before,after)])['passed'])
    def test_prompt_schedule_and_payload_disagreement_rejected(self):
        spec={'id':'a','stage':'request_prompt','handler':'replace_prompt_sequence','parameters':{'delivery':'once_per_scheduled_request',
            'steps':[{'at_s':0.,'replacement':'a'},{'at_s':1.,'replacement':'b'}]}}
        self.a['timing/activation_time_s'][()]=12.;app=self.a.create_group('applications')
        app['original_json']=[json.dumps('clean'),json.dumps('clean')];app['attacked_json']=[json.dumps('a'),json.dumps('b')]
        app['physics_step']=[2400,2600];app['timestamp_s']=[12.,13.]
        root=self.c.create_group('model_boundaries');root.attrs['complete']=True
        for i,after in enumerate(['a','b']):
            r=root.create_group(str(i));r.attrs.update(request_id=i,physics_step=2400+i*200,sim_time_s=12.+i,receipt_status='inference_returned')
            r['raw_actions']=np.zeros((1,10));r['prompt/original']='clean';r['prompt/attacked']=after;r['prompt/consumed']=after
            synthetic_receipt(r)
        self.assertTrue(verify_effect(self.f,self.a,'request_prompt',expected_attack=spec)['passed'])
        root['1'].attrs['sim_time_s']=13.5
        self.assertIn('PROMPT_SCHEDULE_TIME_MISMATCH',verify_effect(self.f,self.a,'request_prompt',expected_attack=spec)['issues'])
        root['1'].attrs['sim_time_s']=13.
        app['attacked_json'][0]=json.dumps('clean')
        self.assertIn('APPLICATION_PAYLOAD_CONTRADICTS_CAPTURED_MUTATION',verify_effect(self.f,self.a,'request_prompt',expected_attack=spec)['issues'])
    def test_object_relative_direction_rejects_wrong_axis_or_hold_drift(self):
        from .attack_parameter_proof import direction_vector
        definition={'target_prim':'/Target','mode':'principal_long_axis','magnitude':.02}
        spec={'id':'a','parameters':{'direction_contract':definition}}
        vertices=[[x,y,z] for x in [-.1,.1] for y in [-.4,.4] for z in [-.02,.02]]
        witness={'consumer':'attack_direction_resolution','attack_id':'a','direction_evidence_kind':'first_application_geometry',
            'definition':definition,'target_points_world':vertices,'target_binding':[{'collider':'/Target/Mesh'}],
            'resolved_world_vector':[0,.02,0],'frozen_activation_key':['a',3.]}
        np.testing.assert_allclose(direction_vector(spec,[witness]),[0,.02,0],atol=1e-12)
        held=dict(witness,direction_evidence_kind='held_vector_consumption',resolved_world_vector=[.02,0,0])
        with self.assertRaises(ValueError):direction_vector(spec,[witness,held])
        witness['resolved_world_vector']=[.02,0,0]
        with self.assertRaises(ValueError):direction_vector(spec,[witness])

    def test_expected_unknown_handler_does_not_get_parameter_pass(self):
        result=self.check({'id':'a','stage':'sensor_lidar','handler':'unimplemented','parameters':{}},[(1,[0],[1])])
        self.assertFalse(result['passed']);self.assertIn('PARAMETER_SEMANTICS_NOT_VERIFIED:unimplemented',result['issues'])

if __name__=='__main__':unittest.main()
