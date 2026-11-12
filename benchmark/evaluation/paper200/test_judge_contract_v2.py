import json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from paper200.evidence_contract import canonical_digest
from paper200.contract_verification import verify_contract,compare_pair
from paper200.attack_proof import verify_effect
from paper200.test_neural_consumption import synthetic_receipt


def blob(group,name,value):group.create_dataset(name,data=np.frombuffer(json.dumps(value).encode(),np.uint8))


def contract_file(file):
    m=file.create_group('metadata');g=file.create_group('extensions/paper200_collection')
    contract={'safety_profile':{'rules':{'S12':{'objects':[{'prim':'/Basket'}]}}},'attacks':[]}
    blob(g,'runner_judge_contract_json',contract);m.attrs['runner_judge_contract_sha256']=canonical_digest(contract)
    blob(file.create_group('extensions/m20_safety_v1'),'profile_json',contract['safety_profile'])
    return contract


class JudgeContractTests(unittest.TestCase):
    def test_full_content_not_rule_id(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            c=contract_file(f)
            self.assertTrue(verify_contract(f,c)['passed'])
            c['safety_profile']['rules']['S12']['objects'][0]['prim']='/TaskTarget'
            self.assertIn('FROZEN_RUNNER_JUDGE_CONTRACT_MISMATCH',verify_contract(f,c)['issues'])

    def test_self_declared_contract_is_not_trusted_expected_contract(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            contract_file(f)
            self.assertFalse(verify_contract(f)['passed'])

    def test_runtime_profile_drift_same_identifier_rejected(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            c=contract_file(f);g=f['extensions/m20_safety_v1'];del g['profile_json']
            blob(g,'profile_json',{'rules':{'S12':{'objects':[]}}})
            self.assertIn('RECORDED_SAFETY_PROFILE_CONTENT_MISMATCH',verify_contract(f,c)['issues'])

    def test_missing_formal_initial_state_is_not_pair(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as a,h5py.File(Path(d)/'b.h5','w') as b:
            contract_file(a);contract_file(b)
            self.assertIn('PAIRED_FORMAL_INITIAL_STATE_MISSING',compare_pair(a,b)['issues'])

    def test_metadata_annotation_and_count_not_effect(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            attack=f.create_group('attack');app=attack.create_group('applications')
            app.create_dataset('original_json',data=[json.dumps({'values':[1,2],'attack_metadata':{}})])
            app.create_dataset('attacked_json',data=[json.dumps({'values':[1,2],'attack_metadata':{'applied':True}})])
            app.create_dataset('physics_step',data=[1000]);app.create_dataset('timestamp_s',data=[5.])
            self.assertFalse(verify_effect(f,attack,'sensor_joint_encoder')['payload_changed'])

    def test_actual_joint_consumer_must_equal_attacked_values(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            attack=f.create_group('attack');app=attack.create_group('applications')
            app.create_dataset('original_json',data=[json.dumps({'joint_names':['j%d'%i for i in range(12)],'position_rad':[1.]*12})])
            app.create_dataset('attacked_json',data=[json.dumps({'joint_names':['j%d'%i for i in range(12)],'position_rad':[0.]*12})])
            app.create_dataset('physics_step',data=[1000]);app.create_dataset('timestamp_s',data=[5.])
            g=f.create_group('extensions/paper200_collection')
            row={'consumer':'locomotion_onnx','receipt_status':'inference_returned','physics_step':1000,
                'measured_position_rad':[1.]*12,'attacked_position_rad':[0.]*12,
                'consumed_position_rad':[1.]*12,'joint_names':['j%d'%i for i in range(12)],
                'joint_indices':list(range(12)),'observation_joint_names':['j%d'%i for i in range(16)],
                'position_observation_slice':[9,21],'position_default_rad':[0.]*12,'observation':[[0.]*57],'output':[.1,.2]}
            if 'interception_payloads' not in g:
                payload=g.create_group('interception_payloads')
                exact={'runtime_context':{'physics_step':1000,'time_s':5.},
                    'original':{'joint_names':row['joint_names'],'position_rad':row['measured_position_rad']},
                    'attacked':{'joint_names':row['joint_names'],'position_rad':row['attacked_position_rad']}}
                payload.create_dataset('sensor_joint_encoder',data=np.frombuffer(json.dumps(exact).encode(),np.uint8))
            g.create_dataset('consumer_receipts_jsonl',data=np.frombuffer(json.dumps(row).encode(),np.uint8))
            self.assertFalse(verify_effect(f,attack,'sensor_joint_encoder')['passed'])
            del g['consumer_receipts_jsonl'];row['consumed_position_rad']=[0.]*12
            if 'interception_payloads' not in g:
                payload=g.create_group('interception_payloads')
                exact={'runtime_context':{'physics_step':1000,'time_s':5.},
                    'original':{'joint_names':row['joint_names'],'position_rad':row['measured_position_rad']},
                    'attacked':{'joint_names':row['joint_names'],'position_rad':row['attacked_position_rad']}}
                payload.create_dataset('sensor_joint_encoder',data=np.frombuffer(json.dumps(exact).encode(),np.uint8))
            g.create_dataset('consumer_receipts_jsonl',data=np.frombuffer(json.dumps(row).encode(),np.uint8))
            self.assertTrue(verify_effect(f,attack,'sensor_joint_encoder')['passed'])

    def test_agent_repaired_joint_feedback_with_matching_actual_observation(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            attack=f.create_group('attack');app=attack.create_group('applications')
            app.create_dataset('original_json',data=[json.dumps({'joint_names':['j%d'%i for i in range(12)],'position_rad':[1.]*12})])
            app.create_dataset('attacked_json',data=[json.dumps({'joint_names':['j%d'%i for i in range(12)],'position_rad':[0.]*12})])
            app.create_dataset('physics_step',data=[1000]);app.create_dataset('timestamp_s',data=[5.])
            g=f.create_group('extensions/paper200_collection');obs=np.zeros(57);obs[9:21]=1.
            row={'consumer':'locomotion_onnx','receipt_status':'inference_returned','physics_step':1000,
                'measured_position_rad':[1.]*12,'attacked_position_rad':[0.]*12,
                'consumed_position_rad':[1.]*12,'joint_names':['j%d'%i for i in range(12)],
                'joint_indices':list(range(12)),'observation_joint_names':['j%d'%i for i in range(16)],
                'position_observation_slice':[9,21],'position_default_rad':[0.]*12,
                'observation':obs.tolist(),'output':[.1,.2],'Agent_mode':'on'}
            if 'interception_payloads' not in g:
                payload=g.create_group('interception_payloads')
                exact={'runtime_context':{'physics_step':1000,'time_s':5.},
                    'original':{'joint_names':row['joint_names'],'position_rad':row['measured_position_rad']},
                    'attacked':{'joint_names':row['joint_names'],'position_rad':row['attacked_position_rad']}}
                payload.create_dataset('sensor_joint_encoder',data=np.frombuffer(json.dumps(exact).encode(),np.uint8))
            g.create_dataset('consumer_receipts_jsonl',data=np.frombuffer(json.dumps(row).encode(),np.uint8))
            proof=verify_effect(f,attack,'sensor_joint_encoder')
            self.assertTrue(proof['passed']);self.assertTrue(proof['evidence'][-1]['post_Agent_differs'])

    def test_state_and_response_require_exact_interception_to_request_join(self):
        for stage in ('state_feedback','response_action'):
            with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
                attack=f.create_group('attack');app=attack.create_group('applications')
                app.create_dataset('original_json',data=[json.dumps({'sha256':'before','type':'ndarray'})])
                app.create_dataset('attacked_json',data=[json.dumps({'sha256':'after','type':'ndarray'})])
                app.create_dataset('physics_step',data=[1000]);app.create_dataset('timestamp_s',data=[5.])
                g=f.create_group('extensions/paper200_collection');payload=g.create_group('interception_payloads')
                before=np.zeros(25 if stage=='state_feedback' else (1,10));after=before+1
                app['original_json'][0]=json.dumps(before.tolist());app['attacked_json'][0]=json.dumps(after.tolist())
                row={'runtime_context':{'physics_step':1000},'original':before.tolist(),'attacked':after.tolist()}
                payload.create_dataset(stage,data=np.frombuffer(json.dumps(row).encode(),np.uint8))
                boundary=g.create_group('model_boundaries');boundary.attrs['complete']=True
                req=boundary.create_group('1');req.attrs.update(request_id=1,physics_step=1000,receipt_status='inference_returned')
                req.create_dataset('raw_actions',data=np.zeros((1,10)))
                key='state/original' if stage=='state_feedback' else 'returned_actions'
                req.create_dataset(key,data=after)
                if stage=='state_feedback':
                    req.create_dataset('state/consumed',data=before)
                    req.create_dataset('state/attacked',data=after)
                    synthetic_receipt(req)

                self.assertTrue(verify_effect(f,attack,stage)['passed'])
                req.attrs['physics_step']=1001
                self.assertFalse(verify_effect(f,attack,stage)['passed'])

    def test_agent_blocked_prompt_still_has_valid_injection(self):
        with tempfile.TemporaryDirectory() as d,h5py.File(Path(d)/'a.h5','w') as f:
            attack=f.create_group('attack');app=attack.create_group('applications')
            app.create_dataset('original_json',data=[json.dumps('clean')]);app.create_dataset('attacked_json',data=[json.dumps('attack')])
            app.create_dataset('physics_step',data=[1000]);app.create_dataset('timestamp_s',data=[5.])
            g=f.create_group('extensions/paper200_collection/model_boundaries');g.attrs['complete']=True
            req=g.create_group('1');req.attrs.update(physics_step=1000,request_id=1,receipt_status='inference_returned')
            req.create_dataset('raw_actions',data=[[0]*10]);p=req.create_group('prompt')
            for k,v in [('original','clean'),('attacked','attack'),('consumed','guarded')]:p.create_dataset(k,data=v)
            synthetic_receipt(req)
            proof=verify_effect(f,attack,'request_prompt')
            self.assertTrue(proof['passed']);self.assertTrue(proof['evidence'][-1]['post_Agent_differs'])


if __name__=='__main__':unittest.main()
