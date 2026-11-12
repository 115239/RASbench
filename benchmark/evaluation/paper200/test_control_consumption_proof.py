"""Adversarial evidence tests; synthetic fixtures are not physical acceptance."""
import hashlib,json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np
from .attack_proof import verify_effect
from .test_judge_contract_v2 import blob
from .test_neural_consumption import synthetic_receipt

class ControlProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.f=h5py.File(Path(self.tmp.name)/'e.h5','w')
        self.c=self.f.create_group('extensions/paper200_collection');self.a=self.f.create_group('attack')
    def tearDown(self):self.f.close();self.tmp.cleanup()
    def application(self,stage,before,after,step=100):
        app=self.a.create_group('applications')
        app['original_json']=[json.dumps(before)];app['attacked_json']=[json.dumps(after)]
        app['physics_step']=[step];app['timestamp_s']=[step*.005]
        payload=self.c.require_group('interception_payloads')
        blob(payload,stage,{'runtime_context':{'physics_step':step,'command_id':'cmd1','time_s':step*.005},'original':before,'attacked':after})
    def receipts(self,rows):
        if 'consumer_receipts_jsonl' in self.c:del self.c['consumer_receipts_jsonl']
        self.c['consumer_receipts_jsonl']=np.frombuffer(('\n'.join(json.dumps(r) for r in rows)).encode(),np.uint8)
    def test_base_joins_original_attack_then_Agent_and_actual_obs(self):
        self.application('control_base',[.1,0,0],[.4,0,0])
        row={'consumer':'locomotion_onnx','receipt_status':'inference_returned','physics_step':100,'command_id':'cmd1','physics_dt_s':.005,'sim_time_s':.5,
          'original_base_command':[.1,0,0],'attacked_base_command':[.4,0,0],'consumed_base_command':[0,0,0],
          'policy_base_command':[0,0,0],'observation':[[0]*57],'base_command_frame':'base_link',
          'base_command_units':['m/s','m/s','rad/s'],'base_command_scale':[1,1,1],'output':[0]*16,
          'base_command_observation_slice':[6,9]}
        self.receipts([row]);self.assertTrue(verify_effect(self.f,self.a,'control_base')['passed'])
        row['observation'][0][6]=.4;self.receipts([row])
        self.assertFalse(verify_effect(self.f,self.a,'control_base')['passed'])
    def test_full_joint_injection_survives_Agent_block_but_partial_fake_fails(self):
        before={'positions':{'joint%d'%i:.2 for i in range(1,9)},'wheel_velocities':{'wheel':0.}}
        after=json.loads(json.dumps(before));after['positions']['joint3']=.4
        self.application('control_joint',before,after)
        names=list(before['positions'])+['wheel']
        row={'consumer':'articulation_controller.apply_action','receipt_status':'apply_action_returned','physics_step':100,'command_id':'cmd1','physics_dt_s':.005,'sim_time_s':.5,
            'original':before,'attacked':after,'ordinary_clamped':after,'consumed':before,
            'joint_names':names,'joint_indices':list(range(9)),'coordinate_frame':'articulation_joint_local',
            'velocity_units':'rad/s','position_units':{n:'rad' for n in before['positions']}}
        self.receipts([row]);self.assertTrue(verify_effect(self.f,self.a,'control_joint')['passed'])
        row['consumed']={'positions':{'joint3':.2},'wheel_velocities':{'wheel':0.}}
        self.receipts([row]);self.assertFalse(verify_effect(self.f,self.a,'control_joint')['passed'])
    def test_lidar_requires_same_cloud_transform_and_frame_receipt(self):
        self.application('sensor_lidar',{'points':[1]},{'points':[2]})
        p=self.c.create_group('lidar_payloads/one');q=self.c.create_group('lidar_consumers/one')
        for name,value in {'physics_step':100,'original_points':[[1.,0,0],[2,0,0]],'attacked_points':[[2.,0,0]],
            'attacked_sensor_to_world':np.eye(4),'attacked_timestamp_s':.5,'attacked_frame_id':'m20_rtx_lidar'}.items():p[name]=value
        for name,value in {'physics_step':110,'frame_physics_step':100,'points':[[2.,0,0]],'command':[.2,0,0],
            'sensor_to_world':np.eye(4),'timestamp_s':.5,'frame_id':'m20_rtx_lidar'}.items():q[name]=value
        self.assertTrue(verify_effect(self.f,self.a,'sensor_lidar')['passed'])
        q['points'][0,0]=1.;self.assertFalse(verify_effect(self.f,self.a,'sensor_lidar')['passed'])
    def test_material_single_application_requires_real_hold_and_image_join(self):
        self.application('immediate_visual_material',{'/Mesh':'/Old'},{'/Mesh':'/New'})
        blob(self.c,'result_json',{'termination_physics_step':103})
        common={'attack_id':'a','original_materials':{'/Mesh':'/Old'},'actual_materials':{'/Mesh':'/New'},
            'override_material':'/New','until_physics_step':999}
        records=[dict(common,consumer='usd_renderer_binding',receipt_status='bindings_read',physics_step=s) for s in (101,102,103)]
        image=np.zeros((2,2,3),np.uint8)
        rendered=dict(common,consumer='rendered_policy_observation',receipt_status='pixels_read',physics_step=102,
            images={'front':{'shape':[2,2,3],'dtype':'uint8','sha256':hashlib.sha256(image.tobytes()).hexdigest()}})
        records.append(rendered);self.receipts(records)
        root=self.c.create_group('model_boundaries');root.attrs['complete']=True
        request=root.create_group('1');request.attrs.update(physics_step=102,receipt_status='inference_returned')
        request.attrs['request_id']=1
        request['front/original']=image
        request['front/consumed']=image
        synthetic_receipt(request)
        self.assertTrue(verify_effect(self.f,self.a,'immediate_visual_material')['passed'])
        self.receipts(records[1:]);result=verify_effect(self.f,self.a,'immediate_visual_material')
        self.assertFalse(result['passed']);self.assertIn('MATERIAL_HOLD_COVERAGE_INCOMPLETE',result['issues'])

    def test_material_immediate_lifecycle_requires_exact_initial_binding(self):
        self.a.create_group('applications')
        self.a['metadata/attack_id']='a'
        event={'event':'visual_material_override_installed','attack_id':'a','activated_physics_step':100,
               'until_physics_step':999,'original_materials':{'/Mesh':'/Old'},'affected_mesh_prims':['/Mesh'],
               'override_material':'/New','texture_sha256':'a'*64}
        self.a['immediate_effect/lifecycle_event_json']=[json.dumps(event)]
        blob(self.c,'result_json',{'termination_physics_step':103})
        common={'attack_id':'a','original_materials':{'/Mesh':'/Old'},'actual_materials':{'/Mesh':'/New'},
            'override_material':'/New','until_physics_step':999,'texture_sha256':'a'*64}
        records=[dict(common,consumer='usd_renderer_binding',receipt_status='bindings_read',physics_step=s) for s in (100,101,102,103)]
        image=np.zeros((2,2,3),np.uint8)
        records.append(dict(common,consumer='rendered_policy_observation',receipt_status='pixels_read',physics_step=102,
            images={'front':{'shape':[2,2,3],'dtype':'uint8','sha256':hashlib.sha256(image.tobytes()).hexdigest()}}))
        self.receipts(records)
        root=self.c.create_group('model_boundaries');root.attrs['complete']=True
        request=root.create_group('1');request.attrs.update(physics_step=102,receipt_status='inference_returned',request_id=1)
        request['front/original']=image;request['front/consumed']=image;synthetic_receipt(request)
        self.assertTrue(verify_effect(self.f,self.a,'immediate_visual_material')['passed'])
        records[0]['original_materials']={'/Mesh':'/Fabricated'};self.receipts(records)
        result=verify_effect(self.f,self.a,'immediate_visual_material')
        self.assertFalse(result['passed']);self.assertIn('MATERIAL_INSTALL_EVENT_BINDING_MISMATCH',result['issues'])
        records[0]['original_materials']={'/Mesh':'/Old'};self.receipts(records)
        del self.a['immediate_effect/lifecycle_event_json']
        self.assertFalse(verify_effect(self.f,self.a,'immediate_visual_material')['passed'])

if __name__=='__main__':unittest.main()
