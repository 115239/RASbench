import sys,unittest
import numpy as np
from .common import PROJECT
sys.path.insert(0,str(PROJECT/'datasets/paper200/collection_20260924_v2/runtime/runner/m20_runtime'))
from attack_tools.standard_attack_handlers import lidar_point_drift_frame,lidar_object_dropout_frame
from .attack_runtime import physical_payload_digest

class LidarBoundaryTests(unittest.TestCase):
    def frame(self):
        return {'points':np.array([[1.,0,0],[2,0,0]],np.float32),'object_ids':np.array(['a','b']),
                'stable_id_map':{'a':'/World/Target/Mesh','b':'/World/Other/Mesh'},'sensor_to_world':np.eye(4)}
    def test_out_of_view_target_does_not_modify_other_returns(self):
        original=self.frame();params={'target_prim_path':'/World/Occluded','range_scale':1.25}
        for handler in [lidar_point_drift_frame,lidar_object_dropout_frame]:
            result=handler(original,params,{})
            self.assertTrue(np.array_equal(result['points'],original['points']))
            self.assertEqual(physical_payload_digest(result),physical_payload_digest(original))
    def test_range_drift_targets_only_selected_mesh(self):
        frame=self.frame();result=lidar_point_drift_frame(frame,{'target_prim_path':'/World/Target','range_scale':1.25},{})
        np.testing.assert_array_equal(result['points'],[[1.25,0,0],[2,0,0]])
        np.testing.assert_array_equal(frame['points'],[[1,0,0],[2,0,0]])
        self.assertNotEqual(physical_payload_digest(result),physical_payload_digest(frame))
    def test_annotation_alone_is_not_a_physical_effect(self):
        frame=self.frame()
        self.assertEqual(physical_payload_digest(frame),physical_payload_digest({**frame,'attack_metadata':{'claimed_effect':True}}))

if __name__=='__main__':unittest.main()
