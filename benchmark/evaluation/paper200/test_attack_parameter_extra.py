"""Synthetic adversarial checks; these are not physical acceptance evidence."""
import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from .attack_parameter_extra import verify, phantom, image_expected, anchored_points
from .attack_proof import text, rows


class ParameterGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = h5py.File(Path(self.tmp.name)/'e.h5', 'w')
        self.c = self.f.create_group('collection');self.a = self.f.create_group('attack')

    def tearDown(self):
        self.f.close();self.tmp.cleanup()

    def check(self, handler, stage, p):
        return verify(self.c, self.a, dict(id='attack', handler=handler, stage=stage, parameters=p), stage, [], rows, text)

    def sample(self, before, after):
        s = self.c.create_group('lidar_payloads/1')
        s['original_points'] = np.asarray(before, np.float32);s['attacked_points'] = np.asarray(after, np.float32)
        s['physics_step'] = 20
        for prefix in ('original', 'attacked'):
            s[prefix+'_sensor_to_world'] = np.eye(4)
            s[prefix+'_frame_id'] = 'm20_rtx_lidar';s[prefix+'_timestamp_s'] = .1
        return s

    def test_target_dropout_requires_per_sample_map_and_exact_survivors(self):
        s = self.sample([[1, 0, 0], [2, 0, 0], [3, 0, 0]], [[2, 0, 0]])
        s['original_object_ids'] = [1, 2, 3];s['attacked_object_ids'] = [2]
        p = dict(target_prim_path='/Target', include_descendants=True)
        self.assertFalse(self.check('lidar_object_dropout_frame', 'sensor_lidar', p)['passed'])
        s['stable_id_map_json'] = json.dumps({'1':'/Target', '2':'/Other', '3':'/Target/Child'})
        self.assertTrue(self.check('lidar_object_dropout_frame', 'sensor_lidar', p)['passed'])
        s['attacked_points'][0, 0] = 2.01
        self.assertFalse(self.check('lidar_object_dropout_frame', 'sensor_lidar', p)['passed'])

    def test_world_roi_uses_actual_transform(self):
        s = self.sample([[0, 0, 0], [1, 0, 0]], [[1, 0, 0]])
        for prefix in ('original', 'attacked'):
            s[prefix+'_sensor_to_world'][0, 3] = 10
        p = {'world_aabb':{'min':[9.9, -.1, -.1], 'max':[10.1, .1, .1]}}
        self.assertTrue(self.check('lidar_object_dropout_frame', 'sensor_lidar', p)['passed'])
        s['attacked_sensor_to_world'][0, 3] = 11
        self.assertFalse(self.check('lidar_object_dropout_frame', 'sensor_lidar', p)['passed'])

    def test_drift_preserves_unselected_points(self):
        s = self.sample([[1, 0, 0], [2, 0, 0]], [[1, .2, 0], [2, 0, 0]])
        p = {'sensor_aabb':{'min':[.9, -.1, -.1], 'max':[1.1, .1, .1]}, 'translation_xyz_m':[0, .2, 0]}
        self.assertTrue(self.check('lidar_point_drift_frame', 'sensor_lidar', p)['passed'])
        s['attacked_points'][1, 1] = .2
        self.assertFalse(self.check('lidar_point_drift_frame', 'sensor_lidar', p)['passed'])

    def test_seeded_phantom_requires_exact_random_returns_and_seed(self):
        p = dict(count=8, seed=21)
        s = self.sample([[0, 0, 0]], np.concatenate([[[0, 0, 0]], phantom(p, 20)]))
        self.assertTrue(self.check('lidar_phantom_add_frame', 'sensor_lidar', p)['passed'])
        p['seed'] = 22
        self.assertFalse(self.check('lidar_phantom_add_frame', 'sensor_lidar', p)['passed'])

    def test_anchored_phantom_requires_ground_and_trigger_receipt(self):
        p = dict(anchor_frame='world_at_trigger', forward_distance_m=1., width_m=.6,
                 thickness_m=.1, height_m=.8, layers=2, columns=2, rows=2)
        s = self.sample([[0, 0, 0]], [[1, 0, 0]])
        self.assertFalse(self.check('lidar_phantom_add_frame', 'sensor_lidar', p)['passed'])
        receipt = dict(consumer='phantom_anchor', attack_id='attack', trigger_position=[0, 0, .5],
                       trigger_yaw=0., ground_hit_position=[0, 0, 0])
        receipt['world_points'] = [[x, y, z] for x in (.95, 1.05) for y in (-.3, .3) for z in (0, .8)]
        self.assertTrue(np.allclose(anchored_points(p, receipt), receipt['world_points']))
        del s['attacked_points'];s['attacked_points'] = np.concatenate([[[0, 0, 0]], receipt['world_points']]).astype(np.float32)
        self.c['consumer_receipts_jsonl'] = json.dumps(receipt)
        self.assertTrue(self.check('lidar_phantom_add_frame', 'sensor_lidar', p)['passed'])
        receipt['ground_hit_position'] = [0, 0, .1]
        del self.c['consumer_receipts_jsonl'];self.c['consumer_receipts_jsonl'] = json.dumps(receipt)
        self.assertFalse(self.check('lidar_phantom_add_frame', 'sensor_lidar', p)['passed'])

    def image_record(self, p):
        before = np.zeros((6, 6, 3), np.uint8)+90
        ids = np.zeros((6, 6), np.int32);ids[1:5, 2:5] = 3;mask = ids == 3
        record = self.c.create_group('visual_masks/one')
        record['metadata_json'] = json.dumps(dict(attack_id='attack', physics_step=10,
                                                  idToLabels={'3':'/Target/mesh', '0':'/Other'}))
        for name, value in [('image_before', before), ('mask', mask), ('instance_ids', ids)]:
            record[name] = value
        return record, before, mask

    def join_image(self, record, p, before, mask):
        after = image_expected(before, mask, p, record);record['image_after'] = after
        request = self.c.create_group('model_boundaries/1');request.attrs['physics_step'] = 10
        request['front/original'] = before;request['front/attacked'] = after
        return request

    def test_checker_mask_must_join_instance_ids_and_request(self):
        p = dict(target_prim='/Target', tile_px=2, color_a_rgb=[255, 255, 255], color_b_rgb=[0, 0, 0])
        record, before, mask = self.image_record(p);request = self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        record['instance_ids'][1, 2] = 0
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_stripes_angle_wrong_rejected(self):
        p = dict(target_prim='/Target', texture_mode='diagonal_stripes', stripe_angle_deg=30,
                 stripe_period_px=3, bright_fraction=.5, color_a_rgb=[255]*3, color_b_rgb=[0]*3)
        record, before, mask = self.image_record(p);self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        p['stripe_angle_deg'] = 60
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_floor_uses_exact_frozen_texture_bytes(self):
        out = io.BytesIO();Image.fromarray(np.arange(27, dtype=np.uint8).reshape(3, 3, 3)).save(out, format='PNG')
        raw = out.getvalue()
        p = dict(target_prim='/Target', texture_mode='scene_floor_texture', texture_sha256=hashlib.sha256(raw).hexdigest())
        record, before, mask = self.image_record(p);record['texture_asset_bytes'] = np.frombuffer(raw, np.uint8)
        self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        record['texture_asset_bytes'][10] ^= 1
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_glare_and_every_changed_request_need_exact_join(self):
        p = dict(target_prim='/Target', texture_mode='specular_glare', streak_angle_deg=30,
                 streak_width_fraction=.1, secondary_strength=.3, highlight_gain=.8)
        record, before, mask = self.image_record(p);request = self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        second = self.c['model_boundaries'].create_group('2');second.attrs['physics_step'] = 11
        second['front/original'] = before;second['front/attacked'] = request['front/attacked'][...]
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        del self.c['model_boundaries/2']
        p['highlight_gain'] = .2
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_npz_schema_aliases_remain_value_verified(self):
        p = dict(target_prim='/Target', tile_px=2)
        record, before, mask = self.image_record(p);self.join_image(record, p, before, mask)
        record.move('image_before', 'before');record.move('image_after', 'after')
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        record['after'][0, 0, 0] = 100
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_material_requires_frozen_content_hash_and_exact_mesh(self):
        row = dict(consumer='usd_renderer_binding', attack_id='attack', original_materials={'/T/mesh':'/Old'},
                   actual_materials={'/T/mesh':'/New'}, override_material='/New', texture_sha256='abc')
        self.c['consumer_receipts_jsonl'] = json.dumps(row)
        p = dict(expected_affected_mesh_prims=['/T/mesh'])
        self.assertFalse(self.check('install_visual_material_override', 'immediate_visual_material', p)['passed'])
        p['texture_sha256'] = 'abc'
        self.assertTrue(self.check('install_visual_material_override', 'immediate_visual_material', p)['passed'])
        p['expected_affected_mesh_prims'] += ['/Other']
        self.assertFalse(self.check('install_visual_material_override', 'immediate_visual_material', p)['passed'])

    def test_contour_width_and_local_checker_origin_verified(self):
        from scipy.ndimage import binary_erosion
        p = dict(target_prim='/Target', texture_mode='contour_checker', mask_region='inner_contour',
                 contour_width_px=3, tile_px=2, color_a_rgb=[255]*3, color_b_rgb=[0]*3)
        record = self.c.create_group('visual_masks/one')
        record['metadata_json'] = json.dumps(dict(attack_id='attack', physics_step=10,
                                                  idToLabels={'3':'/Target/mesh', '0':'/Other'}))
        before = np.zeros((16, 17, 3), np.uint8)+90
        ids = np.zeros((16, 17), np.int32);ids[2:14, 3:15] = 3
        full = ids == 3;mask = full & ~binary_erosion(full, iterations=3, border_value=0)
        record['image_before'] = before;record['mask'] = mask;record['instance_ids'] = ids
        request = self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        self.assertTrue(np.array_equal(record['image_after'][2, 3], [255]*3))
        self.assertTrue(np.array_equal(record['image_after'][5, 6], before[5, 6]))
        p['contour_width_px'] = 2
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_contour_mask_resize_accepts_readonly_image_buffer(self):
        from scipy.ndimage import binary_erosion
        p = dict(target_prim='/Target', texture_mode='contour_checker', mask_region='inner_contour',
                 contour_width_px=3, tile_px=2, color_a_rgb=[255]*3, color_b_rgb=[0]*3)
        record = self.c.create_group('visual_masks/one')
        record['metadata_json'] = json.dumps(dict(attack_id='attack', physics_step=10,
                                                  idToLabels={'3':'/Target/mesh', '0':'/Other'}))
        before = np.full((16, 16, 3), 90, np.uint8)
        ids = np.zeros((8, 8), np.int32);ids[1:7, 1:7] = 3
        full = np.asarray(Image.fromarray(ids == 3).resize((16, 16), Image.Resampling.NEAREST))
        self.assertFalse(full.flags.writeable)
        mask = full & ~binary_erosion(full, iterations=3, border_value=0)
        record['image_before'] = before;record['mask'] = mask;record['instance_ids'] = ids
        self.join_image(record, p, before, mask)
        self.assertTrue(self.check('paper200_projected_patch', 'request_image', p)['passed'])
        record['mask'][4, 4] = False
        self.assertFalse(self.check('paper200_projected_patch', 'request_image', p)['passed'])

    def test_unknown_lidar_semantics_reported_before_missing_data(self):
        result = self.check('unregistered_lidar', 'sensor_lidar', {})
        self.assertFalse(result['passed'])
        self.assertIn('PARAMETER_SEMANTICS_NOT_VERIFIED:unregistered_lidar', result['issues'])


if __name__ == '__main__':
    unittest.main()
