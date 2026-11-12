import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import read,write,sha
from .repair_contracts import relocate,implementation_identity
from .review_release_contract import repair_public_identity
from .repaired_visual_handlers import transform_image,mask_from_frame,restrict_mask
from .vision_pool import _isolated_identity,_frozen_service_source


class ReleaseContractTests(unittest.TestCase):
    def test_relocation_preserves_historical_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'v2';old=Path(td)/'v1'
            (root/'runtime').mkdir(parents=True);(root/'runtime/source.py').write_text('x=1')
            write(root/'manifests/repair_lineage.json',{'source_root':str(old)})
            value={'model_config':str(old/'runtime/source.py'),'pilot_result':str(old/'pilot/episode/result.json')}
            write(root/'manifests/assistance.json',value);relocate(root)
            got=read(root/'manifests/assistance.json')
            self.assertEqual(got['pilot_result'],value['pilot_result'])
            self.assertEqual(got['model_config'],str(root/'runtime/source.py'))

    def test_source_content_mutation_changes_contract_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('runner','baseline_runner','baseline_code','agent','openpi','judge/v2'):
                path=root/'runtime'/name/'entry.py';path.parent.mkdir(parents=True);path.write_text('x=1')
            for name,value in [('models',{'model':'fixture'}),('model_files_verified',{'passed':True,'files':{}}),
                               ('pi05_transfer',{}),('native_runtime',{})]:write(root/'manifests'/(name+'.json'),value)
            one=implementation_identity(root)
            (root/'runtime/runner/entry.py').write_text('x=2')
            two=implementation_identity(root)
            self.assertNotEqual(one['inventory_sha256'],two['inventory_sha256'])
            self.assertEqual(one['runtime_components']['openpi'],two['runtime_components']['openpi'])

    def test_public_identity_repair_preserves_instruction_and_original(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'v2';root.mkdir();(root/'STOP').touch()
            path=root/'manifests/public/Q1/task.json'
            write(path,{'benchmark_id':'v1','instruction':'observed public task','public_row_sha256':'source'})
            original=path.read_bytes();changes=repair_public_identity(root)
            self.assertEqual(len(changes),1);self.assertEqual(read(path)['benchmark_id'],'v2')
            self.assertEqual(read(path)['instruction'],'observed public task')
            self.assertEqual((root/'manifests/revisions/public_identity_original/Q1/task.json').read_bytes(),original)
            self.assertEqual(repair_public_identity(root),[])

    def test_visual_replica_identity_is_versioned_and_unique(self):
        first=_isolated_identity(Path('/tmp/paper_v2'),1);second=_isolated_identity(Path('/tmp/paper_v2'),1)
        self.assertTrue(first[0].startswith('paper_v2-vision-1-'))
        self.assertNotEqual(first[0],second[0]);self.assertNotEqual(first[2],second[2])
        self.assertGreater(first[1],0)

    def test_vision_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'runtime/frozen';source.mkdir(parents=True)
            p=source/'entry.py';p.write_text('x=1');config={'frozen_source':str(source),'source_sha256':{'entry.py':sha(p)}}
            _frozen_service_source(root,config);p.write_text('x=2')
            with self.assertRaisesRegex(ValueError,'source changed'):_frozen_service_source(root,config)


class VisualTextureTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((32,32,3),80,np.uint8);self.mask=np.zeros((32,32),bool);self.mask[4:28,4:28]=True
        self.mask[12:20,12:20]=False

    def test_diagonal_stripes_are_not_checkerboard_and_respect_occlusion(self):
        p={'texture_mode':'diagonal_stripes','stripe_angle_deg':45,'stripe_period_px':8,
           'bright_fraction':.5,'color_a_rgb':[255]*3,'color_b_rgb':[0]*3}
        result=transform_image(self.image,self.mask,p)
        np.testing.assert_array_equal(result[~self.mask],self.image[~self.mask])
        # Along a +45deg diagonal phase is constant; perpendicular phase changes.
        self.assertEqual(result[6,6,0],result[9,9,0])
        self.assertGreater(len(np.unique(result[self.mask,0])),1)
        np.testing.assert_array_equal(self.image,np.full_like(self.image,80))

    def test_glare_has_actual_saturation_without_recoloring_background(self):
        p={'texture_mode':'specular_glare','streak_angle_deg':25,'streak_width_fraction':.08,
           'highlight_gain':1.8,'secondary_strength':.35}
        result=transform_image(self.image,self.mask,p)
        self.assertTrue(np.any(result[self.mask]==255));self.assertTrue(np.any(result[self.mask]<180))
        np.testing.assert_array_equal(result[~self.mask],self.image[~self.mask])
        self.assertTrue(np.all(result>=self.image))

    def test_floor_mode_uses_supplied_texture_and_zero_mask_no_effect(self):
        texture=np.full((7,9,3),[173,122,89],np.uint8)
        result=transform_image(self.image,self.mask,{'texture_mode':'scene_floor_texture'},texture)
        np.testing.assert_array_equal(result[self.mask],np.tile([173,122,89],(int(self.mask.sum()),1)))
        untouched=transform_image(self.image,np.zeros_like(self.mask),{'texture_mode':'scene_floor_texture'},None)
        np.testing.assert_array_equal(untouched,self.image)

    def test_instance_mask_excludes_countertop_and_other_objects(self):
        frame={'instance_id_segmentation':{'data':np.array([[1,2],[3,1]]),
            'info':{'idToLabels':{'1':'/cabinet/door/mesh','2':'/cabinet/top','3':'/robot'}}}}
        mask,ids,_,_=mask_from_frame(frame,['/cabinet/door'],(2,2))
        self.assertEqual(ids,[1]);np.testing.assert_array_equal(mask,[[True,False],[False,True]])


class DirectionWitnessTests(unittest.TestCase):
    def test_pca_short_axis_is_second_axis_not_cross_z_proxy(self):
        from .attack_direction_witness import resolve_points_direction
        # Rotated rectangular physical shape with distinct three principal axes.
        import itertools
        points=np.array(list(itertools.product([-3.,3.],[-1.,1.],[-.2,.2])))
        angle=.3;rot=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        points=points@rot.T
        short,record=resolve_points_direction(points,None,'principal_short_axis')
        np.testing.assert_allclose(short,rot[:,1],atol=1e-10)
        self.assertEqual(record['selected_eigenvector'],1)

    def test_support_normal_uses_nearest_physical_hull_edge(self):
        from .attack_direction_witness import resolve_points_direction
        target=np.array([[-.9,-.01,1],[-.8,.01,1],[-.9,.0,1.1]])
        support=np.array([[-1,-2,1],[1,-2,1],[1,2,1],[-1,2,1]])
        direction,record=resolve_points_direction(target,support,'nearest_support_edge_outward')
        np.testing.assert_allclose(direction,[-1,0,0]);self.assertAlmostEqual(record['distance_m'],.1333333333)

    def test_contour_preserves_interior_and_outside_occlusion(self):
        mask=np.zeros((20,20),bool);mask[2:18,2:18]=True
        contour=restrict_mask(mask,{'mask_region':'inner_contour','contour_width_px':3})
        self.assertTrue(contour[2,8]);self.assertFalse(contour[10,10]);self.assertFalse(contour[0,0])
        self.assertTrue(np.all(~contour|mask))


if __name__=='__main__':unittest.main()
