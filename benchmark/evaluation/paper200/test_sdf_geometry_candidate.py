"""Regression checks for safe-only native collision bounds, not physical proof."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

CANDIDATE=Path(__file__).resolve().parents[1]/'runner_judge_v2/candidates/sdf_collision_bounds/candidate.py'
spec=importlib.util.spec_from_file_location('safety_tools.sdf_geometry_candidate',CANDIDATE)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


class NativeBoundTests(unittest.TestCase):
    def setUp(self):
        self.t=module.SafetyGeometryTracker.__new__(module.SafetyGeometryTracker)
        self.t._static_materialized_cache={};self.t._posed_materialized_cache={}
        self.robot=module._GeometryRecord('/robot','/robot','Cube',{'size':.1},np.eye(4),None,None,True)
        self.sdf=module._GeometryRecord('/sdf/mesh','/sdf','Mesh',{'convex_parts':[],
            'local_bounds_xyz':(np.full(3,-.1),np.full(3,.1)),'collision_bound_source':'physx_property_query'},
            np.eye(4),None,None,False,'sdf_exact_surface_unavailable_native_bound_only')

    def state(self,x):return {'/robot':{'pose':[x,0.,0.,1.,0.,0.,0.]},'/sdf':{'pose':[0.,0.,0.,1.,0.,0.,0.]}}

    def test_distant_native_bound_certifies_only_lower_bound(self):
        result=self.t._measure_record_sets([self.robot],[self.sdf],self.state(.3),threshold=.1)
        self.assertTrue(result['complete']);self.assertGreaterEqual(result['distance_m'],.1)
        self.assertIn('lower_bound',result['distance_kind']);self.assertFalse(self.sdf.complete)

    def test_near_native_bound_cannot_claim_violation_or_safe(self):
        result=self.t._measure_record_sets([self.robot],[self.sdf],self.state(.12),threshold=.1)
        self.assertFalse(result['complete'])

    def test_absent_owner_pose_stays_unknown(self):
        result=self.t._measure_record_sets([self.robot],[self.sdf],{'/robot':self.state(3)['/robot']},threshold=.1)
        self.assertFalse(result['complete'])

    def test_visual_static_bounds_cannot_certify_missing_collider(self):
        invalid=module._GeometryRecord('/unproved','/unproved','Mesh',{},None,np.eye(4),
            (np.array([100.,100.]),np.array([101.,101.])),False,'unsupported')
        result=self.t._measure_record_sets([self.robot],[invalid],self.state(0),threshold=.1)
        self.assertFalse(result['complete'])

    def test_current_dynamic_bound_replaces_initial_fallback(self):
        state=self.state(0);state['/sdf']['pose'][0]=4.
        bound=self.t._materialize([self.sdf],state)[0]['bounds']
        np.testing.assert_allclose(bound,[[3.9,-.1],[4.1,.1]])


if __name__=='__main__':unittest.main()
