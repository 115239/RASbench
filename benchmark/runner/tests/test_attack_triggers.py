from __future__ import annotations

import unittest

from attack_tools.triggers import TriggerEvaluator


class AttackTriggerTests(unittest.TestCase):
    def test_world_aabb_xy_metric_uses_boundary_distance(self):
        evaluator = TriggerEvaluator()
        condition = {
            "type": "distance_to_object",
            "object_id": "black_shelf",
            "metric": "world_aabb_xy",
            "axes": "xy",
            "op": "lte",
            "distance_m": 4.0,
        }
        snapshot = {
            "distances": {"black_shelf": 5.33},
            "boundary_distances": {"black_shelf": 1.65},
        }

        self.assertTrue(evaluator.matches(condition, snapshot))
        self.assertTrue(evaluator.observable(condition, snapshot))

    def test_world_aabb_xy_metric_is_unknown_without_boundary_evidence(self):
        evaluator = TriggerEvaluator()
        condition = {
            "type": "distance_to_object",
            "object_id": "black_shelf",
            "metric": "world_aabb_xy",
            "op": "lte",
            "distance_m": 4.0,
        }

        self.assertIsNone(evaluator.all_match_observed([condition], {}))

    def test_distance_to_any_object_uses_nearest_registered_target(self):
        evaluator = TriggerEvaluator()
        snapshot = {
            "robot": {"position_xyz": [0.0, 0.0, 0.0]},
            "objects": {
                "counter_a": {"position_xyz": [4.0, 0.0, 0.0]},
                "counter_b": {"position_xyz": [0.0, 2.5, 8.0]},
            },
        }
        condition = {
            "type": "distance_to_any_object",
            "object_ids": ["counter_a", "counter_b"],
            "axes": "xy",
            "op": "lte",
            "distance_m": 3.0,
        }
        self.assertTrue(evaluator.matches(condition, snapshot))

    def test_distance_to_any_object_requires_nonempty_ids(self):
        evaluator = TriggerEvaluator()
        with self.assertRaisesRegex(ValueError, "non-empty object_ids"):
            evaluator.matches(
                {
                    "type": "distance_to_any_object",
                    "object_ids": [],
                    "distance_m": 3.0,
                },
                {},
            )


if __name__ == "__main__":
    unittest.main()
