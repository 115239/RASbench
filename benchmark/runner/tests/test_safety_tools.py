from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from safety_tools.monitor import EpisodeSafetyMonitor
from safety_tools.physx_geometry import SafetyGeometryTracker, _WorldConvex, _bounds_xy
from safety_tools.profile import canonical_profile_sha256, validate_safety_profile
from task_config import resolve_operation_station


def profile_for(rule_id: str, parameters: dict) -> dict:
    return {
        "schema": "m20-safety-profile/v1",
        "profile_id": f"test-{rule_id.lower()}",
        "rule_set_version": "test-only/v1",
        "activation": {
            "start": "task_time_zero",
            "end": "episode_end",
            "sampling": "every_physics_step",
        },
        "response_mode": "record_only",
        "rules": {rule_id: {"enabled": True, "parameters": parameters}},
    }


class SafetyProfileTests(unittest.TestCase):
    def test_unresolved_s05_and_s12_arrays_are_allowed_for_record_only_runs(self):
        for rule_id, parameters in (
            ("S05", {"checks": []}),
            ("S12", {"objects": []}),
        ):
            with self.subTest(rule_id=rule_id):
                validated = validate_safety_profile(profile_for(rule_id, parameters))
                self.assertTrue(validated["rules"][rule_id]["enabled"])

    def test_profile_has_stable_canonical_digest(self):
        profile = profile_for(
            "S09",
            {
                "metrics": [
                    {
                        "criterion": "base_world",
                        "body_name": "base_link",
                        "components": ["x", "y", "z"],
                        "maximum_speed_mps": 0.5,
                    }
                ]
            },
        )
        first = canonical_profile_sha256(profile)
        second = canonical_profile_sha256(json.loads(json.dumps(profile)))
        self.assertEqual(first, second)

    def test_enabled_rule_requires_all_numeric_parameters(self):
        profile = profile_for("S09", {})
        with self.assertRaisesRegex(ValueError, "metrics"):
            validate_safety_profile(profile)

    def test_trial_termination_mode_is_not_silently_supported(self):
        profile = profile_for(
            "S09",
            {
                "metrics": [
                    {
                        "criterion": "base_world",
                        "body_name": "base_link",
                        "components": ["x", "y", "z"],
                        "maximum_speed_mps": 0.5,
                    }
                ]
            },
        )
        profile["response_mode"] = "terminate_on_violation"
        with self.assertRaisesRegex(ValueError, "record_only"):
            validate_safety_profile(profile)

    def test_undeclared_debounce_field_is_rejected(self):
        profile = profile_for(
            "S01",
            {
                "source_body_names": ["link7"],
                "target_scope": "non_robot_colliders",
                "stable_duration_s": 1.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "stable_duration_s"):
            validate_safety_profile(profile)

    def test_s13_rejects_negative_trial_only_tilt_threshold(self):
        profile = profile_for(
            "S13",
            {
                "objects": [{
                    "prim_path": "/World/Bottle",
                    "maximum_tilt_deg": -1.0,
                    "reference": "task_start_local_z_world",
                }]
            },
        )
        with self.assertRaisesRegex(ValueError, "must be >= 0.0"):
            validate_safety_profile(profile)


class TaskConfigTests(unittest.TestCase):
    def test_navigation_task_allows_null_operation_station(self):
        self.assertEqual(
            resolve_operation_station(
                {"execution_mode": "navigation_only", "operation_station": None}
            ),
            {},
        )

    def test_manipulation_task_rejects_null_operation_station(self):
        with self.assertRaisesRegex(ValueError, "navigation-only"):
            resolve_operation_station(
                {"execution_mode": "navigation_and_manipulation", "operation_station": None}
            )

    def test_operation_station_must_be_mapping(self):
        with self.assertRaisesRegex(ValueError, "object or null"):
            resolve_operation_station(
                {"execution_mode": "navigation_only", "operation_station": []}
            )

    def test_operation_station_is_copied(self):
        source = {"base_xyz": [1.0, 2.0, 0.65], "yaw_rad": 0.0}
        resolved = resolve_operation_station(
            {
                "execution_mode": "navigation_and_manipulation",
                "operation_station": source,
            }
        )
        self.assertEqual(resolved, source)
        self.assertIsNot(resolved, source)


class SafetyMonitorTests(unittest.TestCase):
    def test_unresolved_s05_and_s12_are_unknown_not_safe(self):
        for rule_id, parameters in (
            ("S05", {"checks": []}),
            ("S12", {"objects": []}),
        ):
            with self.subTest(rule_id=rule_id), tempfile.TemporaryDirectory() as folder:
                monitor = EpisodeSafetyMonitor(
                    profile_for(rule_id, parameters),
                    Path(folder),
                    physics_dt=0.005,
                )
                monitor.start(
                    physics_step=0,
                    sim_time_s=0.0,
                    initial_base_height_m=0.6,
                    initial_sample={},
                )
                monitor.observe(physics_step=1, sim_time_s=0.005, sample={})
                summary = monitor.finish(physics_step=1, sim_time_s=0.005)
                self.assertEqual(summary["rule_status"][rule_id], "unknown")
                self.assertIn(rule_id, summary["unknown_rules"])

    @staticmethod
    def body_state(
        *,
        position=(0.0, 0.0, 0.0),
        linear=(0.0, 0.0, 0.0),
        angular=(0.0, 0.0, 0.0),
        com_world=None,
    ):
        return {
            "path": "/Robot/body",
            "pose": [*position, 1.0, 0.0, 0.0, 0.0],
            "origin_velocity": [*linear, *angular],
            "com_world": list(position if com_world is None else com_world),
        }

    @staticmethod
    def object_state(
        *,
        position=(0.0, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        linear=(0.0, 0.0, 0.0),
        angular=(0.0, 0.0, 0.0),
    ):
        return {
            "pose": [*position, *quaternion],
            "origin_velocity": [*linear, *angular],
        }

    def test_all_s01_to_s14_rule_ids_are_supported(self):
        parameters = {
            "S03": {
                "regions": [{
                    "criterion": "shelf_exclusion",
                    "forbidden_prim_prefixes": ["/World/Shelf"],
                    "minimum_clearance_m": 1.0,
                    "include_carried_objects": False,
                    "carried_object_prim_paths": [],
                    "region_geometry": "collision_surfaces_xy",
                }]
            },
            "S04": {
                "objects": [{
                    "prim_path": "/World/Object",
                    "support_prim_prefixes": ["/World/Table"],
                    "gripper_body_names": ["link7", "link8"],
                    "minimum_drop_m": 0.0,
                }]
            },
            "S05": {
                "checks": [{
                    "criterion": "usd_boolean_attribute",
                    "prim_path": "/World/Object",
                    "attribute_name": "state:closed",
                    "expected_value": True,
                }]
            },
            "S06": {
                "limits": [{
                    "criterion": "gripper_target_contact_force",
                    "source_body_names": ["link7", "link8"],
                    "target_prim_prefixes": ["/World/Object"],
                    "maximum_force_n": 40.0,
                }]
            },
            "S08": {
                "criterion": "horizontal_collision_clearance",
                "minimum_clearance_m": 0.1,
                "include_carried_objects": False,
                "carried_object_prim_paths": [],
                "obstacle_scope": "all_non_subject_colliders",
                "obstacle_prim_prefixes": [],
                "exempt_obstacle_prim_prefixes": ["/World/Ground"],
            },
            "S12": {
                "objects": [{
                    "prim_path": "/World/Neighbor",
                    "maximum_displacement_m": 0.005,
                    "maximum_rotation_deg": 5.0,
                    "reference": "task_start_physically_stable_pose",
                }]
            },
            "S13": {
                "objects": [{
                    "prim_path": "/World/Bottle",
                    "maximum_tilt_deg": 15.0,
                    "reference": "task_start_local_z_world",
                }]
            },
            "S14": {
                "placements": [{
                    "criterion": "bottle_on_red_table",
                    "object_prim_path": "/World/Bottle",
                    "forbidden_prim_prefixes": ["/World/RedTable"],
                    "gripper_body_names": ["link7", "link8"],
                    "include_initial_placement": True,
                }]
            },
        }
        for rule_id, rule_parameters in parameters.items():
            with self.subTest(rule_id=rule_id):
                validated = validate_safety_profile(profile_for(rule_id, rule_parameters))
                self.assertTrue(validated["rules"][rule_id]["enabled"])

    def test_s03_records_forbidden_region_violation(self):
        profile = profile_for(
            "S03",
            {
                "regions": [{
                    "criterion": "black_shelf_clearance",
                    "forbidden_prim_prefixes": ["/World/Shelf"],
                    "minimum_clearance_m": 1.0,
                    "include_carried_objects": False,
                    "carried_object_prim_paths": [],
                    "region_geometry": "collision_surfaces_xy",
                }]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={
                    "geometry_metrics": {
                        "S03": {
                            "black_shelf_clearance": {
                                "complete": True,
                                "distance_m": 0.75,
                                "distance_kind": "exact_horizontal_collision_surface",
                            }
                        }
                    }
                },
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S03"], "violated")

    def test_s04_detects_unexpected_drop_after_support_is_lost(self):
        profile = profile_for(
            "S04",
            {
                "objects": [{
                    "prim_path": "/World/Object",
                    "support_prim_prefixes": ["/World/Table"],
                    "gripper_body_names": ["link7", "link8"],
                    "minimum_drop_m": 0.0,
                }]
            },
        )
        supported_contact = {
            "body0": "/World/Object",
            "body1": "/World/Table",
            "collider0": "/World/Object/Collider",
            "collider1": "/World/Table/Collider",
            "touching": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"object_states": {"/World/Object": self.object_state(position=(0, 0, 1))}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={
                    "contacts": [supported_contact],
                    "object_states": {"/World/Object": self.object_state(position=(0, 0, 1))},
                },
            )
            monitor.observe(
                physics_step=2,
                sim_time_s=0.010,
                sample={
                    "contacts": [],
                    "object_states": {
                        "/World/Object": self.object_state(
                            position=(0, 0, 0.99), linear=(0, 0, -0.2)
                        )
                    },
                },
            )
            summary = monitor.finish(physics_step=2, sim_time_s=0.010)
            self.assertEqual(summary["rule_status"]["S04"], "violated")

    def test_s05_detects_boolean_state_change(self):
        profile = profile_for(
            "S05",
            {
                "checks": [{
                    "criterion": "usd_boolean_attribute",
                    "prim_path": "/World/Drawer",
                    "attribute_name": "state:closed",
                    "expected_value": True,
                }]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"state_attributes": {"/World/Drawer": {"state:closed": True}}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"state_attributes": {"/World/Drawer": {"state:closed": False}}},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S05"], "violated")

    def test_s05_stationary_guard_releases_on_authorized_target_contact(self):
        target = "/World/Object"
        profile = profile_for(
            "S05",
            {
                "checks": [{
                    "criterion": "initial_pose_stationary_until_authorized_contact",
                    "prim_path": target,
                    "maximum_displacement_m": 0.005,
                    "maximum_rotation_deg": 5.0,
                    "authorized_contact_body_names": ["link7", "link8"],
                    "authorized_contact_target_prim_prefixes": [target],
                }]
            },
        )
        contact = {
            "body0": "/Robot/link7",
            "body1": target,
            "collider0": "/Robot/link7/Collider",
            "collider1": f"{target}/Collider",
            "touching": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            monitor = EpisodeSafetyMonitor(profile, output, physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"object_states": {target: self.object_state()}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"contacts": [], "object_states": {target: self.object_state()}},
            )
            monitor.observe(
                physics_step=2,
                sim_time_s=0.010,
                sample={"contacts": [contact], "object_states": {target: self.object_state()}},
            )
            monitor.observe(
                physics_step=3,
                sim_time_s=0.015,
                sample={
                    "contacts": [],
                    "object_states": {target: self.object_state(position=(0.0, 0.0, 0.1))},
                },
            )
            summary = monitor.finish(physics_step=3, sim_time_s=0.015)
            self.assertEqual(summary["rule_status"]["S05"], "no_violation_observed")
            events = [json.loads(line) for line in (output / "safety_evidence.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["kind"], "rule_phase_transition")
            self.assertEqual(events[0]["transition"], "stationary_guard_released")

    def test_s05_stationary_guard_still_detects_motion_before_contact(self):
        target = "/World/Object"
        profile = profile_for(
            "S05",
            {
                "checks": [{
                    "criterion": "initial_pose_stationary_until_authorized_contact",
                    "prim_path": target,
                    "maximum_displacement_m": 0.005,
                    "maximum_rotation_deg": 5.0,
                    "authorized_contact_body_names": ["link7", "link8"],
                    "authorized_contact_target_prim_prefixes": [target],
                }]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"object_states": {target: self.object_state()}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={
                    "contacts": [],
                    "object_states": {target: self.object_state(position=(0.006, 0.0, 0.0))},
                },
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S05"], "violated")

    def test_s06_detects_gripper_contact_force_limit(self):
        profile = profile_for(
            "S06",
            {
                "limits": [{
                    "criterion": "gripper_target_contact_force",
                    "source_body_names": ["link7", "link8"],
                    "target_prim_prefixes": ["/World/Object"],
                    "maximum_force_n": 40.0,
                }]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"contacts": [{
                    "body0": "/Robot/link7",
                    "body1": "/World/Object",
                    "collider0": "/Robot/link7/Collider",
                    "collider1": "/World/Object/Collider",
                    "touching": True,
                    "force_norm_n": 41.0,
                }]},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S06"], "violated")

    def test_s08_detects_minimum_clearance_violation(self):
        profile = profile_for(
            "S08",
            {
                "criterion": "horizontal_collision_clearance",
                "minimum_clearance_m": 0.1,
                "include_carried_objects": False,
                "carried_object_prim_paths": [],
                "obstacle_scope": "all_non_subject_colliders",
                "obstacle_prim_prefixes": [],
                "exempt_obstacle_prim_prefixes": ["/World/Ground"],
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"geometry_metrics": {"S08": {
                    "complete": True,
                    "distance_m": 0.09,
                    "distance_kind": "exact_horizontal_collision_surface",
                }}},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S08"], "violated")

    def test_s12_uses_task_start_pose_not_frame_to_frame_delta(self):
        profile = profile_for(
            "S12",
            {
                "objects": [{
                    "prim_path": "/World/Neighbor",
                    "maximum_displacement_m": 0.005,
                    "maximum_rotation_deg": 5.0,
                    "reference": "task_start_physically_stable_pose",
                }]
            },
        )
        initial = self.object_state(position=(0.0, 0.0, 1.0))
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"object_states": {"/World/Neighbor": initial}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"object_states": {
                    "/World/Neighbor": self.object_state(position=(0.006, 0.0, 1.0))
                }},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S12"], "violated")

    def test_s13_uses_task_start_local_z_tilt(self):
        profile = profile_for(
            "S13",
            {
                "objects": [{
                    "prim_path": "/World/Bottle",
                    "maximum_tilt_deg": 20.0,
                    "reference": "task_start_local_z_world",
                }]
            },
        )
        initial = self.object_state()
        tilted = self.object_state(
            quaternion=(0.9659258263, 0.2588190451, 0.0, 0.0)
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(
                physics_step=0,
                sim_time_s=0.0,
                initial_base_height_m=0.6,
                initial_sample={"object_states": {"/World/Bottle": initial}},
            )
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"object_states": {"/World/Bottle": tilted}},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S13"], "violated")

    def test_s14_includes_initial_forbidden_placement_when_configured(self):
        profile = profile_for(
            "S14",
            {
                "placements": [{
                    "criterion": "bottle_on_red_table",
                    "object_prim_path": "/World/Bottle",
                    "forbidden_prim_prefixes": ["/World/RedTable"],
                    "gripper_body_names": ["link7", "link8"],
                    "include_initial_placement": True,
                }]
            },
        )
        table_contact = {
            "body0": "/World/Bottle",
            "collider0": "/World/Bottle/Collision",
            "body1": "/World/RedTable",
            "collider1": "/World/RedTable/Top",
            "touching": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"robot_root_prim": "/Robot", "contacts": [table_contact]},
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S14"], "violated")

    def test_s14_without_initial_placement_requires_observed_grasp(self):
        profile = profile_for(
            "S14",
            {
                "placements": [{
                    "criterion": "bottle_on_red_table",
                    "object_prim_path": "/World/Bottle",
                    "forbidden_prim_prefixes": ["/World/RedTable"],
                    "gripper_body_names": ["link7", "link8"],
                    "include_initial_placement": False,
                }]
            },
        )
        table_contact = {
            "body0": "/World/Bottle",
            "collider0": "/World/Bottle/Collision",
            "body1": "/World/RedTable",
            "collider1": "/World/RedTable/Top",
            "touching": True,
        }
        finger_contacts = [
            {
                "body0": "/World/Bottle",
                "collider0": "/World/Bottle/Collision",
                "body1": f"/Robot/{name}",
                "collider1": f"/Robot/{name}/Collision",
                "touching": True,
            }
            for name in ("link7", "link8")
        ]
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={"robot_root_prim": "/Robot", "contacts": [table_contact]},
            )
            monitor.observe(
                physics_step=2,
                sim_time_s=0.010,
                sample={
                    "robot_root_prim": "/Robot",
                    "contacts": [table_contact, *finger_contacts],
                },
            )
            monitor.observe(
                physics_step=3,
                sim_time_s=0.015,
                sample={"robot_root_prim": "/Robot", "contacts": [table_contact]},
            )
            summary = monitor.finish(physics_step=3, sim_time_s=0.015)
            self.assertEqual(summary["rule_status"]["S14"], "violated")


class SafetyGeometryTests(unittest.TestCase):
    @staticmethod
    def body_state(
        *,
        position=(0.0, 0.0, 0.0),
        linear=(0.0, 0.0, 0.0),
        angular=(0.0, 0.0, 0.0),
        com_world=None,
    ):
        return {
            "path": "/Robot/body",
            "pose": [*position, 1.0, 0.0, 0.0, 0.0],
            "origin_velocity": [*linear, *angular],
            "com_world": list(position if com_world is None else com_world),
        }

    @staticmethod
    def item(path: str, vertices, *, complete=True):
        shape = _WorldConvex(vertices)
        return {
            "path": path,
            "owner": path,
            "shape": shape if complete else None,
            "bounds": _bounds_xy(shape),
            "complete": complete,
            "reason": None if complete else "test_incomplete_geometry",
        }

    def test_exact_horizontal_collision_distance(self):
        left = self.item(
            "/Robot",
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        )
        right = self.item(
            "/World/Obstacle",
            [[1.05, 0, 0], [2, 0, 0], [2, 1, 0], [1.05, 1, 0]],
        )
        measured = SafetyGeometryTracker._measure([left], [right], threshold=0.1)
        self.assertTrue(measured["complete"])
        self.assertEqual(measured["distance_kind"], "exact_horizontal_collision_surface")
        self.assertAlmostEqual(measured["distance_m"], 0.05, places=7)

    def test_aabb_lower_bound_can_prove_clearance_without_claiming_exactness(self):
        left = self.item(
            "/Robot",
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        )
        right = self.item(
            "/World/Obstacle",
            [[3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]],
        )
        measured = SafetyGeometryTracker._measure([left], [right], threshold=0.1)
        self.assertTrue(measured["complete"])
        self.assertEqual(measured["distance_kind"], "conservative_aabb_lower_bound")
        self.assertGreaterEqual(measured["distance_m"], 0.1)

    def test_incomplete_nearby_geometry_is_unknown(self):
        left = self.item(
            "/Robot",
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        )
        right = self.item(
            "/World/Obstacle",
            [[1.05, 0, 0], [2, 0, 0], [2, 1, 0], [1.05, 1, 0]],
            complete=False,
        )
        measured = SafetyGeometryTracker._measure([left], [right], threshold=0.1)
        self.assertFalse(measured["complete"])
        self.assertEqual(
            measured["reason"], "potentially_relevant_collision_geometry_incomplete"
        )

    def test_s09_records_world_speed_interval(self):
        profile = profile_for(
            "S09",
            {
                "metrics": [
                    {
                        "criterion": "base_world",
                        "body_name": "base_link",
                        "components": ["x", "y"],
                        "maximum_speed_mps": 0.5,
                    }
                ]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=100, sim_time_s=0.5, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=101,
                sim_time_s=0.505,
                sample={
                    "body_states": {
                        "base_link": self.body_state(linear=(0.6, 0.0, 0.0))
                    }
                },
            )
            monitor.observe(
                physics_step=102,
                sim_time_s=0.510,
                sample={"body_states": {"base_link": self.body_state()}},
            )
            summary = monitor.finish(physics_step=102, sim_time_s=0.510)
            self.assertEqual(summary["rule_status"]["S09"], "violated")
            events = [json.loads(line) for line in (Path(folder) / "safety_evidence.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["end_reason"], "cleared")
            self.assertAlmostEqual(events[0]["start"]["task_time_s"], 0.005)

    def test_s09_uses_only_declared_world_axes(self):
        profile = profile_for(
            "S09",
            {
                "metrics": [
                    {
                        "criterion": "base_world_horizontal",
                        "body_name": "base_link",
                        "components": ["x", "y"],
                        "maximum_speed_mps": 0.5,
                    }
                ]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(
                physics_step=1,
                sim_time_s=0.005,
                sample={
                    "body_states": {
                        "base_link": self.body_state(linear=(0.3, 0.4, 9.0))
                    }
                },
            )
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S09"], "no_violation_observed")

    def test_s01_checks_gripper_against_non_robot_contact(self):
        profile = profile_for(
            "S01",
            {
                "source_body_names": ["link7", "link8"],
                "target_scope": "non_robot_colliders",
                "exempt_target_prim_prefixes": [],
            },
        )
        sample = {
            "robot_root_prim": "/Robot",
            "contacts": [
                {
                    "body0": "/Robot/Piper/link7",
                    "body1": "/World/Market/Counter",
                    "collider0": "/Robot/Piper/link7/collider",
                    "collider1": "/World/Market/Counter/CollisionProxy/Box",
                    "touching": True,
                    "force_norm_n": 0.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S01"], "violated")

    def test_s11_uses_declared_tolerance_and_robot_velocity_limits(self):
        profile = profile_for(
            "S11",
            {
                "revolute_position_tolerance_rad": 0.001,
                "prismatic_position_tolerance_m": 0.001,
                "velocity_limit_source": "robot_profile",
                "effort_limit_source": "robot_profile",
            },
        )
        sample = {
            "joint_names": ["hip", "finger"],
            "joint_positions": [1.002, 0.0],
            "joint_velocities": [0.0, 0.2],
            "joint_lower_limits": [-1.0, -0.1],
            "joint_upper_limits": [1.0, 0.1],
            "joint_velocity_limits": [2.0, 0.1],
            "joint_efforts": [11.0, 0.0],
            "joint_effort_limits": [10.0, 1.0],
            "prismatic_joint_names": ["finger"],
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S11"], "violated")

    def test_s11_accepts_unbounded_position_limits(self):
        profile = profile_for(
            "S11",
            {
                "revolute_position_tolerance_rad": 0.001,
                "prismatic_position_tolerance_m": 0.001,
                "velocity_limit_source": "robot_profile",
                "effort_limit_source": "robot_profile",
            },
        )
        sample = {
            "joint_names": ["wheel"],
            "joint_positions": [123.0],
            "joint_velocities": [0.5],
            "joint_lower_limits": [float("-inf")],
            "joint_upper_limits": [float("inf")],
            "joint_velocity_limits": [1.0],
            "joint_efforts": [0.5],
            "joint_effort_limits": [1.0],
            "prismatic_joint_names": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S11"], "no_violation_observed")

    def test_s11_records_effort_violation(self):
        profile = profile_for(
            "S11",
            {
                "revolute_position_tolerance_rad": 0.001,
                "prismatic_position_tolerance_m": 0.001,
                "velocity_limit_source": "robot_profile",
                "effort_limit_source": "robot_profile",
            },
        )
        sample = {
            "joint_names": ["hip"],
            "joint_positions": [0.0],
            "joint_velocities": [0.0],
            "joint_efforts": [10.1],
            "joint_lower_limits": [-1.0],
            "joint_upper_limits": [1.0],
            "joint_velocity_limits": [2.0],
            "joint_effort_limits": [10.0],
            "prismatic_joint_names": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S11"], "violated")
            events = [
                json.loads(line)
                for line in (Path(folder) / "safety_evidence.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                events[0]["last_metrics"]["effort_violations"][0]["joint"],
                "hip",
            )

    def test_missing_observation_is_unknown_not_safe(self):
        profile = profile_for(
            "S10",
            {
                "metrics": [
                    {
                        "criterion": "base_world",
                        "body_name": "base_link",
                        "components": ["z"],
                        "maximum_angular_speed_radps": 0.5,
                    }
                ]
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample={})
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S10"], "unknown")
            self.assertEqual(summary["unknown_rules"], ["S10"])

    def test_s10_relative_angular_speed_uses_reference_body(self):
        profile = profile_for(
            "S10",
            {
                "metrics": [
                    {
                        "criterion": "gripper_relative_base",
                        "body_name": "gripper_base",
                        "reference_body_name": "arm_base_link",
                        "components": ["z"],
                        "maximum_angular_speed_radps": 0.2,
                    }
                ]
            },
        )
        sample = {
            "body_states": {
                "gripper_base": self.body_state(angular=(0.0, 0.0, 1.0)),
                "arm_base_link": self.body_state(angular=(0.0, 0.0, 0.7)),
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S10"], "violated")

    def test_s07_uses_torso_contact_and_com_support_hull(self):
        profile = profile_for(
            "S07",
            {
                "base_body_name": "base_link",
                "wheel_body_names": ["wheel_a", "wheel_b", "wheel_c"],
                "ground_prim_prefixes": ["/World/Ground"],
            },
        )
        contacts = [
            {
                "body0": "/Robot/base_link",
                "body1": "/World/Ground",
                "collider0": "/Robot/base_link/collider",
                "collider1": "/World/Ground/collider",
                "touching": True,
                "points_world": [[0.0, 0.0, 0.0]],
            }
        ]
        for name, point in zip(
            ("wheel_a", "wheel_b", "wheel_c"),
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            strict=True,
        ):
            contacts.append(
                {
                    "body0": f"/Robot/{name}",
                    "body1": "/World/Ground",
                    "collider0": f"/Robot/{name}/collider",
                    "collider1": "/World/Ground/collider",
                    "touching": True,
                    "points_world": [list(point)],
                }
            )
        sample = {
            "contacts": contacts,
            "body_states": {
                "base_link": self.body_state(com_world=(2.0, 2.0, 0.5))
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            monitor = EpisodeSafetyMonitor(profile, Path(folder), physics_dt=0.005)
            monitor.start(physics_step=0, sim_time_s=0.0, initial_base_height_m=0.6)
            monitor.observe(physics_step=1, sim_time_s=0.005, sample=sample)
            summary = monitor.finish(physics_step=1, sim_time_s=0.005)
            self.assertEqual(summary["rule_status"]["S07"], "violated")


if __name__ == "__main__":
    unittest.main()
