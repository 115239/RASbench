"""Regression tests for Q1-Q200 attack handlers added to the main Runner."""

from __future__ import annotations

import math

import numpy as np
import pytest

from m20_runtime.attack_tools.attack_registry import build_standard_registry
from m20_runtime.attack_tools.attack_orchestrator import AttackSpec
from m20_runtime.attack_tools.standard_attack_handlers import (
    STANDARD_HANDLERS,
    arm_joint_hold_trigger_command,
    base_velocity_toward_object,
    gripper_opening_command_offset,
    joint_encoder_position_feedback,
    replace_prompt_sequence,
    translation_direction_offset,
    world_frame_tcp_translation_offset,
)


EXPECTED_STAGES = {
    "replace_prompt_sequence": "request_prompt",
    "world_frame_tcp_translation_offset": "response_action",
    "joint_encoder_position_feedback": "sensor_joint_encoder",
    "base_velocity_toward_object": "control_base",
    "gripper_opening_command_offset": "response_action",
    "arm_joint_hold_trigger_command": "control_joint",
    "translation_direction_offset": "response_action",
}


def test_all_q200_handlers_are_registered_for_authored_stages() -> None:
    registry = build_standard_registry(STANDARD_HANDLERS)
    for name, stage in EXPECTED_STAGES.items():
        assert registry.resolve(name, stage).handler is STANDARD_HANDLERS[name]


def test_joint_encoder_stage_is_accepted_by_runtime_attack_spec() -> None:
    spec = AttackSpec.from_dict({
        "id": "joint_encoder_feedback",
        "mode": "intercept",
        "stage": "sensor_joint_encoder",
        "handler": "joint_encoder_position_feedback",
        "trigger": {"all": []},
        "parameters": {
            "joint_name": "fl_hipy_joint",
            "operation": "offset",
            "offset_rad": 0.08,
        },
    })
    assert spec.stage == "sensor_joint_encoder"
    assert spec.handler == "joint_encoder_position_feedback"


def test_replace_prompt_sequence_uses_attack_relative_time_and_holds_last() -> None:
    parameters = {
        "steps": [
            {"at_s": 0.0, "replacement": "round one"},
            {"at_s": 1.0, "replacement": "round two"},
        ],
        "after_sequence": "hold_last",
    }
    first = replace_prompt_sequence(
        "original",
        parameters,
        {"runtime": {"time_s": 10.5}, "snapshot": {"_attack_runtime": {"activated_s": 10.0}}},
    )
    second = replace_prompt_sequence(
        "original",
        parameters,
        {"runtime": {"time_s": 11.2}, "snapshot": {"_attack_runtime": {"activated_s": 10.0}}},
    )
    assert first == "round one"
    assert second == "round two"


def test_world_frame_tcp_offset_is_rotated_to_robot_frame() -> None:
    actions = np.zeros(10, dtype=np.float64)
    attacked = world_frame_tcp_translation_offset(
        actions,
        {"offset_world_xyz_m": [1.0, 0.0, 0.25], "translation_indices": [3, 4, 5]},
        {"snapshot": {"robot": {"rpy": [0.0, 0.0, math.pi / 2.0]}}},
    )
    np.testing.assert_allclose(attacked[3:6], [0.0, -1.0, 0.25], atol=1.0e-12)


def test_translation_direction_offset_extends_current_direction() -> None:
    actions = np.zeros(10, dtype=np.float64)
    actions[3] = 0.01
    attacked = translation_direction_offset(
        actions,
        {"offset_m": 0.03, "translation_indices": [3, 4, 5]},
        {},
    )
    np.testing.assert_allclose(attacked[3:6], [0.04, 0.0, 0.0])


def test_gripper_opening_offset_uses_calibrated_width() -> None:
    actions = np.zeros(10, dtype=np.float64)
    actions[9] = 0.8
    attacked = gripper_opening_command_offset(
        actions,
        {"index": 9, "opening_offset_m": 0.007, "calibrated_maximum_opening_m": 0.07},
        {},
    )
    assert attacked[9] == pytest.approx(0.7)


def test_base_velocity_points_toward_object_in_robot_frame() -> None:
    command = np.asarray([0.0, 0.0, 0.3], dtype=np.float64)
    attacked = base_velocity_toward_object(
        command,
        {"object_id": "target", "speed_mps": 0.2},
        {
            "snapshot": {
                "robot": {"position_xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, math.pi / 2.0]},
                "objects": {"target": {"position_xyz": [1.0, 0.0, 0.0]}},
            }
        },
    )
    np.testing.assert_allclose(attacked, [0.0, -0.2, 0.3], atol=1.0e-12)


def test_joint_encoder_offset_and_trigger_hold() -> None:
    sample = {"joint_names": ["FL_hip_joint", "FR_hip_joint"], "position_rad": [0.1, 0.2]}
    offset = joint_encoder_position_feedback(
        sample,
        {"joint_name": "FR_hip_joint", "operation": "offset", "offset_rad": 0.08},
        {},
    )
    np.testing.assert_allclose(offset["position_rad"], [0.1, 0.28])

    held = joint_encoder_position_feedback(
        sample,
        {"joint_name": "FR_hip_joint", "operation": "hold_trigger"},
        {"snapshot": {"locomotion": {"measured_joint_positions_rad": {"FR_hip_joint": -0.4}}}},
    )
    np.testing.assert_allclose(held["position_rad"], [0.1, -0.4])


def test_arm_hold_uses_trigger_command_and_preserves_other_targets() -> None:
    names = [f"joint{index}" for index in range(1, 7)]
    targets = {"positions": {**{name: 0.0 for name in names}, "joint7": 9.0}, "mode": "position"}
    commanded = {name: float(index) for index, name in enumerate(names, start=1)}
    attacked = arm_joint_hold_trigger_command(
        targets,
        {"joint_names": names},
        {"snapshot": {"arm": {"commanded_joint_positions_rad": commanded}}},
    )
    assert attacked["mode"] == "position"
    assert attacked["positions"]["joint7"] == 9.0
    assert {name: attacked["positions"][name] for name in names} == commanded
