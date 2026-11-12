"""Typed registry for attack handlers; arbitrary import paths are forbidden."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
AttackHandler = Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any]

@dataclass(frozen=True)
class AttackRegistration:
    name: str
    handler: AttackHandler
    stages: frozenset[str]
    input_contract: str = "any"
    output_contract: str = "same_as_input"

class AttackRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, AttackRegistration] = {}
    def register(self, name: str, handler: AttackHandler, *, stages: Iterable[str],
                 input_contract: str = "any", output_contract: str = "same_as_input") -> None:
        normalized = str(name).strip()
        if not normalized or "." in normalized or "/" in normalized or normalized in self._entries:
            raise ValueError(f"invalid or duplicate attack handler: {name!r}")
        entry = AttackRegistration(normalized, handler, frozenset(stages), input_contract, output_contract)
        if not entry.stages:
            raise ValueError(f"{name}: at least one stage is required")
        self._entries[normalized] = entry
    def resolve(self, name: str, stage: str) -> AttackRegistration:
        try:
            entry = self._entries[name]
        except KeyError as error:
            raise KeyError(f"unregistered attack handler: {name}") from error
        if stage not in entry.stages:
            raise ValueError(f"{name!r} cannot run at {stage!r}; supported={sorted(entry.stages)}")
        return entry
    def handlers(self) -> dict[str, AttackHandler]:
        return {name: entry.handler for name, entry in self._entries.items()}
    def describe(self) -> list[dict[str, Any]]:
        return [{"name": e.name, "stages": sorted(e.stages), "input_contract": e.input_contract,
                 "output_contract": e.output_contract}
                for e in sorted(self._entries.values(), key=lambda item: item.name)]

def build_standard_registry(handlers: Mapping[str, AttackHandler]) -> AttackRegistry:
    metadata = {
        "paper200_roi_dropout": ({"sensor_lidar"}, "lidar_frame"),
        "paper200_world_action_offset": ({"response_action", "control_ee_target"}, "numeric_action"),
        "paper200_projected_patch": ({"request_image"}, "rgb_image"),
        "paper200_world_imu_bias": ({"sensor_imu"}, "imu_sample"),
        "picture_red_to_green": ({"request_image"}, "rgb_image"),
        "picture_region_patch": ({"request_image"}, "rgb_image"),
        "picture_bboard_target": ({"request_image"}, "rgb_image"),
        "state_offset": ({"request_state", "state_feedback"}, "numeric_state"),
        "state_set_values": ({"request_state", "state_feedback"}, "numeric_state"),
        "replace_prompt": ({"request_prompt"}, "text"),
        "replace_prompt_sequence": ({"request_prompt"}, "text"),
        "append_target_coordinates": ({"request_prompt"}, "text"),
        "scale_actions": ({"response_action", "control_base", "control_joint"}, "numeric_action"),
        "affine_actions": ({"response_action", "control_base", "control_joint", "control_gripper"}, "numeric_action"),
        "gripper_directional_scale": ({"response_action"}, "numeric_action"),
        "gripper_opening_command_offset": ({"response_action", "control_gripper"}, "numeric_action"),
        "base_velocity_toward_object": ({"response_action", "control_base"}, "numeric_action"),
        "world_frame_planar_velocity_offset": ({"response_action"}, "numeric_action"),
        "world_frame_tcp_translation_offset": ({"response_action"}, "numeric_action"),
        "translation_direction_offset": ({"response_action"}, "numeric_action"),
        "state_hold_trigger_arm_positions": ({"request_state", "state_feedback"}, "numeric_state"),
        "gripper_opening_feedback_offset": ({"request_state", "state_feedback"}, "numeric_state"),
        "joint_encoder_position_feedback": ({"sensor_joint_encoder"}, "joint_encoder_sample"),
        "lidar_object_dropout_frame": ({"sensor_lidar"}, "lidar_frame"),
        "lidar_phantom_add_frame": ({"sensor_lidar"}, "lidar_frame"),
        "lidar_point_drift_frame": ({"sensor_lidar"}, "lidar_frame"),
        "imu_sample_transform": ({"sensor_imu"}, "imu_sample"),
        "imu_oscillating_bias": ({"sensor_imu"}, "imu_sample"),
        "force_torque_set_value": ({"sensor_force_torque"}, "force_torque_sample"),
        "wheel_encoder_velocity_bias": ({"sensor_wheel_encoder"}, "wheel_encoder_sample"),
        "joint_target_offset": ({"control_joint"}, "joint_target_mapping"),
        "arm_joint_hold_measured": ({"control_joint"}, "joint_target_mapping"),
        "arm_joint_hold_trigger_command": ({"control_joint"}, "joint_target_mapping"),
        "direct_robot_control": ({"immediate_robot_control"}, "robot_control_adapter"),
        "immediate_gripper_release": ({"immediate_gripper"}, "gripper_control_adapter"),
        "install_lidar_object_dropout": ({"immediate_lidar"}, "lidar_control_adapter"),
        "install_imu_transform": ({"immediate_imu"}, "imu_control_adapter"),
        "install_visual_material_override": (
            {"immediate_visual_material"}, "visual_material_adapter"
        ),
    }
    registry = AttackRegistry()
    for name, handler in handlers.items():
        if name not in metadata:
            raise ValueError(f"missing registration metadata for {name}")
        stages, contract = metadata[name]
        registry.register(name, handler, stages=stages, input_contract=contract)
    return registry
