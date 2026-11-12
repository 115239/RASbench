"""Immediate robot-control, LiDAR, and IMU attack primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def direct_robot_control(target: Any, parameters: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    """Send a bounded simulation command through the scene's control adapter."""

    command = np.asarray(parameters["command"], dtype=np.float32)
    duration_s = float(parameters.get("duration_s", 0.1))
    if command.ndim != 1 or not np.isfinite(command).all():
        raise ValueError("robot command must be a finite one-dimensional vector")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    target.apply_direct_control(command, duration_s=duration_s, context=context)
    return target


def immediate_gripper_release(
    target: Any, parameters: Mapping[str, Any], context: Mapping[str, Any]
) -> Any:
    """Open the M20 gripper immediately without waiting for another VLA query."""

    opening_m = float(parameters.get("opening_m", 0.05))
    hold_s = float(parameters.get("hold_s", 1.0))
    if not 0.0 <= opening_m <= 0.05:
        raise ValueError("opening_m must be within [0.0, 0.05]")
    if hold_s <= 0:
        raise ValueError("hold_s must be positive")
    target.release_gripper_now(opening_m=opening_m, hold_s=hold_s, context=context)
    return target


@dataclass
class ObjectPointDropout:
    """Remove LiDAR returns by stable object ID, with an optional AABB fallback."""

    target_prim_path: str | None = None
    world_aabb_min: np.ndarray | None = None
    world_aabb_max: np.ndarray | None = None
    sensor_aabb_min: np.ndarray | None = None
    sensor_aabb_max: np.ndarray | None = None
    include_descendants: bool = True

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> "ObjectPointDropout":
        bounds = parameters.get("world_aabb")
        sensor_bounds = parameters.get("sensor_aabb")
        return cls(
            target_prim_path=parameters.get("target_prim_path"),
            world_aabb_min=None if bounds is None else np.asarray(bounds["min"], dtype=np.float64),
            world_aabb_max=None if bounds is None else np.asarray(bounds["max"], dtype=np.float64),
            sensor_aabb_min=None if sensor_bounds is None else np.asarray(sensor_bounds["min"], dtype=np.float64),
            sensor_aabb_max=None if sensor_bounds is None else np.asarray(sensor_bounds["max"], dtype=np.float64),
            include_descendants=bool(parameters.get("include_descendants", True)),
        )

    def apply(self, frame: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        points = np.asarray(frame["points"])
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"LiDAR points must be Nx3, got {points.shape}")
        remove = np.zeros(len(points), dtype=bool)
        method = None

        object_ids = frame.get("object_ids")
        stable_id_map = frame.get("stable_id_map")
        if self.target_prim_path and object_ids is not None and stable_id_map is not None:
            target = self.target_prim_path.rstrip("/")
            matching_ids = {
                stable_id for stable_id, path in stable_id_map.items()
                if path == target or (self.include_descendants and str(path).startswith(target + "/"))
            }
            # An occluded or out-of-view valid target has no returns in this frame.
            # Preserve every point; static scene binding validation handles bad paths.
            remove = np.fromiter((value in matching_ids for value in object_ids), bool, count=len(points))
            method = "stable_object_id"
        elif self.world_aabb_min is not None and self.world_aabb_max is not None:
            transform = np.asarray(frame.get("sensor_to_world", np.eye(4)), dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("sensor_to_world must be 4x4")
            homogeneous = np.column_stack([points, np.ones(len(points))])
            world = (transform @ homogeneous.T).T[:, :3]
            remove = np.all((world >= self.world_aabb_min) & (world <= self.world_aabb_max), axis=1)
            method = "world_aabb"
        elif self.sensor_aabb_min is not None and self.sensor_aabb_max is not None:
            if self.sensor_aabb_min.shape != (3,) or self.sensor_aabb_max.shape != (3,):
                raise ValueError("sensor_aabb min/max must be xyz vectors")
            if np.any(self.sensor_aabb_min > self.sensor_aabb_max):
                raise ValueError("sensor_aabb min must not exceed max")
            remove = np.all(
                (points >= self.sensor_aabb_min) & (points <= self.sensor_aabb_max), axis=1
            )
            method = "sensor_aabb"
        else:
            raise ValueError("LiDAR dropout needs target_prim_path, world_aabb, or sensor_aabb")

        attacked: dict[str, Any] = {}
        for key, value in frame.items():
            if key in {"stable_id_map", "sensor_to_world"}:
                attacked[key] = value
                continue
            array = np.asarray(value)
            attacked[key] = array[~remove].copy() if array.ndim > 0 and len(array) == len(points) else value
        audit = {
            "method": method,
            "target_prim_path": self.target_prim_path,
            "clean_points": int(len(points)),
            "removed_points": int(remove.sum()),
            "attacked_points": int((~remove).sum()),
            "removed_indices": np.flatnonzero(remove).tolist(),
        }
        return attacked, audit


def install_lidar_object_dropout(target: Any, parameters: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    attack = ObjectPointDropout.from_parameters(parameters)
    target.install_lidar_filter(attack, context=context)
    return target


@dataclass
class IMUTransform:
    attack: str
    parameters: Mapping[str, Any]

    def apply(self, sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
        result = {key: np.asarray(value, dtype=np.float64).copy() for key, value in sample.items()}
        if self.attack == "bias":
            for key in ("angular_velocity", "linear_acceleration", "projected_gravity"):
                if key in result:
                    result[key] += np.asarray(self.parameters.get(f"{key}_bias", [0, 0, 0]))
        elif self.attack == "scale":
            for key in ("angular_velocity", "linear_acceleration", "projected_gravity"):
                if key in result:
                    result[key] *= np.asarray(self.parameters.get(f"{key}_scale", [1, 1, 1]))
        elif self.attack == "axis_flip":
            signs = np.asarray(self.parameters.get("signs", [-1, 1, 1]))
            for key in ("angular_velocity", "linear_acceleration", "projected_gravity"):
                if key in result:
                    result[key] *= signs
        else:
            raise ValueError(f"unsupported IMU attack: {self.attack!r}")
        return result


def install_imu_transform(target: Any, parameters: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    transform = IMUTransform(str(parameters["attack"]), parameters)
    target.install_imu_transform(transform, context=context)
    return target


def install_visual_material_override(
    target: Any,
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Any:
    """Install a bounded USD material override through the scene adapter."""

    target.install_visual_material_override(parameters, context=context)
    return target


IMMEDIATE_HANDLERS = {
    "direct_robot_control": direct_robot_control,
    "immediate_gripper_release": immediate_gripper_release,
    "install_lidar_object_dropout": install_lidar_object_dropout,
    "install_imu_transform": install_imu_transform,
    "install_visual_material_override": install_visual_material_override,
}
