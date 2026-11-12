"""Stable data contracts shared by scenes, sensors, and attack handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class RobotControlCommand:
    base_velocity: np.ndarray
    arm_action: np.ndarray | None = None
    timestamp_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.base_velocity = np.asarray(self.base_velocity, dtype=np.float32)
        if self.base_velocity.shape != (3,) or not np.isfinite(self.base_velocity).all():
            raise ValueError("base_velocity must be finite [vx, vy, yaw_rate]")
        if self.arm_action is not None:
            self.arm_action = np.asarray(self.arm_action, dtype=np.float32)
            if self.arm_action.ndim != 1 or not np.isfinite(self.arm_action).all():
                raise ValueError("arm_action must be a finite vector")


@dataclass
class IMUSample:
    timestamp_s: float
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray | None = None
    projected_gravity: np.ndarray | None = None
    orientation_wxyz: np.ndarray | None = None
    frame_id: str = "imu"

    def as_mapping(self) -> dict[str, np.ndarray]:
        result = {"angular_velocity": np.asarray(self.angular_velocity, dtype=np.float64)}
        for name in ("linear_acceleration", "projected_gravity", "orientation_wxyz"):
            value = getattr(self, name)
            if value is not None:
                result[name] = np.asarray(value, dtype=np.float64)
        for name, value in result.items():
            expected = (4,) if name == "orientation_wxyz" else (3,)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {expected}")
        return result


@dataclass
class LidarFrame:
    timestamp_s: float
    points: np.ndarray
    object_ids: np.ndarray | None = None
    stable_id_map: Mapping[int, str] | None = None
    intensity: np.ndarray | None = None
    distance: np.ndarray | None = None
    azimuth: np.ndarray | None = None
    elevation: np.ndarray | None = None
    sensor_to_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    frame_id: str = "lidar"

    def as_mapping(self) -> dict[str, Any]:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("LiDAR points must be a finite Nx3 array")
        count = len(points)
        result: dict[str, Any] = {
            "timestamp_s": float(self.timestamp_s),
            "points": points,
            "sensor_to_world": np.asarray(self.sensor_to_world, dtype=np.float64),
            "frame_id": self.frame_id,
        }
        if result["sensor_to_world"].shape != (4, 4):
            raise ValueError("sensor_to_world must be 4x4")
        for name in ("object_ids", "intensity", "distance", "azimuth", "elevation"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if len(array) != count:
                raise ValueError(f"{name} length {len(array)} does not match point count {count}")
            result[name] = array
        if self.stable_id_map is not None:
            result["stable_id_map"] = dict(self.stable_id_map)
        return result
