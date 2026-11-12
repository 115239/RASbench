"""Point-cloud local navigation adapter that emits M20 base commands."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LidarPlannerConfig:
    maximum_speed_mps: float = 0.20
    maximum_yaw_rate_rps: float = 0.45
    stop_distance_m: float = 0.45
    slow_distance_m: float = 1.10
    side_clearance_m: float = 0.42
    goal_tolerance_m: float = 0.12


class M20LidarLocalPlanner:
    """Goal seeking with point-cloud collision sectors and deterministic avoidance."""

    def __init__(self, config: LidarPlannerConfig | None = None):
        self.config = config or LidarPlannerConfig()
        self.last_turn_sign = 1.0

    def command(
        self,
        cloud_sensor: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
        goal_xy: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        cloud = np.asarray(cloud_sensor, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError(f"cloud must be Nx3, got {cloud.shape}")
        delta = np.asarray(goal_xy, dtype=np.float64) - np.asarray(robot_xy, dtype=np.float64)
        distance_to_goal = float(np.linalg.norm(delta))
        heading_error = _wrap(math.atan2(delta[1], delta[0]) - float(robot_yaw))
        if distance_to_goal <= self.config.goal_tolerance_m:
            return np.zeros(3, np.float32), {"state": "goal", "goal_distance_m": distance_to_goal}

        finite = cloud[np.all(np.isfinite(cloud), axis=1)]
        finite = finite[(finite[:, 2] > -0.65) & (finite[:, 2] < 0.75)]
        front = finite[(finite[:, 0] > 0) & (np.abs(finite[:, 1]) <= self.config.side_clearance_m)]
        front_distance = float(front[:, 0].min()) if len(front) else float("inf")
        left = finite[(finite[:, 0] > 0) & (finite[:, 1] > 0)]
        right = finite[(finite[:, 0] > 0) & (finite[:, 1] < 0)]
        left_clearance = float(np.linalg.norm(left[:, :2], axis=1).min()) if len(left) else float("inf")
        right_clearance = float(np.linalg.norm(right[:, :2], axis=1).min()) if len(right) else float("inf")

        if front_distance <= self.config.stop_distance_m:
            if abs(left_clearance - right_clearance) > 0.05:
                self.last_turn_sign = 1.0 if left_clearance > right_clearance else -1.0
            speed = 0.0
            yaw_rate = self.last_turn_sign * self.config.maximum_yaw_rate_rps
            state = "avoid"
        else:
            clearance_scale = np.clip(
                (front_distance - self.config.stop_distance_m)
                / (self.config.slow_distance_m - self.config.stop_distance_m),
                0.2,
                1.0,
            )
            heading_scale = max(0.15, 1.0 - abs(heading_error) / math.pi)
            speed = self.config.maximum_speed_mps * float(clearance_scale) * heading_scale
            yaw_rate = float(np.clip(1.2 * heading_error, -self.config.maximum_yaw_rate_rps, self.config.maximum_yaw_rate_rps))
            state = "track"
        command = np.asarray([speed, 0.0, yaw_rate], dtype=np.float32)
        return command, {
            "state": state,
            "goal_distance_m": distance_to_goal,
            "heading_error_rad": heading_error,
            "front_distance_m": front_distance,
            "left_clearance_m": left_clearance,
            "right_clearance_m": right_clearance,
            "cloud_points": int(len(finite)),
        }
