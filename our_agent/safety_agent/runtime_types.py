"""Allowlisted sensor and actuator interfaces, independent of simulator and evaluator."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = 'on'
    recovery: bool = True
    final_guard: bool = True
    visual_only: bool = False
    max_recoveries_per_phase: int = 2
    recovery_fraction: float = .20
    confirm_samples: int = 2
    resume_samples: int = 3
    recovery_attempt_s: float = 3.
    sensor_max_age_s: float = .5
    visual_max_age_s: float = 1.
    base_speed_mps: float = .5
    yaw_rate_rps: float = 1.
    base_acceleration: float = .6
    yaw_acceleration: float = 1.5
    arm_speed_rps: float = .8
    clearance_m: float = .1
    footprint_radius_m: float = .40
    attitude_stop_rad: float = .8
    prediction_s: float = .25

    def __post_init__(self):
        if self.mode not in {'off', 'shadow', 'on'}:
            raise ValueError('mode must be off/shadow/on')
        if not 0 < self.recovery_fraction <= .2:
            raise ValueError('recovery fraction must be in (0,.2]')
        for k, v in vars(self).items():
            if isinstance(v, float) and (not math.isfinite(v) or v <= 0):
                raise ValueError('invalid ' + k)
        for k in ('max_recoveries_per_phase', 'confirm_samples', 'resume_samples'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError('invalid ' + k)
        if self.max_recoveries_per_phase > 2:
            raise ValueError('at most two recoveries per phase')


@dataclass(frozen=True)
class ObservationFrame:
    episode_id: str
    sequence: int
    time_s: float
    phase: str
    state: np.ndarray
    images: dict = field(default_factory=dict)
    lidar: np.ndarray | None = None
    lidar_time_s: float | None = None
    imu: np.ndarray | None = None
    encoder: np.ndarray | None = None
    calibration: dict = field(default_factory=dict)
    wheel_velocity: np.ndarray | None = None
    wheel_radius_m: float | None = None
    proprioception_time_s: float | None = None

    def __post_init__(self):
        a = np.asarray(self.state)
        if a.shape != (25,) or not np.isfinite(a).all():
            raise ValueError('finite 25-D observed state required')
        if not self.episode_id or type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('invalid observation identity')
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('invalid observation time')
        if self.lidar is not None:
            p = np.asarray(self.lidar)
            if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
                raise ValueError('LiDAR must contain finite sensor-frame xyz only')
            if self.lidar_time_s is None or not math.isfinite(self.lidar_time_s) or self.lidar_time_s > self.time_s:
                raise ValueError('invalid LiDAR timestamp')


@dataclass(frozen=True)
class VisualEvidence:
    episode_id: str
    sequence: int
    time_s: float
    target_visible: bool = False
    target_consistent: bool = False
    tracks_tcp: bool = False
    lift_m: float | None = None
    supported: bool = False
    placement_authorized: bool = False
    views: tuple[str, ...] = ()
    target_boxes: dict = field(default_factory=dict)
    uncertainty: tuple[str, ...] = ()

    def grasp_confirmed(self):
        return (self.target_visible and self.target_consistent and self.tracks_tcp
                and self.lift_m is not None and math.isfinite(self.lift_m) and self.lift_m >= .02
                and len(set(self.views)) >= 2)


@dataclass(frozen=True)
class ExecutionReceipt:
    episode_id: str
    command_id: int
    observation_sequence: int
    source: str
    start_s: float
    end_s: float
    applied_base: tuple[float, float, float]
    applied_joint_positions: tuple = ()
    applied_joint_velocities: tuple = ()

    def __post_init__(self):
        if not self.source or self.command_id < 0 or self.observation_sequence < 0:
            raise ValueError('invalid receipt identity')
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError('invalid executed interval')
        if len(self.applied_base) != 3 or not np.isfinite(self.applied_base).all():
            raise ValueError('invalid applied base')
        for values in (self.applied_joint_positions, self.applied_joint_velocities):
            if any(v is not None and not math.isfinite(v) for v in values):
                raise ValueError('joint receipt uses finite numbers or None for uncontrolled axes')


class TemporalGate:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.sequence = -1
        self.time_s = -1.
        self.receipt_id = -1
        self.receipt_end = -1.

    def observe(self, frame):
        if frame.episode_id != self.episode_id or frame.sequence <= self.sequence or frame.time_s <= self.time_s:
            raise ValueError('stale or foreign observation')
        self.sequence, self.time_s = frame.sequence, frame.time_s

    def receipt(self, receipt):
        if (receipt.episode_id != self.episode_id or receipt.command_id <= self.receipt_id
                or receipt.observation_sequence > self.sequence
                or receipt.start_s < self.receipt_end - 1e-9
                or receipt.end_s < self.time_s - 1e-9):
            raise ValueError('stale, overlapping or foreign receipt')
        self.receipt_id, self.receipt_end = receipt.command_id, receipt.end_s
