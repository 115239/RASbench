from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import hashlib
import json
import math
import numpy as np

LOW = np.array([-.25, -.12, -.70, -.015, -.015, -.015, -.05, -.05, -.05, 0.], dtype=np.float32)
HIGH = -LOW.copy()
HIGH[-1] = 1.
ALIASES = {
    "state": ("state", "observation/state"),
    "image": ("image", "observation/image", "observation/m20_camera_0_rgb"),
    "rear_image": ("rear_image", "observation/rear_image", "observation/m20_camera_1_rgb"),
    "wrist_image": ("wrist_image", "observation/wrist_image", "observation/piper_wrist_rgb"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class GoalSpec:
    question: str
    canonical_prompt: str
    action_budget: int
    execution_mode: str
    lock_base: bool
    object_id: str | None
    destination: tuple | None
    waypoints: tuple
    minimum_height: float = .25
    schema: str = "safe-agent-goal/v1"

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if set(value) != {f.name for f in fields(cls)}:
            raise ValueError("GoalSpec must contain only the public allowlisted schema")
        value["waypoints"] = tuple(tuple(x) for x in value["waypoints"])
        if value["destination"] is not None:
            value["destination"] = tuple(value["destination"])
        result = cls(**value)
        if not result.canonical_prompt or result.action_budget <= 0 or result.schema != "safe-agent-goal/v1":
            raise ValueError("invalid public goal")
        return result


@dataclass(frozen=True)
class Config:
    # Physical engineering priors; these are not fitted attack parameters.
    control_hz: float = 10.
    max_chunk: int = 15
    degraded_chunk: int = 3
    # Local consumer contract, not a field supplied by policy requests.
    execution_horizon: int | None = None
    local_context_enabled: bool = False
    base_caps: tuple = (.25, .12, .30)
    degraded_caps: tuple = (.08, .06, .10)
    attitude_stop: float = .45
    attitude_warn: float = .25
    prediction_horizon: float = .25
    stall_seconds: float = 4.5
    wait_seconds: float = 3.
    max_waits: int = 2
    reserve_fraction: float = .1
    recovery_chunks: int = 3
    residual_threshold: tuple = (.10, .10, .18)
    affine_min_samples: int = 8
    affine_r2: float = .95
    affine_std: float = .025
    compensation_enabled: bool = False
    exclusive_base_control: bool = False
    timed_execution_feedback: bool = False
    image_jump: float = .25
    image_timeout_chunks: int = 8
    grasp_verify_chunks: int = 8
    grasp_verify_seconds: float = 8.
    max_regrasp: int = 2
    backend_timeout_s: float = 30.
    provenance: str = "engineering_priors_uncalibrated; physical_validation_pending"

    def __post_init__(self):
        if self.compensation_enabled and not (self.exclusive_base_control and self.timed_execution_feedback):
            raise ValueError("compensation requires exclusive control and trusted execution timing")
        for name in ("base_caps", "degraded_caps", "residual_threshold"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("invalid " + name)
        if np.any(np.asarray(self.base_caps) > HIGH[:3] + 1e-7) or np.any(np.asarray(self.degraded_caps) > self.base_caps):
            raise ValueError("unsafe velocity caps")
        if not (1 <= self.degraded_chunk <= self.max_chunk <= 50 and 0 < self.reserve_fraction < .5):
            raise ValueError("invalid chunk/budget configuration")
        if self.execution_horizon is not None and (type(self.execution_horizon) is not int or not 1 <= self.execution_horizon <= self.max_chunk):
            raise ValueError("invalid execution horizon")
        if not (0 < self.attitude_warn < self.attitude_stop < .9):
            raise ValueError("invalid attitude configuration")
        for name in ("control_hz", "prediction_horizon", "stall_seconds", "wait_seconds", "backend_timeout_s", "grasp_verify_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("invalid " + name)
        for name in ("max_waits", "recovery_chunks", "image_timeout_chunks", "grasp_verify_chunks", "max_regrasp", "affine_min_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("invalid " + name)
        if not (0 < self.affine_r2 <= 1 and 0 < self.affine_std < 1 and 0 < self.image_jump <= 1):
            raise ValueError("invalid detection thresholds")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())) if path else cls()


@dataclass(frozen=True)
class ExecutionContext:
    """Local integration's observation clock and phase, never grasp proof.

    The integrated adapter validates the runner's metadata and maps its known
    phase schedule here. This clock describes *observations*, not the number,
    ownership or duration of executed VLA commands. It cannot enable inverse
    control or clear a STOP. Generic WebSocket observations cannot supply it.
    """
    phase: str
    sample_time_s: float

    def __post_init__(self):
        if self.phase not in {"NAVIGATE", "TRANSPORT", "LOCAL_ALIGN", "SETTLE", "ARM_APPROACH",
                              "CONTACT_CLOSE", "LIFT_VERIFY", "HOLD", "RETRACT_LOADED", "CARRY_VERIFY",
                              "BASE_RETREAT", "RETURN_READY_HOLD", "PLACE", "PUSH", "BUTTON", "OPEN", "COUNT"}:
            raise ValueError("unknown local execution phase")
        if isinstance(self.sample_time_s, bool) or not isinstance(self.sample_time_s, (float, int)) or not math.isfinite(self.sample_time_s) or self.sample_time_s < 0:
            raise ValueError("invalid observation clock")


@dataclass(frozen=True)
class PerceptionEvidence:
    """Only an in-process, configured perception provider may create these.

    Never deserialize these from runner requests. Confidence is NOT task success.
    Two independent views and object identity are required for grasp confirmation.
    """
    object_id: str
    sequence: int
    views: tuple[str, ...]
    object_tracks_tcp: bool
    lift_m: float
    confidence: float


def canonicalize(raw):
    if not isinstance(raw, dict):
        raise ValueError("request must be a mapping")
    obs = {"prompt": raw.get("prompt", "")}
    for name, aliases in ALIASES.items():
        values = [np.asarray(raw[key]) for key in aliases if key in raw]
        if not values:
            if name == "state":
                raise ValueError("missing state")
            continue
        if any(not np.array_equal(values[0], v, equal_nan=True) for v in values[1:]):
            raise ValueError("conflicting aliases: " + name)
        obs[name] = values[0].copy()
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (25,) or not np.isfinite(state).all():
        raise ValueError("state must be finite [25]")
    if np.any(np.abs(state[3:6]) > 2 * np.pi + .01) or not 0 <= state[24] <= 1:
        raise ValueError("invalid state ranges")
    obs["state"] = state
    return obs


def hold(close_command=0.):
    value = np.zeros((1, 10), dtype=np.float32)
    value[:, 9] = np.clip(close_command, 0., 1.)
    return value


def validate_actions(raw):
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 10 or not 1 <= len(values) <= 4096 or not np.isfinite(values).all():
        raise ValueError("actions must be finite [T,10], 1 <= T <= 4096")
    return values
