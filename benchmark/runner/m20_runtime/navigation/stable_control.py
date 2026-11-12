"""Simulator-independent stability rules for the M20+Piper controller.

The timing contract comes from the verified M20+Piper expert collector: M20
locomotion runs at 50 Hz over 200 Hz physics.  MOVE commands use the learned
locomotion policy; a zero command enters a separate stationary stance that
brakes the wheels and smoothly restores the nominal leg pose.  The helpers live
outside the Isaac entry point so safety-critical state transitions can be
tested on the host without importing Kit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


LOW_LEVEL_CONTROL_PROFILE = "m20-piper-stop-stance-200hz-50hz-v3"
SOURCE_CONTROLLER_ENTRYPOINT = "m20pro_piper/vla/isaac/collect_expert_episode.py"
SOURCE_CONTROLLER_SHA256 = "fe86817d6441575729cc7b28de0ecfc9e0ae64b3ef92db22a0e7981fce5530d8"

# The locomotion ONNX is a velocity-tracking policy and has no verified static
# equilibrium with the combined Piper payload.  STOP therefore has a separate
# actuator contract: zero wheel velocity plus a smoothly blended nominal leg
# stance.  Timeout and explicit callers also use this fail-closed wheel brake.
EXPLICIT_WHEEL_BRAKE_PHASES = frozenset(
    {
        "vla_timeout_brake",
    }
)
ZERO_COMMAND_ATOL = 1.0e-6
STATIONARY_STANCE_BLEND_SECONDS = 0.75
STATIONARY_LEG_STIFFNESS = 200.0
STATIONARY_LEG_DAMPING = 20.0


def is_zero_base_command(
    command: Sequence[float] | np.ndarray,
    *,
    atol: float = ZERO_COMMAND_ATOL,
) -> bool:
    """Return whether a finite three-axis base command requests a full stop."""

    value = np.asarray(command, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"base command must be a finite 3-vector: {value}")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"base command tolerance must be finite and non-negative: {atol}")
    return bool(np.max(np.abs(value)) <= atol)


def should_brake_wheels(
    phase: str,
    command: Sequence[float] | np.ndarray,
    *,
    explicit_brake: bool = False,
) -> bool:
    """Resolve the stationary/fail-closed motor-level wheel brake."""

    stationary = is_zero_base_command(command)

    return bool(
        explicit_brake
        or stationary
        or phase in EXPLICIT_WHEEL_BRAKE_PHASES
    )


def smooth_stationary_leg_targets(
    start_targets: Sequence[float] | np.ndarray,
    goal_targets: Sequence[float] | np.ndarray,
    progress: float,
) -> np.ndarray:
    """Blend twelve leg targets with zero endpoint velocity/acceleration."""

    start = np.asarray(start_targets, dtype=np.float64)
    goal = np.asarray(goal_targets, dtype=np.float64)
    for label, value in (("start_targets", start), ("goal_targets", goal)):
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError(f"{label} must be a finite 12-vector: {value}")
    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be finite and within [0, 1]: {progress}")
    blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    return (start + blend * (goal - start)).astype(np.float32)


def executed_m20_action_history(
    raw_action: Sequence[float] | np.ndarray,
    *,
    brake_wheels: bool,
) -> np.ndarray:
    """Return the normalized action history corresponding to clean targets."""

    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError(f"raw M20 action must be a finite 16-vector: {action}")
    executed = action.copy()
    if brake_wheels:
        executed[12:] = 0.0
    return executed


def smooth_piper_targets(
    start_targets: Sequence[float] | np.ndarray,
    goal_targets: Sequence[float] | np.ndarray,
    progress: float,
) -> np.ndarray:
    """Interpolate seven Piper targets with zero endpoint velocity/acceleration."""

    start = np.asarray(start_targets, dtype=np.float64)
    goal = np.asarray(goal_targets, dtype=np.float64)
    for label, value in (("start_targets", start), ("goal_targets", goal)):
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{label} must be a finite 7-vector: {value}")
    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be finite and within [0, 1]: {progress}")
    blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    return (start + blend * (goal - start)).astype(np.float32)


def accumulate_piper_joint_targets(
    previous_targets: Sequence[float] | np.ndarray | None,
    measured_arm_positions: Sequence[float] | np.ndarray,
    delta_joint: Sequence[float] | np.ndarray,
    lower_limits: Sequence[float] | np.ndarray,
    upper_limits: Sequence[float] | np.ndarray,
    gripper_opening: float,
) -> np.ndarray:
    """Accumulate one IK delta on the commanded Piper target, not measurement.

    Re-latching every zero delta to measured joints turns gravity sag into a
    moving setpoint.  A persistent commanded target keeps the high-PD Piper
    drive restoring toward the last accepted command while still allowing each
    new VLA delta to advance the target.
    """

    measured = np.asarray(measured_arm_positions, dtype=np.float64)
    delta = np.asarray(delta_joint, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    for label, value in (
        ("measured_arm_positions", measured),
        ("delta_joint", delta),
        ("lower_limits", lower),
        ("upper_limits", upper),
    ):
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError(f"{label} must be a finite 6-vector: {value}")
    if np.any(lower > upper):
        raise ValueError("Piper lower limits must not exceed upper limits")
    if not np.isfinite(gripper_opening):
        raise ValueError(f"gripper opening must be finite: {gripper_opening}")

    if previous_targets is None:
        base = measured
    else:
        previous = np.asarray(previous_targets, dtype=np.float64)
        if previous.shape != (7,) or not np.isfinite(previous).all():
            raise ValueError(f"previous Piper target must be a finite 7-vector: {previous}")
        base = previous[:6]

    targets = np.empty(7, dtype=np.float32)
    targets[:6] = np.clip(base + delta, lower, upper)
    targets[6] = float(np.clip(gripper_opening, 0.0, 0.05))
    return targets
