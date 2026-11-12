"""Offline motion, pose, path, and smoothness predicates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .geometry import quaternion_apply_wxyz
from .trace import (
    PredicateValidationError,
    constant_or_vector_series,
    finite_array,
    finite_float,
    pose_series,
    scalar_series,
    vector_series,
)


def evaluate_distance_below(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    position = vector_series(inputs["position"], "position", len(time_s))
    target = constant_or_vector_series(inputs["target_position"], "target_position", len(time_s))
    limit = finite_float(params["max_distance_m"], "max_distance_m", positive=True)
    distance = np.linalg.norm(position - target, axis=1)
    return distance <= limit, {"minimum_distance_m": float(np.min(distance, initial=np.inf)), "max_distance_m": limit}


def evaluate_stationary_near(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    position = vector_series(inputs["position"], "position", len(time_s))
    target = constant_or_vector_series(inputs["target_position"], "target_position", len(time_s))
    velocity = vector_series(inputs["linear_velocity"], "linear_velocity", len(time_s))
    distance_limit = finite_float(params["max_distance_m"], "max_distance_m", positive=True)
    speed_limit = finite_float(params["max_speed_mps"], "max_speed_mps", nonnegative=True)
    distance = np.linalg.norm(position - target, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    return (distance <= distance_limit) & (speed <= speed_limit), {"minimum_distance_m": float(np.min(distance, initial=np.inf)), "minimum_speed_mps": float(np.min(speed, initial=np.inf))}


def evaluate_directional_force(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    force = vector_series(inputs["force"], "force", len(time_s))
    direction = finite_array(params["direction"], "direction", ndim=1, shape_tail=(3,))
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise PredicateValidationError("direction must have nonzero magnitude")
    if not isinstance(params["direction_frame"], str) or not params["direction_frame"].strip():
        raise PredicateValidationError("direction_frame must be a non-empty string")
    threshold = finite_float(params["min_force_n"], "min_force_n", positive=True)
    effective = force @ (direction / norm)
    return effective >= threshold, {"maximum_effective_force_n": float(np.max(effective, initial=-np.inf)), "direction_frame": params["direction_frame"]}


def evaluate_scalar_target(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    value = scalar_series(inputs["value"], "value", len(time_s))
    operator = params["operator"]
    if operator not in {">=", "<="}:
        raise PredicateValidationError("operator must be '>=' or '<='")
    target = finite_float(params["target_value"], "target_value")
    if not isinstance(params["target_unit"], str) or not params["target_unit"].strip():
        raise PredicateValidationError("target_unit must be a non-empty string")
    matched = value >= target if operator == ">=" else value <= target
    return matched, {"observed_min": float(np.min(value, initial=np.inf)), "observed_max": float(np.max(value, initial=-np.inf)), "operator": operator, "target_value": target, "target_unit": params["target_unit"]}


def evaluate_object_stationary(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    velocity = vector_series(inputs["linear_velocity"], "linear_velocity", len(time_s))
    speed_limit = finite_float(params["max_speed_mps"], "max_speed_mps", nonnegative=True)
    speed = np.linalg.norm(velocity, axis=1)
    matched = speed <= speed_limit
    if "angular_velocity" in inputs:
        angular = vector_series(inputs["angular_velocity"], "angular_velocity", len(time_s))
        angular_limit = finite_float(params["max_angular_speed_radps"], "max_angular_speed_radps", nonnegative=True)
        matched &= np.linalg.norm(angular, axis=1) <= angular_limit
    return matched, {"maximum_speed_mps": float(np.max(speed, initial=0.0)), "max_speed_mps": speed_limit}


def evaluate_object_upright(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    local_up = finite_array(params["local_up_axis"], "local_up_axis", ndim=1, shape_tail=(3,))
    world_up = finite_array(params["world_up_axis"], "world_up_axis", ndim=1, shape_tail=(3,))
    if np.linalg.norm(local_up) <= 1e-12 or np.linalg.norm(world_up) <= 1e-12:
        raise PredicateValidationError("up axes must be nonzero")
    local_up /= np.linalg.norm(local_up)
    world_up /= np.linalg.norm(world_up)
    actual = quaternion_apply_wxyz(pose[:, 3:7], np.broadcast_to(local_up, (len(time_s), 3)))
    cosine = np.clip(actual @ world_up, -1.0, 1.0)
    tilt = np.rad2deg(np.arccos(cosine))
    limit = finite_float(params["max_tilt_deg"], "max_tilt_deg", nonnegative=True)
    return tilt <= limit, {"maximum_tilt_deg": float(np.max(tilt, initial=0.0)), "max_tilt_deg": limit}


def evaluate_object_yaw_aligned(
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    time_s: np.ndarray,
):
    """Check world-Z yaw against one trusted target heading.

    The pose quaternion is ``wxyz``.  When ``allow_pi_symmetry`` is true,
    headings separated by pi radians are treated as the same physical
    orientation, which is useful for rectangular parts without a distinguished
    front edge.
    """

    pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    target_yaw = finite_float(params["target_yaw_rad"], "target_yaw_rad")
    tolerance_deg = finite_float(
        params["yaw_tolerance_deg"],
        "yaw_tolerance_deg",
        nonnegative=True,
    )
    if tolerance_deg > 180.0:
        raise PredicateValidationError("yaw_tolerance_deg must be <= 180")
    allow_pi_symmetry = params["allow_pi_symmetry"]
    if not isinstance(allow_pi_symmetry, (bool, np.bool_)):
        raise PredicateValidationError("allow_pi_symmetry must be a boolean")

    w, x, y, z = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    def wrapped_delta(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    error = np.abs(wrapped_delta(yaw - target_yaw))
    if bool(allow_pi_symmetry):
        error = np.minimum(
            error,
            np.abs(wrapped_delta(yaw - target_yaw - np.pi)),
        )
    error_deg = np.rad2deg(error)
    return error_deg <= tolerance_deg, {
        "minimum_yaw_error_deg": float(np.min(error_deg, initial=np.inf)),
        "maximum_yaw_error_deg": float(np.max(error_deg, initial=0.0)),
        "target_yaw_rad": target_yaw,
        "yaw_tolerance_deg": tolerance_deg,
        "allow_pi_symmetry": bool(allow_pi_symmetry),
    }


def evaluate_object_axis_aligned(
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    time_s: np.ndarray,
):
    """Check one trusted object-local axis against a trusted world axis.

    The sampled pose quaternion is ``wxyz``.  Both axes are normalized before
    comparison, so only their directions matter.  Unlike ``object_upright``,
    this embodied-only Predicate does not assume that either direction is
    world-up; it can therefore express an inverted object exactly by aligning
    local ``+Z`` with world ``-Z``.
    """

    pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    local_axis = finite_array(
        params["local_axis"],
        "local_axis",
        ndim=1,
        shape_tail=(3,),
    )
    target_world_axis = finite_array(
        params["target_world_axis"],
        "target_world_axis",
        ndim=1,
        shape_tail=(3,),
    )
    local_norm = float(np.linalg.norm(local_axis))
    target_norm = float(np.linalg.norm(target_world_axis))
    if local_norm <= 1e-12 or target_norm <= 1e-12:
        raise PredicateValidationError("alignment axes must be nonzero")
    local_axis = local_axis / local_norm
    target_world_axis = target_world_axis / target_norm

    max_angle_deg = finite_float(
        params["max_angle_deg"],
        "max_angle_deg",
        nonnegative=True,
    )
    if max_angle_deg > 180.0:
        raise PredicateValidationError("max_angle_deg must be <= 180")

    actual_world_axis = quaternion_apply_wxyz(
        pose[:, 3:7],
        np.broadcast_to(local_axis, (len(time_s), 3)),
    )
    cosine = np.clip(actual_world_axis @ target_world_axis, -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cosine))
    return angle_deg <= max_angle_deg, {
        "minimum_axis_error_deg": float(np.min(angle_deg, initial=np.inf)),
        "maximum_axis_error_deg": float(np.max(angle_deg, initial=0.0)),
        "local_axis": [float(value) for value in local_axis],
        "target_world_axis": [float(value) for value in target_world_axis],
        "max_angle_deg": max_angle_deg,
    }


def evaluate_path_length_below(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    position = vector_series(inputs["position"], "position", len(time_s))
    limit = finite_float(params["max_path_length_m"], "max_path_length_m", positive=True)
    increments = np.linalg.norm(np.diff(position, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    return cumulative <= limit, {"path_length_m": float(cumulative[-1]) if len(cumulative) else 0.0, "max_path_length_m": limit}


def _integrated_squared_jerk(position: np.ndarray, time_s: np.ndarray) -> float:
    if len(time_s) < 4:
        raise PredicateValidationError("integrated squared jerk requires at least four samples")
    velocity = np.gradient(position, time_s, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, time_s, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, time_s, axis=0, edge_order=1)
    squared = np.sum(jerk * jerk, axis=1)
    return float(np.sum(0.5 * (squared[1:] + squared[:-1]) * np.diff(time_s)))


def _sparc(position: np.ndarray, time_s: np.ndarray) -> float:
    if len(time_s) < 4:
        raise PredicateValidationError("SPARC requires at least four samples")
    sample_t = np.linspace(time_s[0], time_s[-1], len(time_s))
    uniform = np.column_stack([np.interp(sample_t, time_s, position[:, axis]) for axis in range(3)])
    speed = np.linalg.norm(np.gradient(uniform, sample_t, axis=0), axis=1)
    spectrum = np.abs(np.fft.rfft(speed))
    if np.max(spectrum, initial=0.0) <= 1e-12:
        return 0.0
    amplitude = spectrum / np.max(spectrum)
    frequency = np.fft.rfftfreq(len(speed), d=float(sample_t[1] - sample_t[0]))
    keep = amplitude >= 0.05
    if np.count_nonzero(keep) < 2:
        return 0.0
    x = frequency[keep] / max(float(frequency[keep][-1]), 1e-12)
    return -float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(amplitude[keep]) ** 2)))


def evaluate_trajectory_smooth(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    position = vector_series(inputs["position"], "position", len(time_s))
    metric = params["metric"]
    if metric == "integrated_squared_jerk":
        value = _integrated_squared_jerk(position, time_s)
        threshold = finite_float(params["max_integrated_squared_jerk"], "max_integrated_squared_jerk", nonnegative=True)
        matched = value <= threshold
    elif metric == "sparc":
        value = _sparc(position, time_s)
        threshold = finite_float(params["min_sparc"], "min_sparc")
        matched = value >= threshold
    else:
        raise PredicateValidationError("metric must be integrated_squared_jerk or sparc")
    return np.full(len(time_s), matched, dtype=np.bool_), {"metric": metric, "value": value, "threshold": threshold}
