"""World-axis-aligned bounding-box predicates.

The immutable geometry evidence for an entity is a local-space AABB.  At
evaluation time it is transformed through the entity pose by transforming all
eight local corners and re-fitting an axis-aligned box in world coordinates.
This is intentionally conservative for rotated geometry, but is deterministic
and needs no mesh access in the Judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .geometry import geometry_aabb, quaternion_apply_wxyz
from .trace import (
    PredicateValidationError,
    constant_or_pose_series,
    finite_array,
    finite_float,
)


def _aabb_corners(local_aabb_m: np.ndarray) -> np.ndarray:
    low, high = local_aabb_m[:3], local_aabb_m[3:]
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def world_aabb_from_local(local_aabb_m: object, pose: object, name: str = "local_aabb_m") -> np.ndarray:
    """Transform a static local AABB through one pose or a pose series.

    ``pose`` may be ``[7]`` or ``[T, 7]`` in ``xyz + wxyz quaternion`` order.
    The returned bounds have shape ``[T, 6]`` (including ``T == 1`` for a
    constant pose), in ``[min_x, min_y, min_z, max_x, max_y, max_z]`` order.
    """

    local = finite_array(local_aabb_m, name, ndim=1, shape_tail=(6,))
    if np.any(local[:3] > local[3:]):
        raise PredicateValidationError(f"{name} has inverted bounds")

    raw_pose = finite_array(pose, "pose")
    if raw_pose.shape == (7,):
        poses = constant_or_pose_series(raw_pose, "pose", 1)
    elif raw_pose.ndim == 2 and raw_pose.shape[1:] == (7,) and len(raw_pose) > 0:
        poses = constant_or_pose_series(raw_pose, "pose", len(raw_pose))
    else:
        raise PredicateValidationError("pose must have shape [7] or [T, 7] with T >= 1")

    corners = _aabb_corners(local)
    world_corners = poses[:, None, :3] + quaternion_apply_wxyz(
        poses[:, None, 3:7], corners[None, :, :]
    )
    return np.concatenate(
        (np.min(world_corners, axis=1), np.max(world_corners, axis=1)), axis=1
    )


def _aabb_series(value: object, name: str) -> np.ndarray:
    bounds = finite_array(value, name)
    if bounds.shape == (6,):
        bounds = bounds[None, :]
    elif bounds.ndim != 2 or bounds.shape[1:] != (6,) or len(bounds) == 0:
        raise PredicateValidationError(f"{name} must have shape [6] or [T, 6] with T >= 1")
    if np.any(bounds[:, :3] > bounds[:, 3:]):
        raise PredicateValidationError(f"{name} has inverted bounds")
    return bounds


def _broadcast_aabbs(first: object, second: object) -> tuple[np.ndarray, np.ndarray]:
    left = _aabb_series(first, "first_aabb")
    right = _aabb_series(second, "second_aabb")
    length = max(len(left), len(right))
    if len(left) not in {1, length} or len(right) not in {1, length}:
        raise PredicateValidationError("AABB sample counts must match or be constant")
    return np.broadcast_to(left, (length, 6)), np.broadcast_to(right, (length, 6))


def aabb_distance(first_aabb: object, second_aabb: object) -> np.ndarray:
    """Return per-sample shortest Euclidean distance between world AABBs.

    Intersecting or merely touching closed AABBs have distance zero.
    """

    first, second = _broadcast_aabbs(first_aabb, second_aabb)
    per_axis_gap = np.maximum(
        np.maximum(first[:, :3] - second[:, 3:], second[:, :3] - first[:, 3:]),
        0.0,
    )
    return np.linalg.norm(per_axis_gap, axis=1)


def aabb_intersects(first_aabb: object, second_aabb: object, tolerance_m: object = 0.0) -> np.ndarray:
    """Return whether closed AABBs overlap, permitting an optional gap tolerance."""

    tolerance = finite_float(tolerance_m, "tolerance_m", nonnegative=True)
    return aabb_distance(first_aabb, second_aabb) <= tolerance


def aabb_fully_inside(
    inner_aabb: object, outer_aabb: object, tolerance_m: object = 0.0
) -> np.ndarray:
    """Return whether each complete closed ``inner_aabb`` lies in ``outer_aabb``.

    A positive tolerance expands the outer bounds uniformly in all axes; it
    never changes the object's frozen local AABB.
    """

    tolerance = finite_float(tolerance_m, "tolerance_m", nonnegative=True)
    inner, outer = _broadcast_aabbs(inner_aabb, outer_aabb)
    return np.all(
        (inner[:, :3] >= outer[:, :3] - tolerance)
        & (inner[:, 3:] <= outer[:, 3:] + tolerance),
        axis=1,
    )


def _geometry_local_aabb(inputs: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        geometry = inputs[key]
    except KeyError as error:
        raise PredicateValidationError(f"inputs is missing {key}") from error
    if not isinstance(geometry, Mapping):
        raise PredicateValidationError(f"{key} must be a geometry mapping")
    return geometry_aabb(geometry, key)


def _pose_input(inputs: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    try:
        return constant_or_pose_series(inputs[key], key, length)
    except KeyError as error:
        raise PredicateValidationError(f"inputs is missing {key}") from error


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredicateValidationError(f"{name} must be a mapping")
    return value


def _time_length(time_s: object) -> int:
    samples = finite_array(time_s, "time_s", ndim=1)
    if len(samples) < 1:
        raise PredicateValidationError("aabb evaluation requires at least one time sample")
    return len(samples)


def evaluate_aabb_distance_below(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Test world-AABB separation of two posed entities at every sample."""

    inputs = _mapping(inputs, "inputs")
    params = _mapping(params, "params")
    length = _time_length(time_s)
    subject_pose = _pose_input(inputs, "subject_pose", length)
    reference_pose = _pose_input(inputs, "reference_pose", length)
    subject_aabb = world_aabb_from_local(
        _geometry_local_aabb(inputs, "subject_geometry"), subject_pose, "subject_geometry.local_aabb_m"
    )
    reference_aabb = world_aabb_from_local(
        _geometry_local_aabb(inputs, "reference_geometry"), reference_pose, "reference_geometry.local_aabb_m"
    )
    threshold = finite_float(params.get("max_distance_m"), "max_distance_m", nonnegative=True)
    distance = aabb_distance(subject_aabb, reference_aabb)
    return distance <= threshold, {
        "max_distance_m": threshold,
        "minimum_aabb_distance_m": float(np.min(distance)),
        "maximum_aabb_distance_m": float(np.max(distance)),
        "matched_sample_count": int(np.count_nonzero(distance <= threshold)),
    }


def evaluate_object_aabb_in_aabb(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compare an object's world AABB against a container/region world AABB.

    ``mode='intersects'`` accepts closed-box contact/overlap (or a gap no
    greater than ``tolerance_m``).  ``mode='fully_inside'`` requires the
    object's whole transformed AABB to be inside the reference AABB, with the
    tolerance expanding only the reference bounds.
    """

    inputs = _mapping(inputs, "inputs")
    params = _mapping(params, "params")
    length = _time_length(time_s)
    object_pose = _pose_input(inputs, "object_pose", length)
    region_pose = _pose_input(inputs, "region_pose", length)
    object_aabb = world_aabb_from_local(
        _geometry_local_aabb(inputs, "object_geometry"), object_pose, "object_geometry.local_aabb_m"
    )
    region_aabb = world_aabb_from_local(
        _geometry_local_aabb(inputs, "region_geometry"), region_pose, "region_geometry.local_aabb_m"
    )
    mode = params.get("mode")
    if mode not in {"intersects", "fully_inside"}:
        raise PredicateValidationError("mode must be 'intersects' or 'fully_inside'")
    tolerance = finite_float(params.get("tolerance_m"), "tolerance_m", nonnegative=True)
    distance = aabb_distance(object_aabb, region_aabb)
    matched = (
        aabb_intersects(object_aabb, region_aabb, tolerance)
        if mode == "intersects"
        else aabb_fully_inside(object_aabb, region_aabb, tolerance)
    )
    return matched, {
        "mode": mode,
        "tolerance_m": tolerance,
        "minimum_aabb_distance_m": float(np.min(distance)),
        "maximum_aabb_distance_m": float(np.max(distance)),
        "matched_sample_count": int(np.count_nonzero(matched)),
    }
