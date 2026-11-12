"""Offline lift, containment, surface, and stack predicates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .contact import contact_present
from .geometry import (
    entity_centroid_world,
    geometry_aabb,
    geometry_footprint,
    geometry_planes,
    point_in_footprint,
    point_in_planes,
    world_point_from_local,
    world_point_to_local,
)
from .trace import (
    PredicateValidationError,
    constant_or_pose_series,
    contact_force_series,
    finite_array,
    finite_float,
    pose_series,
    vector_series,
)


def evaluate_object_picked_up(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    initial = finite_array(inputs["initial_object_pose"], "initial_object_pose", ndim=1, shape_tail=(7,))
    if np.linalg.norm(initial[3:7]) <= 1e-12:
        raise PredicateValidationError("initial_object_pose contains a zero quaternion")
    contact_threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
    height_threshold = finite_float(params["min_height_delta_m"], "min_height_delta_m", positive=True)
    grabbed, norms = contact_present(inputs["gripper_contact_force"], len(time_s), contact_threshold, "gripper_contact_force")
    height = pose[:, 2] - initial[2]
    matched = grabbed & (height >= height_threshold)
    return matched, {
        "maximum_height_delta_m": float(np.max(height, initial=-np.inf)),
        "maximum_gripper_contact_force_n": float(np.max(norms, initial=0.0)),
    }


def _optional_placement_constraints(matched: np.ndarray, inputs: Mapping[str, Any], params: Mapping[str, Any], length: int) -> np.ndarray:
    for key in ("require_detached", "require_stationary"):
        if not isinstance(params[key], (bool, np.bool_)):
            raise PredicateValidationError(f"{key} must be boolean")
    if params["require_detached"]:
        if "gripper_contact_force" not in inputs:
            raise PredicateValidationError("require_detached needs gripper_contact_force")
        threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
        contact, _ = contact_present(inputs["gripper_contact_force"], length, threshold, "gripper_contact_force")
        matched &= ~contact
    if params["require_stationary"]:
        if "linear_velocity" not in inputs:
            raise PredicateValidationError("require_stationary needs linear_velocity")
        velocity = vector_series(inputs["linear_velocity"], "linear_velocity", length)
        limit = finite_float(params["max_speed_mps"], "max_speed_mps", nonnegative=True)
        matched &= np.linalg.norm(velocity, axis=1) <= limit
    return matched


def evaluate_object_in_container(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    length = len(time_s)
    object_pose = pose_series(inputs["object_pose"], "object_pose", length)
    # Formal task_reference poses are immutable [7] constants while physical
    # container state is [T,7].  The predicate contract supports both sources.
    container_pose = constant_or_pose_series(
        inputs["container_pose"], "container_pose", length
    )
    centroid_world = entity_centroid_world(object_pose, inputs["object_geometry"], "object_geometry")
    centroid_local = world_point_to_local(container_pose, centroid_world)
    planes = geometry_planes(inputs["container_geometry"], "container_geometry", "hull_planes_open_top")
    tolerance = finite_float(params["containment_tolerance_m"], "containment_tolerance_m", nonnegative=True)
    matched = point_in_planes(centroid_local, planes, tolerance)
    matched = _optional_placement_constraints(matched, inputs, params, length)
    return matched, {"minimum_max_plane_distance_m": float(np.min(np.max(centroid_local @ planes[:, :3].T + planes[:, 3], axis=1), initial=np.inf))}


def _aabb_corners(aabb: np.ndarray) -> np.ndarray:
    low, high = aabb[:3], aabb[3:]
    return np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=np.float64)


def _object_bottom_in_reference(object_pose: np.ndarray, object_geometry: Mapping[str, Any], reference_pose: np.ndarray) -> np.ndarray:
    corners = _aabb_corners(geometry_aabb(object_geometry, "object_geometry"))
    result = []
    for pose, ref in zip(object_pose, reference_pose, strict=True):
        world = world_point_from_local(np.broadcast_to(pose, (8, 7)), corners)
        local = world_point_to_local(np.broadcast_to(ref, (8, 7)), world)
        result.append(np.min(local[:, 2]))
    return np.asarray(result)


def _upward_support(force_value: object, length: int, threshold: float, cosine: float) -> tuple[np.ndarray, np.ndarray]:
    forces = contact_force_series(force_value, "support_force", length)
    vectors = forces if forces.ndim == 2 else forces.reshape(length, -1, 3)
    if vectors.ndim == 2:
        vectors = vectors[:, None, :]
    norms = np.linalg.norm(vectors, axis=-1)
    upward = vectors[..., 2]
    valid = (upward >= threshold) & (upward >= cosine * norms)
    return np.any(valid, axis=1), np.max(upward, axis=1)


def evaluate_object_on_surface(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    length = len(time_s)
    object_pose = pose_series(inputs["object_pose"], "object_pose", length)
    surface_pose = pose_series(inputs["surface_pose"], "surface_pose", length)
    centroid_world = entity_centroid_world(object_pose, inputs["object_geometry"], "object_geometry")
    centroid_local = world_point_to_local(surface_pose, centroid_world)
    footprint = geometry_footprint(inputs["surface_geometry"], "surface_geometry")
    surface_top = geometry_aabb(inputs["surface_geometry"], "surface_geometry")[5]
    bottom = _object_bottom_in_reference(object_pose, inputs["object_geometry"], surface_pose)
    height_tolerance = finite_float(params["height_tolerance_m"], "height_tolerance_m", nonnegative=True)
    footprint_tolerance = finite_float(params["footprint_tolerance_m"], "footprint_tolerance_m", nonnegative=True)
    force_limit = finite_float(params["min_support_force_n"], "min_support_force_n", nonnegative=True)
    cosine = finite_float(params["min_upward_cosine"], "min_upward_cosine", nonnegative=True)
    if cosine > 1:
        raise PredicateValidationError("min_upward_cosine must not exceed 1")
    support, upward = _upward_support(inputs["support_force"], length, force_limit, cosine)
    matched = (
        point_in_footprint(centroid_local[:, :2], footprint, footprint_tolerance)
        & (np.abs(bottom - surface_top) <= height_tolerance)
        & support
    )
    matched = _optional_placement_constraints(matched, inputs, params, length)
    return matched, {"minimum_surface_gap_m": float(np.min(np.abs(bottom - surface_top), initial=np.inf)), "maximum_upward_force_n": float(np.max(upward, initial=0.0))}


def evaluate_objects_stacked(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    length = len(time_s)
    upper = pose_series(inputs["upper_pose"], "upper_pose", length)
    lower = pose_series(inputs["lower_pose"], "lower_pose", length)
    centroid = entity_centroid_world(upper, inputs["upper_geometry"], "upper_geometry")
    centroid_local = world_point_to_local(lower, centroid)
    footprint = geometry_footprint(inputs["lower_geometry"], "lower_geometry")
    lower_top = geometry_aabb(inputs["lower_geometry"], "lower_geometry")[5]
    upper_bottom = _object_bottom_in_reference(upper, inputs["upper_geometry"], lower)
    gap_limit = finite_float(params["height_tolerance_m"], "height_tolerance_m", nonnegative=True)
    footprint_tol = finite_float(params["footprint_tolerance_m"], "footprint_tolerance_m", nonnegative=True)
    threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
    contact, norms = contact_present(inputs["contact_force"], length, threshold)
    matched = point_in_footprint(centroid_local[:, :2], footprint, footprint_tol) & (np.abs(upper_bottom - lower_top) <= gap_limit) & contact
    return matched, {"minimum_stack_gap_m": float(np.min(np.abs(upper_bottom - lower_top), initial=np.inf)), "maximum_contact_force_n": float(np.max(norms, initial=0.0))}
