"""Pure NumPy local/world geometry operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .trace import (
    PredicateValidationError,
    constant_or_pose_series,
    finite_array,
    finite_float,
    pose_series,
)


def quaternion_apply_wxyz(quaternion: object, point: object) -> np.ndarray:
    q = finite_array(quaternion, "quaternion", shape_tail=(4,))
    p = finite_array(point, "point", shape_tail=(3,))
    try:
        prefix = np.broadcast_shapes(q.shape[:-1], p.shape[:-1])
    except ValueError as error:
        raise PredicateValidationError("quaternion and point prefixes are not broadcastable") from error
    q = np.broadcast_to(q, prefix + (4,))
    p = np.broadcast_to(p, prefix + (3,))
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise PredicateValidationError("quaternion must be nonzero")
    q = q / norm
    w = q[..., :1]
    xyz = q[..., 1:]
    return p + 2.0 * np.cross(xyz, np.cross(xyz, p) + w * p)


def world_point_from_local(pose: np.ndarray, local_point: object) -> np.ndarray:
    point = finite_array(local_point, "local_point", shape_tail=(3,))
    return pose[..., :3] + quaternion_apply_wxyz(pose[..., 3:7], point)


def world_point_to_local(pose: np.ndarray, world_point: object) -> np.ndarray:
    point = finite_array(world_point, "world_point", shape_tail=(3,))
    q = pose[..., 3:7].copy()
    q[..., 1:] *= -1.0
    return quaternion_apply_wxyz(q, point - pose[..., :3])


def point_in_planes(point: object, planes: object, tolerance_m: float = 0.0) -> np.ndarray:
    p = finite_array(point, "point", shape_tail=(3,))
    h = finite_array(planes, "planes", ndim=2, shape_tail=(4,))
    if h.shape[0] < 4:
        raise PredicateValidationError("planes must contain at least four hull planes")
    tolerance = finite_float(tolerance_m, "tolerance_m", nonnegative=True)
    normal_norm = np.linalg.norm(h[:, :3], axis=1)
    if np.any(normal_norm <= 1e-12):
        raise PredicateValidationError("planes must have nonzero normals")
    signed = (p @ h[:, :3].T + h[:, 3]) / normal_norm
    return np.all(signed <= tolerance, axis=-1)


def point_in_footprint(point_xy: object, footprint_xy: object, tolerance_m: float = 0.0) -> np.ndarray:
    points = finite_array(point_xy, "point_xy", shape_tail=(2,))
    polygon = finite_array(footprint_xy, "footprint_xy", ndim=2, shape_tail=(2,))
    if polygon.shape[0] < 3:
        raise PredicateValidationError("footprint_xy must contain at least three vertices")
    tolerance = finite_float(tolerance_m, "tolerance_m", nonnegative=True)
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_norm = np.linalg.norm(edges, axis=1)
    if np.any(edge_norm <= 1e-12):
        raise PredicateValidationError("footprint_xy contains duplicate adjacent vertices")
    rel = points[..., None, :] - polygon
    crosses = (edges[:, 0] * rel[..., 1] - edges[:, 1] * rel[..., 0]) / edge_norm
    return np.all(crosses >= -tolerance, axis=-1) | np.all(crosses <= tolerance, axis=-1)


def geometry_centroid(geometry: Mapping[str, Any], name: str) -> np.ndarray:
    if not isinstance(geometry, Mapping):
        raise PredicateValidationError(f"{name} must be a geometry mapping")
    return finite_array(geometry["local_centroid_m"], f"{name}.local_centroid_m", ndim=1, shape_tail=(3,))


def geometry_aabb(geometry: Mapping[str, Any], name: str) -> np.ndarray:
    value = finite_array(geometry["local_aabb_m"], f"{name}.local_aabb_m", ndim=1, shape_tail=(6,))
    if np.any(value[:3] > value[3:]):
        raise PredicateValidationError(f"{name}.local_aabb_m has inverted bounds")
    return value


def geometry_footprint(geometry: Mapping[str, Any], name: str) -> np.ndarray:
    if not isinstance(geometry, Mapping):
        raise PredicateValidationError(f"{name} must be a geometry mapping")
    return finite_array(geometry["footprint_xy_m"], f"{name}.footprint_xy_m", ndim=2, shape_tail=(2,))


def geometry_planes(geometry: Mapping[str, Any], name: str, key: str) -> np.ndarray:
    if not isinstance(geometry, Mapping):
        raise PredicateValidationError(f"{name} must be a geometry mapping")
    return finite_array(geometry[key], f"{name}.{key}", ndim=2, shape_tail=(4,))


def geometry_hull_vertices(geometry: Mapping[str, Any], name: str) -> np.ndarray:
    """Load a frozen, full-dimensional local convex-hull vertex set.

    ``local_hull_vertices_m`` is deliberately distinct from ``local_aabb_m``:
    an AABB is only a conservative proxy and cannot prove exact containment of
    a rotated or non-box-shaped object.  Missing or degenerate hull evidence is
    therefore an evaluation error rather than a false geometric fact.
    """

    if not isinstance(geometry, Mapping):
        raise PredicateValidationError(f"{name} must be a geometry mapping")
    if "local_hull_vertices_m" not in geometry:
        raise PredicateValidationError(
            f"{name}.local_hull_vertices_m is required for full-object containment"
        )
    vertices = finite_array(
        geometry["local_hull_vertices_m"],
        f"{name}.local_hull_vertices_m",
        ndim=2,
        shape_tail=(3,),
    )
    if len(vertices) < 4:
        raise PredicateValidationError(
            f"{name}.local_hull_vertices_m must contain at least four vertices"
        )
    if len(np.unique(vertices, axis=0)) != len(vertices):
        raise PredicateValidationError(
            f"{name}.local_hull_vertices_m must contain unique vertices"
        )
    if np.linalg.matrix_rank(vertices[1:] - vertices[0], tol=1e-10) < 3:
        raise PredicateValidationError(
            f"{name}.local_hull_vertices_m must define a full-dimensional hull"
        )
    return vertices


def entity_centroid_world(pose: np.ndarray, geometry: Mapping[str, Any] | None, name: str) -> np.ndarray:
    local = np.zeros(3) if geometry is None else geometry_centroid(geometry, name)
    return world_point_from_local(pose, local)


def evaluate_relative_position(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    subject = pose_series(inputs["subject_pose"], "subject_pose", len(time_s))
    reference = constant_or_pose_series(
        inputs["reference_pose"], "reference_pose", len(time_s)
    )
    subject_point = entity_centroid_world(subject, inputs.get("subject_geometry"), "subject_geometry")
    reference_point = entity_centroid_world(reference, inputs.get("reference_geometry"), "reference_geometry")
    delta = subject_point - reference_point
    relation = params["relation"]
    if relation not in {"left", "right", "front", "behind", "above", "below", "next_to"}:
        raise PredicateValidationError("relation is unsupported")
    minimum = finite_float(params["min_separation_m"], "min_separation_m", nonnegative=True)
    maximum = finite_float(params["max_distance_m"], "max_distance_m", positive=True)
    axes = {"right": (0, 1), "left": (0, -1), "front": (1, 1), "behind": (1, -1), "above": (2, 1), "below": (2, -1)}
    distance = np.linalg.norm(delta, axis=1)
    matched = distance <= maximum if relation == "next_to" else axes[relation][1] * delta[:, axes[relation][0]] >= minimum
    return matched, {"relation": relation, "minimum_distance_m": float(np.min(distance, initial=np.inf)), "maximum_distance_m": float(np.max(distance, initial=0.0))}


def evaluate_object_in_region(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
):
    """Test whether an object's frozen local centroid lies in a convex region.

    The region is described by ``hull_planes_full`` in its local frame.  Both
    object and region poses may vary over the episode; the geometry itself is
    immutable Rule Package context.
    """

    object_pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    region_pose = constant_or_pose_series(
        inputs["region_pose"], "region_pose", len(time_s)
    )
    object_centroid_world = entity_centroid_world(
        object_pose, inputs["object_geometry"], "object_geometry"
    )
    object_centroid_region = world_point_to_local(region_pose, object_centroid_world)
    planes = geometry_planes(
        inputs["region_geometry"], "region_geometry", "hull_planes_full"
    )
    tolerance = finite_float(
        params["containment_tolerance_m"],
        "containment_tolerance_m",
        nonnegative=True,
    )
    matched = point_in_planes(object_centroid_region, planes, tolerance)
    return matched, {
        "containment_tolerance_m": tolerance,
        "matched_sample_count": int(np.count_nonzero(matched)),
    }


def evaluate_object_local_point_in_region(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
):
    """Test a trusted object-local functional point against a convex region.

    This is intentionally different from :func:`evaluate_object_in_region`:
    the latter observes the frozen object centroid, while this Predicate is
    for organiser-defined functional points such as a connector tip or tool
    roller centre.  The point is a trusted Rule Package parameter, never a
    value inferred by Judge from mesh naming.
    """

    object_pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    region_pose = constant_or_pose_series(
        inputs["region_pose"], "region_pose", len(time_s)
    )
    local_point = finite_array(
        params["local_point_m"],
        "local_point_m",
        ndim=1,
        shape_tail=(3,),
    )
    point_world = world_point_from_local(object_pose, local_point)
    point_region = world_point_to_local(region_pose, point_world)
    planes = geometry_planes(
        inputs["region_geometry"], "region_geometry", "hull_planes_full"
    )
    tolerance = finite_float(
        params["containment_tolerance_m"],
        "containment_tolerance_m",
        nonnegative=True,
    )
    matched = point_in_planes(point_region, planes, tolerance)
    return matched, {
        "local_point_m": [float(value) for value in local_point],
        "containment_tolerance_m": tolerance,
        "matched_sample_count": int(np.count_nonzero(matched)),
    }


def evaluate_object_fully_in_region(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
):
    """Test whether an object's complete frozen convex hull is in a region.

    Every frozen object-local hull vertex is transformed by the sampled object
    pose into world space and then into the sampled/fixed region frame.  A
    convex object is fully contained in a convex region exactly when all of its
    hull vertices satisfy every region hull plane.
    """

    object_pose = pose_series(inputs["object_pose"], "object_pose", len(time_s))
    region_pose = constant_or_pose_series(
        inputs["region_pose"], "region_pose", len(time_s)
    )
    object_vertices = geometry_hull_vertices(
        inputs["object_geometry"], "object_geometry"
    )
    planes = geometry_planes(
        inputs["region_geometry"], "region_geometry", "hull_planes_full"
    )
    tolerance = finite_float(
        params["containment_tolerance_m"],
        "containment_tolerance_m",
        nonnegative=True,
    )

    object_vertices_world = world_point_from_local(
        object_pose[:, None, :], object_vertices[None, :, :]
    )
    object_vertices_region = world_point_to_local(
        region_pose[:, None, :], object_vertices_world
    )
    vertices_inside = point_in_planes(
        object_vertices_region, planes, tolerance
    )
    matched = np.all(vertices_inside, axis=1)
    return matched, {
        "containment_tolerance_m": tolerance,
        "object_hull_vertex_count": int(len(object_vertices)),
        "matched_sample_count": int(np.count_nonzero(matched)),
    }
