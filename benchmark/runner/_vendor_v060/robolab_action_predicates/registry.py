"""Single authoritative manifest and dispatcher for offline predicates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from .contact import (
    evaluate_force_magnitude_below,
    evaluate_object_grabbed,
    evaluate_object_in_contact,
    evaluate_object_released,
)
from .geometry import (
    evaluate_object_fully_in_region,
    evaluate_object_in_region,
    evaluate_object_local_point_in_region,
    evaluate_relative_position,
)
from .manipulation import (
    evaluate_object_in_container,
    evaluate_object_on_surface,
    evaluate_object_picked_up,
    evaluate_objects_stacked,
)
from .motion import (
    evaluate_directional_force,
    evaluate_distance_below,
    evaluate_object_stationary,
    evaluate_object_axis_aligned,
    evaluate_object_upright,
    evaluate_object_yaw_aligned,
    evaluate_path_length_below,
    evaluate_scalar_target,
    evaluate_stationary_near,
    evaluate_trajectory_smooth,
)
from .trace import PredicateTrace, PredicateValidationError

Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], np.ndarray], tuple[np.ndarray, dict[str, Any]]]

PREDICATE_REGISTRY_SCHEMA = "robolab-action-predicate-registry/v0.5"
PREDICATE_PROFILE_SHARED = "shared"
PREDICATE_PROFILE_EMBODIED = "embodied"


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    params: tuple[str, ...]
    defaults: Mapping[str, Any]
    required_geometry: tuple[str, ...]
    stream_reduce: Mapping[str, str]
    default_temporal: Mapping[str, Any]
    semantic_source: str
    evaluator: Evaluator


def _spec(
    name: str,
    purpose: str,
    inputs: tuple[str, ...],
    params: tuple[str, ...],
    evaluator: Evaluator,
    *,
    defaults: Mapping[str, Any],
    optional_inputs: tuple[str, ...] = (),
    required_geometry: tuple[str, ...] = (),
    stream_reduce: Mapping[str, str] | None = None,
    default_temporal: Mapping[str, Any] | None = None,
    semantic_source: str = "RoboLab predicate_logic",
) -> PredicateSpec:
    return PredicateSpec(
        name=name,
        purpose=purpose,
        inputs=inputs,
        optional_inputs=optional_inputs,
        params=params,
        defaults=MappingProxyType(dict(defaults)),
        required_geometry=required_geometry,
        stream_reduce=MappingProxyType(dict(stream_reduce or {})),
        default_temporal=MappingProxyType(dict(default_temporal or {"mode": "eventually"})),
        semantic_source=semantic_source,
        evaluator=evaluator,
    )


_SPECS = (
    _spec("distance_below", "entity reaches target vicinity", ("position", "target_position"), ("max_distance_m",), evaluate_distance_below, defaults={"max_distance_m": 0.8}),
    _spec("stationary_near", "entity is stopped near target", ("position", "target_position", "linear_velocity"), ("max_distance_m", "max_speed_mps"), evaluate_stationary_near, defaults={"max_distance_m": 0.8, "max_speed_mps": 0.05}),
    _spec("directional_force", "force reaches a threshold along a direction", ("force",), ("direction", "direction_frame", "min_force_n"), evaluate_directional_force, defaults={"direction_frame": "world", "min_force_n": 8.0}, stream_reduce={"force": "any_sample_peak"}),
    _spec("force_magnitude_below", "force magnitude does not exceed a frozen limit", ("force",), ("max_force_n",), evaluate_force_magnitude_below, defaults={}, stream_reduce={"force": "any_sample_peak"}, default_temporal={"mode": "always"}, semantic_source="RoboLab force magnitude safety bound"),
    _spec("scalar_target", "scalar reaches an ordered threshold", ("value",), ("operator", "target_value", "target_unit"), evaluate_scalar_target, defaults={}),
    _spec("object_in_contact", "objects have effective physical contact", ("contact_force",), ("min_contact_force_n",), evaluate_object_in_contact, defaults={"min_contact_force_n": 0.1}, stream_reduce={"contact_force": "any_sample_peak"}, semantic_source="RoboLab object_in_contact/in_contact"),
    _spec("object_grabbed", "target has effective gripper contact", ("gripper_contact_force",), ("min_contact_force_n",), evaluate_object_grabbed, defaults={"min_contact_force_n": 0.1}, stream_reduce={"gripper_contact_force": "any_sample_peak"}, semantic_source="RoboLab object_grabbed"),
    _spec("object_released", "previously grabbed target loses contact", ("gripper_contact_force",), ("min_contact_force_n",), evaluate_object_released, defaults={"min_contact_force_n": 0.1}, optional_inputs=("gripper_open",), stream_reduce={"gripper_contact_force": "any_sample_peak"}, semantic_source="RoboLab object_dropped/release state transition"),
    _spec("object_picked_up", "grabbed object rises above initial height", ("object_pose", "initial_object_pose", "gripper_contact_force"), ("min_contact_force_n", "min_height_delta_m"), evaluate_object_picked_up, defaults={"min_contact_force_n": 0.1, "min_height_delta_m": 0.05}, stream_reduce={"gripper_contact_force": "any_sample_peak"}, semantic_source="RoboLab object_picked_up"),
    _spec("object_in_container", "object centroid is inside the container open-top hull", ("object_pose", "container_pose", "object_geometry", "container_geometry"), ("containment_tolerance_m", "require_detached", "require_stationary", "min_contact_force_n", "max_speed_mps"), evaluate_object_in_container, defaults={"containment_tolerance_m": 0.0, "require_detached": False, "require_stationary": False, "min_contact_force_n": 0.1, "max_speed_mps": 0.05}, optional_inputs=("gripper_contact_force", "linear_velocity"), required_geometry=("object_geometry", "container_geometry"), semantic_source="RoboLab object_in_container/centroid-in-hull"),
    _spec("object_in_region", "object centroid is inside a frozen convex 3D region", ("object_pose", "region_pose", "object_geometry", "region_geometry"), ("containment_tolerance_m",), evaluate_object_in_region, defaults={"containment_tolerance_m": 0.0}, required_geometry=("object_geometry", "region_geometry"), semantic_source="RoboLab frozen-region centroid-in-hull"),
    _spec("object_on_surface", "object lies in footprint with upward support", ("object_pose", "surface_pose", "support_force", "object_geometry", "surface_geometry"), ("height_tolerance_m", "footprint_tolerance_m", "min_support_force_n", "min_upward_cosine", "require_detached", "require_stationary", "min_contact_force_n", "max_speed_mps"), evaluate_object_on_surface, defaults={"height_tolerance_m": 0.02, "footprint_tolerance_m": 0.0, "min_support_force_n": 0.1, "min_upward_cosine": 0.5, "require_detached": False, "require_stationary": False, "min_contact_force_n": 0.1, "max_speed_mps": 0.05}, optional_inputs=("gripper_contact_force", "linear_velocity"), required_geometry=("object_geometry", "surface_geometry"), stream_reduce={"support_force": "any_sample_peak"}, semantic_source="RoboLab object_on_top/is_supported_on_surface"),
    _spec("object_stationary", "linear and optional angular speed stay below limits", ("linear_velocity",), ("max_speed_mps", "max_angular_speed_radps"), evaluate_object_stationary, defaults={"max_speed_mps": 0.05, "max_angular_speed_radps": 0.1}, optional_inputs=("angular_velocity",), default_temporal={"mode": "held_for", "min_duration_s": 0.2}, semantic_source="RoboLab objects_stationary"),
    _spec("object_upright", "local up axis aligns with world up", ("object_pose",), ("local_up_axis", "world_up_axis", "max_tilt_deg"), evaluate_object_upright, defaults={"local_up_axis": [0.0, 0.0, 1.0], "world_up_axis": [0.0, 0.0, 1.0], "max_tilt_deg": 15.0}, default_temporal={"mode": "final"}, semantic_source="RoboLab object_upright"),
    _spec("relative_position", "subject satisfies a named spatial relation", ("subject_pose", "reference_pose"), ("relation", "min_separation_m", "max_distance_m"), evaluate_relative_position, defaults={"min_separation_m": 0.0, "max_distance_m": 0.5}, optional_inputs=("subject_geometry", "reference_geometry"), semantic_source="RoboLab left/right/front/behind/next_to"),
    _spec("objects_stacked", "upper object is supported on lower footprint", ("upper_pose", "lower_pose", "contact_force", "upper_geometry", "lower_geometry"), ("min_contact_force_n", "height_tolerance_m", "footprint_tolerance_m"), evaluate_objects_stacked, defaults={"min_contact_force_n": 0.1, "height_tolerance_m": 0.02, "footprint_tolerance_m": 0.0}, required_geometry=("upper_geometry", "lower_geometry"), stream_reduce={"contact_force": "any_sample_peak"}, semantic_source="RoboLab stacked"),
    _spec("path_length_below", "trajectory path length remains within a limit", ("position",), ("max_path_length_m",), evaluate_path_length_below, defaults={}, default_temporal={"mode": "final"}, semantic_source="RoboLab trajectory_metrics.path_length"),
    _spec("trajectory_smooth", "integrated jerk or SPARC satisfies a limit", ("position",), ("metric", "max_integrated_squared_jerk", "min_sparc"), evaluate_trajectory_smooth, defaults={"metric": "integrated_squared_jerk", "max_integrated_squared_jerk": 100.0, "min_sparc": -2.0}, default_temporal={"mode": "final"}, semantic_source="RoboLab trajectory_metrics integrated jerk/SPARC"),
    _spec("object_fully_in_region", "object's complete frozen convex hull is inside a frozen convex 3D region", ("object_pose", "region_pose", "object_geometry", "region_geometry"), ("containment_tolerance_m",), evaluate_object_fully_in_region, defaults={"containment_tolerance_m": 0.0}, required_geometry=("object_geometry", "region_geometry"), semantic_source="RoboLab frozen full-object convex-hull containment"),
)

# These capabilities execute in the shared Judge but are intentionally exposed
# only by the embodied Generator capability view.  Keeping the default view
# unchanged preserves the safety Generator's allowlist and Registry fingerprint.
_EMBODIED_ONLY_SPECS = (
    _spec(
        "object_axis_aligned",
        "trusted object-local axis aligns with a trusted world direction",
        ("object_pose",),
        ("local_axis", "target_world_axis", "max_angle_deg"),
        evaluate_object_axis_aligned,
        defaults={},
        default_temporal={"mode": "final"},
        semantic_source="RoboLab embodied trusted local-axis alignment",
    ),
    _spec(
        "object_yaw_aligned",
        "object world-Z yaw aligns with a trusted target heading",
        ("object_pose",),
        ("target_yaw_rad", "yaw_tolerance_deg", "allow_pi_symmetry"),
        evaluate_object_yaw_aligned,
        defaults={},
        default_temporal={"mode": "final"},
        semantic_source="RoboLab embodied world-Z yaw alignment",
    ),
    _spec(
        "object_local_point_in_region",
        "trusted object-local functional point is inside a frozen convex region",
        ("object_pose", "region_pose", "region_geometry"),
        ("local_point_m", "containment_tolerance_m"),
        evaluate_object_local_point_in_region,
        defaults={"containment_tolerance_m": 0.0},
        required_geometry=("region_geometry",),
        semantic_source="RoboLab embodied trusted local-point region containment",
    ),
)

_ALL_SPECS = _SPECS + _EMBODIED_ONLY_SPECS
_BY_NAME = MappingProxyType({spec.name: spec for spec in _ALL_SPECS})
PREDICATE_NAMES = tuple(spec.name for spec in _SPECS)
EMBODIED_PREDICATE_NAMES = tuple(spec.name for spec in _ALL_SPECS)


def _specs_for_profile(profile: str) -> tuple[PredicateSpec, ...]:
    if profile == PREDICATE_PROFILE_SHARED:
        return _SPECS
    if profile == PREDICATE_PROFILE_EMBODIED:
        return _ALL_SPECS
    raise PredicateValidationError(f"unsupported predicate profile: {profile!r}")


def predicate_manifest(
    profile: str = PREDICATE_PROFILE_SHARED,
) -> dict[str, Any]:
    """Return the evaluator-independent contract shared by Generator and Judge."""

    specs = _specs_for_profile(profile)
    return {
        "schema": PREDICATE_REGISTRY_SCHEMA,
        "predicates": [
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "inputs": list(spec.inputs),
                "optional_inputs": list(spec.optional_inputs),
                "params": list(spec.params),
                "defaults": dict(spec.defaults),
                "required_geometry": list(spec.required_geometry),
                "stream_reduce": dict(spec.stream_reduce),
                "default_temporal": dict(spec.default_temporal),
                "semantic_source": spec.semantic_source,
            }
            for spec in specs
        ],
    }


def predicate_manifest_sha256(
    profile: str = PREDICATE_PROFILE_SHARED,
) -> str:
    canonical = json.dumps(
        predicate_manifest(profile),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def get_predicate_spec(name: str) -> PredicateSpec:
    try:
        return _BY_NAME[name]
    except (KeyError, TypeError) as error:
        raise PredicateValidationError(f"unsupported predicate: {name!r}") from error


def iter_predicate_specs(
    profile: str = PREDICATE_PROFILE_SHARED,
) -> tuple[PredicateSpec, ...]:
    return _specs_for_profile(profile)


def evaluate_predicate(
    name: str,
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    control_steps: object,
    time_s: object,
) -> PredicateTrace:
    spec = get_predicate_spec(name)
    if not isinstance(inputs, Mapping) or not isinstance(params, Mapping):
        raise PredicateValidationError("inputs and params must be mappings")
    input_keys = set(inputs)
    required_inputs = set(spec.inputs)
    allowed_inputs = required_inputs | set(spec.optional_inputs)
    if not required_inputs <= input_keys or not input_keys <= allowed_inputs:
        raise PredicateValidationError(f"{name} inputs must contain {sorted(required_inputs)} and only optional {sorted(set(spec.optional_inputs))}")
    if not set(params) <= set(spec.params):
        raise PredicateValidationError(f"{name} has unknown params: {sorted(set(params) - set(spec.params))}")
    resolved = dict(spec.defaults)
    resolved.update(params)
    missing = set(spec.params) - set(resolved)
    if missing:
        raise PredicateValidationError(f"{name} is missing params: {sorted(missing)}")
    steps = np.asarray(control_steps)
    times = np.asarray(time_s, dtype=np.float64)
    if steps.ndim != 1 or times.shape != steps.shape:
        raise PredicateValidationError("control_steps and time_s must be matching one-dimensional arrays")
    if not len(steps):
        raise PredicateValidationError("predicate evaluation requires at least one control step")
    PredicateTrace(steps, times, np.zeros(len(steps), dtype=np.bool_), {})
    matched, metrics = spec.evaluator(inputs, resolved, times)
    return PredicateTrace(steps, times, np.asarray(matched, dtype=np.bool_), metrics)
