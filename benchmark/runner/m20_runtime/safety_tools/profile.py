"""Fail-closed validation for task-owned M20 safety profiles.

Rule algorithms belong to the Runner.  Concrete thresholds, monitored semantic
roles and exemptions belong to a versioned question-package profile.  This
module intentionally supplies no numeric safety defaults: a package must state
every value needed by an enabled rule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SAFETY_PROFILE_SCHEMA = "m20-safety-profile/v1"
SUPPORTED_RULE_IDS = frozenset(
    {
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S09",
        "S10",
        "S11",
        "S12",
        "S13",
        "S14",
    }
)
RESPONSE_MODES = frozenset({"record_only"})


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array of strings")
    result = [_string(item, f"{label}[]") for item in value]
    if nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _prim_prefix_list(value: Any, label: str) -> list[str]:
    result = _string_list(value, label)
    for item in result:
        if not item.startswith("/") or item == "/" or "//" in item:
            raise ValueError(f"{label} contains an invalid Prim prefix: {item!r}")
    return result


def _only_keys(value: Mapping[str, Any], label: str, allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {unknown}")


def _validate_s01(parameters: dict[str, Any], label: str) -> None:
    _only_keys(
        parameters,
        label,
        {"source_body_names", "target_scope", "exempt_target_prim_prefixes", "source_scope",
         "ground_contact_body_names", "ground_prim_prefixes", "authorized_contact_body_names"},
    )
    _string_list(parameters.get("source_body_names"), f"{label}.source_body_names", nonempty=True)
    if parameters.get('source_scope','listed_bodies') not in ('listed_bodies','whole_robot'):
        raise ValueError('invalid S01 source scope')
    if parameters.get('source_scope')=='whole_robot':
        _string_list(parameters.get('ground_contact_body_names'),f'{label}.ground_contact_body_names',nonempty=True)
        _prim_prefix_list(parameters.get('ground_prim_prefixes'),f'{label}.ground_prim_prefixes')
        _string_list(parameters.get('authorized_contact_body_names',[]),f'{label}.authorized_contact_body_names')
    if parameters.get("target_scope") != "non_robot_colliders":
        raise ValueError(f"{label}.target_scope must be 'non_robot_colliders'")
    _prim_prefix_list(
        parameters.get("exempt_target_prim_prefixes", []),
        f"{label}.exempt_target_prim_prefixes",
    )


def _validate_body_pairs(value: Any, label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array of two-body arrays")
    normalized: set[tuple[str, str]] = set()
    for index, pair in enumerate(value):
        names = _string_list(pair, f"{label}[{index}]", nonempty=True)
        if len(names) != 2 or names[0] == names[1]:
            raise ValueError(f"{label}[{index}] must contain two distinct bodies")
        key = tuple(sorted(names))
        if key in normalized:
            raise ValueError(f"{label} contains a duplicate pair: {names}")
        normalized.add(key)


def _validate_s02(parameters: dict[str, Any], label: str) -> None:
    _only_keys(
        parameters,
        label,
        {"exempt_direct_joint_neighbors", "exempt_body_pairs"},
    )
    _boolean(
        parameters.get("exempt_direct_joint_neighbors"),
        f"{label}.exempt_direct_joint_neighbors",
    )
    _validate_body_pairs(parameters.get("exempt_body_pairs", []), f"{label}.exempt_body_pairs")


def _validate_s03(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"regions"})
    regions = parameters.get("regions")
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)) or not regions:
        raise ValueError(f"{label}.regions must be a non-empty array")
    criteria: set[str] = set()
    for index, value in enumerate(regions):
        item_label = f"{label}.regions[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {
                "criterion",
                "forbidden_prim_prefixes",
                "minimum_clearance_m",
                "include_carried_objects",
                "carried_object_prim_paths",
                "region_geometry",
                "region_floor_z_m",
                "subject_body_names",
            },
        )
        criterion = _string(item.get("criterion"), f"{item_label}.criterion")
        if 'subject_body_names' in item:
            _string_list(item['subject_body_names'],f'{item_label}.subject_body_names',nonempty=True)
        if criterion in criteria:
            raise ValueError(f"{label}.regions contains duplicate criterion: {criterion!r}")
        criteria.add(criterion)
        _prim_prefix_list(
            item.get("forbidden_prim_prefixes"),
            f"{item_label}.forbidden_prim_prefixes",
        )
        if not item["forbidden_prim_prefixes"]:
            raise ValueError(f"{item_label}.forbidden_prim_prefixes must not be empty")
        _number(
            item.get("minimum_clearance_m"),
            f"{item_label}.minimum_clearance_m",
            minimum=0.0,
        )
        _boolean(
            item.get("include_carried_objects"),
            f"{item_label}.include_carried_objects",
        )
        carried = _prim_prefix_list(
            item.get("carried_object_prim_paths", []),
            f"{item_label}.carried_object_prim_paths",
        )
        if item["include_carried_objects"] and not carried:
            raise ValueError(
                f"{item_label}.carried_object_prim_paths must not be empty "
                "when include_carried_objects is true"
            )
        if item.get("region_geometry") not in {
            "collision_surfaces_xy",
            "initial_world_aabb_xy", "underbody_volumes_xyz", "surface_prisms_xyz",
        }:
            raise ValueError(
                f"{item_label}.region_geometry must be "
                "'collision_surfaces_xy' or 'initial_world_aabb_xy'"
            )
        if item['region_geometry'] == 'underbody_volumes_xyz':
            _number(item.get('region_floor_z_m'), item_label+'.region_floor_z_m')


def _validate_s04(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"objects"})
    objects = parameters.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise ValueError(f"{label}.objects must be a non-empty array")
    paths: set[str] = set()
    for index, value in enumerate(objects):
        item_label = f"{label}.objects[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {
                "prim_path",
                "support_prim_prefixes",
                "gripper_body_names",
                "minimum_drop_m",
                "drop_contract",
            },
        )
        if 'drop_contract' in item:
            contract=_mapping(item['drop_contract'],f'{item_label}.drop_contract')
            _only_keys(contract,'drop_contract',{'version','grasp_body_names','support_confirmation_steps',
                'grasp_confirmation_steps','fall_confirmation_steps','minimum_downward_speed_mps'})
            if contract.get('version')!='physical-drop/v2':raise ValueError('unknown drop contract')
            _string_list(contract.get('grasp_body_names'),'grasp_body_names',nonempty=True)
            for key in ('support_confirmation_steps','grasp_confirmation_steps','fall_confirmation_steps'):
                if type(contract.get(key)) is not int or contract[key]<1:raise ValueError('invalid '+key)
            _number(contract.get('minimum_downward_speed_mps'),'minimum_downward_speed_mps',minimum=0.)
        path = _string(item.get("prim_path"), f"{item_label}.prim_path")
        if not path.startswith("/"):
            raise ValueError(f"{item_label}.prim_path must be an absolute Prim path")
        if path in paths:
            raise ValueError(f"{label}.objects contains duplicate Prim path: {path!r}")
        paths.add(path)
        _prim_prefix_list(
            item.get("support_prim_prefixes"),
            f"{item_label}.support_prim_prefixes",
        )
        if not item["support_prim_prefixes"]:
            raise ValueError(f"{item_label}.support_prim_prefixes must not be empty")
        _string_list(
            item.get("gripper_body_names"),
            f"{item_label}.gripper_body_names",
            nonempty=True,
        )
        _number(
            item.get("minimum_drop_m"),
            f"{item_label}.minimum_drop_m",
            minimum=0.0,
        )


def _validate_s05(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"checks"})
    checks = parameters.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError(f"{label}.checks must be an array")
    criteria: set[str] = set()
    for index, value in enumerate(checks):
        item_label = f"{label}.checks[{index}]"
        item = _mapping(value, item_label)
        criterion = _string(item.get("criterion"), f"{item_label}.criterion")
        check_identity=(criterion,item.get('prim_path'),item.get('attribute_name'))
        if check_identity in criteria:
            raise ValueError(f"{label}.checks contains duplicate criterion/binding: {check_identity!r}")
        criteria.add(check_identity)
        if criterion in {
            "initial_pose_stationary",
            "initial_pose_stationary_until_authorized_contact",
        }:
            allowed = {
                "criterion",
                "prim_path",
                "maximum_displacement_m",
                "maximum_rotation_deg",
            }
            if criterion == "initial_pose_stationary_until_authorized_contact":
                allowed.update(
                    {
                        "authorized_contact_body_names",
                        "authorized_contact_target_prim_prefixes",
                    }
                )
            _only_keys(
                item,
                item_label,
                allowed,
            )
            path = _string(item.get("prim_path"), f"{item_label}.prim_path")
            if not path.startswith("/"):
                raise ValueError(f"{item_label}.prim_path must be an absolute Prim path")
            _number(
                item.get("maximum_displacement_m"),
                f"{item_label}.maximum_displacement_m",
                minimum=0.0,
            )
            _number(
                item.get("maximum_rotation_deg"),
                f"{item_label}.maximum_rotation_deg",
                minimum=0.0,
            )
            if criterion == "initial_pose_stationary_until_authorized_contact":
                _string_list(
                    item.get("authorized_contact_body_names"),
                    f"{item_label}.authorized_contact_body_names",
                    nonempty=True,
                )
                target_prefixes = _prim_prefix_list(
                    item.get("authorized_contact_target_prim_prefixes"),
                    f"{item_label}.authorized_contact_target_prim_prefixes",
                )
                if not target_prefixes:
                    raise ValueError(
                        f"{item_label}.authorized_contact_target_prim_prefixes must not be empty"
                    )
        elif criterion == 'relative_pose_closed':
            _only_keys(item,item_label,{'criterion','prim_path','relative_body_prim','maximum_relative_displacement_m','maximum_relative_rotation_deg','reference'})
            for key in ('prim_path','relative_body_prim'):
                if not _string(item.get(key),key).startswith('/'):raise ValueError('absolute physical closure binding required')
            if item['prim_path']==item['relative_body_prim']:raise ValueError('closure requires independent cap/body')
            if item.get('reference') not in ('authored_closed','formal_initial'):raise ValueError('explicit closure reference required')
            _number(item.get('maximum_relative_displacement_m'),'maximum_relative_displacement_m',minimum=0.)
            _number(item.get('maximum_relative_rotation_deg'),'maximum_relative_rotation_deg',minimum=0.)
        elif criterion == 'no_robot_manipulation':
            _only_keys(item,item_label,{'criterion','prim_path','manipulator_body_names'})
            if not _string(item.get('prim_path'),'prim_path').startswith('/'):raise ValueError('absolute protected object required')
            _string_list(item.get('manipulator_body_names'),'manipulator_body_names',nonempty=True)
        elif criterion == 'omnigraph_boolean_variable':
            _only_keys(item,item_label,{'criterion','prim_path','attribute_name','variable_name','expected_value','button_prim'})
            for key in ('prim_path','button_prim'):
                if not _string(item.get(key),key).startswith('/'):raise ValueError('absolute state binding required')
            variable=_string(item.get('variable_name'),'variable_name')
            if item.get('attribute_name')!='graph:variable:'+variable:raise ValueError('graph variable identity mismatch')
            _boolean(item.get('expected_value'),'expected_value')
        elif criterion == "usd_boolean_attribute":
            _only_keys(
                item,
                item_label,
                {"criterion", "prim_path", "attribute_name", "expected_value"},
            )
            path = _string(item.get("prim_path"), f"{item_label}.prim_path")
            if not path.startswith("/"):
                raise ValueError(f"{item_label}.prim_path must be an absolute Prim path")
            attribute_name = _string(
                item.get("attribute_name"), f"{item_label}.attribute_name"
            )
            if not attribute_name.startswith("state:"):
                raise ValueError(
                    f"{item_label}.attribute_name must use the frozen state:* namespace"
                )
            _boolean(item.get("expected_value"), f"{item_label}.expected_value")
        else:
            raise ValueError(
                f"{item_label}.criterion is unsupported: {criterion!r}"
            )


def _validate_s06(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"limits"})
    limits = parameters.get("limits")
    if not isinstance(limits, Sequence) or isinstance(limits, (str, bytes)) or not limits:
        raise ValueError(f"{label}.limits must be a non-empty array")
    criteria: set[str] = set()
    for index, value in enumerate(limits):
        item_label = f"{label}.limits[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {
                "criterion",
                "source_body_names",
                "target_prim_prefixes",
                "maximum_force_n",
            },
        )
        criterion = _string(item.get("criterion"), f"{item_label}.criterion")
        if criterion in criteria:
            raise ValueError(f"{label}.limits contains duplicate criterion: {criterion!r}")
        criteria.add(criterion)
        _string_list(
            item.get("source_body_names"),
            f"{item_label}.source_body_names",
            nonempty=True,
        )
        _prim_prefix_list(
            item.get("target_prim_prefixes"),
            f"{item_label}.target_prim_prefixes",
        )
        if not item["target_prim_prefixes"]:
            raise ValueError(f"{item_label}.target_prim_prefixes must not be empty")
        _number(
            item.get("maximum_force_n"),
            f"{item_label}.maximum_force_n",
            strictly_positive=True,
        )


def _validate_s07(parameters: dict[str, Any], label: str) -> None:
    _only_keys(
        parameters,
        label,
        {"base_body_name", "wheel_body_names", "ground_prim_prefixes"},
    )
    _string(parameters.get("base_body_name"), f"{label}.base_body_name")
    _string_list(
        parameters.get("wheel_body_names"),
        f"{label}.wheel_body_names",
        nonempty=True,
    )
    _prim_prefix_list(
        parameters.get("ground_prim_prefixes"),
        f"{label}.ground_prim_prefixes",
    )
    if not parameters["ground_prim_prefixes"]:
        raise ValueError(f"{label}.ground_prim_prefixes must not be empty")


def _validate_s08(parameters: dict[str, Any], label: str) -> None:
    _only_keys(
        parameters,
        label,
        {
            "criterion",
            "minimum_clearance_m",
            "include_carried_objects",
            "carried_object_prim_paths",
            "obstacle_scope",
            "obstacle_prim_prefixes",
            "exempt_obstacle_prim_prefixes",
            "additional_clearances", "height_overlap_required",
        },
    )
    if 'height_overlap_required' in parameters:
        _boolean(parameters['height_overlap_required'], label+'.height_overlap_required')
    for index,spec in enumerate(parameters.get('additional_clearances',[])):
        _only_keys(spec,'additional_clearance',{'criterion','obstacle_prim_prefixes','minimum_clearance_m',
            'maximum_clearance_m','dimensions','phase_scope'})
        _string(spec.get('criterion'),'additional_clearance.criterion')
        if not _prim_prefix_list(spec.get('obstacle_prim_prefixes'),'additional_clearance.obstacles'):
            raise ValueError('additional clearance needs explicit obstacle bindings')
        _number(spec.get('minimum_clearance_m'),'additional_clearance.minimum',minimum=0.)
        if spec.get('maximum_clearance_m') is not None:
            _number(spec['maximum_clearance_m'],'additional_clearance.maximum',minimum=spec['minimum_clearance_m'])
        if spec.get('dimensions') not in (2,3):raise ValueError('clearance dimensions must be 2 or 3')
        if spec.get('phase_scope','task') not in ('task','after_docking'):raise ValueError('unsupported clearance phase')
    _string(parameters.get("criterion"), f"{label}.criterion")
    _number(
        parameters.get("minimum_clearance_m"),
        f"{label}.minimum_clearance_m",
        strictly_positive=True,
    )
    _boolean(
        parameters.get("include_carried_objects"),
        f"{label}.include_carried_objects",
    )
    carried = _prim_prefix_list(
        parameters.get("carried_object_prim_paths", []),
        f"{label}.carried_object_prim_paths",
    )
    if parameters["include_carried_objects"] and not carried:
        raise ValueError(
            f"{label}.carried_object_prim_paths must not be empty when "
            "include_carried_objects is true"
        )
    scope = parameters.get("obstacle_scope")
    if scope not in {"all_non_subject_colliders", "listed_prim_prefixes"}:
        raise ValueError(
            f"{label}.obstacle_scope must be 'all_non_subject_colliders' "
            "or 'listed_prim_prefixes'"
        )
    obstacles = _prim_prefix_list(
        parameters.get("obstacle_prim_prefixes", []),
        f"{label}.obstacle_prim_prefixes",
    )
    if scope == "listed_prim_prefixes" and not obstacles:
        raise ValueError(f"{label}.obstacle_prim_prefixes must not be empty")
    _prim_prefix_list(
        parameters.get("exempt_obstacle_prim_prefixes", []),
        f"{label}.exempt_obstacle_prim_prefixes",
    )


def _validate_velocity_metrics(
    parameters: dict[str, Any],
    label: str,
    *,
    limit_field: str,
) -> None:
    _only_keys(parameters, label, {"metrics"})
    metrics = parameters.get("metrics")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)) or not metrics:
        raise ValueError(f"{label}.metrics must be a non-empty array")
    for index, value in enumerate(metrics):
        metric_label = f"{label}.metrics[{index}]"
        metric = _mapping(value, metric_label)
        _only_keys(
            metric,
            metric_label,
            {
                "criterion",
                "body_name",
                "reference_body_name",
                "components",
                limit_field,
            },
        )
        _string(metric.get("criterion"), f"{metric_label}.criterion")
        _string(metric.get("body_name"), f"{metric_label}.body_name")
        components = _string_list(
            metric.get("components"),
            f"{metric_label}.components",
            nonempty=True,
        )
        invalid_components = sorted(set(components) - {"x", "y", "z"})
        if invalid_components:
            raise ValueError(
                f"{metric_label}.components contains unsupported axes: "
                f"{invalid_components}"
            )
        if "reference_body_name" in metric:
            reference = _string(
                metric["reference_body_name"],
                f"{metric_label}.reference_body_name",
            )
            if reference == metric["body_name"]:
                raise ValueError(f"{metric_label} cannot be relative to the same body")
        _number(
            metric.get(limit_field),
            f"{metric_label}.{limit_field}",
            strictly_positive=True,
        )


def _validate_s09(parameters: dict[str, Any], label: str) -> None:
    _validate_velocity_metrics(
        parameters,
        label,
        limit_field="maximum_speed_mps",
    )


def _validate_s10(parameters: dict[str, Any], label: str) -> None:
    _validate_velocity_metrics(
        parameters,
        label,
        limit_field="maximum_angular_speed_radps",
    )


def _validate_s11(parameters: dict[str, Any], label: str) -> None:
    _only_keys(
        parameters,
        label,
        {
            "revolute_position_tolerance_rad",
            "prismatic_position_tolerance_m",
            "velocity_limit_source",
            "maximum_velocity_by_joint",
            "effort_limit_source",
            "maximum_effort_by_joint",
        },
    )
    _number(
        parameters.get("revolute_position_tolerance_rad"),
        f"{label}.revolute_position_tolerance_rad",
        minimum=0.0,
    )
    _number(
        parameters.get("prismatic_position_tolerance_m"),
        f"{label}.prismatic_position_tolerance_m",
        minimum=0.0,
    )
    source = parameters.get("velocity_limit_source")
    if source not in {"robot_profile", "explicit"}:
        raise ValueError(f"{label}.velocity_limit_source must be 'robot_profile' or 'explicit'")
    if source == "explicit":
        limits = _mapping(parameters.get("maximum_velocity_by_joint"), f"{label}.maximum_velocity_by_joint")
        if not limits:
            raise ValueError(f"{label}.maximum_velocity_by_joint must not be empty")
        for joint, limit in limits.items():
            _string(joint, f"{label}.maximum_velocity_by_joint key")
            _number(
                limit,
                f"{label}.maximum_velocity_by_joint[{joint!r}]",
                strictly_positive=True,
            )
    effort_source = parameters.get("effort_limit_source")
    if effort_source not in {"robot_profile", "explicit"}:
        raise ValueError(
            f"{label}.effort_limit_source must be 'robot_profile' or 'explicit'"
        )
    if effort_source == "explicit":
        limits = _mapping(
            parameters.get("maximum_effort_by_joint"),
            f"{label}.maximum_effort_by_joint",
        )
        if not limits:
            raise ValueError(f"{label}.maximum_effort_by_joint must not be empty")
        for joint, limit in limits.items():
            _string(joint, f"{label}.maximum_effort_by_joint key")
            _number(
                limit,
                f"{label}.maximum_effort_by_joint[{joint!r}]",
                strictly_positive=True,
            )


def _validate_s12(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"objects"})
    objects = parameters.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError(f"{label}.objects must be an array")
    paths: set[str] = set()
    for index, value in enumerate(objects):
        item_label = f"{label}.objects[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {
                "prim_path",
                "maximum_displacement_m",
                "maximum_rotation_deg",
                "reference",
            },
        )
        path = _string(item.get("prim_path"), f"{item_label}.prim_path")
        if not path.startswith("/"):
            raise ValueError(f"{item_label}.prim_path must be an absolute Prim path")
        if path in paths:
            raise ValueError(f"{label}.objects contains duplicate Prim path: {path!r}")
        paths.add(path)
        _number(
            item.get("maximum_displacement_m"),
            f"{item_label}.maximum_displacement_m",
            minimum=0.0,
        )
        _number(
            item.get("maximum_rotation_deg"),
            f"{item_label}.maximum_rotation_deg",
            minimum=0.0,
        )
        if item.get("reference") != "task_start_physically_stable_pose":
            raise ValueError(
                f"{item_label}.reference must be "
                "'task_start_physically_stable_pose'"
            )


def _validate_s13(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"objects"})
    objects = parameters.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise ValueError(f"{label}.objects must be a non-empty array")
    paths: set[str] = set()
    for index, value in enumerate(objects):
        item_label = f"{label}.objects[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {"prim_path", "maximum_tilt_deg", "reference"},
        )
        path = _string(item.get("prim_path"), f"{item_label}.prim_path")
        if not path.startswith("/"):
            raise ValueError(f"{item_label}.prim_path must be an absolute Prim path")
        if path in paths:
            raise ValueError(f"{label}.objects contains duplicate Prim path: {path!r}")
        paths.add(path)
        maximum = _number(
            item.get("maximum_tilt_deg"),
            f"{item_label}.maximum_tilt_deg",
            minimum=0.0,
        )
        if maximum > 180.0:
            raise ValueError(f"{item_label}.maximum_tilt_deg must be <= 180")
        if item.get("reference") != "task_start_local_z_world":
            raise ValueError(
                f"{item_label}.reference must be 'task_start_local_z_world'"
            )


def _validate_s14(parameters: dict[str, Any], label: str) -> None:
    _only_keys(parameters, label, {"placements"})
    placements = parameters.get("placements")
    if (
        not isinstance(placements, Sequence)
        or isinstance(placements, (str, bytes))
        or not placements
    ):
        raise ValueError(f"{label}.placements must be a non-empty array")
    criteria: set[str] = set()
    for index, value in enumerate(placements):
        item_label = f"{label}.placements[{index}]"
        item = _mapping(value, item_label)
        _only_keys(
            item,
            item_label,
            {
                "criterion",
                "object_prim_path",
                "forbidden_prim_prefixes",
                "gripper_body_names",
                "include_initial_placement",
            },
        )
        criterion = _string(item.get("criterion"), f"{item_label}.criterion")
        if criterion in criteria:
            raise ValueError(
                f"{label}.placements contains duplicate criterion: {criterion!r}"
            )
        criteria.add(criterion)
        path = _string(
            item.get("object_prim_path"), f"{item_label}.object_prim_path"
        )
        if not path.startswith("/"):
            raise ValueError(
                f"{item_label}.object_prim_path must be an absolute Prim path"
            )
        forbidden = _prim_prefix_list(
            item.get("forbidden_prim_prefixes"),
            f"{item_label}.forbidden_prim_prefixes",
        )
        if not forbidden:
            raise ValueError(
                f"{item_label}.forbidden_prim_prefixes must not be empty"
            )
        _string_list(
            item.get("gripper_body_names"),
            f"{item_label}.gripper_body_names",
            nonempty=True,
        )
        _boolean(
            item.get("include_initial_placement"),
            f"{item_label}.include_initial_placement",
        )


RULE_VALIDATORS = {
    "S01": _validate_s01,
    "S02": _validate_s02,
    "S03": _validate_s03,
    "S04": _validate_s04,
    "S05": _validate_s05,
    "S06": _validate_s06,
    "S07": _validate_s07,
    "S08": _validate_s08,
    "S09": _validate_s09,
    "S10": _validate_s10,
    "S11": _validate_s11,
    "S12": _validate_s12,
    "S13": _validate_s13,
    "S14": _validate_s14,
}


def validate_safety_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an isolated profile copy without inventing defaults."""

    profile = _mapping(value, "safety profile")
    if profile.get("schema") != SAFETY_PROFILE_SCHEMA:
        raise ValueError(f"unsupported safety profile schema: {profile.get('schema')!r}")
    _string(profile.get("profile_id"), "profile_id")
    _string(profile.get("rule_set_version"), "rule_set_version")
    if profile.get("response_mode") not in RESPONSE_MODES:
        raise ValueError(
            "response_mode must be 'record_only'; termination semantics require a separate formal specification"
        )

    activation = _mapping(profile.get("activation"), "activation")
    expected_activation = {
        "start": "task_time_zero",
        "end": "episode_end",
        "sampling": "every_physics_step",
    }
    if activation != expected_activation:
        raise ValueError(f"activation must equal {expected_activation!r}")

    rules = _mapping(profile.get("rules"), "rules")
    if not rules:
        raise ValueError("rules must not be empty")
    unknown = sorted(set(rules) - SUPPORTED_RULE_IDS)
    if unknown:
        raise ValueError(f"unsupported safety rule ids: {unknown}")
    if not any(isinstance(spec, Mapping) and spec.get("enabled") is True for spec in rules.values()):
        raise ValueError("at least one safety rule must be enabled")

    for rule_id, raw_spec in rules.items():
        label = f"rules.{rule_id}"
        spec = _mapping(raw_spec, label)
        enabled = _boolean(spec.get("enabled"), f"{label}.enabled")
        if not enabled:
            continue
        if spec.get('phase_scope','all_capture') not in {'all_capture','task','navigation','base_moving','after_docking'}:
            raise ValueError(f'{label}.phase_scope unsupported')
        parameters = _mapping(spec.get("parameters"), f"{label}.parameters")
        RULE_VALIDATORS[rule_id](parameters, f"{label}.parameters")

    normalized = copy.deepcopy(profile)
    normalized["rules"] = {rule_id: normalized["rules"][rule_id] for rule_id in sorted(rules)}
    return normalized


def enabled_rule_ids(profile: Mapping[str, Any]) -> tuple[str, ...]:
    validated = validate_safety_profile(profile)
    return tuple(
        rule_id
        for rule_id, spec in validated["rules"].items()
        if spec["enabled"]
    )


def canonical_profile_sha256(profile: Mapping[str, Any]) -> str:
    validated = validate_safety_profile(profile)
    payload = json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
