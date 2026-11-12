"""Fail-closed validation for the unified Rule Package v1 contract."""

from __future__ import annotations

import math
import re
import copy
from collections.abc import Mapping
from typing import Any, FrozenSet, Set

from robolab_action_judge import package as legacy_package
from robolab_action_judge.errors import ActionJudgeError
from robolab_action_predicates import (
    EMBODIED_PREDICATE_NAMES,
    SAFETY_V08_PREDICATE_NAMES,
)

from .errors import RulePackageV1Error

RULE_PACKAGE_V1_SCHEMA = "robolab-action-rule-package/v1"
V1_PACKAGE_SCHEMA = RULE_PACKAGE_V1_SCHEMA
TRACE_PLAN_V1_SCHEMA = "robolab-action-trace-plan/v0.1"
CANDIDATE_QUANTIFIER_V1_SCHEMA = "robolab-action-candidate-quantifier/v0.1"

TOP_LEVEL_KEYS = frozenset(
    {"package_schema", "package_id", "task_id", "robot_role", "trajectory_context", "items"}
)
ITEM_REQUIRED_KEYS = frozenset(
    {"id", "kind", "description", "evaluation", "depends_on"}
)
ITEM_OPTIONAL_KEYS = frozenset({"scoring_annotations"})
ITEM_KINDS = frozenset({"score_point", "fatal_check", "outcome_check"})
ATOMIC_EVALUATION_KEYS = frozenset({"predicate", "inputs", "params", "temporal"})
TRACE_EVALUATION_KEYS = frozenset({"schema", "nodes", "verdict"})
CANDIDATE_EVALUATION_KEYS = frozenset({"candidate_quantifier"})
MAX_ITEMS = 4096
MAX_TEXT_LENGTH = 16384
MAX_ANNOTATIONS = 32
MAX_ANNOTATION_LIST = 256
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROBOT_ROLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_ANNOTATION_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

_ALL_PREDICATES: FrozenSet[str] = frozenset(
    set(EMBODIED_PREDICATE_NAMES) | set(SAFETY_V08_PREDICATE_NAMES)
)


def _text(value: object, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RulePackageV1Error("{} must be a non-empty string".format(name))
    if len(value) > MAX_TEXT_LENGTH:
        raise RulePackageV1Error("{} is too long".format(name))
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise RulePackageV1Error("{} contains unsupported characters".format(name))
    return value


def _finite_json(value: object, name: str, *, depth: int = 0) -> None:
    if depth > 16:
        raise RulePackageV1Error("{} exceeds the maximum nesting depth".format(name))
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RulePackageV1Error("{} contains a non-finite number".format(name))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, "{}[{}]".format(name, index), depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RulePackageV1Error("{} keys must be strings".format(name))
            _finite_json(item, "{}.{}".format(name, key), depth=depth + 1)
        return
    raise RulePackageV1Error("{} contains a non-JSON value".format(name))


def _geometry_context(package: Mapping[str, Any]) -> tuple[Set[str], dict[str, Mapping[str, Any]]]:
    context = package["trajectory_context"]
    geometries = {
        str(geometry["entity_id"]): geometry for geometry in context["geometries"]
    }
    return set(geometries), geometries


def _validate_atomic(
    evaluation: Mapping[str, Any],
    name: str,
    geometry_ids: Set[str],
    geometries: Mapping[str, Mapping[str, Any]],
) -> None:
    legacy_package._validate_predicate_call_fields(
        evaluation,
        name,
        geometry_ids,
        allow_v053_region_pose_literal=True,
        allow_numeric_literal=True,
        allowed_predicates=_ALL_PREDICATES,
    )
    legacy_package._validate_temporal(evaluation.get("temporal"), "{}.temporal".format(name))
    legacy_package._validate_full_object_containment_geometry(evaluation, name, geometries)


def _validate_evaluation(
    evaluation: object,
    name: str,
    geometry_ids: Set[str],
    geometries: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(evaluation, Mapping):
        raise RulePackageV1Error("{} must be an object".format(name))
    keys = frozenset(evaluation)
    try:
        if keys == ATOMIC_EVALUATION_KEYS:
            _validate_atomic(evaluation, name, geometry_ids, geometries)
        elif keys == TRACE_EVALUATION_KEYS:
            legacy_package._validate_v053_evaluation(
                evaluation,
                name,
                geometry_ids,
                allow_numeric_literal=True,
                allowed_predicates=_ALL_PREDICATES,
            )
            legacy_package._validate_full_object_containment_geometry(
                {"evaluation": evaluation}, name, geometries
            )
        elif keys == CANDIDATE_EVALUATION_KEYS:
            legacy_package._validate_candidate_quantifier(
                evaluation.get("candidate_quantifier"),
                "{}.candidate_quantifier".format(name),
                geometry_ids,
                geometries,
                allowed_predicates=_ALL_PREDICATES,
                allow_numeric_literal=True,
            )
        else:
            raise RulePackageV1Error(
                "{} must be exactly an atomic predicate, Trace Plan, or candidate quantifier".format(name)
            )
    except ActionJudgeError as error:
        raise RulePackageV1Error("{} is invalid: {}".format(name, error)) from error


def _validate_annotation_scalar(value: object, name: str) -> None:
    if isinstance(value, (str, bool, int)) or value is None:
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise RulePackageV1Error("{} must be a finite JSON scalar".format(name))


def _validate_annotations(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or len(value) > MAX_ANNOTATIONS:
        raise RulePackageV1Error(
            "{} must be an object with at most {} fields".format(name, MAX_ANNOTATIONS)
        )
    for key, annotation in value.items():
        if not isinstance(key, str) or _ANNOTATION_KEY.fullmatch(key) is None:
            raise RulePackageV1Error("{} has an invalid field name".format(name))
        item_name = "{}.{}".format(name, key)
        if isinstance(annotation, list):
            if len(annotation) > MAX_ANNOTATION_LIST:
                raise RulePackageV1Error("{} is too long".format(item_name))
            for index, item in enumerate(annotation):
                _validate_annotation_scalar(item, "{}[{}]".format(item_name, index))
        else:
            _validate_annotation_scalar(annotation, item_name)


def validate_rule_package_v1(package: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one v1 package and return a private, canonical-shape copy."""

    if not isinstance(package, Mapping):
        raise RulePackageV1Error("Rule Package root must be an object")
    if frozenset(package) != TOP_LEVEL_KEYS:
        raise RulePackageV1Error("Rule Package v1 has an invalid top-level key set")
    if package.get("package_schema") != RULE_PACKAGE_V1_SCHEMA:
        raise RulePackageV1Error(
            "package_schema must be '{}'".format(RULE_PACKAGE_V1_SCHEMA)
        )
    _text(package.get("package_id"), "package_id", identifier=True)
    _text(package.get("task_id"), "task_id", identifier=True)
    role = package.get("robot_role")
    if not isinstance(role, str) or _ROBOT_ROLE.fullmatch(role) is None:
        raise RulePackageV1Error("robot_role contains unsupported characters")

    try:
        legacy_package._validate_trajectory_context(
            package.get("trajectory_context"),
            require_v5=True,
            allow_hull_vertices=True,
        )
    except ActionJudgeError as error:
        raise RulePackageV1Error("trajectory_context is invalid: {}".format(error)) from error

    geometry_ids, geometries = _geometry_context(package)
    items = package.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise RulePackageV1Error(
            "items must contain between 1 and {} items".format(MAX_ITEMS)
        )
    seen: Set[str] = set()
    for index, item in enumerate(items):
        name = "items[{}]".format(index)
        if not isinstance(item, Mapping):
            raise RulePackageV1Error("{} must be an object".format(name))
        keys = frozenset(item)
        if not ITEM_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
            ITEM_REQUIRED_KEYS | ITEM_OPTIONAL_KEYS
        ):
            raise RulePackageV1Error("{} has an invalid key set".format(name))
        item_id = _text(item.get("id"), "{}.id".format(name), identifier=True)
        if item_id in seen:
            raise RulePackageV1Error("duplicate Rule Package item id: {}".format(item_id))
        if item.get("kind") not in ITEM_KINDS:
            raise RulePackageV1Error("{}.kind is invalid".format(name))
        _text(item.get("description"), "{}.description".format(name))
        _validate_evaluation(
            item.get("evaluation"), "{}.evaluation".format(name), geometry_ids, geometries
        )
        dependencies = item.get("depends_on")
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(dependency, str) for dependency in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            raise RulePackageV1Error("{}.depends_on must be a unique list of item IDs".format(name))
        if any(dependency not in seen for dependency in dependencies):
            raise RulePackageV1Error(
                "{}.depends_on may reference only earlier items".format(name)
            )
        if "scoring_annotations" in item:
            _validate_annotations(
                item.get("scoring_annotations"), "{}.scoring_annotations".format(name)
            )
        seen.add(item_id)

    _finite_json(package, "Rule Package")
    return copy.deepcopy(dict(package))


__all__ = [
    "CANDIDATE_QUANTIFIER_V1_SCHEMA",
    "RULE_PACKAGE_V1_SCHEMA",
    "TRACE_PLAN_V1_SCHEMA",
    "V1_PACKAGE_SCHEMA",
    "validate_rule_package_v1",
]
