"""Public score-free Judge Core shared by safety and diagnostic competitions.

The successful protocol value is deliberately only an ordered ``ItemId ->
bool`` mapping.  Package identity, evidence validation, diagnostics, dependency
filtering, and scoring remain outside that value.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py

from robolab_action_judge.errors import ActionJudgeError
from robolab_action_judge.package import (
    V052_PACKAGE_SCHEMA,
    V053_PACKAGE_SCHEMA,
    V06_PACKAGE_SCHEMA,
    V07_PACKAGE_SCHEMA,
    V08_PACKAGE_SCHEMA,
    load_rule_package,
    validate_rule_package,
)
from robolab_action_judge.predicate_engine import (
    evaluate_boolean_predicate_items,
    evaluate_v053_process_items,
)
from robolab_action_judge.preflight import inspect_submission_against_rule_package
from robolab_action_judge.trajectory_view import Hdf5TrajectoryView

from v2.common.rule_package_v1 import (
    RULE_PACKAGE_V1_SCHEMA,
    RulePackageV1Error,
    load_rule_package_v1,
    project_rule_package_v1_for_boolean_engine,
)

from .errors import (
    JudgeAuditCallbackError,
    JudgeCoreEvaluationError,
    JudgeCoreInputError,
    JudgeCorePreflightError,
    JudgeCoreUnsupportedError,
)


RawFacts = OrderedDict
RulePackageSource = Union[str, Path, Mapping[str, Any]]
AuditCallback = Callable[[Mapping[str, Any]], None]

_SAFETY_SCHEMAS = frozenset(
    {V06_PACKAGE_SCHEMA, V07_PACKAGE_SCHEMA, V08_PACKAGE_SCHEMA}
)
_DIAGNOSTIC_SCHEMAS = frozenset({V052_PACKAGE_SCHEMA, V053_PACKAGE_SCHEMA})
SUPPORTED_PACKAGE_SCHEMAS = (
    _SAFETY_SCHEMAS | _DIAGNOSTIC_SCHEMAS | frozenset({RULE_PACKAGE_V1_SCHEMA})
)


def evaluate_raw_facts(
    rule_package: RulePackageSource,
    submission_hdf5_path: Union[str, Path],
    *,
    expected_task_id: str,
    expected_robot_role: str,
    audit_callback: Optional[AuditCallback] = None,
) -> RawFacts:
    """Evaluate every Rule Package judgment item into a complete ordered map.

    ``depends_on`` is intentionally ignored.  A normal predicate/Trace failure
    becomes ``False``; malformed inputs, missing evidence, unsupported judging
    modes, or evaluation failures raise a typed error and return no mapping.

    The optional callback is invoked exactly once, after every verdict has been
    computed and completeness has been checked.  Its report is never inserted
    into the returned Raw Facts.
    """

    task_id = _normalized_identity(expected_task_id, "expected_task_id")
    robot_role = _normalized_identity(expected_robot_role, "expected_robot_role")
    package = _load_trusted_package(rule_package)
    if any("final_source_rule" in item.get("scoring_annotations", {}) for item in package.get("items", [])):
        raise JudgeCoreUnsupportedError("final-source packages require the trusted final physical evaluator")
    schema = package.get("package_schema")
    if schema not in SUPPORTED_PACKAGE_SCHEMAS:
        raise JudgeCoreUnsupportedError(
            "public Judge Core does not support Rule Package schema {!r}".format(
                schema
            )
        )
    package_task_id = package.get("task_id")
    if package_task_id != task_id:
        raise JudgeCoreInputError(
            "expected_task_id must exactly match trusted Rule Package task_id: "
            "{!r} != {!r}".format(task_id, package_task_id)
        )
    preflight_package = (
        project_rule_package_v1_for_boolean_engine(package)
        if schema == RULE_PACKAGE_V1_SCHEMA
        else package
    )
    if (
        schema == RULE_PACKAGE_V1_SCHEMA
        and package.get("robot_role") != robot_role
    ):
        raise JudgeCoreInputError(
            "expected_robot_role must exactly match trusted Rule Package robot_role: "
            "{!r} != {!r}".format(robot_role, package.get("robot_role"))
        )

    ordered_items = _ordered_judgment_items(package)
    _reject_external_judges(ordered_items)
    expected_ids = [str(item[1]["id"]) for item in ordered_items]
    _require_unique_ids(expected_ids)

    evidence_path = Path(submission_hdf5_path).expanduser().resolve()
    if not evidence_path.is_file():
        raise JudgeCoreInputError(
            "submission HDF5 does not exist: {}".format(evidence_path)
        )

    try:
        with h5py.File(evidence_path, "r") as hdf5_file:
            preflight = inspect_submission_against_rule_package(
                preflight_package,
                hdf5_file,
                expected_task_id=task_id,
                expected_robot_role=robot_role,
            )
            if preflight.get("status") != "VALID":
                raise JudgeCorePreflightError(
                    "submission HDF5 failed Rule Package preflight",
                    preflight,
                )
            diagnostics = _evaluate_all(
                package,
                ordered_items,
                hdf5_file,
                preflight,
            )
    except JudgeCorePreflightError:
        raise
    except OSError as error:
        raise JudgeCoreInputError(
            "cannot read submission HDF5: {}".format(error)
        ) from error
    except ActionJudgeError as error:
        raise JudgeCoreEvaluationError(str(error)) from error

    actual_ids = [str(item["id"]) for item in diagnostics]
    if actual_ids != expected_ids:
        raise JudgeCoreEvaluationError(
            "predicate engine returned incomplete or reordered judgment items"
        )
    verdicts: List[bool] = []
    for item in diagnostics:
        label = item.get("label")
        if label not in {"PASS", "FAIL"}:
            raise JudgeCoreEvaluationError(
                "predicate engine returned a non-binary label for {!r}".format(
                    item.get("id")
                )
            )
        verdicts.append(label == "PASS")

    # Construct the protocol value only after all items have been validated.
    raw_facts = OrderedDict(zip(expected_ids, verdicts))
    if audit_callback is not None:
        report = _audit_report(schema, diagnostics, preflight)
        try:
            audit_callback(report)
        except Exception as error:
            raise JudgeAuditCallbackError(
                "Judge audit callback failed: {}".format(error)
            ) from error
    return raw_facts


def _normalized_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise JudgeCoreInputError("{} must be normalized non-empty text".format(name))
    return value


def _load_trusted_package(source: RulePackageSource) -> Dict[str, Any]:
    legacy_error: Optional[Exception] = None
    try:
        if isinstance(source, Mapping):
            if source.get("package_schema") == RULE_PACKAGE_V1_SCHEMA:
                return load_rule_package_v1(source)
            package = copy.deepcopy(dict(source))
            validate_rule_package(package)
            return package
        return load_rule_package(source)
    except ActionJudgeError as error:
        legacy_error = error
    except (TypeError, ValueError) as error:
        raise JudgeCoreInputError("invalid trusted Rule Package: {}".format(error)) from error

    # A valid v1 path is expected to be rejected by the frozen legacy loader.
    # Preserve the legacy error for every non-v1 path so old failure behaviour
    # does not silently change.
    try:
        return load_rule_package_v1(source)
    except (RulePackageV1Error, OSError, TypeError, ValueError):
        assert legacy_error is not None
        raise JudgeCoreInputError(
            "invalid trusted Rule Package: {}".format(legacy_error)
        ) from legacy_error


def _ordered_judgment_items(
    package: Mapping[str, Any],
) -> List[Tuple[str, Mapping[str, Any]]]:
    result: List[Tuple[str, Mapping[str, Any]]] = [
        ("items", item) for item in package["items"]
    ]
    if package.get("package_schema") in _DIAGNOSTIC_SCHEMAS:
        result.extend(("fatal_checks", item) for item in package["fatal_checks"])
        if package.get("stage") == "final":
            result.extend(
                ("outcome_checks", item) for item in package["outcome_checks"]
            )
    return result


def _require_unique_ids(item_ids: List[str]) -> None:
    if len(item_ids) != len(set(item_ids)):
        raise JudgeCoreInputError(
            "trusted Rule Package judgment IDs must be unique across all groups"
        )


def _reject_external_judges(
    ordered_items: List[Tuple[str, Mapping[str, Any]]]
) -> None:
    unsupported = [
        str(item["id"])
        for group, item in ordered_items
        if group != "items" and item.get("judge") != "predicate"
    ]
    if unsupported:
        raise JudgeCoreUnsupportedError(
            "public Judge Core cannot execute external trajectory_ai checks: {}".format(
                ", ".join(unsupported)
            )
        )


def _evaluate_all(
    package: Mapping[str, Any],
    ordered_items: List[Tuple[str, Mapping[str, Any]]],
    hdf5_file: h5py.File,
    preflight: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    schema = package["package_schema"]
    engine_package = (
        project_rule_package_v1_for_boolean_engine(package)
        if schema == RULE_PACKAGE_V1_SCHEMA
        else package
    )
    view = Hdf5TrajectoryView(hdf5_file, engine_package["trajectory_context"])
    diagnostics: List[Dict[str, Any]] = []

    process_items = (
        list(engine_package["items"])
        if schema == RULE_PACKAGE_V1_SCHEMA
        else [item for group, item in ordered_items if group == "items"]
    )
    candidate_ids_by_item = None
    if schema in {V08_PACKAGE_SCHEMA, RULE_PACKAGE_V1_SCHEMA}:
        candidate_ids_by_item = {
            str(entry["owner_id"]): tuple(
                str(candidate_id)
                for candidate_id in entry["complete_candidate_ids"]
            )
            for entry in preflight.get("candidate_evidence", [])
            if isinstance(entry, Mapping)
        }
    if schema in {
        V053_PACKAGE_SCHEMA,
        V06_PACKAGE_SCHEMA,
        V07_PACKAGE_SCHEMA,
        V08_PACKAGE_SCHEMA,
        RULE_PACKAGE_V1_SCHEMA,
    }:
        process_results = evaluate_v053_process_items(
            view,
            engine_package["trajectory_context"],
            process_items,
            candidate_ids_by_item=candidate_ids_by_item,
        )
    else:
        process_results = evaluate_boolean_predicate_items(
            view,
            engine_package["trajectory_context"],
            process_items,
            enforce_dependencies=False,
        )
    diagnostics.extend(dict(item) for item in process_results)

    for group_name in ("fatal_checks", "outcome_checks"):
        group_items = [item for group, item in ordered_items if group == group_name]
        if not group_items:
            continue
        group_results = evaluate_boolean_predicate_items(
            view,
            package["trajectory_context"],
            group_items,
            enforce_dependencies=False,
        )
        diagnostics.extend(dict(item) for item in group_results)
    return diagnostics


def _audit_report(
    package_schema: str,
    diagnostics: List[Dict[str, Any]],
    preflight: Mapping[str, Any],
) -> Mapping[str, Any]:
    # A deep copy prevents the callback from mutating engine-owned structures.
    value = {
        "package_schema": package_schema,
        "preflight": copy.deepcopy(dict(preflight)),
        "items": copy.deepcopy(diagnostics),
    }
    return MappingProxyType(value)


__all__ = [
    "AuditCallback",
    "RawFacts",
    "RulePackageSource",
    "SUPPORTED_PACKAGE_SCHEMAS",
    "evaluate_raw_facts",
]
