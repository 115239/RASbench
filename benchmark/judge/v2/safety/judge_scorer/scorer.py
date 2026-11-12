"""Thin v2 adapter around the released Safety task-execution scorer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from robolab_safety_tianchi.task_execution import score_task_execution
from robolab_tianchi_evaluator.models import (
    SUCCESS,
    ArtifactEvaluation,
    AtomicJudgment,
)

from v2.common.rule_package_v1.validation import RULE_PACKAGE_V1_SCHEMA
from v2.common.tianchi import ModuleResult, ModuleStatus


_RULE_PACKAGE_V08_SCHEMA = "robolab-action-rule-package/v0.8"


def score_safety_judge(
    package: Mapping[str, Any],
    raw_facts: Mapping[str, bool],
    weight_policy: Optional[Mapping[str, object]] = None,
) -> ModuleResult:
    """Apply the released Safety dependency and core-weight policy.

    The public Judge Core has already evaluated every Rule Package item.  This
    adapter deliberately rebuilds only the legacy boolean evaluation object;
    no HDF5 or Predicate code is duplicated here.
    """

    scoring_package = _project_task_execution_contract(package)
    _validate_raw_facts(scoring_package, raw_facts)
    judgments = {
        item_id: AtomicJudgment(
            item_id=item_id,
            verdict=verdict,
            kind="item",
        )
        for item_id, verdict in raw_facts.items()
    }
    evaluation = ArtifactEvaluation(
        role="trajectory",
        status=SUCCESS,
        judgments=judgments,
    )
    score = score_task_execution(
        scoring_package,
        evaluation,
        expected_role="trajectory",
        weight_policy=weight_policy,
    )
    if score.status != SUCCESS or score.score_ratio is None:
        return ModuleResult(
            module_id="task_execution",
            module_version=score.scoring_schema,
            status=ModuleStatus.UNDER_REVIEW,
            score_ratio=None,
            report={},
        )
    return ModuleResult(
        module_id="task_execution",
        module_version=score.scoring_schema,
        status=ModuleStatus.EVALUATED,
        score_ratio=score.score_ratio,
        report={
            "role": score.role,
            "raw_awarded_units": score.raw_awarded_units,
            "effective_awarded_units": score.effective_awarded_units,
            "total_units": score.total_units,
            "weight_resolution": _weight_resolution_report(score.weight_resolution),
            "contributions": tuple(
                _contribution_report(item) for item in score.contributions
            ),
        },
    )


def _project_task_execution_contract(
    package: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project unified v1 onto the released score-only v0.8 contract.

    Safety task execution assigns weight exclusively from the canonical item
    ID hierarchy.  Consequently this boundary intentionally reads only
    ``id``, ``kind`` and ``depends_on`` from each v1 item.  In particular,
    neither Predicate/Trace ``evaluation`` content nor
    ``scoring_annotations`` can influence Safety weights.

    Older packages are returned unchanged so the released v0.5-v0.8
    compatibility surface remains available.
    """

    if not isinstance(package, Mapping):
        raise ValueError("Safety Rule Package must be an object")
    if package.get("package_schema") != RULE_PACKAGE_V1_SCHEMA:
        return package

    items = package.get("items")
    if not isinstance(items, list):
        raise ValueError("Safety Rule Package items must be an array")
    projected_items = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                "Safety Rule Package items[{}] must be an object".format(index)
            )
        if item.get("kind") != "score_point":
            raise ValueError(
                "Safety task_execution accepts only score_point items; "
                "items[{}].kind is invalid".format(index)
            )
        projected_items.append(
            {
                "id": item.get("id"),
                "depends_on": item.get("depends_on"),
            }
        )

    # The released scorer validates only that these two container fields have
    # the v0.8 shape.  Their contents have no scoring authority.
    return {
        "package_schema": _RULE_PACKAGE_V08_SCHEMA,
        "task_id": "unified-v1-task-execution",
        "trajectory_context": {},
        "items": projected_items,
    }


def _validate_raw_facts(
    package: Mapping[str, Any], raw_facts: Mapping[str, bool]
) -> None:
    items = package.get("items")
    if not isinstance(items, list):
        raise ValueError("Safety Rule Package items must be an array")
    expected_ids = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError("Safety Rule Package items[{}] must be an object".format(index))
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Safety Rule Package items[{}].id must be text".format(index))
        expected_ids.append(item_id)
    if tuple(raw_facts.keys()) != tuple(expected_ids):
        raise ValueError("Raw Facts IDs/order must exactly match Safety Rule Package items")
    if any(type(value) is not bool for value in raw_facts.values()):
        raise ValueError("Raw Facts verdicts must be exact booleans")


def _weight_resolution_report(value: object) -> object:
    if value is None:
        return None
    return {
        "mode": value.mode,
        "effective_policy": dict(value.effective_policy),
        "resolved_units": dict(value.resolved_units),
        "fallback_reason": value.fallback_reason,
    }


def _contribution_report(value: object) -> Mapping[str, object]:
    return {
        "item_id": value.item_id,
        "raw_verdict": value.raw_verdict,
        "effective_verdict": value.effective_verdict,
        "depends_on": value.depends_on,
        "failed_dependencies": value.failed_dependencies,
        "resolved_units": value.resolved_units,
        "raw_awarded_units": value.raw_awarded_units,
        "effective_awarded_units": value.effective_awarded_units,
    }
