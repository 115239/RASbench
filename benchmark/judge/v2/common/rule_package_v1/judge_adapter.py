"""Project Rule Package v1 into the frozen boolean-evaluation engine shape.

This module is deliberately competition-neutral.  The projection is an
implementation adapter only: it does not add scoring, apply ``depends_on``, or
interpret ``kind``/``scoring_annotations``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, List

from robolab_action_judge.package import V08_PACKAGE_SCHEMA

from .validation import validate_rule_package_v1


def project_rule_package_v1_for_boolean_engine(
    package: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a private legacy-engine-shaped view of a validated v1 package.

    The frozen Action Judge engine already has the required atomic, Trace Plan,
    and per-candidate ``exists`` evaluators.  V1 changes their envelope, not
    their boolean semantics, so a small projection keeps one implementation of
    those semantics while the public v1 contract remains independent of any
    competition.
    """

    validated = validate_rule_package_v1(package)
    projected_items: List[Dict[str, Any]] = []
    for item in validated["items"]:
        evaluation = item["evaluation"]
        projected: Dict[str, Any] = {
            "id": str(item["id"]),
            "description": str(item["description"]),
            "depends_on": list(item["depends_on"]),
        }
        if "predicate" in evaluation:
            projected.update(copy.deepcopy(dict(evaluation)))
        elif "candidate_quantifier" in evaluation:
            projected["candidate_quantifier"] = copy.deepcopy(
                evaluation["candidate_quantifier"]
            )
        else:
            projected["evaluation"] = copy.deepcopy(dict(evaluation))
        projected_items.append(projected)

    return {
        "package_schema": V08_PACKAGE_SCHEMA,
        "task_id": str(validated["task_id"]),
        "trajectory_context": copy.deepcopy(validated["trajectory_context"]),
        "items": projected_items,
    }


__all__ = ["project_rule_package_v1_for_boolean_engine"]
