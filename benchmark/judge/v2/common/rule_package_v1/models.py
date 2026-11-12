"""Small immutable-at-the-boundary models for Rule Package v1."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from .validation import RULE_PACKAGE_V1_SCHEMA, validate_rule_package_v1


@dataclass(frozen=True)
class RulePackageItemV1:
    id: str
    kind: str
    description: str
    evaluation: Mapping[str, Any]
    depends_on: Tuple[str, ...] = ()
    scoring_annotations: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "evaluation": copy.deepcopy(dict(self.evaluation)),
            "depends_on": list(self.depends_on),
            "scoring_annotations": copy.deepcopy(dict(self.scoring_annotations)),
        }


@dataclass(frozen=True)
class RulePackageV1:
    package_id: str
    task_id: str
    robot_role: str
    trajectory_context: Mapping[str, Any]
    items: Tuple[RulePackageItemV1, ...]
    package_schema: str = RULE_PACKAGE_V1_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        package = {
            "package_schema": self.package_schema,
            "package_id": self.package_id,
            "task_id": self.task_id,
            "robot_role": self.robot_role,
            "trajectory_context": copy.deepcopy(dict(self.trajectory_context)),
            "items": [item.to_dict() for item in self.items],
        }
        validate_rule_package_v1(package)
        return package


def parse_rule_package_v1(package: Mapping[str, Any]) -> RulePackageV1:
    """Validate and copy an untrusted mapping into the public v1 model."""

    validated = validate_rule_package_v1(package)
    items = tuple(
        RulePackageItemV1(
            id=str(item["id"]),
            kind=str(item["kind"]),
            description=str(item["description"]),
            evaluation=copy.deepcopy(dict(item["evaluation"])),
            depends_on=tuple(item["depends_on"]),
            scoring_annotations=copy.deepcopy(dict(item.get("scoring_annotations", {}))),
        )
        for item in validated["items"]
    )
    return RulePackageV1(
        package_schema=str(validated["package_schema"]),
        package_id=str(validated["package_id"]),
        task_id=str(validated["task_id"]),
        robot_role=str(validated["robot_role"]),
        trajectory_context=copy.deepcopy(dict(validated["trajectory_context"])),
        items=items,
    )


__all__ = ["RulePackageItemV1", "RulePackageV1", "parse_rule_package_v1"]
