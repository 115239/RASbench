"""Parse and validate Rubric v0.5 with explicit score_groups."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SAFETY_RUBRIC_V05_SCHEMA = "robolab-safety-rubric/v0.5"


class RubricV05Error(ValueError):
    """Raised when rubric v0.5 configuration is malformed."""


@dataclass(frozen=True)
class ScoreGroup:
    """One configurable scoring group with all_of logic."""

    item_id: str
    description: str
    all_of: tuple[str, ...]
    max_units: int


def parse_score_groups(
    rubric: Mapping[str, Any],
) -> tuple[ScoreGroup, ...]:
    """Extract and validate score_groups from a v0.5 rubric.

    Returns validated score groups.  Raises RubricV05Error on any
    structural or logical problem.
    """

    if not isinstance(rubric, Mapping):
        raise RubricV05Error("rubric must be an object")

    rubric_schema = rubric.get("rubric_schema")
    if rubric_schema not in (SAFETY_RUBRIC_V05_SCHEMA, "robolab-safety-rubric/v0.4"):
        raise RubricV05Error(
            "rubric_schema must be {} (v0.4 accepted for compat)".format(
                SAFETY_RUBRIC_V05_SCHEMA
            )
        )

    execution = rubric.get("task_execution")
    if not isinstance(execution, Mapping):
        raise RubricV05Error("rubric.task_execution must be an object")

    # Collect all known score point IDs from subtasks
    known_ids: set[str] = set()
    subtasks = execution.get("subtasks", ())
    if isinstance(subtasks, list):
        for subtask in subtasks:
            if not isinstance(subtask, Mapping):
                continue
            for point in subtask.get("score_points", ()):
                if not isinstance(point, Mapping):
                    continue
                point_id = point.get("id")
                if isinstance(point_id, str) and point_id:
                    known_ids.add(point_id)

    # Parse score_groups
    raw_groups = execution.get("score_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise RubricV05Error(
            "rubric.task_execution.score_groups must be a non-empty array"
        )

    groups: list[ScoreGroup] = []
    seen_item_ids: set[str] = set()

    for index, raw in enumerate(raw_groups):
        field = "score_groups[{}]".format(index)
        if not isinstance(raw, Mapping):
            raise RubricV05Error("{} must be an object".format(field))

        item_id = _required_text(raw.get("item_id"), "{}.item_id".format(field))
        if item_id in seen_item_ids:
            raise RubricV05Error(
                "duplicate item_id in score_groups: {!r}".format(item_id)
            )
        seen_item_ids.add(item_id)

        description = _required_text(
            raw.get("description"), "{}.description".format(field)
        )

        all_of = raw.get("all_of")
        if not isinstance(all_of, list) or not all_of:
            raise RubricV05Error(
                "{}.all_of must be a non-empty array of score-point IDs".format(field)
            )
        if len(all_of) != len(set(all_of)):
            raise RubricV05Error(
                "{}.all_of must contain unique IDs".format(field)
            )

        normalized_all_of: list[str] = []
        for ref in all_of:
            if not isinstance(ref, str) or not ref:
                raise RubricV05Error(
                    "{}.all_of entries must be non-empty text".format(field)
                )
            if known_ids and ref not in known_ids:
                raise RubricV05Error(
                    "{}.all_of references unknown score-point ID: {!r}".format(
                        field, ref
                    )
                )
            normalized_all_of.append(ref)

        max_units = raw.get("max_units")
        if not isinstance(max_units, int) or max_units <= 0:
            raise RubricV05Error(
                "{}.max_units must be a positive integer".format(field)
            )

        groups.append(
            ScoreGroup(
                item_id=item_id,
                description=description,
                all_of=tuple(normalized_all_of),
                max_units=max_units,
            )
        )

    return tuple(groups)


def validate_score_groups(
    groups: Sequence[ScoreGroup],
) -> int:
    """Validate groups and return the dynamic total max_units.

    Does not require max_units to sum to exactly 100.
    """
    if not groups:
        raise RubricV05Error("at least one score group is required")
    return sum(group.max_units for group in groups)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RubricV05Error("{} must be non-empty text".format(field))
    return value


__all__ = [
    "SAFETY_RUBRIC_V05_SCHEMA",
    "RubricV05Error",
    "ScoreGroup",
    "parse_score_groups",
    "validate_score_groups",
]