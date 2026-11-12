"""Pure task-configuration validation shared by the M20 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_operation_station(task_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an isolated station mapping with explicit null semantics."""

    value = task_config.get("operation_station")
    execution_mode = str(task_config.get("execution_mode", "navigation_only")).strip().lower()
    if value is None:
        if execution_mode not in {
            "navigation",
            "navigation_only",
            "manipulation",
            "manipulation_only",
            "grasp",
            "grasp_only",
        }:
            raise ValueError(
                "operation_station may be null only for navigation-only or "
                "target-driven manipulation-only tasks"
            )
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("operation_station must be an object or null")
    return dict(value)
