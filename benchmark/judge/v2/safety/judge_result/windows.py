"""Build full and post_a1…post_a6 observation windows from attack markers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

FIXED_ATTACK_IDS = ("A1", "A2", "A3", "A4", "A5", "A6")


@dataclass(frozen=True)
class ScopeWindow:
    """One observation window (full or post_aN)."""

    scope_id: str
    status: str  # "available", "not_reached", "unavailable"
    start_time_s: Optional[float]
    end_time_s: Optional[float]
    start_physics_step: Optional[int]
    end_physics_step: Optional[int]
    attack_id: Optional[str]
    marker_time_s: Optional[float]
    marker_physics_step: Optional[int]
    validation_issues: tuple[str, ...] = ()


def build_scopes(
    attack_markers: Iterable[Any],
    runtime_start_s: float,
    runtime_end_s: float,
    runtime_start_step: int = 0,
    runtime_end_step: Optional[int] = None,
    *,
    attack_registry: Optional[Mapping[str, Any]] = None,
) -> dict[str, ScopeWindow]:
    """Build full + post_a1…post_a6 windows from markers and runtime bounds.

    Args:
        attack_markers: Iterable of objects with attack_id, time_s, physics_step,
            effect_applied, marker_only attributes.
        runtime_start_s: Episode start time in seconds.
        runtime_end_s: Episode end time in seconds.
        runtime_start_step: Episode start physics step.
        runtime_end_step: Episode end physics step (optional).
        attack_registry: Trusted attack registry for expected attack IDs.

    Returns:
        Dict mapping scope_id to ScopeWindow.  Always contains "full" and
        "post_a1" through "post_a6".
    """

    # Index markers by attack_id
    markers_by_id: dict[str, Any] = {}
    for marker in attack_markers:
        aid = getattr(marker, "attack_id", None)
        if aid:
            markers_by_id[aid] = marker

    # Validate runtime
    time_ok = (
        isinstance(runtime_start_s, (int, float))
        and isinstance(runtime_end_s, (int, float))
        and runtime_start_s >= 0
        and runtime_end_s > runtime_start_s
    )
    if not time_ok:
        full_issues = ("INVALID_RUNTIME_TIMEBASE",)
        scopes = {
            "full": ScopeWindow(
                scope_id="full",
                status="unavailable",
                start_time_s=None,
                end_time_s=None,
                start_physics_step=None,
                end_physics_step=None,
                attack_id=None,
                marker_time_s=None,
                marker_physics_step=None,
                validation_issues=full_issues,
            ),
        }
        for attack_id in FIXED_ATTACK_IDS:
            key = "post_{}".format(attack_id.lower())
            scopes[key] = ScopeWindow(
                scope_id=key,
                status="unavailable",
                start_time_s=None,
                end_time_s=None,
                start_physics_step=None,
                end_physics_step=None,
                attack_id=attack_id,
                marker_time_s=None,
                marker_physics_step=None,
                validation_issues=("INVALID_RUNTIME_TIMEBASE",),
            )
        return scopes

    full = ScopeWindow(
        scope_id="full",
        status="available",
        start_time_s=float(runtime_start_s),
        end_time_s=float(runtime_end_s),
        start_physics_step=runtime_start_step,
        end_physics_step=runtime_end_step,
        attack_id=None,
        marker_time_s=None,
        marker_physics_step=None,
    )

    scopes: dict[str, ScopeWindow] = {"full": full}

    expected_ids = _expected_attack_ids(attack_registry)

    for attack_id in FIXED_ATTACK_IDS:
        key = "post_{}".format(attack_id.lower())
        marker = markers_by_id.get(attack_id)

        if marker is None:
            scopes[key] = ScopeWindow(
                scope_id=key,
                status="not_reached",
                start_time_s=None,
                end_time_s=None,
                start_physics_step=None,
                end_physics_step=None,
                attack_id=attack_id,
                marker_time_s=None,
                marker_physics_step=None,
            )
            continue

        marker_time = getattr(marker, "time_s", None)
        marker_step = getattr(marker, "physics_step", None)

        if not isinstance(marker_time, (int, float)) or marker_time < 0:
            scopes[key] = ScopeWindow(
                scope_id=key,
                status="unavailable",
                start_time_s=None,
                end_time_s=None,
                start_physics_step=None,
                end_physics_step=None,
                attack_id=attack_id,
                marker_time_s=None,
                marker_physics_step=None,
                validation_issues=("INVALID_MARKER_TIMEBASE",),
            )
            continue

        if marker_time >= runtime_end_s:
            scopes[key] = ScopeWindow(
                scope_id=key,
                status="unavailable",
                start_time_s=None,
                end_time_s=None,
                start_physics_step=None,
                end_physics_step=None,
                attack_id=attack_id,
                marker_time_s=float(marker_time),
                marker_physics_step=marker_step,
                validation_issues=("MARKER_AFTER_EPISODE_END",),
            )
            continue

        scopes[key] = ScopeWindow(
            scope_id=key,
            status="available",
            start_time_s=float(marker_time),
            end_time_s=float(runtime_end_s),
            start_physics_step=marker_step,
            end_physics_step=runtime_end_step,
            attack_id=attack_id,
            marker_time_s=float(marker_time),
            marker_physics_step=marker_step,
        )

    return scopes


def derive_condition_id(attack_markers: Iterable[Any]) -> str:
    """Derive clean/attack from whether any marker had effect_applied=true."""
    for marker in attack_markers:
        if getattr(marker, "effect_applied", False):
            return "attack"
    return "clean"


def _expected_attack_ids(
    registry: Optional[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Extract expected attack IDs from the registry config section."""
    if not isinstance(registry, Mapping):
        return FIXED_ATTACK_IDS
    raw_list = registry.get("attack_registry")
    if isinstance(raw_list, list):
        ids = []
        for entry in raw_list:
            if isinstance(entry, Mapping):
                aid = entry.get("attack_id")
                if isinstance(aid, str) and aid:
                    ids.append(aid)
        if ids:
            return tuple(ids)
    return FIXED_ATTACK_IDS


__all__ = [
    "FIXED_ATTACK_IDS",
    "ScopeWindow",
    "build_scopes",
    "derive_condition_id",
]