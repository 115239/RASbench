"""Immutable result models for safety-constraint auditing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional


class ConstraintVerdict(str, Enum):
    """Externally visible verdict for one constraint in one scope."""

    VIOLATED = "violated"
    CLEAR = "clear"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class EvidencePoint:
    physics_step: int
    time_s: float
    value: Optional[Any] = None


@dataclass(frozen=True)
class AttackMarkerPoint:
    """One HDF5 attack marker, whether or not an effect was applied."""

    attack_id: str
    physics_step: int
    time_s: float
    row_index: int
    first_confirmed_after_observation_gap: bool = False
    effect_applied: bool = True
    marker_only: bool = False


# Compatibility for callers that imported the v1 name.  The object now models
# every marker and exposes ``effect_applied`` explicitly.
AppliedAttackPoint = AttackMarkerPoint


@dataclass(frozen=True)
class AttackRelation:
    """Objective timing relation between a violation and attack markers.

    This model deliberately does NOT perform causal attribution.
    ``preexisting_at_attack`` and ``started_after_attack`` are purely
    temporal observations.
    """

    marker_relation: str
    preexisting_at_attack: bool = False
    started_after_attack: bool = False
    attack_marker: Optional[AttackMarkerPoint] = None
    latency_s: Optional[float] = None
    latency_steps: Optional[int] = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))

    @property
    def classification(self) -> str:
        """Compatibility alias for the v1 in-process API."""
        return self.marker_relation

    @property
    def actual_attack_attribution(self) -> str:
        """Deprecated.  Use ``preexisting_at_attack`` and ``started_after_attack``
        instead of causal-attribution strings.  Always returns ``not_evaluated``.
        """
        return "not_evaluated"

    @property
    def attribution(self) -> str:
        """Deprecated compatibility alias."""
        return self.actual_attack_attribution

    @property
    def attack(self) -> Optional[AttackMarkerPoint]:
        """Compatibility alias for the v1 in-process API."""
        return self.attack_marker


@dataclass(frozen=True)
class ViolationWindow:
    type_id: str
    criterion: str
    objects: tuple[str, ...]
    start: EvidencePoint
    end: EvidencePoint
    end_reason: str
    first: EvidencePoint
    worst: EvidencePoint
    last_observed: EvidencePoint
    threshold: Any
    unit: str
    comparison: str
    row_index: int
    attack_relation: Optional[AttackRelation] = None

    @property
    def sort_key(self) -> tuple:
        return (
            self.start.physics_step,
            self.end.physics_step,
            self.criterion,
            self.objects,
            self.row_index,
        )


@dataclass(frozen=True)
class ConstraintTypeAudit:
    """Audit result for one safety type."""

    type_id: str
    verdict: ConstraintVerdict
    reason_codes: tuple[str, ...] = ()
    incomplete_observation: bool = False
    windows: tuple[ViolationWindow, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, str) or not self.type_id.strip():
            raise ValueError("constraint type_id must be non-empty text")
        if not isinstance(self.verdict, ConstraintVerdict):
            raise TypeError("constraint verdict must be a ConstraintVerdict")
        normalized_reasons = tuple(sorted(set(self.reason_codes)))
        object.__setattr__(self, "reason_codes", normalized_reasons)
        object.__setattr__(self, "windows", tuple(self.windows))
        if self.verdict is ConstraintVerdict.UNKNOWN and not normalized_reasons:
            raise ValueError("unknown constraint verdict requires a reason code")
        if self.verdict is not ConstraintVerdict.VIOLATED and self.windows:
            raise ValueError("only violated constraints may expose violation windows")


@dataclass(frozen=True)
class ScopeSafetyAudit:
    """Stable aggregation of all applicable constraint results in one scope."""

    scope_id: str
    status: str
    per_type: Mapping[str, str]
    violated_types: tuple[str, ...]
    unknown_types: tuple[str, ...]
    any_violation: Optional[bool]
    violation_type_count: Optional[int]
    type_results: Mapping[str, ConstraintTypeAudit]

    @classmethod
    def from_type_results(
        cls,
        scope_id: str,
        type_results: Mapping[str, ConstraintTypeAudit],
    ) -> ScopeSafetyAudit:
        ordered_ids = tuple(sorted(type_results))
        ordered_results = {type_id: type_results[type_id] for type_id in ordered_ids}
        for type_id, result in ordered_results.items():
            if result.type_id != type_id:
                raise ValueError(
                    f"constraint result key {type_id!r} does not match type_id {result.type_id!r}"
                )

        per_type = {
            type_id: result.verdict.value
            for type_id, result in ordered_results.items()
        }
        violated = tuple(
            type_id
            for type_id, result in ordered_results.items()
            if result.verdict is ConstraintVerdict.VIOLATED
        )
        unknown = tuple(
            type_id
            for type_id, result in ordered_results.items()
            if result.verdict is ConstraintVerdict.UNKNOWN
            or result.incomplete_observation
        )

        applicable = tuple(
            result
            for result in ordered_results.values()
            if result.verdict is not ConstraintVerdict.NOT_APPLICABLE
        )
        if not applicable:
            status = "not_applicable"
            any_violation = None
            violation_count = None
        elif unknown:
            status = "incomplete"
            any_violation = True if violated else None
            violation_count = None
        else:
            status = "ok"
            any_violation = bool(violated)
            violation_count = len(violated)

        return cls(
            scope_id=scope_id,
            status=status,
            per_type=MappingProxyType(per_type),
            violated_types=violated,
            unknown_types=unknown,
            any_violation=any_violation,
            violation_type_count=violation_count,
            type_results=MappingProxyType(ordered_results),
        )


__all__ = [
    "AppliedAttackPoint",
    "AttackMarkerPoint",
    "AttackRelation",
    "ConstraintTypeAudit",
    "ConstraintVerdict",
    "EvidencePoint",
    "ScopeSafetyAudit",
    "ViolationWindow",
]
