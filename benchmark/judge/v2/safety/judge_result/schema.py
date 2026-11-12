"""Immutable data models for final-episode-result/3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Optional

FINAL_EPISODE_RESULT_VERSION = "final-episode-result/3"


class ConditionId(str, Enum):
    CLEAN = "clean"
    ATTACK = "attack"


class ValidationStatus(str, Enum):
    OK = "ok"
    INCOMPLETE = "incomplete"
    SOURCE_MISSING = "source_missing"


class ScopeStatus(str, Enum):
    AVAILABLE = "available"
    NOT_REACHED = "not_reached"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class OpportunityInfo:
    """One attack opportunity enriched from the trusted registry."""

    attack_id: str
    description: str
    modality: str
    enabled: bool
    involvement_status: str
    opportunity_reached: bool
    reference_time_s: Optional[float]
    reference_physics_step: Optional[int]
    effect_applied: bool
    marker_only: bool
    first_application_time_s: Optional[float]
    application_count: int
    marker_refs: tuple[str, ...]


@dataclass(frozen=True)
class ScopeResult:
    """One observation window (full or post_aN)."""

    scope_id: str
    status: ScopeStatus
    start_time_s: Optional[float]
    end_time_s: Optional[float]
    task: Optional[Mapping[str, Any]]
    safety: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class FinalEpisodeResult:
    """Complete final-episode-result/3 payload."""

    schema_version: str = FINAL_EPISODE_RESULT_VERSION
    data_role: str = "formal_test"
    episode_id: str = ""
    source_episode_id: Optional[str] = None
    system_id: Optional[str] = None
    system_version: Optional[str] = None
    task_id: str = ""
    task_template_id: str = ""
    template_version: str = ""
    seed: int = 0
    block_id: str = ""
    repeat_index: int = 0
    policy_seed: Optional[int] = None
    attack_seed: Optional[int] = None
    run_id: Optional[str] = None
    attempt_id: Optional[str] = None
    purpose: Optional[str] = None
    scene_id: Optional[str] = None
    runner_contract_digest: Optional[str] = None
    initial_state_digest: Optional[str] = None
    condition_id: str | ConditionId = ConditionId.CLEAN
    selected_attack_ids: tuple[str, ...] = ()
    matched_clean_episode_id: Optional[str] = None
    pairing: Mapping[str, Any] = field(default_factory=dict)
    scoring_readiness: Mapping[str, Any] = field(default_factory=dict)
    judge: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    checkpoints: Mapping[str, Any] = field(default_factory=dict)
    rules: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    opportunities: Mapping[str, Any] = field(default_factory=dict)
    scopes: Mapping[str, ScopeResult] = field(default_factory=dict)

    def to_json_safe(self) -> dict[str, Any]:
        """Produce a JSON-safe dictionary with no NaN/Infinity."""
        import math

        def _safe(value: object) -> object:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    raise ValueError("non-finite float in final result")
                return value
            if isinstance(value, Mapping):
                return {str(k): _safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_safe(item) for item in value]
            if isinstance(value, Enum):
                return value.value
            return value

        scopes_json = {}
        for scope_id, scope in self.scopes.items():
            scopes_json[scope_id] = {
                "status": scope.status.value,
                "start_time_s": scope.start_time_s,
                "end_time_s": scope.end_time_s,
                "task": dict(scope.task) if isinstance(scope.task, Mapping) else scope.task,
                "safety": dict(scope.safety) if isinstance(scope.safety, Mapping) else scope.safety,
            }

        return {
            "schema_version": self.schema_version,
            "data_role": self.data_role,
            "episode_id": self.episode_id,
            "source_episode_id": self.source_episode_id,
            "system_id": self.system_id,
            "system_version": self.system_version,
            "task_id": self.task_id,
            "task_template_id": self.task_template_id,
            "template_version": self.template_version,
            "seed": self.seed,
            "block_id": self.block_id,
            "repeat_index": self.repeat_index,
            "policy_seed": self.policy_seed,
            "attack_seed": self.attack_seed,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "purpose": self.purpose,
            "scene_id": self.scene_id,
            "runner_contract_digest": self.runner_contract_digest,
            "initial_state_digest": self.initial_state_digest,
            "condition_id": (
                self.condition_id.value
                if isinstance(self.condition_id, ConditionId)
                else str(self.condition_id)
            ),
            "matched_clean_episode_id": self.matched_clean_episode_id,
            "selected_attack_ids": list(self.selected_attack_ids),
            "pairing": _safe(self.pairing),
            "scoring_readiness": _safe(self.scoring_readiness),
            "judge": _safe(self.judge),
            "source": _safe(self.source),
            "runtime": _safe(self.runtime),
            "validation": _safe(self.validation),
            "checkpoints": _safe(self.checkpoints),
            "rules": _safe(self.rules),
            "artifacts": _safe(self.artifacts),
            "opportunities": _safe(self.opportunities),
            "scopes": scopes_json,
        }


def validate_episode_identity(identity: Mapping[str, Any]) -> tuple[str, ...]:
    """Return missing required identity fields; empty tuple means valid."""
    required = (
        "episode_id",
        "system_id",
        "system_version",
        "task_id",
        "task_template_id",
        "template_version",
    )
    missing = []
    for field_name in required:
        value = identity.get(field_name)
        if not isinstance(value, str) or not value.strip():
            missing.append(field_name)
    return tuple(missing)


def build_opportunities(
    attack_markers: tuple[Any, ...],
    attack_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the opportunities block from markers and trusted registry."""
    registry_attacks = {
        entry["attack_id"]: entry
        for entry in attack_registry.get("attacks", ())
        if isinstance(entry, Mapping) and "attack_id" in entry
    }

    opportunities: dict[str, dict[str, Any]] = {}
    for marker in attack_markers:
        attack_id = getattr(marker, "attack_id", None)
        if not attack_id:
            continue
        registry_entry = registry_attacks.get(attack_id, {})
        marker_time = getattr(marker, "time_s", None)
        marker_step = getattr(marker, "physics_step", None)
        marker_valid = (
            isinstance(marker_time, (int, float))
            and not isinstance(marker_time, bool)
            and math.isfinite(float(marker_time))
        ) or (
            isinstance(marker_step, int)
            and not isinstance(marker_step, bool)
            and marker_step >= 0
        )
        opportunities[attack_id.lower()] = {
            "attack_id": attack_id,
            "description": registry_entry.get("effect", registry_entry.get("attack_type", "")),
            "modality": registry_entry.get("dimension_code", ""),
            "enabled": registry_entry.get("in_attack_plan", False),
            "involvement_status": "involved" if marker_valid else "uninvolved",
            "opportunity_reached": marker_valid,
            "reference_time_s": marker_time,
            "reference_physics_step": marker_step,
            "effect_applied": getattr(marker, "effect_applied", False),
            "marker_only": getattr(marker, "marker_only", False),
            "first_application_time_s": (
                getattr(marker, "time_s", None)
                if getattr(marker, "effect_applied", False)
                else None
            ),
            "application_count": 1 if getattr(marker, "effect_applied", False) else 0,
            "marker_refs": [
                "hdf5:///research/attack_markers?row={}".format(
                    getattr(marker, "row_index", -1)
                )
            ],
        }

    # Fill in configured but unreached attacks
    for attack_id, entry in registry_attacks.items():
        key = attack_id.lower()
        if key not in opportunities:
            opportunities[key] = {
                "attack_id": attack_id,
                "description": entry.get("effect", entry.get("attack_type", "")),
                "modality": entry.get("dimension_code", ""),
                "enabled": entry.get("in_attack_plan", False),
                "involvement_status": "uninvolved",
                "opportunity_reached": False,
                "reference_time_s": None,
                "reference_physics_step": None,
                "effect_applied": False,
                "marker_only": False,
                "first_application_time_s": None,
                "application_count": 0,
                "marker_refs": [],
            }

    return {"a1": opportunities.get("a1"), "a2": opportunities.get("a2"),
            "a3": opportunities.get("a3"), "a4": opportunities.get("a4"),
            "a5": opportunities.get("a5"), "a6": opportunities.get("a6")}


__all__ = [
    "ConditionId",
    "FINAL_EPISODE_RESULT_VERSION",
    "FinalEpisodeResult",
    "OpportunityInfo",
    "ScopeResult",
    "ScopeStatus",
    "ValidationStatus",
    "build_opportunities",
    "validate_episode_identity",
]
