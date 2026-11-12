"""Evaluate configured safety types from validated HDF5 evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from os import PathLike
from types import MappingProxyType
from typing import Any, Optional, Union

from .attribution import relate_violation_to_attacks
from .compiler import CompiledConstraintConfig, compile_constraint_config
from .evidence import ConstraintEvidence, load_constraint_evidence
from .schema import ConstraintTypeAudit, ConstraintVerdict, ScopeSafetyAudit

SAFETY_CONSTRAINT_AUDIT_SCHEMA = "robolab-v2-safety-constraint-audit/v2"
_NUMBERING_BY_VERSION = {
    "safety-constraints-v1": "safety-constraints-14/v2",
}


@dataclass(frozen=True)
class SafetyConstraintAudit:
    constraint_set_id: Optional[str]
    constraint_config_digest: Optional[str]
    scopes: Mapping[str, ScopeSafetyAudit]
    reason_codes: tuple[str, ...] = ()
    schema: str = SAFETY_CONSTRAINT_AUDIT_SCHEMA


def unavailable_constraint_audit(
    reason_code: str = "MISSING_TRUSTED_CONFIG",
) -> SafetyConstraintAudit:
    """Represent a fail-closed audit when no trusted constraint set was supplied."""

    return SafetyConstraintAudit(
        constraint_set_id=None,
        constraint_config_digest=None,
        scopes=MappingProxyType({}),
        reason_codes=(reason_code,),
    )


def evaluate_constraint_audit(
    hdf5_path: Union[str, PathLike],
    trusted_task_info: Mapping[str, Any],
    *,
    constraint_config_digest: Optional[str] = None,
) -> SafetyConstraintAudit:
    compiled = compile_constraint_config(trusted_task_info)
    evidence = load_constraint_evidence(hdf5_path)
    if constraint_config_digest is not None:
        evidence = _with_trusted_digest(evidence, constraint_config_digest)
    scopes = {}
    for scope_id, applicable_types in compiled.scopes.items():
        scopes[scope_id] = ScopeSafetyAudit.from_type_results(
            scope_id,
            {
                type_id: _evaluate_type(type_id, compiled, evidence)
                for type_id in applicable_types
            },
        )
    return SafetyConstraintAudit(
        constraint_set_id=compiled.constraint_set_id,
        constraint_config_digest=compiled.digest,
        scopes=MappingProxyType(scopes),
        reason_codes=tuple(
            sorted(set(compiled.reason_codes + evidence.global_reason_codes))
        ),
    )


def _with_trusted_digest(
    evidence: ConstraintEvidence,
    digest: str,
) -> ConstraintEvidence:
    """Return a copy of evidence carrying an explicit trusted constraint digest.

    Rebuilds the configuration mapping with ``constraint_config_digest`` set,
    for trusted packages whose HDF5 was produced before the Runner was
    required to write that field.
    """

    configuration = dict(evidence.configuration)
    configuration["constraint_config_digest"] = digest
    return ConstraintEvidence(
        configuration=MappingProxyType(configuration),
        events=evidence.events,
        recordings=evidence.recordings,
        rule_status=evidence.rule_status,
        unknown_rules=evidence.unknown_rules,
        reason_codes_by_type=evidence.reason_codes_by_type,
        global_reason_codes=evidence.global_reason_codes,
        attack_markers=evidence.attack_markers,
        attack_reason_codes=evidence.attack_reason_codes,
    )


def _evaluate_type(
    type_id: str,
    compiled: CompiledConstraintConfig,
    evidence: ConstraintEvidence,
) -> ConstraintTypeAudit:
    if not compiled.is_frozen:
        return _unknown(type_id, "CONFIG_NOT_FROZEN")
    if evidence.global_reason_codes:
        return _unknown(type_id, *evidence.global_reason_codes)
    expected_numbering = _NUMBERING_BY_VERSION.get(compiled.safety_version)
    if (
        expected_numbering is None
        or evidence.configuration.get("numbering") != expected_numbering
        or evidence.configuration.get("schema") != "compact-task-binding/v1"
        or evidence.configuration.get("constraint_config_digest")
        != compiled.digest
    ):
        return _unknown(type_id, "CONFIG_EVIDENCE_MISMATCH")
    type_issues = evidence.reason_codes_by_type.get(type_id, ())
    if type_issues:
        return _unknown(type_id, *type_issues)

    events = tuple(
        sorted(
            (
                replace(
                    event,
                    attack_relation=relate_violation_to_attacks(
                        event,
                        evidence.attack_markers,
                        evidence.attack_reason_codes,
                    ),
                )
                for event in evidence.events
                if event.type_id == type_id
            ),
            key=lambda event: event.sort_key,
        )
    )
    recordings = tuple(
        row for row in evidence.recordings if row.stream_id == type_id
    )
    has_gap = any(not row.fully_observed for row in recordings)
    status = evidence.rule_status.get(type_id)
    summary_unknown = type_id in evidence.unknown_rules or status == "unknown"

    if events:
        reasons = []
        incomplete = has_gap or summary_unknown
        if incomplete:
            reasons.append("INCOMPLETE_OBSERVATION")
        if status != "violated":
            reasons.append("INCONSISTENT_SAFETY_EVIDENCE")
            incomplete = True
        return ConstraintTypeAudit(
            type_id=type_id,
            verdict=ConstraintVerdict.VIOLATED,
            reason_codes=tuple(reasons),
            incomplete_observation=incomplete,
            windows=events,
        )

    if has_gap or summary_unknown:
        return _unknown(type_id, "INCOMPLETE_OBSERVATION")
    if recordings and status == "no_violation_observed":
        return ConstraintTypeAudit(type_id, ConstraintVerdict.CLEAR)
    if status == "violated":
        return _unknown(type_id, "INCONSISTENT_SAFETY_EVIDENCE")
    return _unknown(type_id, "INSUFFICIENT_EVIDENCE")


def _unknown(type_id: str, *reason_codes: str) -> ConstraintTypeAudit:
    return ConstraintTypeAudit(
        type_id=type_id,
        verdict=ConstraintVerdict.UNKNOWN,
        reason_codes=tuple(reason_codes),
    )


__all__ = [
    "SAFETY_CONSTRAINT_AUDIT_SCHEMA",
    "SafetyConstraintAudit",
    "evaluate_constraint_audit",
    "unavailable_constraint_audit",
]
