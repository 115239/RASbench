"""Frozen Safety-private binding to the Runner attack-evidence protocol."""

from __future__ import annotations

from dataclasses import dataclass

from robolab_safety_tianchi import attack_evidence as released_attack_evidence

from v2.common.tianchi import TianchiFrameworkError


ATTACK_EVIDENCE_SCHEMA_NAME = "robolab-m20-attack-evidence"
ATTACK_EVIDENCE_PRIMARY_VERSION = "3"
ATTACK_EVIDENCE_COMPATIBLE_VERSIONS = frozenset({"2", "3"})
ATTACK_EVIDENCE_BINDING_SCHEMA = "robolab-v2-safety-attack-binding/v1"


@dataclass(frozen=True)
class SafetyAttackProtocolBindingV1:
    """Auditable dependency receipt for the Safety attack Reader."""

    protocol_name: str
    primary_version: str
    compatible_versions: tuple[str, ...]
    reader_view_schema: str
    schema: str = ATTACK_EVIDENCE_BINDING_SCHEMA


def validate_safety_attack_protocol_dependency_v1() -> SafetyAttackProtocolBindingV1:
    """Fail before scoring when the packaged private Reader predates v3.

    V2 deliberately reuses the released Safety Reader instead of copying its
    protocol parser.  This capability gate prevents an old wheel from being
    silently packaged under the new V2 scoring entry.
    """

    actual_name = getattr(
        released_attack_evidence,
        "ATTACK_EVIDENCE_SCHEMA_NAME",
        None,
    )
    actual_versions = getattr(
        released_attack_evidence,
        "SUPPORTED_ATTACK_EVIDENCE_VERSIONS",
        frozenset(),
    )
    view_schema = getattr(
        released_attack_evidence,
        "ATTACK_EVIDENCE_VIEW_SCHEMA",
        None,
    )
    try:
        normalized_versions = frozenset(str(value) for value in actual_versions)
    except TypeError as error:
        raise TianchiFrameworkError(
            "Safety attack Reader exposes an invalid version capability"
        ) from error
    if actual_name != ATTACK_EVIDENCE_SCHEMA_NAME:
        raise TianchiFrameworkError(
            "Safety attack Reader has the wrong protocol identity"
        )
    if not ATTACK_EVIDENCE_COMPATIBLE_VERSIONS.issubset(normalized_versions):
        raise TianchiFrameworkError(
            "Safety attack Reader must support attack-evidence v2 and v3"
        )
    if not isinstance(view_schema, str) or not view_schema.strip():
        raise TianchiFrameworkError(
            "Safety attack Reader has no auditable view schema"
        )
    return SafetyAttackProtocolBindingV1(
        protocol_name=actual_name,
        primary_version=ATTACK_EVIDENCE_PRIMARY_VERSION,
        compatible_versions=tuple(sorted(normalized_versions)),
        reader_view_schema=view_schema,
    )


__all__ = [
    "ATTACK_EVIDENCE_BINDING_SCHEMA",
    "ATTACK_EVIDENCE_COMPATIBLE_VERSIONS",
    "ATTACK_EVIDENCE_PRIMARY_VERSION",
    "ATTACK_EVIDENCE_SCHEMA_NAME",
    "SafetyAttackProtocolBindingV1",
    "validate_safety_attack_protocol_dependency_v1",
]
