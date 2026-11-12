"""Serialize safety-constraint audits into the episode result v3 shape."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Union

from .checker import SafetyConstraintAudit
from .schema import AttackMarkerPoint, AttackRelation, EvidencePoint, ViolationWindow

FINAL_EPISODE_RESULT_FORMAT = "final-episode-result/3"


def export_constraint_audit(audit: SafetyConstraintAudit) -> dict[str, Any]:
    """Build a JSON-safe result fragment without changing any score."""

    scopes = {}
    for scope_id, scope in audit.scopes.items():
        windows = {}
        evidence_refs = {}
        for type_id in scope.violated_types:
            type_windows = scope.type_results[type_id].windows
            if not type_windows:
                continue
            windows[type_id] = [_window_payload(item) for item in type_windows]
            evidence_refs[type_id] = {
                "hdf5_path": "/research/safety_events",
                "row_indices": [item.row_index for item in type_windows],
                "row_count": len(type_windows),
                "interval_count": len({_interval_key(item) for item in type_windows}),
            }

        reason_codes = set(audit.reason_codes)
        for result in scope.type_results.values():
            reason_codes.update(result.reason_codes)
        scopes[scope_id] = {
            "safety": {
                "status": scope.status,
                "per_type": dict(scope.per_type),
                "applicable_safety_types": sorted(scope.type_results.keys()),
                "violated_types": list(scope.violated_types),
                "unknown_types": list(scope.unknown_types),
                "any_violation": scope.any_violation,
                "violation_type_count": scope.violation_type_count,
                "violation_windows": windows,
                "evidence_refs": evidence_refs,
                "reason_codes": sorted(reason_codes),
                "judge": {
                    "component": "v2.safety.constraints",
                    "schema": audit.schema,
                    "constraint_set_id": audit.constraint_set_id,
                    "constraint_config_digest": audit.constraint_config_digest,
                    "score_effect": "none",
                },
            }
        }
    return {"format": FINAL_EPISODE_RESULT_FORMAT, "scopes": scopes}


def merge_constraint_audit(
    final_result: Mapping[str, Any],
    audit: SafetyConstraintAudit,
) -> dict[str, Any]:
    """Merge the exported safety nodes into one formal episode result."""

    if not isinstance(final_result, Mapping):
        raise TypeError("final_result must be a mapping")
    merged = deepcopy(dict(final_result))
    existing_scopes = merged.get("scopes", {})
    if not isinstance(existing_scopes, Mapping):
        raise TypeError("final_result.scopes must be a mapping")
    merged_scopes = deepcopy(dict(existing_scopes))
    fragment = export_constraint_audit(audit)
    for scope_id, exported_scope in fragment["scopes"].items():
        current_scope = merged_scopes.get(scope_id, {})
        if not isinstance(current_scope, Mapping):
            raise TypeError(
                f"final_result.scopes.{scope_id!s} must be a mapping"
            )
        updated_scope = deepcopy(dict(current_scope))
        updated_scope["safety"] = deepcopy(exported_scope["safety"])
        merged_scopes[scope_id] = updated_scope
    merged["schema_version"] = FINAL_EPISODE_RESULT_FORMAT
    merged["scopes"] = merged_scopes
    return merged


def save_final_episode_result(
    output_path: Union[str, PathLike],
    final_result: Mapping[str, Any],
    audit: SafetyConstraintAudit,
) -> Path:
    """Atomically save one formal episode JSON with its constraint audit."""

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = merge_constraint_audit(final_result, audit)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name!s}.",
            suffix=".tmp",
            dir=str(target.parent),
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
        raise
    return target


def _window_payload(window: ViolationWindow) -> dict[str, Any]:
    return {
        "type_id": window.type_id,
        "criterion": window.criterion,
        "objects": list(window.objects),
        "start": _point_payload(window.start),
        "end": {
            **_point_payload(window.end),
            "end_reason": window.end_reason,
        },
        "markers": {
            "first": _point_payload(window.first),
            "worst": _point_payload(window.worst),
            "last_observed": _point_payload(window.last_observed),
        },
        "threshold": {
            "value": window.threshold,
            "unit": window.unit,
            "comparison": window.comparison,
        },
        "source": {
            "hdf5_path": "/research/safety_events",
            "row_index": window.row_index,
        },
        "attack_relation": _attack_relation_payload(window),
    }


def _attack_relation_payload(window: ViolationWindow) -> dict[str, Any]:
    relation = window.attack_relation or AttackRelation(
        "unknown",
        preexisting_at_attack=False,
        started_after_attack=False,
        reason_codes=("ATTACK_RELATION_NOT_EVALUATED",),
    )
    return {
        "marker_relation": relation.marker_relation,
        "preexisting_at_attack": relation.preexisting_at_attack,
        "started_after_attack": relation.started_after_attack,
        "attack_marker": (
            _attack_payload(relation.attack_marker)
            if relation.attack_marker is not None
            else None
        ),
        "violation_start": _point_payload(window.start),
        "latency": (
            {
                "time_s": relation.latency_s,
                "physics_steps": relation.latency_steps,
            }
            if relation.latency_s is not None
            and relation.latency_steps is not None
            else None
        ),
        "reason_codes": list(relation.reason_codes),
    }


def _attack_payload(attack: AttackMarkerPoint) -> dict[str, Any]:
    return {
        "attack_id": attack.attack_id,
        "physics_step": attack.physics_step,
        "time_s": attack.time_s,
        "effect_applied": attack.effect_applied,
        "marker_only": attack.marker_only,
        "first_confirmed_after_observation_gap": (
            attack.first_confirmed_after_observation_gap
        ),
        "source": {
            "hdf5_path": "/research/attack_markers",
            "row_index": attack.row_index,
        },
    }


def _point_payload(point: EvidencePoint) -> dict[str, Any]:
    payload = {
        "physics_step": point.physics_step,
        "time_s": point.time_s,
    }
    if point.value is not None:
        payload["value"] = point.value
    return payload


def _interval_key(window: ViolationWindow) -> tuple:
    return (
        window.start.physics_step,
        window.end.physics_step,
        window.first.physics_step,
        window.worst.physics_step,
        window.last_observed.physics_step,
        window.end_reason,
    )


__all__ = [
    "FINAL_EPISODE_RESULT_FORMAT",
    "export_constraint_audit",
    "merge_constraint_audit",
    "save_final_episode_result",
]
