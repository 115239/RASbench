"""Generate a lightweight analysis_summary.json from a full episode result.

The summary drops detailed violation windows (objects, thresholds, markers,
attack_relation per event) and keeps only the per-type verdict table, task
scores, and episode metadata.  Designed for batch clean–attack pairing and
block-level statistics without parsing the full ~100 MB result.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

ANALYSIS_SUMMARY_SCHEMA = "robolab-v2-analysis-summary/v1"


def generate_summary(
    full_result: Mapping[str, Any],
    *,
    source_result_path: Optional[str] = None,
    rubric_version: str = "robolab-safety-rubric/v0.5",
    safety_config_id: str = "",
    safety_config_digest: str = "",
    judge_component: str = "v2.safety.scoring_modules.entry",
) -> dict[str, Any]:
    """Extract a compact analysis summary from a full episode result.

    Args:
        full_result: The complete ``final-episode-result/3`` dict.
        source_result_path: Path to the full result file for traceability.
        rubric_version: Rubric schema version used for scoring.
        safety_config_id: Constraint set ID (from frozen task_info).
        safety_config_digest: SHA-256 digest of the frozen safety config.
        judge_component: Judge component identifier.

    Returns:
        A JSON-safe dict suitable for saving as ``analysis_summary.json``.
    """

    # ── 1. Format & source ──────────────────────────────────────────
    summary: dict[str, Any] = {
        "schema_version": ANALYSIS_SUMMARY_SCHEMA,
        "source_result": source_result_path or "",
        "source_result_schema": full_result.get("schema_version", ""),
        "judge": {
            "component": judge_component,
        },
        "rules": dict(full_result.get("rules", {})) or {
            "rubric_schema": rubric_version,
            "safety_constraint_set_id": safety_config_id,
            "safety_constraint_config_digest": safety_config_digest,
        },
    }
    summary["rules"].pop("runner_judge_contract_validation", None)

    # ── 2. Identity & pairing ───────────────────────────────────────
    for key in (
        "episode_id", "question_id", "system_id", "system_version", "task_id",
        "task_template_id", "template_version",
        "seed", "block_id", "repeat_index", "policy_seed", "attack_seed",
        "run_id", "attempt_id", "purpose", "scene_id", "runner_contract_digest",
        "initial_state_digest", "source_episode_id",
        "condition_id", "matched_clean_episode_id",
        "selected_attack_ids",
    ):
        if key in full_result:
            summary[key] = full_result[key]
    for key in ("pairing", "scoring_readiness", "identity_sources", "artifacts"):
        if key in full_result:
            summary[key] = deepcopy(full_result[key])
    source = full_result.get("source", {})
    if isinstance(source, Mapping):
        summary["source"] = {
            key: deepcopy(source.get(key))
            for key in (
                "hdf5_sha256", "hdf5_file_name", "hdf5_uri", "rubric_sha256",
                "rule_package_sha256", "artifact_role", "exporter_version",
                "paper200_collection_schema", "runner_variant", "collection_declaration",
            ) if key in source
        }

    # ── 3. Validation ───────────────────────────────────────────────
    summary["validation"] = deepcopy(full_result.get("validation", {}))
    rules = full_result.get("rules", {})
    contract_validation = (
        rules.get("runner_judge_contract_validation")
        if isinstance(rules, Mapping) else None
    )
    if isinstance(contract_validation, Mapping):
        issues = contract_validation.get("issues", [])
        differences = contract_validation.get("differences", [])
        summary["runner_judge_contract_validation"] = {
            "status": contract_validation.get("status"),
            "issues": list(dict.fromkeys(issues)) if isinstance(issues, list) else [],
            "differences": deepcopy(differences) if isinstance(differences, list) else [],
        }

    # ── 4. Attack opportunities ─────────────────────────────────────
    opportunities = full_result.get("opportunities", {})
    summary_opps: dict[str, Any] = {}
    for key, opp in sorted(opportunities.items()):
        if not isinstance(opp, Mapping):
            continue
        summary_opps[key] = {
            field: deepcopy(opp.get(field))
            for field in (
                "attack_id", "description", "modality", "handler", "stage",
                "enabled", "recorded", "missing_fields", "triggered",
                "involvement_status", "opportunity_reached", "effect_activated",
                "effect_evidence_recorded", "effect_applied", "marker_only",
                "reference_time_s", "reference_physics_step",
                "first_application_time_s", "first_application_physics_step",
                "last_application_time_s", "end_time_s", "activation_time_s",
                "deactivation_time_s", "configured_duration_s",
                "effective_duration_s", "application_count",
                "changed_application_count", "status", "stop_reason",
                "registry_validation", "application_evidence_validation", "evidence_ref",
                "attack_contract_validation", "opportunity_evidence_validation",
                "temporal_evidence_validation",
            )
        }
        summary_opps[key]["attack_id"] = opp.get("attack_id", key)
    summary["opportunities"] = summary_opps

    # ── 5. Windows ──────────────────────────────────────────────────
    scopes = full_result.get("scopes", {})
    summary_scopes: dict[str, Any] = {}
    scope_ids = list(scopes.keys()) if isinstance(scopes, Mapping) else []
    if "full" in scope_ids:
        scope_ids = ["full"] + [scope_id for scope_id in scope_ids if scope_id != "full"]
    for scope_id in scope_ids:
        scope = scopes.get(scope_id, {})
        if not isinstance(scope, Mapping):
            continue

        # Extract attack_id from scope_id (post_a1 → A1)
        attack_id = scope.get("attack_id")
        if attack_id is None and scope_id.startswith("post_"):
            attack_id = scope_id[5:]

        window_entry: dict[str, Any] = {
            "scope_id": scope_id,
            "attack_id": attack_id,
            "status": scope.get("status"),
            "start_time_s": scope.get("start_time_s"),
            "end_time_s": scope.get("end_time_s"),
            "start_physics_step": scope.get("start_physics_step"),
            "end_physics_step": scope.get("end_physics_step"),
        }

        # ── 5a. Task score ──────────────────────────────────────────
        task = scope.get("task")
        if isinstance(task, Mapping):
            window_entry["task"] = {
                "status": task.get("status"),
                "score_100": task.get("score_100"),
                "awarded_units": task.get("awarded_units"),
                "max_units": task.get("max_units"),
                "completed_before_window": task.get("completed_before_window"),
                "completed_in_window": task.get("completed_in_window"),
            }
        else:
            window_entry["task"] = None

        # ── 5b. Safety status (per_type ONLY, no event details) ─────
        safety = scope.get("safety")
        if isinstance(safety, Mapping):
            window_entry["safety"] = {
                "status": safety.get("status"),
                "per_type": dict(safety.get("per_type", {})),
                "applicable_safety_types": safety.get("applicable_safety_types", []),
                "violated_types": safety.get("violated_types", []),
                "unknown_types": safety.get("unknown_types", []),
                "unknown_reason_codes_by_type": deepcopy(
                    safety.get("unknown_reason_codes_by_type", {})
                ),
                "any_violation": safety.get("any_violation"),
                "violation_type_count": safety.get("violation_type_count"),
                "reason_codes": safety.get("reason_codes", []),
                "observation_gaps": deepcopy(safety.get("observation_gaps", {})),
                "evidence_refs": deepcopy(safety.get("evidence_refs", {})),
                "evidence_contract_by_type": deepcopy(
                    safety.get("evidence_contract_by_type", {})
                ),
            }
        else:
            window_entry["safety"] = None

        summary_scopes[scope_id] = window_entry

    summary["scopes"] = summary_scopes
    full_scope = summary_scopes.get("full", {})
    full_safety = full_scope.get("safety") if isinstance(full_scope, Mapping) else None
    summary["unknown_safety_types"] = sorted(set(
        map(str, full_safety.get("unknown_types", []))
    )) if isinstance(full_safety, Mapping) else []

    # ── 6. Runtime ──────────────────────────────────────────────────
    runtime = full_result.get("runtime", {})
    if isinstance(runtime, Mapping):
        summary["runtime"] = deepcopy(runtime)

    return summary


def save_summary(
    output_path: Union[str, PathLike],
    summary: Mapping[str, Any],
) -> Path:
    """Atomically save the analysis summary JSON."""

    text = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_name = None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
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


__all__ = [
    "ANALYSIS_SUMMARY_SCHEMA",
    "generate_summary",
    "save_summary",
]
