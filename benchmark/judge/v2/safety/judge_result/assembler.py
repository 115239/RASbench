"""Unified final-episode-result/3 assembly and atomic save."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from v2.safety.judge_result.schema import (
    FINAL_EPISODE_RESULT_VERSION,
    validate_episode_identity,
)


class AssemblyError(ValueError):
    """Raised when the final result cannot be assembled."""


def assemble_final_result(
    *,
    identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    judge: Mapping[str, Any],
    source: Mapping[str, Any],
    opportunities: Mapping[str, Any],
    scopes: Mapping[str, Mapping[str, Any]],
    validation: Optional[Mapping[str, Any]] = None,
    condition_id: str = "clean",
    matched_clean_episode_id: Optional[str] = None,
    checkpoints: Optional[Mapping[str, Any]] = None,
    pairing: Optional[Mapping[str, Any]] = None,
    scoring_readiness: Optional[Mapping[str, Any]] = None,
    rules: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble a complete final-episode-result/3 payload.

    Required identity fields: episode_id, system_id, task_id,
    task_template_id, template_version.

    Args:
        identity: Episode identity fields.
        runtime: Runtime information (start_time_s, end_time_s, etc.).
        judge: Judge metadata.
        source: Source evidence metadata.
        opportunities: Built opportunities block.
        scopes: Per-scope results, each with "task" and "safety" keys.
        validation: Validation status and issues.
        condition_id: "clean" or "attack".
        matched_clean_episode_id: For attack episodes, the matching clean episode.
        checkpoints: Optional checkpoint references.

    Returns:
        JSON-safe dictionary ready for serialization.
    """

    missing = validate_episode_identity(identity)
    validation_dict = dict(validation) if isinstance(validation, Mapping) else {}
    issues = list(validation_dict.get("issues", ()))

    if missing and "SYSTEM_IDENTITY_NOT_PROVIDED" not in issues:
        issues.append(
            "MISSING_IDENTITY_FIELDS:{}".format(",".join(missing))
        )

    status = validation_dict.get("status", "ok")
    if missing and status != "invalid":
        status = "incomplete"

    assembled_scopes: dict[str, Any] = {}
    scope_ids = list(scopes.keys()) if isinstance(scopes, Mapping) else []
    if "full" not in scope_ids:
        scope_ids.insert(0, "full")
    else:
        scope_ids = ["full"] + [scope_id for scope_id in scope_ids if scope_id != "full"]
    for scope_id in scope_ids:
        scope_data = scopes.get(scope_id, {}) if isinstance(scopes, Mapping) else {}
        if not isinstance(scope_data, Mapping):
            scope_data = {}

        assembled_scopes[scope_id] = {
            "status": scope_data.get("status", "not_reached"),
            "start_time_s": scope_data.get("start_time_s"),
            "end_time_s": scope_data.get("end_time_s"),
            "start_physics_step": scope_data.get("start_physics_step"),
            "end_physics_step": scope_data.get("end_physics_step"),
            "attack_id": scope_data.get("attack_id"),
            "task": scope_data.get("task"),
            "safety": scope_data.get("safety"),
        }

    return {
        "schema_version": FINAL_EPISODE_RESULT_VERSION,
        "data_role": identity.get("data_role", "formal_test"),
        "episode_id": identity.get("episode_id", ""),
        "source_episode_id": identity.get("source_episode_id"),
        "question_id": identity.get("question_id", ""),
        "system_id": identity.get("system_id", ""),
        "system_version": identity.get("system_version", ""),
        "task_id": identity.get("task_id", ""),
        "task_template_id": identity.get("task_template_id", ""),
        "template_version": identity.get("template_version", ""),
        "seed": identity.get("seed", 0),
        "block_id": identity.get("block_id", ""),
        "repeat_index": identity.get("repeat_index", 0),
        "policy_seed": identity.get("policy_seed"),
        "attack_seed": identity.get("attack_seed"),
        "run_id": identity.get("run_id"),
        "attempt_id": identity.get("attempt_id"),
        "purpose": identity.get("purpose"),
        "scene_id": identity.get("scene_id"),
        "runner_contract_digest": identity.get("runner_contract_digest"),
        "initial_state_digest": identity.get("initial_state_digest"),
        "condition_id": condition_id,
        "selected_attack_ids": list(identity.get("selected_attack_ids", [])),
        "matched_clean_episode_id": matched_clean_episode_id,
        "pairing": dict(pairing) if isinstance(pairing, Mapping) else {},
        "scoring_readiness": (
            dict(scoring_readiness) if isinstance(scoring_readiness, Mapping) else {}
        ),
        "identity_sources": dict(identity.get("identity_sources", {})),
        "judge": dict(judge) if isinstance(judge, Mapping) else {},
        "source": dict(source) if isinstance(source, Mapping) else {},
        "runtime": dict(runtime) if isinstance(runtime, Mapping) else {},
        "validation": {
            **validation_dict,
            "status": status,
            "issues": list(issues),
        },
        "checkpoints": dict(checkpoints) if isinstance(checkpoints, Mapping) else {},
        "rules": dict(rules) if isinstance(rules, Mapping) else {},
        "artifacts": dict(artifacts) if isinstance(artifacts, Mapping) else {},
        "opportunities": dict(opportunities) if isinstance(opportunities, Mapping) else {},
        "scopes": assembled_scopes,
    }


def save_final_result(
    output_path: Union[str, PathLike],
    final_result: Mapping[str, Any],
) -> Path:
    """Atomically save the final episode result JSON.

    Uses temp file + os.replace for atomicity.  Ensures UTF-8, no NaN,
    and trailing newline.
    """

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Validate no NaN/Infinity before serializing
    text = json.dumps(
        final_result,
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
            prefix=".{}.".format(target.name),
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
    "AssemblyError",
    "assemble_final_result",
    "save_final_result",
]
