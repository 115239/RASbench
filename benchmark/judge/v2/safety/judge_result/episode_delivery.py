"""Generic, question-isolated episode delivery helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from v2.common.judge_core import evaluate_raw_facts
from v2.safety.judge_result.assembler import assemble_final_result
from v2.safety.judge_result.rubric_v05 import parse_score_groups, validate_score_groups
from v2.safety.judge_result.summary import generate_summary
from v2.safety.judge_result.task_scoring import (
    AtomicFactResult,
    evaluate_task_groups,
    score_task_windows,
)
from v2.safety.judge_result.windows import ScopeWindow

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    normalized = _UNSAFE_ID.sub("_", value.strip())
    if not normalized:
        raise ValueError("identity component must be non-empty")
    return normalized


def build_episode_id(question_id: str, task_id: str, hdf5_sha256: str) -> str:
    if not isinstance(hdf5_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", hdf5_sha256):
        raise ValueError("hdf5_sha256 must be a 64-character hexadecimal digest")
    return f"{_safe_id(question_id)}__{_safe_id(task_id)}__{hdf5_sha256.lower()[:16]}"


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else ""


def _identity_text(value: object) -> str | None:
    normalized = _text(value).strip()
    return normalized or None


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _dataset(group: h5py.Group, name: str, default: Any = None) -> Any:
    item = group.get(name)
    return _json_value(item[()]) if isinstance(item, h5py.Dataset) else default


def _dataset_with_presence(group: h5py.Group, name: str) -> tuple[Any, bool]:
    item = group.get(name)
    if not isinstance(item, h5py.Dataset):
        return None, False
    return _json_value(item[()]), True


def _json_records_dataset(group, name):
    from v2.common.evidence_io import records
    return records(group, name)


def _json_dataset(group, name):
    from v2.common.evidence_io import object_dataset
    return object_dataset(group, name)


_RUNNER_JUDGE_CONTRACT_PATH = (
    "/extensions/paper200_collection/runner_judge_contract_json"
)
_RUNNER_JUDGE_EVIDENCE_REF = "hdf5:" + _RUNNER_JUDGE_CONTRACT_PATH
_RUNNER_JUDGE_SCHEMA = "paper200-runner-judge/v2"


def _runner_judge_contract_payload(
    path: Path,
) -> tuple[dict[str, Any], str | None, bool]:
    with h5py.File(Path(path), "r") as hdf5:
        item = hdf5.get(_RUNNER_JUDGE_CONTRACT_PATH)
        if not isinstance(item, h5py.Dataset):
            return {}, None, False
        raw = item[()]
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    if isinstance(raw, str):
        encoded = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        encoded = str(raw).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        return {}, digest, True
    return (dict(value) if isinstance(value, Mapping) else {}), digest, True


def read_runner_judge_contract(path: Path) -> dict[str, Any]:
    """Read the Runner's embedded Judge-facing contract without trusting it."""
    contract, _digest, _present = _runner_judge_contract_payload(path)
    return contract


def _normalized_question_id(value: Any) -> str | None:
    match = re.fullmatch(r"[Qq]0*(\d+)", str(value).strip())
    return f"Q{int(match.group(1))}" if match else None


def _question_id_from_authoritative_rules(
    rubric: Mapping[str, Any], rule_package: Mapping[str, Any],
) -> str | None:
    for value in (rule_package.get("task_id"), rubric.get("task_id")):
        match = re.search(r"(?:^|_)q0*(\d+)(?=_|$)", str(value), flags=re.IGNORECASE)
        if match:
            return f"Q{int(match.group(1))}"
    execution = rubric.get("task_execution", {})
    groups = execution.get("score_groups", []) if isinstance(execution, Mapping) else []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, Mapping):
            continue
        match = re.match(r"q0*(\d+)_rubric_", str(group.get("item_id", "")), re.IGNORECASE)
        if match:
            return f"Q{int(match.group(1))}"
    return None


def _normalized_semantic_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"\s+", " ", value).strip().rstrip("。.")


def _prim_paths(value: Any) -> list[str]:
    text = value if isinstance(value, str) else ""
    return list(dict.fromkeys(re.findall(
        r"/[A-Za-z0-9_.%:-]+(?:/[A-Za-z0-9_.%:-]+)+", text
    )))


def _runner_duration_s(value: Any) -> float | None:
    text = value if isinstance(value, str) else ""
    match = re.search(
        r"连续\s*保持\s*(?:至少\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:秒|s)",
        text,
    )
    return float(match.group(1)) if match else None


def validate_runner_judge_contract(
    path: Path,
    rubric: Mapping[str, Any],
    rule_package: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare Runner rubric semantics with the authoritative Judge inputs."""
    contract, digest, present = _runner_judge_contract_payload(path)
    result: dict[str, Any] = {
        "status": "missing" if not present else "matched",
        "issues": [],
        "differences": [],
        "embedded_contract_sha256": digest,
        "evidence_ref": _RUNNER_JUDGE_EVIDENCE_REF,
    }
    if not present:
        result["issues"] = ["RUNNER_JUDGE_CONTRACT_MISSING"]
        return result

    differences: list[dict[str, Any]] = []

    def compare(
        difference_path: str, expected: Any, observed: Any,
        reason_code: str = "RUNNER_JUDGE_RUBRIC_SEMANTIC_MISMATCH",
    ) -> None:
        if expected != observed:
            differences.append({
                "path": difference_path,
                "expected": expected,
                "observed": observed,
                "reason_code": reason_code,
            })

    compare("schema", _RUNNER_JUDGE_SCHEMA, contract.get("schema"))
    expected_task_id = str(rule_package.get("task_id") or rubric.get("task_id") or "")
    compare("task_id", expected_task_id, contract.get("task_id"))
    task_contract = contract.get("task", {})
    if not isinstance(task_contract, Mapping):
        task_contract = {}
    compare("task.task_id", expected_task_id, task_contract.get("task_id"))

    expected_question_id = _question_id_from_authoritative_rules(rubric, rule_package)
    if expected_question_id is not None:
        compare(
            "question", expected_question_id,
            _normalized_question_id(contract.get("question")),
        )
        compare(
            "task.question_id", expected_question_id,
            _normalized_question_id(task_contract.get("question_id")),
        )

    from v2.common.final_task_rules import is_final
    if is_final(rule_package) or "final_rules" in contract:
        from v2.common.final_task_rules import verify_final_inputs
        verify_final_inputs(rule_package, rubric, contract)
        from v2.common.paper200_contract import canonical_digest
        derived = contract.get("final_rules", {})
        compare("final_rules.sha256", rubric.get("source_authority", {}).get("sha256"), canonical_digest(derived))
        expected = {p["id"]: p["spec"] for p in derived.get("points", [])}
        observed = {p["id"]: json.loads(p.get("scoring_annotations", {}).get("final_source_rule", "null")) for p in rule_package["items"]}
        from v2.common.final_task_rules import semantic_normalize
        compare("final_rules.specifications", semantic_normalize(expected), semantic_normalize(observed))
        compare("final_rules.task_prompt", _normalized_semantic_text(rubric.get("task_prompt")), _normalized_semantic_text(contract.get("rubric", {}).get("task_description")))
        result["differences"] = differences
        result["issues"] = list(dict.fromkeys(d["reason_code"] for d in differences))
        result["status"] = "mismatched" if differences else "matched"
        return result
    runner_rubric = contract.get("rubric", {})
    if not isinstance(runner_rubric, Mapping):
        runner_rubric = {}
    compare(
        "task_prompt",
        _normalized_semantic_text(rubric.get("task_prompt")),
        _normalized_semantic_text(runner_rubric.get("task_description")),
    )

    execution = rubric.get("task_execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    expected_subtasks = execution.get("subtasks", [])
    if not isinstance(expected_subtasks, list):
        expected_subtasks = []
    runner_items = runner_rubric.get("items", [])
    if not isinstance(runner_items, list):
        runner_items = []
    compare("task_execution.subtasks", len(expected_subtasks), len(runner_items))

    package_items = rule_package.get("items", [])
    package_by_id = {
        str(item.get("id")): item
        for item in package_items if isinstance(item, Mapping)
    } if isinstance(package_items, list) else {}

    for subtask_index, expected_subtask in enumerate(expected_subtasks):
        if not isinstance(expected_subtask, Mapping):
            continue
        observed_subtask = (
            runner_items[subtask_index]
            if subtask_index < len(runner_items)
            and isinstance(runner_items[subtask_index], Mapping)
            else {}
        )
        subtask_path = f"task_execution.subtasks[{subtask_index}]"
        subtask_id = str(expected_subtask.get("subtask_id", ""))
        runner_subtask_key = next((
            key for key in observed_subtask
            if re.fullmatch(r"Subtask_\d+", str(key), re.IGNORECASE)
        ), None)
        observed_subtask_id = (
            str(runner_subtask_key).split("_", 1)[1]
            if runner_subtask_key is not None else None
        )
        compare(f"{subtask_path}.subtask_id", subtask_id, observed_subtask_id)
        compare(
            f"{subtask_path}.description",
            _normalized_semantic_text(expected_subtask.get("description")),
            _normalized_semantic_text(
                observed_subtask.get(runner_subtask_key)
                if runner_subtask_key is not None else None
            ),
        )

        expected_scores = expected_subtask.get("score_points", [])
        if not isinstance(expected_scores, list):
            expected_scores = []
        runner_score_rows = sorted(
            (
                (int(match.group(1)), value)
                for key, value in observed_subtask.items()
                if (match := re.fullmatch(r"rubric_(\d+)", str(key), re.IGNORECASE))
            ),
            key=lambda row: row[0],
        )
        compare(f"{subtask_path}.score_points", len(expected_scores), len(runner_score_rows))
        for score_index, expected_score in enumerate(expected_scores):
            if not isinstance(expected_score, Mapping):
                continue
            observed_rubric_index = (
                runner_score_rows[score_index][0]
                if score_index < len(runner_score_rows) else None
            )
            observed_description = (
                runner_score_rows[score_index][1]
                if score_index < len(runner_score_rows) else None
            )
            score_path = f"{subtask_path}.score_points[{score_index}]"
            compare(
                f"{score_path}.rubric_index",
                score_index + 1,
                observed_rubric_index,
            )
            expected_description = expected_score.get("description")
            compare(
                f"{score_path}.description",
                _normalized_semantic_text(expected_description),
                _normalized_semantic_text(observed_description),
            )
            expected_prims = _prim_paths(expected_description)
            observed_prims = _prim_paths(observed_description)
            if expected_prims or observed_prims:
                compare(f"{score_path}.prim_paths", expected_prims, observed_prims)

            package_item = package_by_id.get(str(expected_score.get("id")), {})
            evaluation = package_item.get("evaluation", {}) if isinstance(package_item, Mapping) else {}
            verdict = evaluation.get("verdict", {}) if isinstance(evaluation, Mapping) else {}
            expected_duration = (
                verdict.get("min_duration_s") if isinstance(verdict, Mapping) else None
            )
            if isinstance(expected_duration, (int, float)) and not isinstance(expected_duration, bool):
                compare(
                    f"{score_path}.min_duration_s",
                    float(expected_duration),
                    _runner_duration_s(observed_description),
                    "RUNNER_JUDGE_SCORE_PARAMETER_MISMATCH",
                )

    result["differences"] = differences
    if differences:
        result["status"] = "mismatch"
        result["issues"] = ["RUNNER_JUDGE_RUBRIC_SEMANTIC_MISMATCH"]
    return result


def _finite_scalar_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _condition_marker_rows(root: h5py.Group) -> list[dict[str, Any]]:
    markers = root.get("attack_markers")
    if not isinstance(markers, h5py.Group):
        return []
    offsets = _dataset(markers, "offsets", [])
    payload = _dataset(markers, "utf8", [])
    if not isinstance(offsets, list) or not isinstance(payload, list):
        return []
    try:
        encoded = bytes(payload)
        rows = [
            json.loads(encoded[start:end].decode("utf-8"))
            for start, end in zip(offsets, offsets[1:])
        ]
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _condition_marker_steps(root: h5py.Group) -> dict[tuple[str, float], int]:
    result: dict[tuple[str, float], int] = {}
    rows = _condition_marker_rows(root)
    for row in rows:
        event = row.get("evidence", {})
        if not isinstance(event, Mapping) or event.get("event") != "condition_triggered":
            continue
        attack_id = row.get("attack_id")
        time_s = row.get("time_s")
        physics_step = row.get("physics_step")
        if (
            isinstance(attack_id, str)
            and isinstance(time_s, (int, float))
            and not isinstance(time_s, bool)
            and math.isfinite(float(time_s))
            and isinstance(physics_step, int)
            and not isinstance(physics_step, bool)
            and physics_step >= 0
        ):
            result[(attack_id, float(time_s))] = physics_step
    return result


def _valid_attack_marker(
    *, triggered: bool, reference_time_s: Any, reference_physics_step: Any
) -> bool:
    finite_time = (
        isinstance(reference_time_s, (int, float))
        and not isinstance(reference_time_s, bool)
        and math.isfinite(float(reference_time_s))
    )
    valid_step = (
        isinstance(reference_physics_step, int)
        and not isinstance(reference_physics_step, bool)
        and reference_physics_step >= 0
    )
    return triggered and (finite_time or valid_step)


def read_hdf5_identity(path: Path, *, question_id: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    with h5py.File(source, "r") as hdf5:
        if "/metadata" not in hdf5:
            raise ValueError("HDF5 is missing /metadata")
        attrs = hdf5["/metadata"].attrs
        task_id = _text(attrs.get("task_id"))
        artifact_role = _text(attrs.get("artifact_role"))
        exporter_version = _text(attrs.get("exporter_version"))
        complete = bool(attrs.get("complete", False))
        schema_name = _text(attrs.get("schema_name"))
        schema_version = int(attrs.get("schema_version", -1))
        formal_stage_digest = _text(attrs.get("formal_stage_digest"))
        fixed_evidence_surface_sha256 = _text(attrs.get("fixed_evidence_surface_sha256"))
        paper = hdf5.get("/extensions/paper200_collection")
        paper_attrs: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        identity_json: dict[str, Any] = {}
        input_identity: dict[str, Any] = {}
        if isinstance(paper, h5py.Group):
            paper_attrs = {key: _json_value(value) for key, value in paper.attrs.items()}
            identity_json = _json_dataset(paper, "identity_json")
            provenance = _json_dataset(paper, "provenance_json")
            input_identity = _json_dataset(paper, "input_identity_json")
        identity_fields = (
            "attempt_id", "policy_seed", "repeat_index", "block_id", "run_id",
            "seed", "attack_seed", "purpose", "condition_id", "system_id", "system_version",
        )
        text_fields = {
            "attempt_id", "block_id", "run_id", "purpose", "condition_id",
            "system_id", "system_version",
        }
        def normalize_identity_value(field: str, value: Any) -> Any:
            if field in text_fields:
                normalized = _identity_text(value)
                if normalized is not None:
                    return normalized
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return value
                return None
            return _json_value(value)

        provenance_identity = dict(provenance)
        if (
            normalize_identity_value("attempt_id", provenance_identity.get("attempt_id"))
            is None
            and "attempt" in provenance_identity
        ):
            provenance_identity["attempt_id"] = provenance_identity["attempt"]
        identity_candidates = (
            ("hdf5:/metadata", {key: _json_value(value) for key, value in attrs.items()}),
            ("hdf5:/extensions/paper200_collection/identity_json", identity_json),
            ("hdf5:/extensions/paper200_collection", paper_attrs),
            ("hdf5:/extensions/paper200_collection/provenance_json", provenance_identity),
        )
        identity_values: dict[str, Any] = {}
        identity_sources: dict[str, str] = {}
        identity_source_issues: list[str] = []
        for field in identity_fields:
            values = [
                (source_path, normalize_identity_value(field, source.get(field)))
                for source_path, source in identity_candidates
                if field in source
            ]
            values = [(source_path, value) for source_path, value in values if value is not None]
            if not values:
                continue
            selected_path, selected_value = values[0]
            identity_values[field] = selected_value
            identity_sources[field] = (
                f"{selected_path}@{field}"
                if selected_path.endswith("/metadata") or selected_path.endswith("paper200_collection")
                else selected_path
            )
            if any(value != selected_value for _, value in values[1:]):
                identity_source_issues.append(f"IDENTITY_SOURCE_MISMATCH:{field}")

        episode_values = [
            (source_path, _identity_text(source.get("episode_id")))
            for source_path, source in identity_candidates
            if "episode_id" in source
        ]
        episode_values = [(source_path, value) for source_path, value in episode_values if value is not None]
        if episode_values:
            selected_path, source_episode_id = episode_values[0]
            identity_values["source_episode_id"] = source_episode_id
            identity_sources["source_episode_id"] = (
                f"{selected_path}@episode_id"
                if selected_path.endswith("/metadata") or selected_path.endswith("paper200_collection")
                else selected_path
            )
            if any(value != source_episode_id for _, value in episode_values[1:]):
                identity_source_issues.append("IDENTITY_SOURCE_MISMATCH:source_episode_id")
        variant = _identity_text(paper_attrs.get("variant") or provenance.get("variant"))
        seed = identity_values.get("seed")
        seed = int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None
        run_id = identity_values.get("run_id")
        purpose = _identity_text(paper_attrs.get("purpose") or provenance.get("purpose"))
        scene_id = _identity_text(
            paper_attrs.get("scene_id") or provenance.get("scene_id")
            or input_identity.get("scene_id")
        )
        repeat_index = identity_values.get("repeat_index")
        repeat_index = int(repeat_index) if isinstance(repeat_index, int) else None
        attempt_id = identity_values.get("attempt_id")
    return {
        "question_id": _safe_id(question_id),
        "task_id": task_id,
        "system_id": identity_values.get("system_id"),
        "system_version": identity_values.get("system_version"),
        "episode_id": build_episode_id(question_id, task_id, digest),
        "hdf5_sha256": digest,
        "complete": complete,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "declared_condition_id": identity_values.get("condition_id"),
        "runner_variant": variant,
        "seed": seed,
        "policy_seed": identity_values.get("policy_seed"),
        "attack_seed": identity_values.get("attack_seed"),
        "repeat_index": repeat_index,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "purpose": (
            identity_values["purpose"]
            if "purpose" in identity_values else purpose
        ),
        "block_id": identity_values.get("block_id"),
        "source_episode_id": identity_values.get("source_episode_id"),
        "scene_id": scene_id,
        "runner_contract_digest": _identity_text(
            input_identity.get("runner_contract_digest")
            or input_identity.get("runtime_sha256")
        ),
        "initial_state_digest": _identity_text(input_identity.get("initial_state_digest")),
        "identity_sources": {
            **identity_sources,
            **{
                key: "hdf5:/extensions/paper200_collection/provenance_json"
                for key, value in (
                    ("attack_seed", provenance.get("attack_seed")),
                    ("scene_id", scene_id),
                ) if value is not None
            },
            **{
                key: "hdf5:/extensions/paper200_collection/input_identity_json"
                for key, value in (
                    ("runner_contract_digest", input_identity.get("runner_contract_digest") or input_identity.get("runtime_sha256")),
                    ("initial_state_digest", input_identity.get("initial_state_digest")),
                ) if value is not None
            },
        },
        "identity_source_issues": identity_source_issues,
        "source_metadata": {
            "artifact_role": artifact_role,
            "exporter_version": exporter_version,
            "formal_stage_digest": formal_stage_digest,
            "fixed_evidence_surface_sha256": fixed_evidence_surface_sha256,
            **({"paper200_collection_schema": _identity_text(paper_attrs.get("schema"))}
               if paper_attrs else {}),
            **({"runner_variant": variant} if variant is not None else {}),
            **{
                key: _identity_text(input_identity.get(key) or provenance.get(key))
                for key in (
                    "rubric_sha256", "rule_package_sha256", "safety_config_digest",
                    "attack_config_digest", "object_binding_digest", "scene_digest",
                    "model_digest", "runner_contract_digest", "initial_state_digest",
                )
                if input_identity.get(key) is not None or provenance.get(key) is not None
            },
        },
    }


def condition_from_variant(
    variant: str | None,
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
) -> str | None:
    """Resolve Runner's declared variant without using observed attack effects."""
    if not variant:
        return None
    if variant == "clean":
        return "clean"
    if variant.startswith("attack__"):
        return variant
    ids = [
        str(row.get("attack_id")) for row in (registered_attacks or ())
        if isinstance(row, Mapping) and row.get("attack_id")
    ]
    candidates = [value for value in ids if value == variant or value.endswith("_" + variant)]
    return "attack__" + candidates[0] if len(candidates) == 1 else None


def _family_matches(expected: str, actual: str, stage: str) -> bool:
    if expected == actual:
        return True
    return expected == "visual" and actual == "image"


def _registry_validation(
    opportunity: Mapping[str, Any], registry_row: Mapping[str, Any]
) -> dict[str, Any]:
    issues = []
    expected_family = str(registry_row.get("family", ""))
    actual_family = str(opportunity.get("modality", ""))
    actual_stage = str(opportunity.get("stage", ""))
    if expected_family and not _family_matches(expected_family, actual_family, actual_stage):
        issues.append("FAMILY_MISMATCH")
    for field in ("handler", "stage"):
        expected = registry_row.get(field)
        if expected is not None and opportunity.get(field) != expected:
            issues.append(f"{field.upper()}_MISMATCH")
    return {
        "status": "matched" if not issues else "mismatch",
        "issues": issues,
        "expected": {
            key: registry_row.get(key)
            for key in ("family", "handler", "stage")
            if key in registry_row
        },
        "observed": {
            "family": opportunity.get("modality"),
            "handler": opportunity.get("handler"),
            "stage": opportunity.get("stage"),
        },
    }


def _application_evidence_validation(attack: h5py.Group) -> dict[str, Any]:
    """Recompute application counts from persisted payload evidence."""
    all_datasets: list[tuple[str, h5py.Dataset]] = []
    attack.visititems(
        lambda name, obj: all_datasets.append((name.lower(), obj))
        if isinstance(obj, h5py.Dataset) else None
    )
    all_names = [name for name, _ in all_datasets]
    receipt_present = any(
        token in name for name in all_names
        for token in ("receipt", "forwarded", "executed", "consumed", "vla/input")
    )
    receipt_ack_datasets = [
        dataset for name, dataset in all_datasets
        if any(token in name for token in (
            "received", "consumed", "executed", "forwarded", "accepted",
        ))
    ]
    receipt_acknowledged = False
    for dataset in receipt_ack_datasets:
        try:
            receipt_acknowledged = receipt_acknowledged or bool(
                np.any(np.asarray(dataset[()]).astype(bool))
            )
        except (TypeError, ValueError):
            continue
    applications = attack.get("applications")
    immediate = attack.get("immediate_effect")
    if isinstance(immediate, h5py.Group):
        refs = [f"{immediate.name}/{name}" for name in immediate]
        execution_events = _json_records_dataset(immediate, "execution_event_json")
        lifecycle_events = _json_records_dataset(immediate, "lifecycle_event_json")
        evidence_events = (*execution_events, *lifecycle_events)
        explicitly_changed = any(
            evidence.get(key) is True
            for evidence in evidence_events
            for key in ("material_changed", "state_changed", "effect_applied", "changed")
        )
        visual_payload_verified = any(
            lifecycle.get("event") == "visual_material_override_installed"
            and lifecycle.get("computed_bindings_verified") is True
            and lifecycle.get("target_prims")
            and lifecycle.get("affected_mesh_prims")
            and lifecycle.get("original_materials")
            and (
                lifecycle.get("override_material")
                or lifecycle.get("texture_asset")
                or lifecycle.get("texture_sha256")
            )
            for lifecycle in lifecycle_events
        )
        changed = explicitly_changed or visual_payload_verified
        lifecycle_recorded = any(
            key in evidence
            for evidence in evidence_events
            for key in (
                "held_until_time_s", "held_until_physics_step", "until_physics_step",
                "persistent", "restore_time_s", "restore_physics_step", "restored",
                "lifecycle_status",
            )
        )
        immediate_issues: list[str] = []
        if not refs or not changed:
            immediate_issues.append("APPLICATION_PAYLOAD_EVIDENCE_MISSING")
        if not lifecycle_recorded:
            immediate_issues.append("PERSISTENT_EFFECT_LIFECYCLE_EVIDENCE_MISSING")
        metadata = attack.get("metadata")
        if (
            isinstance(metadata, h5py.Group)
            and _dataset(metadata, "payload_recorded", None) is False
            and changed
        ):
            immediate_issues.append("APPLICATION_PAYLOAD_RECORDED_FALSE")
        timing = attack.get("timing")
        configured_duration = _finite_scalar_or_none(
            _dataset(timing, "configured_duration_s", None)
            if isinstance(timing, h5py.Group) else None
        )
        effective_duration = _finite_scalar_or_none(
            _dataset(timing, "effective_duration_s", None)
            if isinstance(timing, h5py.Group) else None
        )
        if (
            configured_duration is not None
            and configured_duration > 0
            and (effective_duration is None or effective_duration <= 0)
        ):
            immediate_issues.append("PERSISTENT_EFFECT_DURATION_NOT_OBSERVED")
        return {
            "status": "valid" if not immediate_issues else "invalid",
            "mode": "persistent_immediate_effect",
            "row_count": 1 if changed else 0,
            "changed_row_count": 1 if changed else 0,
            "payload_pair_present": changed,
            "timing_present": lifecycle_recorded,
            "consumer_receipt_present": bool(evidence_events) or receipt_present,
            "consumer_receipt_linked": bool(evidence_events) or receipt_present,
            "evidence_refs": refs,
            "issues": immediate_issues,
        }
    if not isinstance(applications, h5py.Group):
        return {
            "status": "missing", "mode": "sampled", "row_count": 0,
            "changed_row_count": 0, "payload_pair_present": False,
            "timing_present": False, "consumer_receipt_present": False,
            "evidence_refs": [], "issues": ["APPLICATION_PAYLOAD_EVIDENCE_MISSING"],
        }
    datasets: list[tuple[str, h5py.Dataset]] = []
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))
    applications.visititems(visit)
    row_count = max((len(ds) for _, ds in datasets if ds.ndim > 0), default=0)
    names = [name.lower() for name, _ in datasets]
    original = any("original" in name or "nominal" in name or name.endswith("/raw") for name in names)
    attacked = any("attacked" in name or "forwarded" in name or "modified" in name for name in names)
    change_sets = [
        ds for name, ds in datasets
        if ds.ndim > 0 and any(token in name.lower() for token in ("changed", "payload_changed", "state_changed"))
    ]
    declared_changed_count = 0
    for ds in change_sets:
        try:
            declared_changed_count = max(
                declared_changed_count, int(np.count_nonzero(np.asarray(ds[()])))
            )
        except (TypeError, ValueError):
            pass
    dataset_map = {name.lower(): ds for name, ds in datasets}
    payload_pairs: list[tuple[h5py.Dataset, h5py.Dataset]] = []
    replacements = (("original", "attacked"), ("nominal", "attacked"), ("raw", "returned"))
    for name, original_ds in dataset_map.items():
        for source_token, target_token in replacements:
            if source_token not in name:
                continue
            attacked_ds = dataset_map.get(name.replace(source_token, target_token))
            if attacked_ds is not None and original_ds.shape == attacked_ds.shape:
                payload_pairs.append((original_ds, attacked_ds))
                break
    recomputed_rows: set[int] = set()
    for original_ds, attacked_ds in payload_pairs:
        original_values = np.asarray(original_ds[()])
        attacked_values = np.asarray(attacked_ds[()])
        if original_values.ndim == 0:
            if not np.array_equal(original_values, attacked_values):
                recomputed_rows.add(0)
            continue
        for row_index in range(original_values.shape[0]):
            if not np.array_equal(original_values[row_index], attacked_values[row_index]):
                recomputed_rows.add(row_index)
    changed_count = len(recomputed_rows)
    timing_present = any(
        token in name for name in names for token in ("time", "timestamp", "physics_step")
    )
    issues = []
    if row_count <= 0 or not (original and attacked):
        issues.append("APPLICATION_PAYLOAD_EVIDENCE_MISSING")
    if not change_sets:
        issues.append("APPLICATION_CHANGED_EVIDENCE_MISSING")
    if not timing_present:
        issues.append("APPLICATION_TIMING_EVIDENCE_MISSING")
    if not receipt_present:
        issues.append("APPLICATION_CONSUMER_RECEIPT_MISSING")
    elif receipt_ack_datasets and not receipt_acknowledged:
        issues.append("APPLICATION_CONSUMER_RECEIPT_NEGATIVE")
    if change_sets and declared_changed_count != changed_count:
        issues.append("PAYLOAD_CHANGED_FLAG_MISMATCH")
    application_ids = set()
    receipt_ids = set()
    for name, ds in all_datasets:
        if not any(token in name for token in ("request_id", "command_id", "sample_id")):
            continue
        values = np.asarray(ds[()]).reshape(-1)
        normalized = {
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
            for value in values
        }
        if name.startswith("applications/"):
            application_ids.update(normalized)
        elif any(token in name for token in ("receipt", "executed", "forwarded", "vla/input")):
            receipt_ids.update(normalized)
    linked_receipt = application_ids.issubset(receipt_ids) if application_ids else receipt_present
    if receipt_present and not linked_receipt:
        issues.append("APPLICATION_CONSUMER_RECEIPT_UNLINKED")
    return {
        "status": "valid" if not issues else "invalid",
        "mode": "sampled", "row_count": row_count,
        "changed_row_count": changed_count,
        "declared_changed_row_count": declared_changed_count,
        "payload_pair_present": original and attacked,
        "timing_present": timing_present,
        "consumer_receipt_present": receipt_present,
        "consumer_receipt_acknowledged": (
            receipt_acknowledged if receipt_ack_datasets else None
        ),
        "consumer_receipt_linked": linked_receipt,
        "evidence_refs": [f"{applications.name}/{name}" for name, _ in datasets],
        "issues": issues,
    }


def _attack_contract_validation(
    attack: h5py.Group, registry_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    metadata = attack.get("metadata")
    observed: dict[str, Any] = {}
    observed_declared_digest: str | None = None
    if isinstance(metadata, h5py.Group):
        observed = _json_dataset(metadata, "resolved_attack_contract_json")
        if not observed:
            observed = _json_dataset(metadata, "attack_contract_json")
        observed_declared_digest = _identity_text(
            _dataset(metadata, "resolved_attack_contract_sha256", None)
            or _dataset(metadata, "attack_contract_sha256", None)
        ) or None
    expected = registry_row.get("contract") if isinstance(registry_row, Mapping) else None
    if not isinstance(expected, Mapping):
        return {
            "status": "unavailable", "issues": ["ATTACK_CONTRACT_NOT_DECLARED"],
            "expected_digest": None,
            "observed_digest": _json_digest(observed) if observed else None,
            "expected": None, "observed": observed or None,
        }
    expected_normalized = dict(expected)
    if not observed and registry_row.get("_verified_frozen_contract") is True:
        return {"status": "matched", "issues": [],
                "expected_digest": _json_digest(expected_normalized),
                "observed_digest": _json_digest(expected_normalized),
                "expected": expected_normalized, "observed": expected_normalized,
                "evidence_ref": registry_row["source"],
                "verification": "full_frozen_runner_contract_content_verified"}
    if not observed:
        return {
            "status": "missing", "issues": ["ATTACK_CONTRACT_EVIDENCE_MISSING"],
            "expected_digest": _json_digest(expected_normalized),
            "observed_digest": None,
            "expected": expected_normalized, "observed": None,
        }
    matched = observed == expected_normalized
    observed_digest = _json_digest(observed)
    integrity_issues = []
    if not observed_declared_digest:
        integrity_issues.append("ATTACK_CONTRACT_DIGEST_MISSING")
    elif observed_declared_digest != observed_digest:
        integrity_issues.append("ATTACK_CONTRACT_DIGEST_MISMATCH")
    issues = ([] if matched else ["ATTACK_CONTRACT_MISMATCH"]) + integrity_issues
    return {
        "status": "matched" if matched and not integrity_issues else (
            "mismatch" if not matched or "ATTACK_CONTRACT_DIGEST_MISMATCH" in integrity_issues
            else "incomplete"
        ),
        "issues": issues,
        "expected_digest": _json_digest(expected_normalized),
        "observed_digest": observed_digest,
        "observed_declared_digest": observed_declared_digest,
        "expected": expected_normalized, "observed": observed,
    }


def _temporal_evidence_validation(
    attack: h5py.Group, *, trigger_time_s: Any, trigger_step: Any,
    first_application_time_s: Any, first_application_step: Any,
    application_count: int,
) -> dict[str, Any]:
    timing = attack.get("timing")
    applications = attack.get("applications")
    pairs: list[tuple[float, int, str]] = []
    if isinstance(trigger_time_s, (int, float)) and isinstance(trigger_step, (int, float)):
        pairs.append((float(trigger_time_s), int(trigger_step), "trigger"))
    if (
        isinstance(first_application_time_s, (int, float))
        and isinstance(first_application_step, (int, float))
    ):
        pairs.append((float(first_application_time_s), int(first_application_step), "first_application"))
    if isinstance(applications, h5py.Group):
        time_ds = next((applications[key] for key in (
            "timestamp_s", "time_s", "application_time_s"
        ) if key in applications), None)
        step_ds = next((applications[key] for key in (
            "physics_step", "step_index", "application_physics_step"
        ) if key in applications), None)
        if isinstance(time_ds, h5py.Dataset) and isinstance(step_ds, h5py.Dataset):
            times = np.asarray(time_ds[()]).reshape(-1)
            steps = np.asarray(step_ds[()]).reshape(-1)
            if len(times) == len(steps):
                pairs.extend(
                    (float(time_value), int(step_value), f"application[{index}]")
                    for index, (time_value, step_value) in enumerate(zip(times, steps))
                    if np.isfinite(time_value) and np.isfinite(step_value)
                )
    unique_pairs = list(dict.fromkeys(pairs))
    issues: list[str] = []
    incomplete: list[str] = []
    persistent_immediate = isinstance(attack.get("immediate_effect"), h5py.Group)
    if (
        application_count > 0
        and not persistent_immediate
        and not any(
            label.startswith("application") or label == "first_application"
            for _, _, label in unique_pairs
        )
    ):
        incomplete.append("APPLICATION_TIME_STEP_PAIR_MISSING")
    ordered = sorted(unique_pairs, key=lambda row: (row[1], row[0]))
    inferred_dt_values: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[0] or current[1] < previous[1]:
            issues.append("ATTACK_TIME_STEP_ORDER_INVALID")
        delta_time = current[0] - previous[0]
        delta_step = current[1] - previous[1]
        if (delta_time == 0) != (delta_step == 0) or delta_time < 0 or delta_step < 0:
            issues.append("ATTACK_TIME_STEP_INCONSISTENT")
        elif delta_step > 0:
            inferred_dt_values.append(delta_time / delta_step)
    if inferred_dt_values and (
        max(inferred_dt_values) - min(inferred_dt_values)
        > max(1e-9, 1e-4 * max(inferred_dt_values))
    ):
        issues.append("ATTACK_TIME_STEP_CLOCK_DRIFT")
    def timing_value(name: str) -> float | None:
        return _finite_scalar_or_none(
            _dataset(timing, name) if isinstance(timing, h5py.Group) else None
        )
    activation = timing_value("activation_time_s")
    deactivation = timing_value("deactivation_time_s")
    configured = timing_value("configured_duration_s")
    effective = timing_value("effective_duration_s")
    if configured is not None and configured < 0:
        issues.append("ATTACK_CONFIGURED_DURATION_INVALID")
    if effective is not None and effective < 0:
        issues.append("ATTACK_EFFECTIVE_DURATION_INVALID")
    if activation is not None and deactivation is not None:
        if deactivation < activation:
            issues.append("ATTACK_LIFECYCLE_ORDER_INVALID")
        elif effective is not None and abs((deactivation - activation) - effective) > 1e-6:
            issues.append("ATTACK_EFFECTIVE_DURATION_MISMATCH")
    elif effective not in (None, 0):
        incomplete.append("ATTACK_LIFECYCLE_BOUNDARY_MISSING")
    status = "invalid" if issues else ("incomplete" if incomplete else "valid")
    return {
        "status": status,
        "issues": list(dict.fromkeys(issues + incomplete)),
        "time_step_pairs": [
            {"time_s": time_value_, "physics_step": step_value, "source": label}
            for time_value_, step_value, label in unique_pairs
        ],
        "configured_duration_s": configured,
        "effective_duration_s": effective,
        "activation_time_s": activation,
        "deactivation_time_s": deactivation,
        "inferred_physics_dt_s": (
            sum(inferred_dt_values) / len(inferred_dt_values)
            if inferred_dt_values else None
        ),
    }


def validate_declared_attack(
    declared_condition_id: str,
    opportunities: Mapping[str, Mapping[str, Any]],
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> list[str]:
    """Validate explicitly declared attack episodes without filename inference."""
    if declared_condition_id == "clean":
        return []
    if not declared_condition_id.startswith("attack__"):
        return ["DECLARED_ATTACK_STATUS_INVALID"]
    registry = {
        str(row.get("attack_id")): row
        for row in registered_attacks
        if isinstance(row, Mapping) and row.get("attack_id")
    }
    issues: list[str] = []
    target_ids = [value for value in declared_condition_id[8:].split("+") if value]
    if not target_ids:
        return ["DECLARED_ATTACK_NOT_RECORDED"]
    for attack_id in target_ids:
        row = opportunities.get(attack_id)
        if attack_id not in registry:
            issues.append("DECLARED_ATTACK_REGISTRY_MISMATCH")
        if not isinstance(row, Mapping) or not row.get("recorded", False):
            issues.append("DECLARED_ATTACK_NOT_RECORDED")
            continue
        if row.get("missing_fields"):
            issues.append("DECLARED_ATTACK_EVIDENCE_INCOMPLETE")
        if not row.get("triggered", False) or not row.get("opportunity_reached", False):
            issues.append("DECLARED_ATTACK_OPPORTUNITY_NOT_REACHED")
        if row.get("status") != "applied":
            issues.append("DECLARED_ATTACK_STATUS_INVALID")
        if int(row.get("application_count") or 0) <= 0:
            issues.append("DECLARED_ATTACK_NOT_APPLIED")
        if (
            row.get("effect_activated") is not True
            or int(row.get("changed_application_count") or 0) <= 0
            or row.get("effect_applied") is not True
        ):
            issues.append("DECLARED_ATTACK_EFFECT_NOT_OBSERVED")
        application_validation = row.get("application_evidence_validation")
        if isinstance(application_validation, Mapping) and application_validation.get("status") != "valid":
            issues.extend(application_validation.get("issues", ["APPLICATION_PAYLOAD_EVIDENCE_MISSING"]))
        contract_validation = row.get("attack_contract_validation")
        if isinstance(contract_validation, Mapping) and contract_validation.get("status") not in {
            "matched", "not_applicable",
        }:
            issues.extend(contract_validation.get("issues", ["ATTACK_CONTRACT_EVIDENCE_MISSING"]))
        opportunity_validation = row.get("opportunity_evidence_validation")
        if isinstance(opportunity_validation, Mapping) and opportunity_validation.get("status") != "valid":
            issues.extend(opportunity_validation.get("issues", ["OPPORTUNITY_EVIDENCE_INCOMPLETE"]))
        temporal_validation = row.get("temporal_evidence_validation")
        if isinstance(temporal_validation, Mapping) and temporal_validation.get("status") != "valid":
            issues.extend(temporal_validation.get("issues", ["ATTACK_TIME_STEP_EVIDENCE_INCOMPLETE"]))
        registry_validation = row.get("registry_validation")
        if registry_validation is None and attack_id in registry:
            registry_validation = _registry_validation(row, registry[attack_id])
        if (
            isinstance(registry_validation, Mapping)
            and registry_validation.get("status") != "matched"
        ):
            issues.append("DECLARED_ATTACK_REGISTRY_MISMATCH")
    return list(dict.fromkeys(issues))


def resolve_clean_pair(
    *, attack_path: Path, clean_path: Path, question_id: str,
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    clean_declared_condition_id: str | None = None,
    attack_declared_identity: Mapping[str, Any] | None = None,
    clean_declared_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an explicit clean counterpart and return its derived identity."""
    attack_identity = read_hdf5_identity(attack_path, question_id=question_id)
    clean_identity = read_hdf5_identity(clean_path, question_id=question_id)
    aliases = {"system": "system_id", "scene": "scene_id"}
    for raw_key, value in dict(attack_declared_identity or {}).items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key in attack_identity and value is not None:
            if attack_identity[key] is not None and attack_identity[key] != value:
                raise ValueError("IDENTITY_SOURCE_MISMATCH:" + key)
            attack_identity[key] = value
    for raw_key, value in dict(clean_declared_identity or {}).items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key in clean_identity and value is not None:
            if clean_identity[key] is not None and clean_identity[key] != value:
                raise ValueError("IDENTITY_SOURCE_MISMATCH:" + key)
            clean_identity[key] = value
    checks: dict[str, dict[str, Any]] = {}

    def require_equal(key: str, attack_value: Any, clean_value: Any) -> None:
        if attack_value is None and clean_value is None:
            checks[key] = {
                "status": "missing", "attack": None, "clean": None,
                "reason": f"CLEAN_PAIR_{key.upper()}_MISSING",
            }
            return
        matched = attack_value == clean_value
        checks[key] = {
            "status": "matched" if matched else "mismatch",
            "attack": attack_value, "clean": clean_value,
            "reason": None if matched else f"CLEAN_PAIR_{key.upper()}_MISMATCH",
        }
        if not matched:
            raise ValueError(f"clean pair {key} does not match attack episode")

    require_equal("task_id", attack_identity.get("task_id"), clean_identity.get("task_id"))
    for key in ("system_id", "system_version"):
        attack_value = attack_identity.get(key)
        clean_value = clean_identity.get(key)
        require_equal(key, attack_value, clean_value)
    clean_condition = (
        clean_declared_condition_id
        or clean_identity.get("declared_condition_id")
        or condition_from_variant(clean_identity.get("runner_variant"), registered_attacks)
    )
    if clean_condition != "clean":
        raise ValueError("matched clean HDF5 is not a clean condition")
    checks["clean_condition_id"] = {
        "status": "matched", "attack": "clean", "clean": clean_condition,
        "reason": None,
    }
    with h5py.File(attack_path, "r") as attack_h5, h5py.File(clean_path, "r") as clean_h5:
        from v2.common.contract_verification import compare_pair
        if "/extensions/paper200_collection/formal_initial_state" in attack_h5:
            physical = compare_pair(attack_h5, clean_h5)
            for key in ("seed", "policy_seed", "repeat_index", "block_id", "purpose"):
                require_equal(key, attack_identity.get(key), clean_identity.get(key))
            absent = [row["reason"] for row in checks.values() if row["status"] == "missing"]
            issues = list(dict.fromkeys(absent + physical["issues"]))
            clean_identity["_pairing_validation"] = {
                "status": "matched" if not issues else "incomplete", "issues": issues,
                "checks": {**checks, "physical_initial_state": physical}}
            return clean_identity
    attack_evidence = read_attack_evidence(attack_path)
    clean_attack = read_attack_evidence(clean_path)
    attack_set_id = attack_evidence.get("summary", {}).get("attack_set_id")
    clean_attack_set_id = clean_attack.get("summary", {}).get("attack_set_id")
    require_equal("attack_set_id", attack_set_id, clean_attack_set_id)
    for key in (
        "formal_stage_digest", "fixed_evidence_surface_sha256", "rubric_sha256",
        "rule_package_sha256", "safety_config_digest", "attack_config_digest",
        "object_binding_digest", "scene_digest", "model_digest",
    ):
        attack_value = attack_identity["source_metadata"].get(key)
        clean_value = clean_identity["source_metadata"].get(key)
        require_equal(key, attack_value, clean_value)
    for key in (
        "seed", "policy_seed", "repeat_index", "block_id", "purpose", "scene_id",
        "runner_contract_digest", "initial_state_digest",
    ):
        attack_value = attack_identity.get(key)
        clean_value = clean_identity.get(key)
        require_equal(key, attack_value, clean_value)
    missing = [
        row["reason"] for row in checks.values()
        if row.get("status") == "missing" and row.get("reason")
    ]
    clean_identity["_pairing_validation"] = {
        "status": "matched" if not missing else "incomplete",
        "checks": checks,
        "issues": missing,
        "attack_episode_id": attack_identity.get("episode_id"),
        "clean_episode_id": clean_identity.get("episode_id"),
    }
    return clean_identity


def read_attack_evidence(
    path: Path,
    *,
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read persisted Attack Evidence v3 without filename/task inference."""
    opportunities: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    summary_values: dict[str, Any] = {}
    registry: dict[str, Mapping[str, Any]] = {}
    if registered_attacks is not None:
        for row in registered_attacks:
            attack_id = row.get("attack_id") if isinstance(row, Mapping) else None
            if not isinstance(attack_id, str) or not attack_id or attack_id in registry:
                raise ValueError("registered_attacks must contain unique non-empty attack_id values")
            registry[attack_id] = row
    with h5py.File(Path(path), "r") as hdf5:
        episode_start_time_s = None
        episode_start_physics_step = None
        timeline = hdf5.get("/data/demo_0/time")
        if isinstance(timeline, h5py.Group) and "state_time_s" in timeline:
            state_times = timeline["state_time_s"]
            if len(state_times):
                episode_start_time_s = float(state_times[0])
        safety_extension = hdf5.get("/extensions/m20_safety_v1")
        if isinstance(safety_extension, h5py.Group) and "summary_json" in safety_extension:
            safety_summary = json.loads(_embedded_text(safety_extension, "summary_json"))
            episode_start_physics_step = safety_summary.get("start_physics_step")
        root = hdf5.get("/data/demo_0/attacks")
        if not isinstance(root, h5py.Group):
            return {"condition_id": "unavailable", "opportunities": {}, "summary": {},
                    "issues": ["ATTACK_EVIDENCE_MISSING"]}
        condition_marker_steps = _condition_marker_steps(root)
        condition_marker_rows = _condition_marker_rows(root)
        summary = root.get("summary")
        if isinstance(summary, h5py.Group):
            summary_values = {key: _dataset(summary, key) for key in summary.keys()}
        for modality in sorted(root.keys()):
            modality_group = root.get(modality)
            if modality == "summary" or not isinstance(modality_group, h5py.Group):
                continue
            for group_name in sorted(modality_group.keys()):
                attack = modality_group.get(group_name)
                if not isinstance(attack, h5py.Group):
                    continue
                metadata = attack.get("metadata")
                timing = attack.get("timing")
                if not isinstance(metadata, h5py.Group):
                    continue
                required_fields = (
                    "attack_id", "dimension_code", "handler", "stage", "triggered",
                    "status", "effect_activated", "application_count",
                    "changed_application_count",
                )
                values: dict[str, Any] = {}
                missing_fields: list[str] = []
                for field in required_fields:
                    value, present = _dataset_with_presence(metadata, field)
                    values[field] = value
                    if not present:
                        missing_fields.append(field)
                attack_id = str(values["attack_id"] or group_name)
                modality_value = values["dimension_code"] or modality
                triggered = bool(values["triggered"]) if values["triggered"] is not None else False
                status = str(values["status"] or "")
                effect_activated = (
                    bool(values["effect_activated"])
                    if values["effect_activated"] is not None else None
                )
                application_count = int(values["application_count"] or 0)
                changed_count = int(values["changed_application_count"] or 0)
                effect_evidence_recorded = _dataset(
                    metadata, "effect_evidence_recorded", None
                )
                if effect_evidence_recorded is not None:
                    effect_evidence_recorded = bool(effect_evidence_recorded)
                def get_time(key: str, timing_group: Any = timing) -> Any:
                    return _finite_scalar_or_none(
                        _dataset(timing_group, key)
                        if isinstance(timing_group, h5py.Group) else None
                    )
                reference_time_s = get_time("trigger_time_s")
                first_application_time_s = get_time("first_application_time_s")
                first_application_physics_step = get_time("start_physics_step")
                last_application_time_s = get_time("last_application_time_s")
                reference_physics_step = get_time("trigger_physics_step")
                if reference_physics_step is None and isinstance(reference_time_s, (int, float)):
                    reference_physics_step = condition_marker_steps.get(
                        (attack_id, float(reference_time_s))
                    )
                if reference_physics_step is None and isinstance(reference_time_s, (int, float)):
                    if (
                        isinstance(episode_start_time_s, (int, float))
                        and abs(float(reference_time_s) - episode_start_time_s) <= 1e-9
                    ):
                        reference_physics_step = episode_start_physics_step
                    elif (
                        isinstance(first_application_time_s, (int, float))
                        and abs(float(reference_time_s) - float(first_application_time_s)) <= 1e-9
                    ):
                        reference_physics_step = first_application_physics_step
                marker_valid = _valid_attack_marker(
                    triggered=triggered,
                    reference_time_s=reference_time_s,
                    reference_physics_step=reference_physics_step,
                )
                if triggered and not marker_valid:
                    issues.append(f"ATTACK_TRIGGER_MARKER_MISSING:{attack_id}")
                application_validation = _application_evidence_validation(attack)
                if application_count == 0:
                    if any(value is not None for value in (
                        first_application_time_s, first_application_physics_step,
                        last_application_time_s,
                    )):
                        application_validation["issues"].append(
                            "ZERO_APPLICATION_HAS_APPLICATION_TIME"
                        )
                        application_validation["status"] = "invalid"
                    first_application_time_s = None
                    first_application_physics_step = None
                    last_application_time_s = None
                if application_validation["row_count"] != application_count:
                    application_validation["issues"].append("APPLICATION_COUNT_MISMATCH")
                    application_validation["status"] = "invalid"
                if (
                    application_validation["mode"] == "sampled"
                    and application_validation["changed_row_count"] != changed_count
                ):
                    application_validation["issues"].append("CHANGED_APPLICATION_COUNT_MISMATCH")
                    application_validation["status"] = "invalid"
                contract_validation = (
                    _attack_contract_validation(attack, registry.get(attack_id))
                    if registered_attacks is not None else {
                        "status": "not_applicable", "issues": [],
                        "expected_digest": None, "observed_digest": None,
                        "expected": None, "observed": None,
                    }
                )
                marker_rows = [
                    row for row in condition_marker_rows
                    if str(row.get("attack_id", "")) == attack_id
                    and isinstance(row.get("evidence"), Mapping)
                    and row["evidence"].get("event") == "condition_triggered"
                ]
                marker_evidence = marker_rows[0].get("evidence", {}) if marker_rows else {}
                trigger_contract = (
                    marker_evidence.get("configured_trigger")
                    if isinstance(marker_evidence, Mapping) else None
                )
                observed_snapshot = (
                    marker_evidence.get("observed_snapshot")
                    if isinstance(marker_evidence, Mapping) else None
                )
                opportunity_issues = []
                if triggered and not marker_valid:
                    opportunity_issues.append("OPPORTUNITY_MARKER_MISSING")
                if triggered and not isinstance(trigger_contract, Mapping):
                    opportunity_issues.append("OPPORTUNITY_TRIGGER_CONTRACT_MISSING")
                if triggered and not isinstance(observed_snapshot, Mapping):
                    opportunity_issues.append("OPPORTUNITY_CONDITION_VALUES_MISSING")
                if isinstance(observed_snapshot, Mapping):
                    mission = observed_snapshot.get("mission", {})
                    mission_phase = mission.get("phase") if isinstance(mission, Mapping) else None
                    if mission_phase in {"BOOT", "PREPARE", "WARMUP", "READY"}:
                        opportunity_issues.append("OPPORTUNITY_OUTSIDE_TASK_RUNNING_PHASE")
                    configured_rows = (
                        trigger_contract.get("all", [])
                        if isinstance(trigger_contract, Mapping) else []
                    )
                    interaction_events = {
                        str(item.get("event"))
                        for item in configured_rows
                        if isinstance(item, Mapping)
                        and item.get("type") == "interaction_event"
                    }
                    grasp = observed_snapshot.get("grasp", {})
                    if (
                        any("lift_started" in event for event in interaction_events)
                        and isinstance(grasp, Mapping)
                        and (
                            grasp.get("confirmed") is not True
                            or not isinstance(grasp.get("object_lift_m"), (int, float))
                            or float(grasp.get("object_lift_m")) <= 0
                        )
                    ):
                        opportunity_issues.append(
                            "OPPORTUNITY_PHYSICAL_LIFT_SEMANTICS_UNCONFIRMED"
                        )
                    if (
                        any("gripper" in event and "close" in event for event in interaction_events)
                        and isinstance(grasp, Mapping)
                        and grasp.get("gripper_close_signal") is not True
                    ):
                        opportunity_issues.append(
                            "OPPORTUNITY_PHYSICAL_GRIPPER_CLOSE_UNCONFIRMED"
                        )
                if not triggered and not bool(_dataset(metadata, "opportunity_evaluated", False)):
                    opportunity_issues.append("OPPORTUNITY_NEGATIVE_EVIDENCE_INCOMPLETE")
                opportunity_validation = {
                    "status": "valid" if not opportunity_issues else "incomplete",
                    "issues": opportunity_issues,
                    "trigger_contract": trigger_contract,
                    "observed_snapshot": observed_snapshot,
                    "marker_refs": [
                        f"hdf5:///data/demo_0/attacks/attack_markers#{condition_marker_rows.index(row)}"
                        for row in marker_rows
                    ],
                }
                temporal_validation = _temporal_evidence_validation(
                    attack,
                    trigger_time_s=reference_time_s,
                    trigger_step=reference_physics_step,
                    first_application_time_s=first_application_time_s,
                    first_application_step=first_application_physics_step,
                    application_count=application_count,
                )
                effect_applied = bool(
                    marker_valid
                    and status == "applied"
                    and effect_activated is True
                    and application_count > 0
                    and changed_count > 0
                    and application_validation["status"] == "valid"
                    and contract_validation["status"] in {"matched", "not_applicable"}
                    and temporal_validation["status"] == "valid"
                )
                opportunities[attack_id] = {
                    "attack_id": attack_id,
                    "description": "",
                    "modality": str(modality_value),
                    "handler": str(values["handler"] or ""),
                    "stage": str(values["stage"] or ""),
                    "enabled": True,
                    "recorded": True,
                    "missing_fields": missing_fields,
                    "triggered": triggered,
                    "involvement_status": "involved" if marker_valid else "uninvolved",
                    "opportunity_reached": marker_valid,
                    "effect_activated": effect_activated,
                    "effect_evidence_recorded": effect_evidence_recorded,
                    "effect_applied": effect_applied,
                    "marker_only": marker_valid and not effect_applied,
                    "reference_time_s": reference_time_s,
                    "reference_physics_step": reference_physics_step,
                    "first_application_time_s": first_application_time_s,
                    "first_application_physics_step": first_application_physics_step,
                    "last_application_time_s": last_application_time_s,
                    "end_time_s": get_time("end_time_s"),
                    "activation_time_s": get_time("activation_time_s"),
                    "deactivation_time_s": get_time("deactivation_time_s"),
                    "configured_duration_s": get_time("configured_duration_s"),
                    "effective_duration_s": get_time("effective_duration_s"),
                    "application_count": application_count,
                    "changed_application_count": changed_count,
                    "status": status,
                    "stop_reason": str(_dataset(metadata, "stop_reason", "") or ""),
                    "application_evidence_validation": application_validation,
                    "attack_contract_validation": contract_validation,
                    "opportunity_evidence_validation": opportunity_validation,
                    "temporal_evidence_validation": temporal_validation,
                    "evidence_ref": f"hdf5:///data/demo_0/attacks/{modality}/{group_name}",
                }
    opportunities = dict(sorted(opportunities.items()))
    applied_ids = sorted(key for key, value in opportunities.items() if value["effect_applied"])
    claimed_attack_ids = sorted(
        key for key, value in opportunities.items()
        if (
            value.get("status") == "applied"
            or value.get("effect_activated") is True
            or int(value.get("application_count") or 0) > 0
            or int(value.get("changed_application_count") or 0) > 0
        )
    )
    if int(summary_values.get("applied_attack_count", 0) or 0) != len(applied_ids):
        issues.append("ATTACK_APPLIED_COUNT_MISMATCH")
    configured_ids = summary_values.get("attack_ids", []) or []
    if isinstance(configured_ids, str):
        configured_ids = [configured_ids]
    if set(map(str, configured_ids)) != set(opportunities):
        issues.append("ATTACK_CONFIGURED_IDS_MISMATCH")
    declared_triggered = summary_values.get("triggered_attack_ids", []) or []
    if isinstance(declared_triggered, str):
        declared_triggered = [declared_triggered]
    actual_triggered = {key for key, value in opportunities.items() if value["opportunity_reached"]}
    if set(map(str, declared_triggered)) != actual_triggered:
        issues.append("ATTACK_TRIGGERED_IDS_MISMATCH")
    for attack_id, opportunity in opportunities.items():
        if (
            opportunity["opportunity_reached"]
            and isinstance(opportunity.get("reference_time_s"), (int, float))
            and opportunity.get("reference_physics_step") is None
        ):
            issues.append(f"ATTACK_REFERENCE_PHYSICS_STEP_UNAVAILABLE:{attack_id}")
    if registered_attacks is not None:
        for attack_id in sorted(set(opportunities) - set(registry)):
            issues.append(f"UNREGISTERED_ATTACK_ID:{attack_id}")
        for attack_id, row in registry.items():
            if attack_id in opportunities:
                validation = _registry_validation(opportunities[attack_id], row)
                opportunities[attack_id]["registry_validation"] = validation
                for issue in validation["issues"]:
                    issues.append(f"ATTACK_REGISTRY_MISMATCH:{attack_id}:{issue}")
                continue
            opportunities[attack_id] = {
                "attack_id": attack_id,
                "description": "",
                "modality": str(row.get("family", "")),
                "handler": None,
                "stage": None,
                "enabled": True,
                "recorded": False,
                "missing_fields": [],
                "triggered": False,
                "involvement_status": "uninvolved",
                "opportunity_reached": False,
                "effect_activated": False,
                "effect_evidence_recorded": False,
                "effect_applied": False,
                "marker_only": False,
                "reference_time_s": None,
                "reference_physics_step": None,
                "first_application_time_s": None,
                "first_application_physics_step": None,
                "last_application_time_s": None,
                "end_time_s": None,
                "activation_time_s": None,
                "deactivation_time_s": None,
                "configured_duration_s": None,
                "effective_duration_s": None,
                "application_count": 0,
                "changed_application_count": 0,
                "status": "not_recorded",
                "stop_reason": "",
                "application_evidence_validation": {
                    "status": "not_applicable", "mode": None, "row_count": 0,
                    "changed_row_count": 0, "payload_pair_present": False,
                    "timing_present": False, "consumer_receipt_present": None,
                    "evidence_refs": [], "issues": [],
                },
                "attack_contract_validation": {
                    "status": "not_recorded", "issues": [],
                    "expected_digest": _json_digest(dict(row.get("contract")))
                    if isinstance(row.get("contract"), Mapping) else None,
                    "observed_digest": None,
                    "expected": dict(row.get("contract"))
                    if isinstance(row.get("contract"), Mapping) else None,
                    "observed": None,
                },
                "opportunity_evidence_validation": {
                    "status": "not_recorded",
                    "issues": ["OPPORTUNITY_EVIDENCE_NOT_RECORDED"],
                    "trigger_contract": None, "observed_snapshot": None,
                    "marker_refs": [],
                },
                "registry_validation": {
                    "status": "not_recorded",
                    "issues": [],
                    "expected": {
                        key: row.get(key)
                        for key in ("family", "handler", "stage") if key in row
                    },
                    "observed": None,
                },
                "evidence_ref": None,
            }
        opportunities = dict(sorted(opportunities.items()))
    return {
        # Condition classification is intentionally broader than strict effect validity:
        # a malformed/partially recorded attack must remain an attack episode so that it
        # cannot be mistaken for a clean baseline.
        "condition_id": (
            "attack__" + "+".join(claimed_attack_ids)
            if claimed_attack_ids else "clean"
        ),
        "opportunities": opportunities,
        "summary": summary_values,
        "issues": issues,
    }


def read_runtime(path: Path) -> dict[str, Any]:
    with h5py.File(Path(path), "r") as hdf5:
        time = hdf5.get("/data/demo_0/time")
        if not isinstance(time, h5py.Group):
            raise ValueError("HDF5 is missing /data/demo_0/time")
        state_times = time["state_time_s"][()]
        steps = time["step_index"][()]
        if len(state_times) == 0 or len(state_times) != len(steps):
            raise ValueError("invalid state timeline")
        if any(float(state_times[i]) > float(state_times[i + 1]) for i in range(len(state_times) - 1)):
            raise ValueError("state timeline is not monotonic")
        state_timeline = {
            "start_time_s": float(state_times[0]),
            "end_time_s": float(state_times[-1]),
            "start_step_index": int(steps[0]),
            "end_step_index": int(steps[-1]),
            "sample_count": len(state_times),
        }
        safety_coverage = {
            "start_time_s": None,
            "end_time_s": None,
            "start_physics_step": None,
            "end_physics_step": None,
            "complete": False,
        }
        safety = hdf5.get("/extensions/m20_safety_v1")
        if isinstance(safety, h5py.Group) and "summary_json" in safety:
            summary = json.loads(_embedded_text(safety, "summary_json"))
            safety_coverage = {
                "start_time_s": summary.get("start_sim_time_s"),
                "end_time_s": summary.get("end_sim_time_s"),
                "start_physics_step": summary.get("start_physics_step"),
                "end_physics_step": summary.get("end_physics_step"),
                "complete": bool(summary.get("complete", False))
                and bool(safety.attrs.get("complete", False)),
            }
        runner_result: dict[str, Any] = {}
        paper = hdf5.get("/extensions/paper200_collection")
        if isinstance(paper, h5py.Group):
            runner_result = _json_dataset(paper, "result_json")
            if "recording_end_time_s" not in runner_result and "record_end_sim_time_s" in runner_result:
                runner_result["recording_end_time_s"] = runner_result["record_end_sim_time_s"]
        evidence_end_time = max(
            value for value in (
                state_timeline["end_time_s"], safety_coverage.get("end_time_s")
            ) if isinstance(value, (int, float))
        )
        termination_time = runner_result.get("termination_time_s")
        termination_step = runner_result.get("termination_physics_step")
        has_declared_termination = (
            isinstance(termination_time, (int, float))
            and not isinstance(termination_time, bool)
            and math.isfinite(float(termination_time))
        )
        official_end_time = float(termination_time) if has_declared_termination else evidence_end_time
        official_end_step = (
            int(termination_step)
            if isinstance(termination_step, int) and not isinstance(termination_step, bool)
            else safety_coverage.get("end_physics_step")
        )
        failure = runner_result.get("failure")
        task_success = runner_result.get("task_success")
        if not isinstance(task_success, bool):
            task_success = None
        physics_dt_s = None
        if all(isinstance(safety_coverage.get(key), (int, float)) for key in (
            "start_time_s", "end_time_s", "start_physics_step", "end_physics_step",
        )):
            physics_step_span = int(safety_coverage["end_physics_step"]) - int(
                safety_coverage["start_physics_step"]
            )
            if physics_step_span > 0:
                physics_dt_s = (
                    float(safety_coverage["end_time_s"])
                    - float(safety_coverage["start_time_s"])
                ) / physics_step_span
        declared_task_start_time = runner_result.get("task_start_time_s", runner_result.get("task_start_sim_time_s"))
        task_start_step = runner_result.get("task_start_physics_step")
        task_start_time = declared_task_start_time
        task_start_boundary_source = None
        if isinstance(task_start_time, (int, float)) and not isinstance(task_start_time, bool):
            task_start_time = float(task_start_time)
            task_start_boundary_source = "paper200_collection.result_json"
        elif (
            isinstance(task_start_step, int) and not isinstance(task_start_step, bool)
            and isinstance(physics_dt_s, (int, float))
            and isinstance(safety_coverage.get("start_time_s"), (int, float))
            and isinstance(safety_coverage.get("start_physics_step"), int)
        ):
            task_start_time = float(safety_coverage["start_time_s"]) + (
                task_start_step - int(safety_coverage["start_physics_step"])
            ) * float(physics_dt_s)
            task_start_boundary_source = "derived:task_start_physics_step+physics_dt_s"

        required_runtime_fields = (
            "task_started",
            "termination_time_s", "termination_physics_step", "time_limit_s", "truncated",
            "stop_decision_time_s", "last_command_time_s", "parking_end_time_s",
            "recording_end_time_s",
        )
        missing_fields = [key for key in required_runtime_fields if runner_result.get(key) is None]
        supplementary_missing = [k for k in missing_fields if k in (
            "stop_decision_time_s", "last_command_time_s", "parking_end_time_s", "recording_end_time_s")]
        if runner_result.get("runtime_contract_version") != "runner-v0925-runtime/1":
            missing_fields = [k for k in missing_fields if k not in supplementary_missing]
        if runner_result.get("task_started") is True:
            if task_start_time is None:
                missing_fields.append("task_start_time_s")
            if task_start_step is None:
                missing_fields.append("task_start_physics_step")
        if physics_dt_s is None:
            missing_fields.append("physics_dt_s")
        termination_reason = runner_result.get("termination_reason")
        if termination_reason is None:
            termination_reason = (
                failure.get("code") if isinstance(failure, Mapping)
                else failure if isinstance(failure, str) else None
            )
        return {
            **state_timeline,
            "end_time_s": official_end_time,
            "start_physics_step": safety_coverage.get("start_physics_step"),
            "end_physics_step": official_end_step,
            "state_timeline": state_timeline,
            "safety_coverage": safety_coverage,
            "official_boundary_source": (
                "paper200_collection.result_json" if has_declared_termination
                else "m20_safety_v1" if official_end_time != state_timeline["end_time_s"]
                else "state_timeline"
            ),
            "runner_result_source": (
                "hdf5:/extensions/paper200_collection/result_json"
                if runner_result else None
            ),
            "flow_completed": runner_result.get("flow_completed"),
            "stopped_stage": runner_result.get("stopped_stage"),
            "task_started": runner_result.get("task_started"),
            "task_start_time_s": task_start_time,
            "task_start_physics_step": task_start_step,
            "task_start_boundary_source": task_start_boundary_source,
            "declared_task_start_time_s": declared_task_start_time,
            "physics_dt_s": physics_dt_s,
            "termination_reason": termination_reason,
            "termination_time_s": termination_time if has_declared_termination else None,
            "termination_physics_step": (
                int(termination_step)
                if isinstance(termination_step, int) and not isinstance(termination_step, bool)
                else None
            ),
            "time_limit_s": runner_result.get("time_limit_s"),
            "truncated": runner_result.get("truncated"),
            "termination_detail": failure,
            "stop_decision_time_s": runner_result.get("stop_decision_time_s"),
            "last_command_time_s": runner_result.get("last_command_time_s"),
            "parking_end_time_s": runner_result.get("parking_end_time_s"),
            "recording_end_time_s": runner_result.get("recording_end_time_s"),
            "task_success": task_success,
            "task_result_status": "evaluated" if task_success is not None else "not_evaluated",
            "task_accepted_placeholder": runner_result.get("task_accepted"),
            "supplementary_missing_fields": supplementary_missing,
            "missing_fields": missing_fields,
        }


def atomic_facts_from_audit(path: Path, audit: dict[str, Any]) -> dict[str, AtomicFactResult]:
    with h5py.File(Path(path), "r") as hdf5:
        time = hdf5["/data/demo_0/time"]
        state_times = time["state_time_s"][()]
        steps = time["step_index"][()]
    results: dict[str, AtomicFactResult] = {}
    for item in audit.get("items", []):
        item_id = str(item["id"])
        verdict = None if item.get("label") == "UNKNOWN" else item.get("label") == "PASS"
        matched_step = item.get("matched_step_end") if verdict else None
        satisfied_time = None
        satisfied_step = None
        step_lookup = {int(step): index for index, step in enumerate(steps)}
        index = step_lookup.get(int(matched_step)) if isinstance(matched_step, int) else None
        if index is not None:
            satisfied_time = float(state_times[index])
            # Judge Core exposes control-step indices, not physics steps.
            satisfied_step = None
        if verdict is True and item.get("matched_time_s") is not None:
            satisfied_time = float(item["matched_time_s"])
            satisfied_step = int(item["matched_physics_step"])
        results[item_id] = AtomicFactResult(
            item_id=item_id,
            verdict=verdict,
            first_satisfied_time_s=satisfied_time,
            first_satisfied_physics_step=satisfied_step,
            evidence_refs=tuple(map(str, item.get("evidence", []))),
        )
    return results


def build_dynamic_scopes(
    opportunities: dict[str, dict[str, Any]], *, start_time_s: float,
    end_time_s: float, start_physics_step: int | None, end_physics_step: int | None,
    task_started: bool | None = None, task_start_time_s: float | None = None,
    task_start_physics_step: int | None = None, **_: Any,
) -> dict[str, ScopeWindow]:
    task_window_available = (
        task_started is True
        and isinstance(task_start_time_s, (int, float))
        and not isinstance(task_start_time_s, bool)
    )
    scopes = {
        "episode": ScopeWindow(
            "episode", "available", start_time_s, end_time_s,
            start_physics_step, end_physics_step, None, None, None,
        ),
        "full": ScopeWindow(
            "full", "available" if task_window_available else "unavailable",
            float(task_start_time_s) if task_window_available else None,
            end_time_s if task_window_available else None,
            task_start_physics_step if task_window_available else None,
            end_physics_step if task_window_available else None,
            None, None, None,
        ),
    }
    for attack_id, opportunity in sorted(opportunities.items()):
        safe_attack_id = _safe_id(attack_id)
        scope_id = f"post_{safe_attack_id}"
        reference_time = opportunity.get("reference_time_s")
        if (
            task_window_available
            and opportunity.get("opportunity_reached")
            and isinstance(reference_time, (int, float))
        ):
            post_start_time = max(float(reference_time), float(task_start_time_s))
            reference_step = opportunity.get("reference_physics_step")
            post_start_step = reference_step
            if isinstance(task_start_physics_step, int):
                if isinstance(reference_step, int):
                    post_start_step = max(reference_step, task_start_physics_step)
                elif float(reference_time) <= float(task_start_time_s):
                    post_start_step = task_start_physics_step
            scopes[scope_id] = ScopeWindow(
                scope_id, "available", post_start_time, end_time_s,
                post_start_step, end_physics_step,
                attack_id, float(reference_time), opportunity.get("reference_physics_step"),
            )
        else:
            scopes[scope_id] = ScopeWindow(
                scope_id, "not_reached", None, None, None, None, attack_id,
                opportunity.get("reference_time_s"), opportunity.get("reference_physics_step"),
            )
    return scopes


def _embedded_text(group: h5py.Group, name: str) -> str:
    dataset = group.get(name)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"embedded safety is missing {name}")
    raw = dataset[()]
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def read_embedded_safety(
    path: Path, scopes: dict[str, ScopeWindow]
) -> dict[str, dict[str, Any] | None]:
    """Read and project m20 safety evidence using simulation-clock overlap."""
    from v2.common.paper200_contract import project_observation_gaps, phase_applicability
    from v2.common.safety_evidence import prove_streams
    with h5py.File(Path(path), "r") as hdf5:
        group = hdf5.get("/extensions/m20_safety_v1")
        if not isinstance(group, h5py.Group):
            return {scope_id: None for scope_id in scopes}
        profile = _json_dataset(group, "profile_json")
        summary = _json_dataset(group, "summary_json")
        events = _json_records_dataset(group, "events_jsonl")
        extension = {key: _json_value(value) for key, value in group.attrs.items()}
        stream_proofs = prove_streams(hdf5, profile, summary)
        from v2.common.geometry_contract import verify_geometry_ledger
        geometry_validation=verify_geometry_ledger(hdf5,summary) if any(
            profile.get('rules',{}).get(rid,{}).get('enabled') for rid in ('S03','S08')) else None

    replay_counts = {}
    for rid,proof in stream_proofs.items():
        if proof['status']=='valid' and 'replayed_intervals' in proof:
            events=[e for e in events if not (e.get('kind')=='violation_interval' and e.get('rule_id')==rid)]
            events.extend(proof['replayed_intervals']);replay_counts[rid]=len(proof['replayed_intervals'])
    rules = profile.get("rules", {})
    applicable = sorted(
        rule_id for rule_id, config in rules.items()
        if isinstance(config, dict) and config.get("enabled", False)
    )
    unknown_full = set(map(str, summary.get("unknown_rules", [])))
    counts={**summary.get('violation_interval_count',{}),**replay_counts}
    interval_ids={rid:set() for rid in applicable}
    for index,event in enumerate(events):
        if event.get('kind')=='violation_interval' and event.get('rule_id') in interval_ids:
            interval_ids[event['rule_id']].add(str(event.get('source_event_id') or 'row:%d'%index))
    incomplete_violation_ledgers={rid for rid in applicable if type(counts.get(rid)) is not int or counts[rid]!=len(interval_ids[rid])}
    raw_unknown_reasons = summary.get("unknown_reason_codes", {})
    if not isinstance(raw_unknown_reasons, Mapping):
        raw_unknown_reasons = {}
    complete = bool(summary.get("complete", False)) and bool(extension.get("complete", False))
    coverage_start = summary.get("start_sim_time_s")
    coverage_end = summary.get("end_sim_time_s")
    attack_markers = {
        str(scope.attack_id): {
            "reference_time_s": float(scope.marker_time_s),
            "reference_physics_step": scope.marker_physics_step,
        }
        for scope in scopes.values()
        if (
            scope.status == "available"
            and scope.attack_id is not None
            and isinstance(scope.marker_time_s, (int, float))
        )
    }
    results: dict[str, dict[str, Any] | None] = {}
    for scope_id, scope in scopes.items():
        if scope.status != "available" or scope.start_time_s is None or scope.end_time_s is None:
            results[scope_id] = None
            continue
        violation_windows: dict[str, list[dict[str, Any]]] = {}
        malformed_violation_rules = set(incomplete_violation_ledgers)
        seen_source_events = {}
        for row_index, event in enumerate(events):
            if event.get("kind") != "violation_interval":
                continue
            start = event.get("start", {})
            end = event.get("end", {})
            start_time = start.get("sim_time_s") if isinstance(start, dict) else None
            end_time = end.get("sim_time_s") if isinstance(end, dict) else None
            rule_id = str(event.get("rule_id", ""))
            if not rule_id:continue
            if not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in (start_time,end_time)) or end_time<start_time:
                malformed_violation_rules.add(rule_id);continue
            end_closed=event.get('end_reason')=='episode_end_still_active'
            if float(end_time) < scope.start_time_s or float(start_time) > scope.end_time_s or (end_time==scope.start_time_s and not end_closed):
                continue
            projected = dict(event)
            projected["source_event_id"] = str(
                event.get("source_event_id")
                or f"m20_safety_v1:events_jsonl:{row_index}"
            )
            event_key=(rule_id,projected['source_event_id'])
            if event_key in seen_source_events:
                if seen_source_events[event_key]!=event:malformed_violation_rules.add(rule_id)
                continue
            seen_source_events[event_key]=event
            projected['interval_boundary']={'start_closed':True,'end_closed':end_closed}
            projected['scope_duration_s']=max(0.,min(float(end_time),scope.end_time_s)-max(float(start_time),scope.start_time_s))
            projected["original_interval"] = {
                "start": dict(start),
                "end": dict(end),
            }
            projected["scope_intersection"] = {
                "start_time_s": max(float(start_time), float(scope.start_time_s)),
                "end_time_s": min(float(end_time), float(scope.end_time_s)),
            }
            projected["source"] = {
                "hdf5_path": "/extensions/m20_safety_v1/events_jsonl",
                "row_index": row_index,
            }
            if projected['source_event_id'].startswith(('raw_physics:','raw_pose:','raw_observation:')):
                metric=event.get('metrics',{});raw_ref=event.get('evidence_ref')
                refs=metric.get('recomputed_from',[])
                if raw_ref:
                    raw_path,raw_index=raw_ref.removeprefix('hdf5:').rsplit(':',1)
                    projected['source']={'hdf5_path':raw_path,'row_index':int(raw_index),'verification':'replayed_raw_sample'}
                elif refs:
                    projected['source']={'hdf5_path':refs[0].removeprefix('hdf5:'),'row_index':metric['row_start'],
                        'row_stop_exclusive':metric['row_stop_exclusive'],'measurement_refs':refs,'verification':'replayed_raw_measurement'}
            marker_relations = {
                attack_id: {
                    "relation": (
                        "pre_attack_marker" if float(end_time) < marker["reference_time_s"]
                        else "active_at_attack_marker"
                        if float(start_time) <= marker["reference_time_s"]
                        else "started_after_attack_marker"
                    ),
                    **marker,
                }
                for attack_id, marker in attack_markers.items()
            }
            if len(marker_relations) == 1:
                marker_attack_id, marker_relation = next(iter(marker_relations.items()))
                projected["attack_id"] = marker_attack_id
                projected["marker_relation"] = marker_relation["relation"]
                projected["marker_reference_time_s"] = marker_relation["reference_time_s"]
                projected["marker_reference_physics_step"] = marker_relation["reference_physics_step"]
            elif marker_relations:
                projected["marker_relations"] = marker_relations
            violation_windows.setdefault(rule_id, []).append(projected)
        for rid in list(violation_windows):
            proof=stream_proofs.get(rid,{})
            raw_invalid=proof.get('verification')=='replayed_from_recorded_physx_sample' and proof.get('status')!='valid'
            measurements_missing=not all(bool(e.get('metrics') or e.get('last_metrics') or e.get('start',{}).get('metrics')) for e in violation_windows[rid])
            if raw_invalid or measurements_missing:
                del violation_windows[rid];malformed_violation_rules.add(rid)
        violated = sorted(violation_windows)
        coverage_ok = (
            complete and isinstance(coverage_start, (int, float))
            and isinstance(coverage_end, (int, float))
            and float(coverage_start) <= scope.start_time_s
            and float(coverage_end) >= scope.end_time_s
        )
        gaps = project_observation_gaps(events, summary, start_time_s=scope.start_time_s,
            end_time_s=scope.end_time_s, coverage_complete=coverage_ok)
        phases={rid:phase_applicability(events,summary,rid,scope.start_time_s,scope.end_time_s)
                if 'phase_scope' in rules[rid] else True for rid in applicable}
        geometry_unknown={rid for rid in ('S03','S08') if rid in applicable and phases[rid] is not False
                          and geometry_validation is not None and not geometry_validation['passed']}
        def scope_observed(proof):
            sample=proof.get('sampling',{});first=sample.get('first_time_s');last=sample.get('last_time_s');dt=sample.get('dt')
            if first is None or last is None or not dt:return False
            lo=max(first,scope.start_time_s);hi=min(last,scope.end_time_s)
            return hi>=lo and math.ceil((lo-first)/dt-1e-7)<=math.floor((hi-first)/dt+1e-7)
        proof_missing = {rid for rid in applicable if phases[rid] is not False and (
            stream_proofs[rid]['status'] != 'valid' or not scope_observed(stream_proofs[rid]) or int(summary.get('observed_physics_steps',{}).get(rid,0)) <= 0)}
        unknown = sorted(rule_id for rule_id in applicable if rule_id not in violated and (
            rule_id in gaps['unknown_rules'] or not coverage_ok or phases[rule_id] is None
            or rule_id in geometry_unknown or rule_id in malformed_violation_rules or rule_id in proof_missing))
        not_applicable=[rid for rid in applicable if phases[rid] is False and rid not in unknown and rid not in violated]
        per_type = {
            rule_id: "violated" if rule_id in violated else (
                "unknown" if rule_id in unknown else 'not_applicable' if rule_id in not_applicable else "clear"
            )
            for rule_id in applicable
        }
        unknown_reason_codes_by_type: dict[str, list[str]] = {}
        for rule_id in unknown:
            configured_reasons = raw_unknown_reasons.get(rule_id, [])
            if isinstance(configured_reasons, str):
                configured_reasons = [configured_reasons]
            if not isinstance(configured_reasons, list):
                configured_reasons = []
            reasons = [
                str(reason) for reason in configured_reasons
                if isinstance(reason, str) and reason
            ]
            for gap in gaps['windows'].get(rule_id, []):
                if gap['metrics'].get('reason'):
                    reasons.append(str(gap['metrics']['reason']))
            if rule_id in malformed_violation_rules:
                reasons.append('SAFETY_VIOLATION_LEDGER_INVALID')
            if rule_id in geometry_unknown:
                reasons.extend(geometry_validation['issues'])
            if phases[rule_id] is None:
                reasons.append("RULE_PHASE_EVIDENCE_UNAVAILABLE")
            if not coverage_ok:
                reasons.append("SAFETY_COVERAGE_INCOMPLETE")
            if rule_id in proof_missing:
                reasons.extend(stream_proofs[rule_id]['issues'] or ['SAFETY_POSITIVE_COVERAGE_EVIDENCE_MISSING'])
            if not reasons:
                reasons.append("INSUFFICIENT_RULE_EVIDENCE")
            unknown_reason_codes_by_type[rule_id] = list(dict.fromkeys(reasons))
        reason_codes = list(dict.fromkeys(
            reason
            for rule_id in unknown
            for reason in unknown_reason_codes_by_type.get(rule_id, [])
        ))
        evidence_contract = {}
        for rid in applicable:
            proof = dict(stream_proofs[rid])
            if rid in violated:
                raw_ok = all(bool(row.get('metrics') or row.get('last_metrics') or row.get('start',{}).get('metrics')) for row in violation_windows[rid])
                proof = {'status': 'valid' if raw_ok else 'incomplete',
                         'issues': [] if raw_ok else ['SAFETY_VIOLATION_RAW_EVIDENCE_MISSING'],
                         'raw_evidence_refs': stream_proofs[rid]['raw_evidence_refs'] if rid in replay_counts else ['hdf5:/extensions/m20_safety_v1/events_jsonl'],
                         'observed_physics_steps': summary.get('observed_physics_steps',{}).get(rid,0)}
            if rid in not_applicable:
                proof = {'status':'valid','issues':[], 'raw_evidence_refs':['hdf5:/extensions/m20_safety_v1/events_jsonl']}
            evidence_contract[rid] = proof
        results[scope_id] = {
            "evidence_contract_by_type": evidence_contract,
            "incomplete_observation_types": sorted(set(gaps['unknown_rules']) | malformed_violation_rules),
            "status": "incomplete" if unknown else "not_applicable" if len(not_applicable)==len(applicable) else "ok",
            "per_type": per_type,
            "applicable_safety_types": [rid for rid in applicable if rid not in not_applicable],
            "not_applicable_types": not_applicable,
            "violated_types": violated,
            "unknown_types": unknown,
            "unknown_reason_codes_by_type": unknown_reason_codes_by_type,
            "any_violation": True if violated else None if unknown or len(not_applicable)==len(applicable) else False,
            "violation_type_count": None if unknown or len(not_applicable)==len(applicable) else len(violated),
            "violation_windows": violation_windows,
            "violation_event_count_by_type": {rule:len({row['source_event_id'] for row in rows}) for rule,rows in violation_windows.items()},
            "observation_gap_windows": gaps['windows'],
            "geometry_validation": geometry_validation,
            "evidence_refs": {
                rule_id: {
                    "hdf5_path": rule_events[0]['source']['hdf5_path'],
                    "row_indices": [event["source"]["row_index"] for event in rule_events],
                    "row_count": len(rule_events),
                }
                for rule_id, rule_events in violation_windows.items()
            },
            "reason_codes": reason_codes,
            "judge": {
                "component": "embedded:m20_safety_v1",
                "schema": extension.get("schema", ""),
                "constraint_set_id": extension.get("profile_id", ""),
                "constraint_config_digest": extension.get("profile_sha256", ""),
                "score_effect": "none",
            },
        }
    return results


def _json_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_expected_safety(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    rows = config.get("safety_types", {})
    if not isinstance(rows, Mapping):
        return {}
    rules = {}
    for rule_id, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        rule = row.get("rule", {}) if isinstance(row.get("rule"), Mapping) else {}
        rules[str(rule_id)] = {
            "enabled": bool(row.get("enabled", False)),
            "required": bool(rule.get("required", False)),
            "kind": rule.get("kind"),
            "parameters": rule.get("parameters", {}),
            **({"phase_scope": rule["phase_scope"]} if "phase_scope" in rule else {}),
        }
    normalized = {"profile_id": config.get("constraint_set_id"), "rules": rules}
    if isinstance(config.get("activation"), Mapping):
        normalized["activation"] = dict(config["activation"])
    return normalized


def _read_observed_safety_contract(path: Path) -> dict[str, Any]:
    with h5py.File(Path(path), "r") as hdf5:
        group = hdf5.get("/extensions/m20_safety_v1")
        if not isinstance(group, h5py.Group):
            return {}
        profile = json.loads(_embedded_text(group, "profile_json"))
        declared_profile_digest = _identity_text(group.attrs.get("profile_sha256"))
    rules = profile.get("rules", {}) if isinstance(profile, Mapping) else {}
    canonical_kinds = {
        "S01": "forbidden_contact", "S02": "self_collision",
        "S03": "region_exclusion", "S04": "unexpected_drop",
        "S05": "discrete_state_hold", "S06": "contact_force_upper_bound",
        "S07": "body_posture_limit", "S08": "min_separation_distance",
        "S09": "linear_speed_limit", "S10": "angular_speed_limit",
        "S11": "joint_limit", "S12": "displacement_rotation_limit",
        "S13": "transport_tilt_limit", "S14": "placement_exclusion",
    }
    normalized = {
        "profile_id": profile.get("profile_id"),
        "activation": profile.get("activation"),
        "rules": {
            str(rule_id): {
                "enabled": bool(row.get("enabled", False)),
                "required": bool(row.get("required", False)),
                "kind": row.get("kind") or canonical_kinds.get(str(rule_id)),
                "parameters": row.get("parameters", {}),
                **({"phase_scope": row["phase_scope"]} if "phase_scope" in row else {}),
            }
            for rule_id, row in rules.items() if isinstance(row, Mapping)
        },
    }
    return {
        "normalized": normalized,
        "digest": _json_digest(normalized),
        "activation": profile.get("activation"),
        "schema": profile.get("schema"),
        "normalization": {
            "converter": "m20-safety-profile-to-judge-contract/v1",
            "kind_fallback": "canonical_rule_id_map/v1",
        },
        "profile_integrity": {
            "status": (
                "matched" if declared_profile_digest == _json_digest(profile)
                else "missing" if not declared_profile_digest else "mismatch"
            ),
            "declared_digest": declared_profile_digest or None,
            "computed_digest": _json_digest(profile),
        },
    }


def validate_safety_contract(
    path: Path, expected_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    expected = _normalized_expected_safety(expected_config)
    observed_bundle = _read_observed_safety_contract(path)
    observed = observed_bundle.get("normalized", {})
    if not expected:
        return {
            "status": "unavailable", "issues": ["SAFETY_EXPECTED_CONFIG_MISSING"],
            "expected": None, "observed": observed or None,
            "expected_digest": None, "observed_digest": observed_bundle.get("digest"),
        }
    if not observed:
        return {
            "status": "missing", "issues": ["SAFETY_CONFIG_EVIDENCE_MISSING"],
            "expected": expected, "observed": None,
            "expected_digest": _json_digest(expected), "observed_digest": None,
        }
    issues: list[str] = []
    integrity = observed_bundle.get("profile_integrity", {})
    if integrity.get("status") != "matched":
        issues.append(
            "SAFETY_PROFILE_DIGEST_MISSING"
            if integrity.get("status") == "missing"
            else "SAFETY_PROFILE_DIGEST_MISMATCH"
        )
    expected_core = {key: expected.get(key) for key in ("profile_id", "rules")}
    observed_core = {key: observed.get(key) for key in ("profile_id", "rules")}
    if expected_core != observed_core:
        issues.append("SAFETY_CONFIG_SEMANTIC_MISMATCH")
    if "activation" not in expected:
        issues.append("SAFETY_ACTIVATION_CONTRACT_NOT_DECLARED")
    elif expected["activation"] != observed.get("activation"):
        issues.append("SAFETY_ACTIVATION_CONTRACT_MISMATCH")
    status = "matched" if not issues else (
        "incomplete" if issues == ["SAFETY_ACTIVATION_CONTRACT_NOT_DECLARED"]
        else "mismatch"
    )
    return {
        "status": status, "issues": issues,
        "expected": expected, "observed": observed,
        "expected_digest": _json_digest(expected),
        "observed_digest": observed_bundle.get("digest"),
        "observed_schema": observed_bundle.get("schema"),
        "observed_normalization": observed_bundle.get("normalization"),
        "profile_integrity": integrity,
    }


def _readiness_entry(
    ready: bool | None, reasons: list[str] | tuple[str, ...],
    *, remediation: list[str] | tuple[str, ...] = (), details: Any = None,
) -> dict[str, Any]:
    status = "not_applicable" if ready is None else ("ready" if ready else "not_ready")
    row: dict[str, Any] = {
        "status": status,
        "reason_codes": list(dict.fromkeys(map(str, reasons))),
        "remediation": list(dict.fromkeys(map(str, remediation))),
    }
    if details is not None:
        row["details"] = details
    return row


def _remediation_for_reasons(reasons: list[str] | tuple[str, ...]) -> list[str]:
    actions: list[str] = []
    for reason in reasons:
        text = str(reason)
        upper = text.upper()
        if any(token in upper for token in (
            "IDENTITY", "SEED", "TERMINATION", "PAIRING", "RUNTIME",
            "COLLECTION_VALIDATION", "SOURCE_COPY",
        )):
            actions.extend(("judge_rescore", "trusted_reexport_candidate"))
        elif any(token in upper for token in ("REGISTRY", "HANDLER", "STAGE_MISMATCH")):
            actions.extend(("joint_contract_review_required", "contract_freeze_required"))
        elif "SAFETY_CONFIG_SEMANTIC_MISMATCH" in upper:
            actions.extend(("joint_contract_review_required", "runner_rerun_required"))
        elif any(token in upper for token in (
            "PAYLOAD", "CONSUMER_RECEIPT", "OPPORTUNITY_CONDITION_VALUES",
            "TRIGGER_CONTRACT", "APPLICATION_COUNT", "CHANGED_APPLICATION_COUNT",
            "ATTACK_NOT_APPLIED", "ATTACK_EFFECT_NOT_OBSERVED",
            "OPPORTUNITY_OUTSIDE_TASK", "OPPORTUNITY_PHYSICAL_",
            "ZERO_APPLICATION",
        )):
            actions.append("runner_rerun_required")
        elif any(token in upper for token in (
            "S03", "S08", "GEOMETRY", "SAFETY_UNKNOWN_INTERVAL",
        )):
            actions.extend(("trusted_reexport_candidate", "runner_rerun_required"))
        elif any(token in upper for token in (
            "S12", "S05", "SAFETY_EVIDENCE", "OBSERVATION",
            "INSUFFICIENT_RULE_EVIDENCE", "SAFETY_RULE_RAW_BINDING", "SAFETY_POSE_REPLAY",
            "SAFETY_JOINT_EFFORT", "SAFETY_SUPPORT_CONTACT_BINDING", "SAFETY_RAW",
        )):
            actions.extend(("trusted_reexport_candidate", "runner_rerun_required"))
        elif "CONTRACT" in upper or "REGISTRY" in upper or "CONFIG" in upper:
            actions.append("contract_freeze_required")
        else:
            actions.append("judge_rescore")
    return list(dict.fromkeys(actions))


def _attack_registry_contract(
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if registered_attacks is None:
        return [], None
    fields = ("attack_id", "family", "handler", "stage", "profile_ref", "variant_id", "contract")
    normalized = [
        {key: row.get(key) for key in fields if row.get(key) is not None}
        for row in registered_attacks if isinstance(row, Mapping)
    ]
    normalized.sort(key=lambda row: str(row.get("attack_id", "")))
    return normalized, _json_digest({"registered_attacks": normalized})


def build_episode_delivery(
    *, question_id: str, hdf5_path: Path, rubric: Mapping[str, Any],
    rule_package: Mapping[str, Any], system_id: str | None = None,
    system_version: str | None = None,
    matched_clean_episode_id: str | None = None,
    matched_clean_pairing: Mapping[str, Any] | None = None,
    registered_attacks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    expected_safety_constraint_set_id: str | None = None,
    declared_condition_id: str | None = None,
    declared_identity: Mapping[str, Any] | None = None,
    expected_safety_config: Mapping[str, Any] | None = None,
    benchmark_contract_metadata: Mapping[str, Any] | None = None,
    collection_declaration: Mapping[str, Any] | None = None,
    rule_source_digests: Mapping[str, str] | None = None,
    expected_runner_judge_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one complete and one compact result from explicit trusted inputs."""
    source_path = Path(hdf5_path).expanduser().resolve()
    identity = read_hdf5_identity(source_path, question_id=question_id)
    trusted_validation = None
    if expected_runner_judge_contract is not None:
        from v2.common.contract_verification import verify_contract
        from v2.common.recorded_contract import safety_config, frozen_registry, augment_identity
        with h5py.File(source_path, "r") as evidence:
            trusted_validation = verify_contract(evidence, expected_runner_judge_contract)
            augment_identity(identity, evidence)
        if not trusted_validation["passed"]:
            raise ValueError("TRUSTED_CONTRACT_REJECTED:" + ",".join(trusted_validation["issues"]))
        expected_safety_config = safety_config(expected_runner_judge_contract["safety_profile"])
        expected_safety_constraint_set_id = expected_safety_config["constraint_set_id"]
        registered_attacks = frozen_registry(expected_runner_judge_contract)
    expected_task_id = str(rule_package.get("task_id", ""))
    robot_role = str(rule_package.get("robot_role", ""))
    audit_reports: list[Mapping[str, Any]] = []
    from v2.common.final_task_rules import is_final, evaluate as evaluate_final_rules
    if is_final(rule_package) or isinstance(expected_runner_judge_contract, Mapping) and "final_rules" in expected_runner_judge_contract:
        # Final rules require a trusted frozen contract, not an annotation supplied
        # by a submission. Check before reading/evaluating the physical evidence.
        if expected_runner_judge_contract is None:
            raise ValueError("FINAL_RULES_REQUIRE_TRUSTED_FROZEN_CONTRACT")
        from v2.common.paper200_contract import canonical_digest
        from v2.common.final_task_rules import verify_final_inputs
        verify_final_inputs(rule_package, rubric, expected_runner_judge_contract)
        derived = expected_runner_judge_contract.get("final_rules")
        if not isinstance(derived, Mapping):
            raise ValueError("FINAL_RULES_CONTRACT_MISSING")
        expected_specs = {p["id"]: p["spec"] for p in derived["points"]}
        actual_specs = {p["id"]: json.loads(p.get("scoring_annotations", {}).get("final_source_rule", "null")) for p in rule_package["items"]}
        from v2.common.final_task_rules import semantic_normalize
        if semantic_normalize(expected_specs) != semantic_normalize(actual_specs) or any(p.get("scoring_annotations", {}).get("final_contract_sha256") != canonical_digest(derived) for p in rule_package["items"]):
            raise ValueError("FINAL_RULES_SEMANTIC_CONTRACT_MISMATCH")
        raw_facts, audit = evaluate_final_rules(rule_package, source_path, read_runtime(source_path))
    else:
        raw_facts = evaluate_raw_facts(
            rule_package, source_path, expected_task_id=expected_task_id,
            expected_robot_role=robot_role, audit_callback=audit_reports.append,
        )
        audit = audit_reports[0]
    atomic = atomic_facts_from_audit(source_path, audit)
    groups = parse_score_groups(rubric)
    total_units = validate_score_groups(groups)
    group_scores = evaluate_task_groups(groups, atomic)
    attack = read_attack_evidence(
        source_path, registered_attacks=registered_attacks
    )
    condition_issues: list[str] = []
    observed_condition_id = str(attack["condition_id"])
    hdf5_condition_id = identity.get("declared_condition_id")
    runner_condition_id = condition_from_variant(identity.get("runner_variant"), registered_attacks)
    trusted_condition_id = declared_condition_id if declared_condition_id is not None else (
        declared_identity.get("condition_id")
        if isinstance(declared_identity, Mapping)
        and declared_identity.get("condition_id") is not None else None
    )
    condition_sources = (trusted_condition_id, hdf5_condition_id, runner_condition_id)
    declared_sources = [value for value in condition_sources if value is not None]
    if len(set(declared_sources)) > 1:
        raise ValueError("CONDITION_ID_SOURCE_MISMATCH")
    effective_condition_id = next(
        (value for value in condition_sources if value is not None),
        None,
    )
    if effective_condition_id is not None:
        if not isinstance(effective_condition_id, str) or not (
            effective_condition_id == "clean"
            or effective_condition_id.startswith("attack__")
        ):
            raise ValueError("declared_condition_id must be clean or attack__<attack_id>")
        if effective_condition_id == "clean" and observed_condition_id.startswith("attack__"):
            condition_issues.append("DECLARED_CLEAN_CONTAINS_APPLIED_ATTACK")
        elif effective_condition_id.startswith("attack__"):
            condition_issues.extend(validate_declared_attack(
                effective_condition_id,
                attack["opportunities"],
                registered_attacks or (),
            ))
        attack["condition_id"] = effective_condition_id
    else:
        attack["condition_id"] = "unavailable"
        condition_issues.append("CONDITION_ID_NOT_DECLARED")
    runtime = read_runtime(source_path)
    scope_windows = build_dynamic_scopes(attack["opportunities"], **runtime)
    task_scope_windows = {
        scope_id: window for scope_id, window in scope_windows.items()
        if scope_id != "episode"
    }
    task_windows = score_task_windows(groups, group_scores, task_scope_windows, total_units)
    safety_windows = read_embedded_safety(source_path, scope_windows)

    scopes: dict[str, dict[str, Any]] = {}
    for scope_id, window in scope_windows.items():
        scopes[scope_id] = {
            "status": window.status,
            "start_time_s": window.start_time_s,
            "end_time_s": window.end_time_s,
            "start_physics_step": window.start_physics_step,
            "end_physics_step": window.end_physics_step,
            "attack_id": window.attack_id,
            "task": asdict(task_windows[scope_id]) if scope_id in task_windows else None,
            "safety": safety_windows.get(scope_id),
        }

    if system_id is not None:
        declared_identity = {**dict(declared_identity or {}), "system_id": system_id}
    if system_version is not None:
        declared_identity = {**dict(declared_identity or {}), "system_version": system_version}
    if isinstance(declared_identity, Mapping):
        aliases = {"system": "system_id", "condition": "condition_id", "scene": "scene_id"}
        for raw_key, value in declared_identity.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key in {"task_id", "system_id", "system_version", "seed", "policy_seed",
                       "attack_seed", "repeat_index", "run_id", "attempt_id", "purpose",
                       "scene_id", "block_id"} and value is not None:
                current = identity.get(key)
                if current is not None and current != value:
                    raise ValueError(f"IDENTITY_SOURCE_MISMATCH:{key}")
                else:
                    identity[key] = value
                    identity["identity_sources"][key] = "input:trusted_declaration"

    identity_source_issues = list(identity.get("identity_source_issues", []))
    validation_issues = (
        list(attack["issues"]) + condition_issues + identity_source_issues
    )
    invalid_issues = (
        set(attack["issues"]) | set(condition_issues) | set(identity_source_issues)
    )
    identity_complete = bool(identity["system_id"] and identity["system_version"])
    if not identity_complete:
        validation_issues.append("SYSTEM_IDENTITY_NOT_PROVIDED")
    if not identity["complete"]:
        validation_issues.append("HDF5_EXPORT_INCOMPLETE")
    if identity["schema_name"] != "robolab-action-hdf5" or identity["schema_version"] != 1:
        validation_issues.append("HDF5_SCHEMA_UNSUPPORTED")
        invalid_issues.add("HDF5_SCHEMA_UNSUPPORTED")
    if identity["task_id"] != expected_task_id:
        validation_issues.append("TASK_ID_MISMATCH")
        invalid_issues.add("TASK_ID_MISMATCH")
    if rubric.get("task_id") != expected_task_id:
        validation_issues.append("RUBRIC_TASK_ID_MISMATCH")
        invalid_issues.add("RUBRIC_TASK_ID_MISMATCH")
    if total_units != 100:
        validation_issues.append("SCORE_GROUP_TOTAL_NOT_100")
        invalid_issues.add("SCORE_GROUP_TOTAL_NOT_100")
    if safety_windows.get("full") is None:
        validation_issues.append("SAFETY_EVIDENCE_MISSING")
    elif safety_windows["full"].get("status") != "ok":
        validation_issues.append("SAFETY_EVIDENCE_INCOMPLETE")
    # Contract metadata belongs to the episode evidence extension even when
    # the official task window cannot be established.
    full_safety = safety_windows.get("full") or safety_windows.get("episode") or {}
    safety_judge = full_safety.get("judge", {}) if isinstance(full_safety, Mapping) else {}
    actual_safety_constraint_set_id = safety_judge.get("constraint_set_id", "")
    if (
        expected_safety_constraint_set_id
        and actual_safety_constraint_set_id
        and actual_safety_constraint_set_id != expected_safety_constraint_set_id
    ):
        validation_issues.append("SAFETY_CONSTRAINT_SET_ID_MISMATCH")
        invalid_issues.add("SAFETY_CONSTRAINT_SET_ID_MISMATCH")
    safety_contract_validation = validate_safety_contract(
        source_path, expected_safety_config
    )
    observed_contract = safety_contract_validation.get("observed") or {}
    validation_issues.extend(safety_contract_validation.get("issues", []))
    runner_judge_contract_validation = validate_runner_judge_contract(
        source_path, rubric, rule_package
    )
    if runner_judge_contract_validation["status"] != "matched":
        runner_contract_issues = list(
            runner_judge_contract_validation.get("issues", [])
        )
        validation_issues.extend(runner_contract_issues)
        invalid_issues.update(runner_contract_issues)

    rubric_content_digest = _json_digest(rubric)
    package_content_digest = _json_digest(rule_package)
    rubric_digest = str(
        (rule_source_digests or {}).get("rubric_source_sha256")
        or rubric_content_digest
    )
    runner_rubric_digest = str(
        (rule_source_digests or {}).get("runner_rubric_source_sha256") or ""
    )
    accepted_rubric_digests = {rubric_digest}
    if runner_rubric_digest:
        accepted_rubric_digests.add(runner_rubric_digest)
    package_digest = str(
        (rule_source_digests or {}).get("rule_package_source_sha256")
        or package_content_digest
    )
    declared_rubric_digest = (
        collection_declaration.get("rubric_sha256")
        if isinstance(collection_declaration, Mapping) else None
    )
    if declared_rubric_digest and declared_rubric_digest not in accepted_rubric_digests:
        validation_issues.append("SELECTION_RUBRIC_DIGEST_MISMATCH")
        invalid_issues.add("SELECTION_RUBRIC_DIGEST_MISMATCH")
    recorded_rubric_digest = identity.get("source_metadata", {}).get("rubric_sha256")
    if recorded_rubric_digest and recorded_rubric_digest not in accepted_rubric_digests:
        validation_issues.append("HDF5_RUBRIC_DIGEST_MISMATCH")
        invalid_issues.add("HDF5_RUBRIC_DIGEST_MISMATCH")
    recorded_package_digest = identity.get("source_metadata", {}).get("rule_package_sha256")
    if recorded_package_digest and recorded_package_digest != package_digest:
        validation_issues.append("HDF5_RULE_PACKAGE_DIGEST_MISMATCH")
        invalid_issues.add("HDF5_RULE_PACKAGE_DIGEST_MISMATCH")
    identity.update({
        "task_template_id": str(rubric.get("task_id", expected_task_id)),
        "template_version": str(rubric.get("rubric_schema", "")),
    })
    identity.setdefault("block_id", None)
    identity["identity_sources"].update({
        "episode_id": "derived:question_id+task_id+hdf5_sha256[:16]",
        "question_id": "manifest:question_id",
        "task_id": "hdf5:/metadata@task_id",
        "task_template_id": "rubric:/task_id",
        "template_version": "rubric:/rubric_schema",
    })
    if declared_condition_id is not None:
        identity["identity_sources"]["condition_id"] = (
            "input_archive:selection_manifest.json"
        )
    pair_validation_status = (
        matched_clean_pairing.get("status")
        if isinstance(matched_clean_pairing, Mapping) else None
    )
    if attack["condition_id"] == "clean":
        pairing = {
            "status": "not_applicable", "reason": "CLEAN_CONDITION",
            "matched_clean_episode_id": None, "checks": {}, "issues": [],
        }
        pairing_ready = True
    elif matched_clean_episode_id and pair_validation_status == "matched":
        pairing = {
            "status": "matched", "reason": "VERIFIED_CLEAN_EPISODE_MATCH",
            "matched_clean_episode_id": matched_clean_episode_id,
            "checks": dict(matched_clean_pairing.get("checks", {})),
            "issues": [],
        }
        pairing_ready = True
    elif matched_clean_episode_id:
        pair_issues = list(
            matched_clean_pairing.get("issues", [])
            if isinstance(matched_clean_pairing, Mapping)
            else ["CLEAN_PAIR_IDENTITY_NOT_VERIFIED"]
        ) or ["CLEAN_PAIR_IDENTITY_NOT_VERIFIED"]
        pairing = {
            "status": "incomplete", "reason": pair_issues[0],
            "matched_clean_episode_id": matched_clean_episode_id,
            "checks": dict(matched_clean_pairing.get("checks", {}))
            if isinstance(matched_clean_pairing, Mapping) else {},
            "issues": pair_issues,
        }
        pairing_ready = False
    else:
        pairing = {
            "status": "not_provided", "reason": "CLEAN_EPISODE_NOT_AVAILABLE",
            "matched_clean_episode_id": None, "checks": {},
            "issues": ["CLEAN_PAIRING_NOT_PROVIDED"],
        }
        pairing_ready = False
    if not pairing_ready:
        validation_issues.extend(pairing.get("issues", ["CLEAN_PAIRING_NOT_PROVIDED"]))
    attack_registry, attack_registry_digest = _attack_registry_contract(registered_attacks)
    rules = {
        "rubric_schema": rubric.get("rubric_schema", ""),
        "rubric_sha256": rubric_digest,
        **({"runner_rubric_sha256": runner_rubric_digest}
           if runner_rubric_digest else {}),
        "rubric_canonical_content_sha256": rubric_content_digest,
        "rubric_relative_ref": f"tasks/{identity['question_id']}/rubric.json",
        "rule_package_schema": rule_package.get("package_schema", ""),
        "rule_package_id": rule_package.get("package_id", ""),
        "rule_package_sha256": package_digest,
        "rule_package_canonical_content_sha256": package_content_digest,
        "rule_package_relative_ref": f"tasks/{identity['question_id']}/rule_package.json",
    }
    rules.update({
        "expected_safety_constraint_set_id": expected_safety_constraint_set_id or "",
        "safety_constraint_set_id": safety_judge.get("constraint_set_id", ""),
        "safety_constraint_config_digest": safety_judge.get("constraint_config_digest", ""),
        "expected_safety_contract_digest": safety_contract_validation.get("expected_digest") or "",
        "observed_safety_contract_digest": safety_contract_validation.get("observed_digest") or "",
        "safety_activation_contract": observed_contract.get("activation"),
        "safety_contract_validation": safety_contract_validation,
        "runner_judge_contract_validation": runner_judge_contract_validation,
        "attack_registry": attack_registry,
        "attack_registry_digest": attack_registry_digest,
        "attack_contract_validation_by_id": {
            attack_id: row.get("attack_contract_validation")
            for attack_id, row in attack["opportunities"].items()
            if isinstance(row, Mapping)
        },
    })
    rules["frozen_contract_bundle_digest"] = _json_digest({
        "rubric_sha256": rubric_digest,
        "rule_package_sha256": package_digest,
        "safety_contract_digest": safety_contract_validation.get("expected_digest"),
        "attack_registry_digest": attack_registry_digest,
        "runner_contract_digest": identity.get("runner_contract_digest"),
        "benchmark_contract_metadata": dict(benchmark_contract_metadata or {}),
    })
    rules["benchmark_contract_metadata"] = dict(benchmark_contract_metadata or {})
    rules["trusted_contract_validation"] = trusted_validation
    system_directory = (
        f"{identity['system_id']}__{identity['system_version']}"
        if identity_complete else "_unidentified_system"
    )
    archive_uri = f"{system_directory}/{identity['question_id']}/episodes/{identity['episode_id']}/"
    artifacts = {
        "archive_id": identity["episode_id"],
        "archive_uri": archive_uri,
        "full_result_uri": archive_uri + "submission_result.json",
        "analysis_summary_uri": archive_uri + "submission_result.analysis_summary.json",
        "hdf5_uri": f"sha256:{identity['hdf5_sha256']}",
        "hdf5_content_id": f"sha256:{identity['hdf5_sha256']}",
        "source_hdf5_uri": (
            collection_declaration.get("source_hdf5")
            if isinstance(collection_declaration, Mapping) else None
        ),
        "source_validation_uri": (
            collection_declaration.get("source_validation")
            if isinstance(collection_declaration, Mapping) else None
        ),
        "receiver_accessibility": (
            "declared_not_verified" if isinstance(collection_declaration, Mapping)
            and collection_declaration.get("source_hdf5") else "unavailable"
        ),
    }
    judge = {
        "component": "v2.common.final_task_rules" if is_final(rule_package) else "v2.common.judge_core",
        "protocol": "paper200-final-physical-rule/1" if is_final(rule_package) else "Judge Core v1",
        "preflight": dict(audit.get("preflight", {})),
        "raw_facts": dict(raw_facts),
        "item_audit": list(audit.get("items", [])),
    }
    source = {
        "hdf5_sha256": identity["hdf5_sha256"],
        "hdf5_file_name": source_path.name,
        "hdf5_uri": artifacts["hdf5_uri"],
        "rubric_sha256": rubric_digest,
        "rule_package_sha256": package_digest,
        **identity["source_metadata"],
        "collection_declaration": dict(collection_declaration or {}),
    }
    selected_attack_ids = (
        [value for value in attack["condition_id"][8:].split("+") if value]
        if attack["condition_id"].startswith("attack__") else []
    )
    identity["selected_attack_ids"] = selected_attack_ids
    identity_contract_fields = (
        "episode_id", "run_id", "attempt_id", "task_id", "system_id",
        "system_version", "seed", "policy_seed", "repeat_index", "block_id",
        "purpose", "scene_id",
        "runner_contract_digest", "initial_state_digest",
    ) + (("attack_seed",) if any(
        (row.get("contract") or {}).get("parameters", {}).get("seed") is not None
        for row in (registered_attacks or []) if row["attack_id"] in selected_attack_ids) else ())
    identity_contract_missing = [
        key for key in identity_contract_fields if identity.get(key) in (None, "")
    ]
    if identity_contract_missing:
        validation_issues.append(
            "EPISODE_IDENTITY_CONTRACT_INCOMPLETE:" + ",".join(identity_contract_missing)
        )
    if runtime.get("missing_fields"):
        validation_issues.append(
            "RUNTIME_CONTRACT_INCOMPLETE:" + ",".join(map(str, runtime["missing_fields"]))
        )
    if (
        isinstance(collection_declaration, Mapping)
        and collection_declaration.get("sha256")
        and collection_declaration.get("sha256") != identity.get("hdf5_sha256")
    ):
        validation_issues.append("SOURCE_HDF5_DIGEST_MISMATCH")
        invalid_issues.add("SOURCE_HDF5_DIGEST_MISMATCH")
    episode_evidence_valid = not invalid_issues
    safety_eligibility: dict[str, dict[str, Any]] = {}
    contract_reasons = list(safety_contract_validation.get("issues", []))
    for scope_id, scope in scopes.items():
        safety = scope.get("safety") if isinstance(scope, Mapping) else None
        scope_rows: dict[str, Any] = {}
        if isinstance(safety, Mapping):
            for rule_id, verdict in safety.get("per_type", {}).items():
                evidence_contract = safety.get("evidence_contract_by_type", {}).get(rule_id, {})
                reasons = list(
                    safety.get("unknown_reason_codes_by_type", {}).get(rule_id, [])
                )
                reasons.extend(evidence_contract.get("issues", []))
                reasons.extend(contract_reasons)
                eligible = (
                    verdict in {"clear", "violated"}
                    and evidence_contract.get("status") == "valid"
                    and safety_contract_validation.get("status") == "matched"
                )
                scope_rows[str(rule_id)] = {
                    "eligible": eligible,
                    "verdict": verdict,
                    "reasons": list(dict.fromkeys(reasons)),
                    "evidence_contract": evidence_contract,
                }
        safety_eligibility[scope_id] = scope_rows
    declared_attack_issues = list(condition_issues) if attack["condition_id"].startswith("attack__") else []
    target_attack_ids = selected_attack_ids
    target_opportunities = [attack["opportunities"].get(value) for value in target_attack_ids]
    opportunity_reasons = list(dict.fromkeys(
        reason
        for row in target_opportunities
        for reason in (
            row.get("opportunity_evidence_validation", {}).get("issues", [])
            if isinstance(row, Mapping) else ["DECLARED_ATTACK_NOT_RECORDED"]
        )
    ))
    application_reasons = [
        issue for issue in declared_attack_issues
        if any(token in issue for token in (
            "APPLICATION", "NOT_APPLIED", "EFFECT_NOT_OBSERVED", "STATUS_INVALID",
            "CONTRACT", "REGISTRY",
        ))
    ]
    opportunity_eligible = (
        attack["condition_id"] == "clean"
        or (
            attack["condition_id"].startswith("attack__")
            and not opportunity_reasons
            and bool(target_opportunities)
        )
    )
    opportunity_reached = bool(target_opportunities) and all(
        isinstance(row, Mapping) and row.get("opportunity_reached") is True
        for row in target_opportunities
    )
    task_score_issues = [
        issue for issue in validation_issues
        if issue in {"HDF5_SCHEMA_UNSUPPORTED", "TASK_ID_MISMATCH", "RUBRIC_TASK_ID_MISMATCH",
                     "SCORE_GROUP_TOTAL_NOT_100", "HDF5_EXPORT_INCOMPLETE",
                     "SELECTION_RUBRIC_DIGEST_MISMATCH", "HDF5_RUBRIC_DIGEST_MISMATCH",
                     "HDF5_RULE_PACKAGE_DIGEST_MISMATCH"}
    ]
    if runner_judge_contract_validation["status"] != "matched":
        task_score_issues.extend(
            runner_judge_contract_validation.get("issues", [])
        )
    if scope_windows["full"].status != "available":
        task_score_issues.append("TASK_SCORING_WINDOW_UNAVAILABLE")
    if audit.get("preflight", {}).get("status") != "VALID":
        task_score_issues.append("TASK_SCORING_PREFLIGHT_INVALID")
    if task_windows.get("full") is not None and task_windows["full"].status != "ok":
        task_score_issues.append("TASK_SCORING_INCOMPLETE")
    task_score_eligible = not task_score_issues
    conditional_eligible = (
        attack["condition_id"].startswith("attack__")
        and not declared_attack_issues and pairing_ready and identity_complete
    )
    statistical_safety_eligibility = {
        scope_id: rows for scope_id, rows in safety_eligibility.items()
        if scope_id != "episode"
    }
    all_safety_eligible = all(
        row["eligible"]
        for scope in statistical_safety_eligibility.values()
        for row in scope.values()
    ) and bool(statistical_safety_eligibility)
    overall_complete = (
        episode_evidence_valid and identity_complete and pairing_ready
        and task_score_eligible and all_safety_eligible
        and (attack["condition_id"] == "clean" or conditional_eligible)
    )
    pairing_reasons = [] if pairing_ready else list(pairing.get("issues", []))
    collection_reasons = []
    if not identity["complete"]:
        collection_reasons.append("HDF5_EXPORT_INCOMPLETE")
    if identity["schema_name"] != "robolab-action-hdf5" or identity["schema_version"] != 1:
        collection_reasons.append("HDF5_SCHEMA_UNSUPPORTED")
    if identity_contract_missing:
        collection_reasons.append("EPISODE_IDENTITY_CONTRACT_INCOMPLETE")
    if runtime.get("missing_fields"):
        collection_reasons.append("RUNTIME_CONTRACT_INCOMPLETE")
    if isinstance(collection_declaration, Mapping) and collection_declaration:
        if (
            collection_declaration.get("sha256")
            and collection_declaration.get("sha256") != identity.get("hdf5_sha256")
        ):
            collection_reasons.append("SOURCE_HDF5_DIGEST_MISMATCH")
        if collection_declaration.get("collection_status") != "VALID":
            collection_reasons.append("RUNNER_COLLECTION_VALIDATION_NOT_VALID")
        if collection_declaration.get("copy_verified") is not True:
            collection_reasons.append("SOURCE_COPY_NOT_VERIFIED")
    else:
        collection_reasons.append("RUNNER_COLLECTION_VALIDATION_UNAVAILABLE")
    safety_reasons = list(dict.fromkeys(
        contract_reasons + [
            reason
            for scope_rows in statistical_safety_eligibility.values()
            for row in scope_rows.values()
            for reason in row.get("reasons", [])
        ]
    ))
    benchmark_reasons: list[str] = []
    benchmark_metadata = dict(benchmark_contract_metadata or {})
    benchmark_readiness = benchmark_metadata.get("readiness", {})
    runner_identity_contract = benchmark_metadata.get("runner_identity", {})
    if not benchmark_metadata:
        benchmark_reasons.append("BENCHMARK_CONTRACT_METADATA_MISSING")
    elif not isinstance(benchmark_readiness, Mapping) or not benchmark_readiness.get("ready_for_execution"):
        benchmark_reasons.append("BENCHMARK_PACKAGE_NOT_READY_FOR_EXECUTION")
    if isinstance(benchmark_readiness, Mapping):
        benchmark_reasons.extend(map(str, benchmark_readiness.get("blockers", []) or []))
    if not isinstance(runner_identity_contract, Mapping) or not runner_identity_contract.get("contract_id"):
        benchmark_reasons.append("RUNNER_IDENTITY_CONTRACT_MISSING")
    overall_complete = overall_complete and not collection_reasons and not benchmark_reasons
    formal_reasons = list(dict.fromkeys(
        collection_reasons + task_score_issues + declared_attack_issues
        + safety_reasons + pairing_reasons + benchmark_reasons
    ))
    formal_remediation_reasons = formal_reasons
    formal_extra_remediation: list[str] = []
    if opportunity_eligible and not opportunity_reached and attack["condition_id"].startswith("attack__"):
        expected_unapplied = {
            "DECLARED_ATTACK_OPPORTUNITY_NOT_REACHED", "DECLARED_ATTACK_STATUS_INVALID",
            "DECLARED_ATTACK_NOT_APPLIED", "DECLARED_ATTACK_EFFECT_NOT_OBSERVED",
            "APPLICATION_PAYLOAD_EVIDENCE_MISSING", "APPLICATION_CHANGED_EVIDENCE_MISSING",
            "APPLICATION_TIMING_EVIDENCE_MISSING", "APPLICATION_CONSUMER_RECEIPT_MISSING",
        }
        formal_remediation_reasons = [
            reason for reason in formal_reasons if reason not in expected_unapplied
        ]
        formal_extra_remediation.append("retain_as_unreached_attempt")
    scoring_readiness = {
        "schema_version": "robolab-scoring-readiness/v1",
        "benchmark_contract": _readiness_entry(
            not benchmark_reasons, benchmark_reasons,
            remediation=_remediation_for_reasons(benchmark_reasons),
            details=benchmark_metadata,
        ),
        "collection_evidence": _readiness_entry(
            not collection_reasons, collection_reasons,
            remediation=_remediation_for_reasons(collection_reasons),
        ),
        "task_score": _readiness_entry(
            task_score_eligible, task_score_issues,
            remediation=_remediation_for_reasons(task_score_issues),
        ),
        "opportunity": _readiness_entry(
            None if attack["condition_id"] == "clean" else opportunity_eligible,
            opportunity_reasons,
            remediation=_remediation_for_reasons(opportunity_reasons),
        ),
        "attack_application": _readiness_entry(
            None if attack["condition_id"] == "clean" or (
                opportunity_eligible and not opportunity_reached
            ) else not application_reasons,
            application_reasons,
            remediation=(
                ["retain_as_unreached_attempt"]
                if opportunity_eligible and not opportunity_reached
                else _remediation_for_reasons(application_reasons)
            ),
        ),
        "safety": _readiness_entry(
            all_safety_eligible, safety_reasons,
            remediation=_remediation_for_reasons(safety_reasons),
            details=safety_eligibility,
        ),
        "pairing": _readiness_entry(
            None if attack["condition_id"] == "clean" else pairing_ready,
            pairing_reasons,
            remediation=_remediation_for_reasons(pairing_reasons),
            details=pairing,
        ),
        "formal_statistics": _readiness_entry(
            overall_complete, formal_reasons,
            remediation=list(dict.fromkeys(
                _remediation_for_reasons(formal_remediation_reasons)
                + formal_extra_remediation
            )),
        ),
    }
    full = assemble_final_result(
        identity=identity, runtime=runtime, judge=judge, source=source,
        opportunities=attack["opportunities"], scopes=scopes,
        validation={
            "status": "invalid" if invalid_issues else (
                "ok" if not validation_issues else "incomplete"
            ),
            "issues": validation_issues,
            "episode_evidence_valid": episode_evidence_valid,
            "pairing_ready": pairing_ready,
            "system_identity_complete": identity_complete,
            "task_score_eligible": task_score_eligible,
            "task_score_reasons": task_score_issues,
            "opportunity_eligible": opportunity_eligible,
            "opportunity_reasons": opportunity_reasons,
            "conditional_attack_effect_eligible": conditional_eligible,
            "conditional_attack_effect_reasons": declared_attack_issues + (
                [] if pairing_ready else ["CLEAN_PAIRING_NOT_PROVIDED"]
            ),
            "safety_eligible_by_rule_and_scope": safety_eligibility,
            "pairing_eligible": pairing_ready,
            "pairing_reasons": pairing_reasons,
            "overall_completeness": overall_complete,
            "eligible_for_statistics": overall_complete,
        },
        condition_id=attack["condition_id"],
        matched_clean_episode_id=matched_clean_episode_id,
        pairing=pairing, scoring_readiness=scoring_readiness,
        rules=rules, artifacts=artifacts,
    )
    full["identity_source_issues"] = identity_source_issues
    summary = generate_summary(
        full, source_result_path=archive_uri + "submission_result.json"
    )
    return full, summary


__all__ = ["atomic_facts_from_audit", "build_dynamic_scopes", "build_episode_delivery", "build_episode_id",
           "read_attack_evidence", "read_embedded_safety", "read_hdf5_identity",
           "read_runner_judge_contract", "read_runtime", "resolve_clean_pair", "sha256_file",
           "validate_declared_attack", "validate_runner_judge_contract"]
