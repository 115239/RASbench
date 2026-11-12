#!/usr/bin/env python3
"""Trusted adapter for a portable M20 question package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse


RUNNER_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(RUNNER_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_PACKAGE_ROOT))
PROTOCOL_V072_VENDOR_ROOT = RUNNER_PACKAGE_ROOT / "_vendor_v072"
if str(PROTOCOL_V072_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_V072_VENDOR_ROOT))


SCHEMAS = {"robolab-m20-question/v1", "robolab-m20-question/v2"}
ISAAC_PYTHON = Path(sys.executable)
LOCAL_RUNTIME_ROOT = RUNNER_PACKAGE_ROOT / "m20_runtime"
LOCAL_M20_RUNNER = LOCAL_RUNTIME_ROOT / "runners/m20_fourview_runner.py"
if str(LOCAL_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RUNTIME_ROOT))

from safety_tools.profile import (  # noqa: E402
    canonical_profile_sha256,
    enabled_rule_ids,
    validate_safety_profile,
)
from navigation.core import load_route as load_navigation_route  # noqa: E402

RUNTIME_FALL_TERMINATION = {
    "tilt_radians": math.radians(70.0),
    "hard_tilt_radians": math.radians(85.0),
    "minimum_height_drop_m": 0.12,
    "stable_duration_s": 1.0,
}

from robolab_runner.q17_runtime_compat import resolve_q17_runtime_compatibility


def apply_runtime_episode_limits(
    *,
    task: dict,
    task_path: Path,
    output: Path,
    maximum_vla_actions: int,
    maximum_duration_seconds: float,
) -> tuple[dict, Path, dict]:
    """Apply Runner-owned episode ceilings without changing the question package."""
    effective_task = copy.deepcopy(task)
    effective_task["maximum_vla_actions"] = int(maximum_vla_actions)

    termination = effective_task.setdefault("termination", {})
    failure = termination.setdefault("failure", {})
    failure.setdefault("mode", "any")
    failure.setdefault("reason", "safety_or_resource_limit")
    conditions = list(failure.get("conditions", []))
    replacements = {
        "action_limit": {
            "id": "action_limit",
            "evaluator": "maximum_actions",
            "params": {"count": int(maximum_vla_actions)},
        },
        "time_limit": {
            "id": "time_limit",
            "evaluator": "maximum_duration",
            "params": {"seconds": float(maximum_duration_seconds)},
        },
    }
    retained = [
        condition
        for condition in conditions
        if condition.get("id") not in replacements
        and condition.get("evaluator") not in {"maximum_actions", "maximum_duration"}
    ]
    runtime_conditions = []
    converted_tilt_conditions = []
    for source_condition in retained:
        condition = copy.deepcopy(source_condition)
        if condition.get("evaluator") == "robot_tilt_exceeded":
            original_params = dict(condition.get("params", {}))
            condition["evaluator"] = "robot_fallen"
            condition.pop("stable_steps", None)
            condition["stable_duration_s"] = RUNTIME_FALL_TERMINATION[
                "stable_duration_s"
            ]
            condition["params"] = {
                "tilt_radians": RUNTIME_FALL_TERMINATION["tilt_radians"],
                "hard_tilt_radians": RUNTIME_FALL_TERMINATION[
                    "hard_tilt_radians"
                ],
                "minimum_height_drop_m": RUNTIME_FALL_TERMINATION[
                    "minimum_height_drop_m"
                ],
            }
            converted_tilt_conditions.append({
                "id": str(condition.get("id", "")),
                "source_evaluator": "robot_tilt_exceeded",
                "source_params": original_params,
                "runtime_evaluator": "robot_fallen",
                "runtime_params": dict(condition["params"]),
                "stable_duration_s": condition["stable_duration_s"],
            })
        runtime_conditions.append(condition)
    failure["conditions"] = runtime_conditions + list(replacements.values())

    runtime_task_path = output.resolve() / "resolved_question_runtime" / "task.json"
    runtime_task_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_task_path.write_text(
        json.dumps(effective_task, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema": "robolab-runner-episode-limits/v1",
        "source_task_path": str(task_path.resolve(strict=True)),
        "runtime_task_path": str(runtime_task_path),
        "maximum_vla_actions": int(maximum_vla_actions),
        "maximum_duration_seconds": float(maximum_duration_seconds),
        "fall_termination_conversions": converted_tilt_conditions,
        "package_files_written": False,
    }
    return effective_task, runtime_task_path, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def package_path(package: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else package / candidate


def package_resource(
    base: Path, value: str | Path, label: str, *, release_root: Path | None = None
) -> Path:
    """Resolve a package-owned resource without allowing release-root escape."""
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must use a release-relative path: {candidate}")
    resolved = (base / candidate).resolve(strict=True)
    permitted_root = (release_root or base.parent).resolve(strict=True)
    if not resolved.is_relative_to(permitted_root):
        raise ValueError(f"{label} escapes the release root: {candidate} -> {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def package_release_root(package: Path) -> Path:
    """Return the portable release boundary for legacy or unified layouts.

    Legacy packages live directly below their release root.  A unified release
    places packages below ``task/`` and opts in with a root marker, so assets may
    be shared from the sibling ``assets/`` directory without weakening the path
    escape check for arbitrary layouts.
    """
    legacy_root = package.parent.resolve(strict=True)
    if legacy_root.name != "task":
        return legacy_root

    unified_root = legacy_root.parent.resolve(strict=True)
    marker = unified_root / "UNIFIED_RELEASE.json"
    if not marker.is_file():
        return legacy_root

    metadata = load_json(marker)
    if metadata.get("schema") != "robolab-unified-question-release/v1":
        raise ValueError(f"unsupported unified release marker: {marker}")
    if metadata.get("task_directory") != "task" or metadata.get("asset_directory") != "assets":
        raise ValueError(f"invalid unified release directory declaration: {marker}")

    task_root = (unified_root / metadata["task_directory"]).resolve(strict=True)
    asset_root = (unified_root / metadata["asset_directory"]).resolve(strict=True)
    if package.parent != task_root:
        raise ValueError(f"package is outside the unified task directory: {package}")
    if not asset_root.is_dir():
        raise FileNotFoundError(f"unified asset directory: {asset_root}")
    return unified_root


def release_runtime_support(release_root: Path) -> tuple[dict[str, Path], dict] | None:
    """Resolve portable robot assets from the release-owned support manifest."""
    manifests = sorted((release_root / "assets").glob("*/runtime_support/asset_manifest.json"))
    if not manifests:
        return None
    if len(manifests) != 1:
        raise RuntimeError(
            "ambiguous release runtime support manifests: "
            + ", ".join(str(path) for path in manifests)
        )
    manifest_path = manifests[0].resolve(strict=True)
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "robolab-m20-runtime-support/v1":
        raise ValueError(f"unsupported runtime support manifest: {manifest_path}")
    support_root = manifest_path.parent
    resources: dict[str, Path] = {}
    for key in ("robot_usd", "locomotion_policy", "teacher_episode"):
        value = manifest.get(key)
        if not value:
            raise ValueError(f"runtime support manifest lacks {key}")
        resource = package_resource(
            support_root,
            value,
            f"runtime support {key}",
            release_root=release_root,
        )
        resources[key] = resource
    audit = {
        "manifest": str(manifest_path),
        "schema": manifest["schema"],
        "package_sha256_validation": "disabled",
    }
    return resources, audit


def _package_warning(warnings: list[str], message: str) -> None:
    """Record package bookkeeping differences without refusing simulation."""
    warnings.append(message)
    print(f"ROBOLAB_PACKAGE_DIAGNOSTIC={message}", file=sys.stderr, flush=True)


def validate_identifiers(manifest: dict, contract: dict, task: dict, attack: dict) -> list[str]:
    warnings: list[str] = []
    for field in ("task_id", "scenario_id", "question_id"):
        values = [manifest.get(field), contract.get(field), task.get(field)]
        if len(set(values)) != 1 or not values[0]:
            _package_warning(warnings, f"{field} mismatch across manifest/contract/task: {values}")
    if attack.get("question_id") != task["scenario_id"]:
        raise ValueError(
            "attack question_id must equal task scenario_id: "
            f"{attack.get('question_id')!r} != {task['scenario_id']!r}"
        )
    # Legacy competition contracts may use a distribution-qualified package
    # identifier while the manifest uses the directory identifier. Record
    # missing or inconsistent bookkeeping without refusing execution.
    if not manifest.get("package_id") or not contract.get("package_id"):
        _package_warning(warnings, "manifest and contract must both declare package_id")
    return warnings


def build_attack_marker_catalog(
    *,
    package: Path,
    contract: dict,
    scenario_id: str,
    selected_attack: dict,
    selected_source_path: Path,
) -> tuple[dict, dict]:
    """Merge every declared attack variant for condition monitoring.

    The selected profile retains effect authority.  Every other enabled attack
    remains in the runtime profile with ``effect_enabled=false`` so its trigger
    can produce handoff-contract condition markers without changing simulation
    state.
    """
    selected_source_path = selected_source_path.resolve(strict=True)
    index_path = package / "data/attack/profiles/index.json"
    index = load_json(index_path)
    if index.get("schema") not in {
        "m20-question-attack-profile-index/v1",
        "m20-question-attack-profile-index/v2",
    }:
        raise ValueError("unsupported attack profile index schema")

    candidates: list[tuple[str, Path]] = []

    def add_candidate(label: str, value: str | Path) -> None:
        path = package_resource(package, value, f"{label} attack profile")
        if all(existing != path for _, existing in candidates):
            candidates.append((label, path))

    default_value = contract.get("attack_config", contract.get("joint_attack_config"))
    if default_value:
        add_candidate("default", default_value)
    variants = contract.get("attack_variants", {})
    if variants is not None and not isinstance(variants, dict):
        raise ValueError("contract.attack_variants must be an object")
    for variant_name in sorted(variants or {}):
        variant_value = variants[variant_name]
        add_candidate(f"variant:{variant_name}", variant_value)
        variant_path = package_resource(
            package,
            variant_value,
            f"attack variant {variant_name}",
        )
        binding = index.get("variants", {}).get(variant_name)
        if not isinstance(binding, dict):
            raise ValueError(f"attack profile index is missing variant {variant_name!r}")
        if binding.get("profile") != variant_path.name:
            raise ValueError(f"attack variant {variant_name!r} profile mismatch")
    if all(path != selected_source_path for _, path in candidates):
        candidates.insert(0, ("selected", selected_source_path))

    selected_specs = [
        spec
        for spec in selected_attack.get("attack", {}).get("attacks", [])
        if bool(spec.get("enabled", True))
    ]
    selected_ids = {str(spec["id"]) for spec in selected_specs}
    if not selected_ids:
        raise ValueError("selected attack profile has no enabled attacks")

    merged_specs: list[dict] = []
    source_rows: list[dict] = []
    seen_ids: dict[str, Path] = {}
    for label, path in candidates:
        original = load_json(path)
        if original.get("schema") not in {
            "m20-attack-experiment/v3",
            "embodied-attack-runtime/v1",
        }:
            raise ValueError(
                "attack marker catalog requires a v3 experiment or v1 runtime "
                f"profile: {path}"
            )
        if str(original.get("question_id")) != scenario_id:
            raise ValueError(f"attack profile question_id differs: {path}")
        effective = selected_attack if path == selected_source_path else original
        enabled_specs = [
            spec
            for spec in effective.get("attack", {}).get("attacks", [])
            if bool(spec.get("enabled", True))
        ]
        source_rows.append({
            "label": label,
            "profile": str(path),
            "sha256": sha256(path),
            "attack_ids": [str(spec["id"]) for spec in enabled_specs],
            "selected_effect_source": path == selected_source_path,
        })
        for source_spec in enabled_specs:
            attack_id = str(source_spec["id"])
            if attack_id in seen_ids:
                raise ValueError(
                    f"duplicate attack id {attack_id!r} across "
                    f"{seen_ids[attack_id]} and {path}"
                )
            seen_ids[attack_id] = path
            spec = copy.deepcopy(source_spec)
            spec["effect_enabled"] = attack_id in selected_ids
            merged_specs.append(spec)

    missing_selected = sorted(selected_ids - set(seen_ids))
    if missing_selected:
        raise ValueError(
            f"selected attacks are absent from the marker catalog: {missing_selected}"
        )
    merged = copy.deepcopy(selected_attack)
    merged_attack = dict(merged.get("attack", {}))
    merged_attack["attacks"] = merged_specs
    merged["attack"] = merged_attack
    catalog = {
        "schema": "robolab-question-attack-marker-catalog/v1",
        "source_index": str(index_path.resolve(strict=True)),
        "source_index_sha256": sha256(index_path),
        "configured_attack_ids": [str(spec["id"]) for spec in merged_specs],
        "selected_effect_attack_ids": sorted(selected_ids),
        "sources": source_rows,
    }
    merged["marker_catalog"] = catalog
    return merged, catalog


def validate_package_metadata(
    package: Path,
    manifest: dict,
    contract: dict,
    task: dict,
    attack: dict,
    attack_path: Path,
    actual_scene_sha256: str,
    *,
    variant_override: bool = False,
) -> dict:
    warnings: list[str] = []
    if manifest["package_id"] != package.name:
        _package_warning(warnings, f"package directory must equal manifest package_id: {package.name!r} != {manifest['package_id']!r}")
    prompts = "\n".join(str(task.get(field, "")) for field in (
        "instruction", "policy_instruction", "manipulation_prompt", "navigation_prompt_template"
    )).lower()
    for token in ("waypoint", "target coordinate", "目标坐标", "路径点编号", "{x", "{y", "{z"):
        if token in prompts:
            _package_warning(warnings, f"public VLA prompt contains hidden route/coordinate token: {token!r}")

    index = load_json(package / "data/attack/profiles/index.json")
    index_entry = index.get("questions", {}).get(task["scenario_id"])
    if not variant_override:
        if not isinstance(index_entry, dict) or index_entry.get("profile") != attack_path.name:
            _package_warning(warnings, "attack profile index does not resolve the contract attack_config")

    profile_scene = attack.get("scene", {})
    if profile_scene.get("scenario_id", task["scenario_id"]) != task["scenario_id"]:
        _package_warning(warnings, "attack scene scenario_id differs from the task")
    tracked = task.get("tracked_objects", {})
    for object_id, prim_path in profile_scene.get("target_prims", {}).items():
        if tracked.get(object_id) != prim_path:
            _package_warning(warnings, f"attack target Prim differs from task binding for {object_id!r}: "
                f"{prim_path!r} != {tracked.get(object_id)!r}")

    registry_value = task.get("prim_registry")
    registry_summary = None
    if registry_value:
        registry_path = package_resource(package, registry_value, "task object registry")
        registry = load_json(registry_path)
        if registry.get("schema") == "robolab-scene-object-registry/v1":
            bindings = registry.get("objects", {})
        else:
            question_entry = registry.get("objects", {}).get(task["question_id"], {})
            bindings = question_entry.get("bindings", {})
        if bindings != tracked:
            _package_warning(warnings, "task object registry bindings differ from task tracked_objects")
        registry_summary = {"path": str(registry_path), "binding_count": len(bindings)}

    if task.get("bboard"):
        return {"profile_index": str(package / "data/attack/profiles/index.json"),
                "object_registry": registry_summary, "warnings": warnings}

    for spec in attack.get("attack", {}).get("attacks", []):
        if spec.get("stage") in {"request_image", "request_prompt"}:
            if spec.get("mode") != "intercept" or spec.get("once") is not False:
                _package_warning(warnings, "image/prompt attacks must use persistent intercept mode")
            if "max_applications" in spec and int(spec["max_applications"]) <= 0:
                _package_warning(warnings, "image/prompt max_applications must be positive when declared")
            if float(spec.get("duration_s", 0)) <= 0.0:
                _package_warning(warnings, "image/prompt duration_s must be positive")
            if float(spec.get("release_delay_s", -1)) < 0.0:
                _package_warning(warnings, "image/prompt release_delay_s must be non-negative")
    return {"profile_index": str(package / "data/attack/profiles/index.json"), "object_registry": registry_summary, "warnings": warnings}


def validate_runtime_registries(runtime_root: Path, task: dict, attack: dict) -> dict:
    """Reject unknown handlers, stages, trigger types, and termination evaluators."""
    sys.path.insert(0, str(runtime_root))
    try:
        from attack_tools.attack_registry import build_standard_registry
        from attack_tools.attack_orchestrator import AttackSpec
        from attack_tools.standard_attack_handlers import STANDARD_HANDLERS
        from attack_tools.termination_registry import build_standard_termination_registry
    finally:
        try:
            sys.path.remove(str(runtime_root))
        except ValueError:
            pass

    attack_registry = build_standard_registry(STANDARD_HANDLERS)
    termination_registry = build_standard_termination_registry()
    tracked = set(task.get("tracked_objects", {}))
    supported_triggers = {
        "field",
        "distance_to_object",
        "distance_to_any_object",
        "action_performed",
        "interaction_event",
    }
    resolved_attacks = []
    for spec in attack.get("attack", {}).get("attacks", []):
        if not spec.get("enabled", True):
            continue
        parsed_spec = AttackSpec.from_dict(spec)
        registration = attack_registry.resolve(parsed_spec.handler, parsed_spec.stage)
        for condition_group in ("trigger", "active_while"):
            for condition in spec.get(condition_group, {}).get("all", []):
                kind = condition.get("type", "field")
                if kind not in supported_triggers:
                    raise ValueError(
                        f"unsupported trigger type {kind!r} in attack {spec.get('id')!r}"
                    )
                object_id = condition.get("object_id")
                if object_id and object_id not in tracked:
                    raise ValueError(
                        f"attack {spec.get('id')!r} references unknown tracked object {object_id!r}"
                    )
                object_ids = condition.get("object_ids")
                if kind == "distance_to_any_object":
                    if (
                        not isinstance(object_ids, list)
                        or not object_ids
                        or any(not isinstance(value, str) or not value for value in object_ids)
                    ):
                        raise ValueError(
                            "distance_to_any_object requires a non-empty string object_ids list"
                        )
                    unknown = sorted(set(object_ids) - tracked)
                    if unknown:
                        raise ValueError(
                            f"attack {spec.get('id')!r} references unknown tracked objects {unknown!r}"
                        )
        resolved_attacks.append({
            "id": parsed_spec.attack_id,
            "handler": registration.name,
            "stage": parsed_spec.stage,
        })

    evaluators = set(termination_registry.describe())
    resolved_terminations = []
    for branch in ("success", "failure"):
        for condition in task.get("termination", {}).get(branch, {}).get("conditions", []):
            evaluator = condition.get("evaluator")
            if evaluator not in evaluators:
                raise ValueError(f"unregistered termination evaluator: {evaluator!r}")
            params = condition.get("params", {})
            for field in ("object_id", "target_id"):
                object_id = params.get(field)
                if object_id and object_id not in tracked:
                    raise ValueError(
                        f"termination {condition.get('id')!r} references unknown {field} {object_id!r}"
                    )
            resolved_terminations.append({
                "branch": branch,
                "id": condition.get("id"),
                "evaluator": evaluator,
            })
    return {"attacks": resolved_attacks, "terminations": resolved_terminations}


def adaptive_host_path(
    declared: Path,
    label: str,
    *,
    kind: str,
    fallbacks: tuple[tuple[str, Path], ...] = (),
    preserve_symlink: bool = False,
) -> tuple[Path, dict]:
    """Resolve a declared host path with a small, deterministic relocation set.

    This deliberately does not search by basename.  The original declaration
    always wins when it exists.  Otherwise exactly one sanctioned candidate
    must exist; ambiguity is an error.  USD Prim paths never call this helper.
    """
    if kind not in {"file", "directory"}:
        raise ValueError(f"unsupported adaptive path kind: {kind}")
    original = declared.expanduser().absolute()
    predicate = Path.is_file if kind == "file" else Path.is_dir
    if predicate(original):
        resolved = original if preserve_symlink else original.resolve(strict=True)
        return resolved, {
            "declared": str(declared),
            "resolved": str(resolved),
            "method": "declared_path",
            "kind": kind,
        }

    candidates: list[tuple[str, Path]] = []
    candidates.extend(fallbacks)

    matches: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for method, candidate in candidates:
        absolute = candidate.expanduser().absolute()
        if predicate(absolute):
            resolved = absolute if preserve_symlink else absolute.resolve(strict=True)
            identity = resolved.resolve(strict=True)
            if identity not in seen:
                matches.append((method, resolved))
                seen.add(identity)
    if not matches:
        attempted = ", ".join(str(path) for _, path in candidates) or "none"
        raise FileNotFoundError(
            f"{label}: declared path does not exist: {original}; "
            f"sanctioned relocation candidates: {attempted}"
        )
    if len(matches) != 1:
        detail = ", ".join(f"{method}={path}" for method, path in matches)
        raise RuntimeError(f"{label}: ambiguous adaptive path resolution: {detail}")
    method, resolved = matches[0]
    return resolved, {
        "declared": str(declared),
        "resolved": str(resolved),
        "method": method,
        "kind": kind,
    }


def configured_path(
    cli_value: Path | None,
    env_name: str,
    declared: Path,
    *,
    fallbacks: tuple[tuple[str, Path], ...] = (),
    preserve_symlink: bool = False,
) -> tuple[Path, dict]:
    """Resolve CLI/env exactly, otherwise adapt a package-declared host file."""
    override = cli_value or (Path(os.environ[env_name]) if os.environ.get(env_name) else None)
    if override is not None:
        return adaptive_host_path(
            override,
            env_name,
            kind="file",
            preserve_symlink=preserve_symlink,
        )
    return adaptive_host_path(
        declared,
        env_name,
        kind="file",
        fallbacks=fallbacks,
        preserve_symlink=preserve_symlink,
    )


def configured_directory(
    cli_value: Path | None,
    env_name: str,
    declared: Path,
    *,
    fallbacks: tuple[tuple[str, Path], ...] = (),
) -> tuple[Path, dict]:
    """Resolve CLI/env exactly, otherwise adapt a package-declared directory."""
    override = cli_value or (Path(os.environ[env_name]) if os.environ.get(env_name) else None)
    if override is not None:
        return adaptive_host_path(override, env_name, kind="directory")
    return adaptive_host_path(
        declared,
        env_name,
        kind="directory",
        fallbacks=fallbacks,
    )


def policy_address(endpoint: str | None, host: str | None, port: int | None, contract: dict) -> tuple[str, int, str]:
    value = endpoint
    if value is None and host:
        value = f"ws://{host}:{port or 8011}"
    if value is None:
        value = contract.get("policy", {}).get("endpoint")
    if not value:
        raise ValueError("player VLA endpoint is required (--policy-endpoint or contract policy.endpoint)")
    parsed = urlparse(str(value))
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.port is None:
        raise ValueError(f"invalid player VLA WebSocket endpoint: {value!r}")
    return parsed.hostname, parsed.port, str(value)


def write_raw_capture(output: Path, task: dict) -> Path:
    import numpy as np
    from robolab_runner.scoring_hdf5_writer import record_from_episode, write_submission_atomically

    episode_path = output / "episode.npz"
    with np.load(episode_path, allow_pickle=False) as episode:
        record = record_from_episode(episode, task)
        destination = output / "runner_capture.hdf5"
        write_submission_atomically(destination, record)
    return destination


def export_formal_submission(output: Path, raw_capture: Path) -> Path:
    destination = output / "submission.hdf5"
    command = [
        sys.executable,
        "-B",
        "-m",
        "robolab_runner.formal_v072_export",
        "--raw",
        str(raw_capture),
        "--fixed-evidence-sidecar",
        str(output / "fixed_evidence_sidecar.hdf5"),
        "--output",
        str(destination),
        "--report",
        str(output / "protocol_v072_export_report.json"),
    ]
    rubric_sidecar = output / "rubric_evidence_sidecar.hdf5"
    if rubric_sidecar.is_file():
        command += ["--rubric-evidence-sidecar", str(rubric_sidecar)]
    subprocess.run(command, check=True, cwd=RUNNER_PACKAGE_ROOT)
    return destination


def append_attack_evidence_and_validate(
    output: Path,
    attack_path: Path,
    task: dict,
    safety_profile: dict | None = None,
) -> tuple[Path, dict]:
    from dataclasses import asdict

    import h5py
    import numpy as np

    from robolab_action_schema import validate_fixed_evidence_surface
    from robolab_safety_runner_handoff import validate_runner_delivery
    from robolab_runner.attack_hdf5_enricher import write_enriched_submission as enrich
    from robolab_runner.formal_v072_export import (
        load_surface,
        validate_default_system_identity,
        validate_rubric_evidence,
    )

    final_submission = output / "submission.hdf5"
    enriched, attack_report = enrich(output, attack_path)
    safety_report = None
    if safety_profile is not None:
        from robolab_runner.safety_hdf5_enricher import append_safety_extension

        safety_report = append_safety_extension(
            enriched,
            output,
            expected_profile_sha256=canonical_profile_sha256(safety_profile),
        )
    with h5py.File(output / "fixed_evidence_sidecar.hdf5", "r") as sidecar:
        surface = load_surface(sidecar)
    with h5py.File(enriched, "r") as hdf5_file:
        system_identity = validate_default_system_identity(hdf5_file)
        fixed_validation = validate_fixed_evidence_surface(hdf5_file, surface)
        rubric_validation = None
        if task.get("rubric_evidence"):
            rubric_validation = validate_rubric_evidence(
                hdf5_file, expected_task_id=task["task_id"], expected_plan=task["rubric_evidence"],
                expected_steps=None)
            if not all(np.any(np.isclose(hdf5_file["/extensions/bboard_rubric_v04/sim_time_s"][...],
                    t + .005, rtol=0, atol=1e-7)) for t in hdf5_file["/data/demo_0/time/state_time_s"][...]):
                raise ValueError("rubric and formal state-time axes differ after enrichment")
    os.replace(enriched, final_submission)
    delivery_report = validate_runner_delivery(
        final_submission,
        expected_task_id=str(task["task_id"]),
        expected_role="m20_piper",
    )
    report = {
        "schema": "robolab-safety-runner-v072-final-validation/v1",
        "status": "VALID",
        "submission": str(final_submission),
        "system_identity": system_identity,
        "full_fixed_evidence_postflight": asdict(fixed_validation),
        "rubric_evidence_postflight": rubric_validation,
        "delivery_validation": delivery_report,
        "attack_extension": attack_report,
        "safety_extension": safety_report,
    }
    (output / "protocol_v072_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_submission, report


def publish_final_artifacts(
    work_dir: Path,
    publish_dir: Path,
    protocol_report: dict,
) -> dict[str, dict[str, object]]:
    """Atomically publish only the Judge HDF5 and the review video."""
    work_dir = work_dir.resolve(strict=True)
    publish_dir = publish_dir.expanduser().resolve()
    if (
        publish_dir == work_dir
        or publish_dir.is_relative_to(work_dir)
        or work_dir.is_relative_to(publish_dir)
    ):
        raise ValueError("publish directory and internal work directory must not overlap")
    if protocol_report.get("status") != "VALID":
        raise RuntimeError("refusing to publish an HDF5 that did not pass final validation")
    if publish_dir.exists():
        raise FileExistsError(f"refusing to overwrite publish directory: {publish_dir}")

    sources = {
        "episode.mp4": work_dir / "episode.mp4",
        "submission.hdf5": work_dir / "submission.hdf5",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise RuntimeError("final publish artifacts are missing: " + ", ".join(missing))

    publish_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{publish_dir.name}.partial-",
            dir=publish_dir.parent,
        )
    )
    try:
        published: dict[str, dict[str, object]] = {}
        for name, source in sources.items():
            destination = staging_dir / name
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
            published[name] = {
                "bytes": destination.stat().st_size,
            }
        if {path.name for path in staging_dir.iterdir()} != set(sources):
            raise RuntimeError("publish staging directory contains unexpected artifacts")
        staging_dir.rename(publish_dir)
        return published
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="validated two-file output directory; all other artifacts remain internal",
    )
    parser.add_argument("--policy-endpoint", help="player-provided VLA WebSocket endpoint")
    parser.add_argument("--policy-host", help="legacy player VLA host")
    parser.add_argument("--policy-port", type=int, help="legacy player VLA port")
    parser.add_argument("--isaac-python", type=Path, help="overrides ROBOLAB_ISAAC_PYTHON")
    parser.add_argument("--runtime-script", type=Path, help="overrides ROBOLAB_M20_RUNTIME")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="overrides ROBOLAB_M20_RUNTIME_ROOT and the package runtime_root",
    )
    parser.add_argument(
        "--locomotion-policy",
        type=Path,
        help="player-provided low-level M20 locomotion policy.onnx",
    )
    parser.add_argument(
        "--teacher-episode",
        type=Path,
        help="overrides ROBOLAB_M20_TEACHER_EPISODE and the package teacher_episode",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--capture-hz", type=int, default=10)
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="recorded video FPS; defaults to capture-hz",
    )
    parser.add_argument(
        "--disable-follow-video",
        action="store_true",
        help=(
            "do not create the optional fourth-person follow camera or MP4; "
            "the three robot-mounted VLA cameras remain enabled and are still "
            "written to the episode and submission HDF5"
        ),
    )
    parser.add_argument(
        "--disable-lidar",
        action="store_true",
        help=(
            "skip RTX LiDAR for an explicitly non-LiDAR attack run; RGB, IMU, "
            "state, prompt and action evidence remain enabled"
        ),
    )
    parser.add_argument(
        "--vla-action-hz",
        type=int,
        default=None,
        help="high-level action rate; defaults to capture-hz",
    )
    parser.add_argument(
        "--route-forward-speed-scale",
        type=float,
        default=1.0,
        help="bounded scale for Runner-authored route-controller planar commands",
    )
    parser.add_argument("--base-vx-limit", type=float, default=0.30)
    parser.add_argument("--base-yaw-rate-limit", type=float, default=0.70)
    parser.add_argument("--maximum-vla-actions", type=int, default=20000)
    parser.add_argument("--maximum-duration-seconds", type=float, default=600.0)
    parser.add_argument(
        "--resume-completed-collector",
        action="store_true",
        help="finalize artifacts already produced by a persistent main-Runner batch",
    )
    parser.add_argument("--attack-mode", choices=("on", "off"), default="on")
    parser.add_argument(
        "--attack-variant",
        help=(
            "select one named profile from contract.attack_variants; "
            "the selected profile must contain exactly one enabled attack"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate and resolve the package without starting Isaac Sim or contacting the VLA",
    )
    parser.add_argument(
        "--navigation-mode",
        choices=("vla", "lidar", "hybrid"),
        help="override task navigation_mode; hybrid keeps VLA with a last-moment LiDAR guard",
    )
    parser.add_argument(
        "--kinematic-base",
        action="store_true",
        help=(
            "internal integration-test backend: execute accepted 3-D base commands "
            "by deterministic root-pose interpolation instead of the 57-to-16 ONNX"
        ),
    )
    parser.add_argument("--experiment-seed", type=int)
    args = parser.parse_args()
    if args.video_fps is None:
        args.video_fps = args.capture_hz
    if args.vla_action_hz is None:
        args.vla_action_hz = args.capture_hz
    if args.capture_hz <= 0 or 200 % args.capture_hz:
        raise ValueError("--capture-hz must be a positive divisor of 200")
    if args.video_fps != args.capture_hz:
        raise ValueError("--video-fps must equal --capture-hz")
    if args.vla_action_hz <= 0 or 50 % args.vla_action_hz:
        raise ValueError("--vla-action-hz must be a positive divisor of 50")
    if not 0.0 < args.route_forward_speed_scale <= 1.5:
        raise ValueError("--route-forward-speed-scale must be in (0, 1.5]")
    if not 0.0 < args.base_vx_limit <= 0.48:
        raise ValueError("--base-vx-limit must be in (0, 0.48]")
    if not 0.0 < args.base_yaw_rate_limit <= 0.95:
        raise ValueError("--base-yaw-rate-limit must be in (0, 0.95]")
    if args.maximum_vla_actions <= 0:
        raise ValueError("--maximum-vla-actions must be positive")
    if (
        not math.isfinite(args.maximum_duration_seconds)
        or args.maximum_duration_seconds <= 0.0
    ):
        raise ValueError("--maximum-duration-seconds must be positive and finite")

    package = args.package_root.resolve(strict=True)
    release_root = package_release_root(package)
    output = args.output_dir.resolve()
    publish_dir = args.publish_dir.expanduser().resolve() if args.publish_dir else None
    if publish_dir is not None and (
        publish_dir == output
        or publish_dir.is_relative_to(output)
        or output.is_relative_to(publish_dir)
    ):
        raise ValueError("publish directory and internal work directory must not overlap")
    if publish_dir is not None and publish_dir.exists():
        raise FileExistsError(f"refusing to overwrite publish directory: {publish_dir}")
    output_nonempty = output.exists() and any(output.iterdir())
    if output_nonempty and not args.resume_completed_collector:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    if args.resume_completed_collector and not output_nonempty:
        raise FileNotFoundError(f"persistent collector output is missing: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract_paths = sorted(package.glob("contracts/*/contract.json"))
    if len(contract_paths) != 1:
        raise ValueError(f"expected exactly one M20 contract, found {len(contract_paths)}")
    contract = load_json(contract_paths[0])
    package_task_path = package / "data/task/task.json"
    task = load_json(package_task_path)
    if args.experiment_seed is not None:
        task["seed"] = args.experiment_seed
    environment = load_json(package / "data/env/environment.json")
    manifest = load_json(package / "package_manifest.json")
    runtime_support_result = release_runtime_support(release_root)
    attack_variants = contract.get("attack_variants", {})
    if args.attack_variant is not None:
        if not isinstance(attack_variants, dict) or args.attack_variant not in attack_variants:
            available = sorted(attack_variants) if isinstance(attack_variants, dict) else []
            raise ValueError(
                f"unknown attack variant {args.attack_variant!r}; available={available}"
            )
        attack_config_value = attack_variants[args.attack_variant]
    else:
        attack_config_value = contract.get("attack_config", contract.get("joint_attack_config"))
    if not attack_config_value:
        raise ValueError("contract does not declare attack_config")
    attack_path = package_resource(package, attack_config_value, "attack config")
    attack = load_json(attack_path)
    if args.attack_variant is not None:
        enabled_attacks = [
            spec
            for spec in attack.get("attack", {}).get("attacks", [])
            if bool(spec.get("enabled", True))
        ]
        if len(enabled_attacks) != 1:
            raise ValueError(
                f"attack variant {args.attack_variant!r} must enable exactly one attack, "
                f"found {len(enabled_attacks)}"
            )
    safety_profile = None
    safety_path = None
    safety_audit = {
        "schema": "robolab-m20-safety-profile-resolution/v1",
        "enabled": False,
        "reason": "question contract does not declare safety_config",
    }
    safety_config_value = contract.get("safety_config")
    if safety_config_value:
        if "safety_profile" in task:
            raise ValueError(
                "task embeds safety_profile while contract also declares safety_config"
            )
        safety_path = package_resource(
            package,
            safety_config_value,
            "safety config",
            release_root=release_root,
        )
        safety_profile = validate_safety_profile(load_json(safety_path))
        task = copy.deepcopy(task)
        task["safety_profile"] = safety_profile
        task["safety_profile_sha256"] = canonical_profile_sha256(safety_profile)
        safety_audit = {
            "schema": "robolab-m20-safety-profile-resolution/v1",
            "enabled": True,
            "source": str(safety_path),
            "source_file_sha256": sha256(safety_path),
            "canonical_profile_sha256": task["safety_profile_sha256"],
            "profile_id": safety_profile["profile_id"],
            "rule_set_version": safety_profile["rule_set_version"],
            "enabled_rules": list(enabled_rule_ids(safety_profile)),
            "activation": dict(safety_profile["activation"]),
            "response_mode": safety_profile["response_mode"],
        }
    package_profile_root = package_path(
        package,
        environment.get("attack_profile_root", "data/attack/profiles"),
    ).resolve(strict=True)
    if contract.get("schema") not in SCHEMAS:
        raise ValueError("unsupported M20 question contract")
    package_diagnostics = validate_identifiers(manifest, contract, task, attack)

    policy_host, policy_port, policy_endpoint = policy_address(
        args.policy_endpoint, args.policy_host, args.policy_port, contract
    )

    scene = package_resource(
        package, environment["scene_usd"], "scene USD", release_root=release_root
    )
    actual_scene_sha256 = sha256(scene)
    declared_robot = package_path(package, environment["robot_usd"])
    if runtime_support_result is not None:
        runtime_support, runtime_support_audit = runtime_support_result
        robot = runtime_support["robot_usd"]
        robot_resolution = {
            "declared": str(declared_robot),
            "resolved": str(robot),
            "method": "release_runtime_support_manifest",
            "kind": "file",
        }
    else:
        runtime_support = {}
        runtime_support_audit = None
        robot, robot_resolution = adaptive_host_path(
            declared_robot,
            "M20 USD",
            kind="file",
        )
    actual_robot_sha256 = sha256(robot)
    declared_locomotion = package_path(package, environment["locomotion_policy"])
    if args.locomotion_policy is not None:
        locomotion, locomotion_resolution = adaptive_host_path(
            args.locomotion_policy,
            "player locomotion policy",
            kind="file",
        )
        locomotion_resolution.update({
            "declared": str(declared_locomotion),
            "configured": str(args.locomotion_policy),
            "method": "player_override",
        })
    elif runtime_support:
        locomotion = runtime_support["locomotion_policy"]
        locomotion_resolution = {
            "declared": str(declared_locomotion),
            "resolved": str(locomotion),
            "method": "release_runtime_support_manifest",
            "kind": "file",
        }
    else:
        locomotion, locomotion_resolution = adaptive_host_path(
            declared_locomotion,
            "M20 locomotion policy",
            kind="file",
        )
    if locomotion.suffix.lower() != ".onnx" or locomotion.stat().st_size <= 0:
        raise ValueError(f"M20 locomotion policy must be a non-empty .onnx file: {locomotion}")
    execution_mode = str(task.get("execution_mode", "navigation_only")).strip().lower()
    manipulation_only = execution_mode in {
        "manipulation",
        "manipulation_only",
        "grasp",
        "grasp_only",
    }
    if manipulation_only:
        spawn = environment.get("spawn")
        runtime_scene = environment.get("manipulation_runtime_scene")
        if not isinstance(spawn, dict):
            raise ValueError("manipulation-only task requires environment.spawn")
        if not isinstance(runtime_scene, dict):
            raise ValueError(
                "manipulation-only task requires environment.manipulation_runtime_scene"
            )
        spawn_xyz = spawn.get("xyz")
        spawn_yaw_rad = spawn.get("yaw_rad")
        if (
            not isinstance(spawn_xyz, list)
            or len(spawn_xyz) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in spawn_xyz
            )
        ):
            raise ValueError("environment.spawn.xyz must be a finite xyz array")
        if (
            isinstance(spawn_yaw_rad, bool)
            or not isinstance(spawn_yaw_rad, (int, float))
            or not math.isfinite(float(spawn_yaw_rad))
            or not -math.pi <= float(spawn_yaw_rad) <= math.pi
        ):
            raise ValueError("environment.spawn.yaw_rad must be within [-pi, pi]")
        bundle_manifest_value = environment.get("scene_bundle_manifest")
        bundle_manifest = (
            package_resource(
                package,
                bundle_manifest_value,
                "scene bundle manifest",
                release_root=release_root,
            )
            if bundle_manifest_value
            else None
        )
        route_payload = {
            "schema": "m20-piper-known-map-scene-route/v1",
            "name": f"{task['task_id']}_runtime_bootstrap",
            "maximum_duration_s": float(args.maximum_duration_seconds),
            "controller": {
                "forward_speed_mps": 0.16,
                "heading_gain": 1.0,
                "maximum_yaw_rate_rps": 0.3,
                "rotate_in_place_threshold_rad": 0.38,
                "slowdown_distance_m": 0.4,
                "arrival_tolerance_m": 0.25,
                "settle_seconds": 1.5,
                "brake_seconds": 1.0,
            },
            "spawn": {
                "xyz": [float(value) for value in spawn_xyz],
                "yaw_rad": float(spawn_yaw_rad),
            },
            # This single stationary point is an internal compatibility value
            # for the low-level collector.  It is not a task route and is never
            # exposed to the policy or used to plan the grasp approach.
            "waypoints": [[float(spawn_xyz[0]), float(spawn_xyz[1])]],
            "scene": {
                "path": str(scene),
                "default_prim": runtime_scene.get("default_prim"),
                "source_physics_scene": runtime_scene.get("source_physics_scene"),
                "minimum_collision_prim_count": runtime_scene.get(
                    "minimum_collision_prim_count"
                ),
                "camera_mode": runtime_scene.get("camera_mode"),
                "robot_radius_m": runtime_scene.get("robot_radius_m"),
                "obstacle_aabbs_xy_m": [
                    {
                        "name": "runtime_bootstrap_sentinel",
                        "minimum": [90.0, 90.0],
                        "maximum": [91.0, 91.0],
                    }
                ],
            },
        }
        if bundle_manifest is not None:
            route_payload["scene"]["bundle_manifest"] = str(bundle_manifest)
        route = output / "resolved_manipulation_bootstrap_route.json"
        route.write_text(
            json.dumps(route_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validated_route = None
    else:
        route = package_resource(package, environment["route"], "route")
        route_payload = load_json(route)
        validated_route = load_navigation_route(route)
        route_scene = route_payload.get("scene", {})
        route_scene_path = package_resource(
            route.parent,
            route_scene.get("path", ""),
            "route scene USD",
            release_root=release_root,
        )
        if route_scene_path != scene:
            _package_warning(
                package_diagnostics,
                f"environment and route resolve different scene USDs: "
                f"{scene} != {route_scene_path}",
            )
        bundle_manifest_value = route_scene.get("bundle_manifest")
        bundle_manifest = (
            package_resource(
                route.parent,
                bundle_manifest_value,
                "scene bundle manifest",
                release_root=release_root,
            )
            if bundle_manifest_value
            else None
        )
    package_metadata = validate_package_metadata(
        package,
        manifest,
        contract,
        task,
        attack,
        attack_path,
        actual_scene_sha256,
        variant_override=args.attack_variant is not None,
    )
    package_diagnostics.extend(package_metadata["warnings"])
    selected_package_attack_path = attack_path
    runtime_question = resolve_q17_runtime_compatibility(
        task=task,
        attack=attack,
        task_path=package_task_path,
        attack_path=attack_path,
        profile_root=package_profile_root,
        output=output,
    )
    task = runtime_question.task
    selected_runtime_attack = runtime_question.attack
    selected_runtime_attack_path = runtime_question.attack_path
    runtime_task_path = runtime_question.task_path
    attack, attack_catalog = build_attack_marker_catalog(
        package=package,
        contract=contract,
        scenario_id=str(contract["scenario_id"]),
        selected_attack=selected_runtime_attack,
        selected_source_path=selected_package_attack_path,
    )
    runtime_profile_root = output / "resolved_attack_profile"
    runtime_profile_root.mkdir(
        parents=True,
        exist_ok=bool(args.resume_completed_collector),
    )
    attack_path = runtime_profile_root / "all_attack_definitions.json"
    attack_path.write_text(
        json.dumps(attack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime_index = {
        "schema": "m20-question-attack-profile-index/v1",
        "mode": "runner-all-condition-markers-selected-effect",
        "questions": {
            str(contract["scenario_id"]): {
                "profile": attack_path.name,
            }
        },
        "marker_catalog": attack_catalog,
    }
    (runtime_profile_root / "index.json").write_text(
        json.dumps(runtime_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    task, runtime_task_path, episode_limits = apply_runtime_episode_limits(
        task=task,
        task_path=runtime_task_path,
        output=output,
        maximum_vla_actions=args.maximum_vla_actions,
        maximum_duration_seconds=args.maximum_duration_seconds,
    )
    package_diagnostics.extend(validate_identifiers(manifest, contract, task, attack))
    declared_runtime_script = package_path(
        package,
        environment.get("runtime_script", "m20_runtime/runners/m20_fourview_runner.py"),
    )
    if args.runtime_script is not None or os.environ.get("ROBOLAB_M20_RUNTIME"):
        m20_runner, runtime_script_resolution = configured_path(
            args.runtime_script,
            "ROBOLAB_M20_RUNTIME",
            declared_runtime_script,
        )
    else:
        m20_runner, runtime_script_resolution = adaptive_host_path(
            LOCAL_M20_RUNNER,
            "runner-local M20 runtime",
            kind="file",
        )
        runtime_script_resolution.update({
            "declared": str(declared_runtime_script),
            "method": "runner_self_location",
        })

    declared_teacher = package_path(
        package,
        environment.get("teacher_episode", "teacher/episode_000055.npz"),
    )
    if args.teacher_episode is not None or os.environ.get("ROBOLAB_M20_TEACHER_EPISODE"):
        teacher, teacher_resolution = configured_path(
            args.teacher_episode,
            "ROBOLAB_M20_TEACHER_EPISODE",
            declared_teacher,
        )
        if args.teacher_episode is not None:
            teacher_resolution.update({
                "declared": str(declared_teacher),
                "configured": str(args.teacher_episode),
                "method": "player_override",
            })
    elif runtime_support:
        teacher = runtime_support["teacher_episode"]
        teacher_resolution = {
            "declared": str(declared_teacher),
            "resolved": str(teacher),
            "method": "release_runtime_support_manifest",
            "kind": "file",
        }
    else:
        teacher, teacher_resolution = adaptive_host_path(
            declared_teacher,
            "M20 teacher episode",
            kind="file",
        )

    declared_runtime_root = package_path(
        package,
        environment.get("runtime_root", "m20_runtime"),
    )
    if args.runtime_root is not None or os.environ.get("ROBOLAB_M20_RUNTIME_ROOT"):
        runtime_root, runtime_root_resolution = configured_directory(
            args.runtime_root,
            "ROBOLAB_M20_RUNTIME_ROOT",
            declared_runtime_root,
        )
    else:
        runtime_root, runtime_root_resolution = adaptive_host_path(
            LOCAL_RUNTIME_ROOT,
            "runner-local M20 runtime root",
            kind="directory",
        )
        runtime_root_resolution.update({
            "declared": str(declared_runtime_root),
            "method": "runner_self_location",
        })
    registry_resolution = validate_runtime_registries(runtime_root, task, attack)
    isaac_python, isaac_python_resolution = configured_path(
        args.isaac_python,
        "ROBOLAB_ISAAC_PYTHON",
        ISAAC_PYTHON,
        preserve_symlink=True,
    )

    configured_attacks = (
        list(attack.get("attack", {}).get("attacks", []))
        if attack.get("schema")
        in {"m20-attack-experiment/v3", "embodied-attack-runtime/v1"}
        else []
    )
    runtime_config = {
        "schema": "embodied-attack-runtime/v1",
        "experiment": {"name": task["task_id"], "seed": task["seed"], "output_dir": str(output)},
        "attack": {
            "enabled": args.attack_mode == "on",
            "marker_only": args.attack_mode == "off",
            # Every attack definition declared by the question remains a
            # hypothetical condition source.  ``effect_enabled`` selects at
            # most the requested experiment effect; attack_mode=off disables
            # every effect while retaining all condition markers.
            "attacks": configured_attacks,
            "evidence_dir": str(output / "attack_evidence"),
        },
        "marker_catalog": attack_catalog,
    }
    runtime_config_path = output / "resolved_vla_boundary_config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")

    command = [
        str(isaac_python), str(m20_runner),
        "--route", str(route),
        "--teacher-episode", str(teacher),
        "--task-config", str(runtime_task_path),
        "--output", str(output),
        "--seed", str(task["seed"]),
        "--gpu", str(args.gpu),
        "--portable-root", str(output / ".kit"),
        "--capture-hz", str(args.capture_hz),
        "--video-fps", str(args.video_fps),
        "--vla-action-hz", str(args.vla_action_hz),
        "--route-forward-speed-scale", str(args.route_forward_speed_scale),
        "--base-vx-limit", str(args.base_vx_limit),
        "--base-yaw-rate-limit", str(args.base_yaw_rate_limit),
        "--policy-host", policy_host, "--policy-port", str(policy_port),
        "--maximum-vla-actions", str(task["maximum_vla_actions"]),
        "--navigation-mode", str(args.navigation_mode or task.get("navigation_mode", "vla")),
    ]
    if not args.disable_follow_video:
        command.append("--record-video")
    if args.disable_lidar:
        command.append("--disable-lidar")
    if task.get("execution_mode", "navigation_only") == "navigation_only":
        command.append("--navigation-only")
    if manipulation_only:
        command.append("--manipulation-only")
    if args.kinematic_base:
        command.append("--kinematic-base")
    legacy_joint_attack = attack.get("schema") == "m20-joint-command-attack/v1"
    if args.attack_mode == "on" and legacy_joint_attack:
        command.extend(["--joint-attack-config", str(attack_path)])
    env = os.environ.copy()
    packages_root = Path(__file__).resolve().parents[2]
    python_paths = [
        str(runtime_root),
        str(PROTOCOL_V072_VENDOR_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    question_id = contract.get("scenario_id")
    # AppLauncher selects the requested physical GPU through --gpu/device and
    # renderer settings.  Do not remap CUDA device ordinals here: Omniverse
    # Vulkan keeps physical ordinals, and the mismatch breaks CUDA/RTX interop.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({
        "PYTHONPATH": os.pathsep.join(python_paths),
        "M20PRO_PIPER_ROOT": str(robot.parents[1]),
        "M20_ROBOT_USD": str(robot),
        "M20_LOCOMOTION_POLICY": str(locomotion),
        "M20_RUNTIME_ROOT": str(runtime_root),
        "M20_ATTACK_ENABLED": "1" if args.attack_mode == "on" else "0",
        "M20_ATTACK_EVIDENCE_DIR": str(output / "attack_evidence"),
        "M20_QUESTION_PACKAGE_ROOT": str(package),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    })
    if question_id:
        env["M20_QUESTION_ID"] = str(question_id)
        env["M20_ATTACK_PROFILE_ROOT"] = str(runtime_profile_root)
    else:
        env["M20_EXPERIMENT_CONFIG"] = str(runtime_config_path)
    preflight = {
        "schema": "robolab-m20-question-preflight/v1",
        "contract": contract,
        "attack_mode": args.attack_mode,
        "attack_variant": args.attack_variant,
        "available_attack_variants": (
            sorted(attack_variants) if isinstance(attack_variants, dict) else []
        ),
        "attack_catalog": {
            **attack_catalog,
            "effects_globally_enabled": args.attack_mode == "on",
            "selected_runtime_profile": str(selected_runtime_attack_path),
            "merged_runtime_profile": str(attack_path),
        },
        "policy_endpoint": policy_endpoint,
        "scene": {"path": str(scene), "sha256": actual_scene_sha256},
        "scene_bundle_manifest": {
            "path": str(bundle_manifest) if bundle_manifest is not None else None,
            "validation": "disabled",
        },
        "route": {
            "required_by_task": not manipulation_only,
            "path": str(route),
            "schema": route_payload.get("schema"),
            "name": (
                route_payload["name"]
                if manipulation_only
                else validated_route.name
            ),
            "waypoint_count": (
                0 if manipulation_only else len(validated_route.waypoints_xy)
            ),
            "maximum_duration_s": (
                float(route_payload["maximum_duration_s"])
                if manipulation_only
                else validated_route.maximum_duration_s
            ),
            "runtime_bootstrap_only": manipulation_only,
        },
        "release_runtime_support": runtime_support_audit,
        "registry_resolution": registry_resolution,
        "path_resolution": {
            "robot_usd": robot_resolution,
            "locomotion_policy": locomotion_resolution,
            "runtime_script": runtime_script_resolution,
            "runtime_root": runtime_root_resolution,
            "teacher_episode": teacher_resolution,
            "isaac_python": isaac_python_resolution,
        },
        "package_metadata": package_metadata,
        "package_diagnostics": package_diagnostics,
        "runtime_question_compatibility": runtime_question.audit,
        "episode_limits": episode_limits,
        "safety_profile": safety_audit,
        "robot": {"path": str(robot), "sha256": actual_robot_sha256},
        "locomotion_policy": {"path": str(locomotion), "sha256": sha256(locomotion)},
        "teacher_episode": {"path": str(teacher), "sha256": sha256(teacher)},
        "command": command,
    }
    (output / "runner_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0
    if args.resume_completed_collector:
        required_collector = [
            output / "result.json",
            output / "episode.npz",
            output / "vla_trace.json",
            output / "fixed_evidence_sidecar.hdf5",
            output / "runner.log",
        ]
        missing_collector = [str(path) for path in required_collector if not path.is_file()]
        if missing_collector:
            raise RuntimeError(
                f"persistent collector outputs are incomplete: {missing_collector}"
            )
        completed = subprocess.CompletedProcess(command, 0)
    else:
        with (output / "runner.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

    collector_error_path = output / "error.json"
    if collector_error_path.is_file():
        collector_error = load_json(collector_error_path)
        error_type = collector_error.get("error_type", "RuntimeError")
        message = collector_error.get("message", "collector failed without an error message")
        traceback_text = collector_error.get("traceback", "")
        detail = f"M20 collector failed ({error_type}): {message}"
        if traceback_text:
            detail += f"\nCollector traceback:\n{traceback_text.rstrip()}"
        raise RuntimeError(detail)

    if completed.returncode != 0:
        raise RuntimeError(
            f"M20 collector exited with code {completed.returncode}; see {output / 'runner.log'}"
        )

    required = [
        output / "result.json", output / "episode.npz",
        output / "vla_trace.json",
        output / "attack_evidence/vla_boundary_audit.jsonl",
        output / "fixed_evidence_sidecar.hdf5",
    ]
    if task.get("rubric_evidence"):
        required += [output / "rubric_evidence_sidecar.hdf5", output / "rubric_evidence_contract.json"]
    if safety_profile is not None:
        required += [
            output / "resolved_safety_profile.json",
            output / "safety_evidence.jsonl",
            output / "safety_summary.json",
        ]
    if not args.disable_follow_video:
        required.append(output / "navigation.mp4")
    if args.attack_mode == "on" and legacy_joint_attack:
        required.append(output / "joint_attack_trace.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"M20 runner exit={completed.returncode}; missing outputs: {missing}")
    result = load_json(output / "result.json")
    raw_capture = write_raw_capture(output, task)
    export_formal_submission(output, raw_capture)
    final_submission, protocol_report = append_attack_evidence_and_validate(
        output,
        attack_path,
        task,
        safety_profile,
    )
    if (output / "navigation.mp4").is_file():
        os.replace(output / "navigation.mp4", output / "episode.mp4")

    vla_audit_lines = sum(1 for _ in (output / "attack_evidence/vla_boundary_audit.jsonl").open(encoding="utf-8"))
    record = {
        "schema": "robolab-m20-question-record/v1",
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "attack_mode": args.attack_mode,
        "attack_variant": args.attack_variant,
        "runner_exit_code": completed.returncode,
        "simulation_completed": True,
        "package_diagnostics": package_diagnostics,
        "task_accepted": bool(result.get("accepted", False)),
        "vla_boundary_records": vla_audit_lines,
        "attack_profile": str(attack_path),
        "attack_catalog": {
            **attack_catalog,
            "effects_globally_enabled": args.attack_mode == "on",
            "selected_runtime_profile": str(selected_runtime_attack_path),
        },
        "attack_runtime_enabled": args.attack_mode == "on",
        "safety_profile": safety_audit,
        "runtime_question_compatibility": runtime_question.audit,
        "episode_limits": episode_limits,
        "artifacts": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in {
                "submission": final_submission,
                "runner_capture": raw_capture,
                "vla_trace": output / "vla_trace.json",
                **(
                    {"video": output / "episode.mp4"}
                    if (output / "episode.mp4").is_file()
                    else {}
                ),
                **(
                    {"joint_attack_trace": output / "joint_attack_trace.jsonl"}
                    if args.attack_mode == "on" and legacy_joint_attack
                    else {}
                ),
                **(
                    {
                        "safety_profile": output / "resolved_safety_profile.json",
                        "safety_evidence": output / "safety_evidence.jsonl",
                        "safety_summary": output / "safety_summary.json",
                    }
                    if safety_profile is not None
                    else {}
                ),
            }.items()
        },
        "protocol_v072_validation": protocol_report,
    }
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if publish_dir is not None:
        record["published_artifacts"] = publish_final_artifacts(
            output,
            publish_dir,
            protocol_report,
        )
        try:
            shutil.rmtree(output)
        except OSError as error:
            print(
                f"warning: could not remove internal work directory {output}: {error}",
                file=sys.stderr,
            )
        print(f"ROBOLAB_FINAL_OUTPUT={publish_dir}", flush=True)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
