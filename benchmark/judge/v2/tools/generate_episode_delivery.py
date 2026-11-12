"""Generate generic full + summary Judge deliveries from HDF5 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import h5py

from v2.safety.judge_result.assembler import save_final_result
from v2.safety.judge_result.episode_delivery import (
    build_episode_delivery,
    condition_from_variant,
    read_hdf5_identity,
    resolve_clean_pair,
    sha256_file,
)
from v2.safety.judge_result.summary import save_summary

_UNSAFE_PATH = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    safe = _UNSAFE_PATH.sub("_", value.strip())
    if not safe or safe in {".", ".."}:
        raise ValueError("unsafe empty path identity")
    return safe


def _resolve_input(value: str, manifest_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    workspace_candidate = candidate.resolve()
    return (manifest_dir / candidate).resolve() if (manifest_dir / candidate).exists() else workspace_candidate


def extract_hdf5_archive(archive_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Safely extract only HDF5 members, flattening names into a private directory."""
    source = Path(archive_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(source) as archive:
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            normalized = info.filename.replace("\\", "/")
            member = PurePosixPath(normalized)
            mode = info.external_attr >> 16
            if (
                member.is_absolute()
                or ".." in member.parts
                or (member.parts and ":" in member.parts[0])
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe ZIP member path: {info.filename}")
            if info.is_dir() or member.suffix.lower() not in {".hdf5", ".h5"}:
                continue
            target = destination / f"{len(extracted) + 1:06d}_{_safe_component(member.name)}"
            with archive.open(info) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            extracted.append(target)
    if not extracted:
        raise ValueError("HDF5 archive contains no .hdf5 or .h5 files")
    return extracted


def load_archive_selection_manifest(
    archive_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load the archive's trusted file ledger, keyed by declared SHA-256."""
    source = Path(archive_path).expanduser().resolve()
    with zipfile.ZipFile(source) as archive:
        candidates = sorted(
            name for name in archive.namelist()
            if PurePosixPath(name.replace("\\", "/")).name == "selection_manifest.json"
        )
        if not candidates:
            return {}
        if len(candidates) != 1:
            raise ValueError("HDF5 archive contains multiple selection_manifest.json files")
        payload = json.loads(archive.read(candidates[0]))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("selection_manifest.json files must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"selection manifest files[{position}] must be an object")
        digest = row.get("sha256")
        task_id = row.get("task_id")
        condition = row.get("condition")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"selection manifest files[{position}] has invalid sha256")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"selection manifest files[{position}] has invalid task_id")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"selection manifest files[{position}] has invalid condition")
        key = digest.lower()
        if key in indexed:
            raise ValueError(f"selection manifest contains duplicate sha256: {key}")
        indexed[key] = dict(row)
    return indexed


def load_directory_selection_manifest(root: str | Path) -> dict[str, dict[str, Any]]:
    candidates = sorted(Path(root).expanduser().resolve().rglob("selection_manifest.json"))
    if not candidates:
        return {}
    if len(candidates) != 1:
        raise ValueError("HDF5 directory contains multiple selection_manifest.json files")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("selection_manifest.json files must be an array")
    indexed = {}
    for position, row in enumerate(rows):
        digest = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"selection manifest files[{position}] has invalid sha256")
        if digest.lower() in indexed:
            raise ValueError(f"selection manifest contains duplicate sha256: {digest.lower()}")
        indexed[digest.lower()] = dict(row)
    return indexed


def _rule_task_index(rubric_archive: str | Path) -> dict[str, str]:
    archive_path = Path(rubric_archive).expanduser().resolve()
    index: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if re.fullmatch(r"tasks/(Q[1-9][0-9]*)/rule_package\.json", name)
        )
        if not names:
            raise ValueError("Rule Package archive contains no task rule packages")
        for name in names:
            question_id = name.split("/")[1]
            package = json.loads(archive.read(name))
            task_id = package.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{name} has no non-empty task_id")
            if task_id in index:
                raise ValueError(f"duplicate task_id in Rule Package archive: {task_id}")
            index[task_id] = question_id
    return index


def _registered_attacks_by_question(rubric_archive: str | Path) -> dict[str, tuple[str, ...]]:
    archive_path = Path(rubric_archive).expanduser().resolve()
    result: dict[str, tuple[str, ...]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(
            value for value in archive.namelist()
            if re.fullmatch(r"tasks/(Q[1-9][0-9]*)/task_info\.json", value)
        ):
            question_id = name.split("/")[1]
            task_info = json.loads(archive.read(name))
            rows = task_info.get("registered_attacks", [])
            result[question_id] = tuple(
                str(row["attack_id"])
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("attack_id"), str)
            )
    return result


def _declared_condition_id(
    label: str, question_id: str, attack_ids: tuple[str, ...]
) -> str:
    value = label.strip()
    if value == "clean":
        return "clean"
    if value.startswith("attack__"):
        attack_id = value[len("attack__"):]
        if attack_id in attack_ids:
            return value
    else:
        candidates = [
            attack_id for attack_id in attack_ids
            if attack_id == value or attack_id.endswith("_" + value)
        ]
        if len(candidates) == 1:
            return "attack__" + candidates[0]
    raise ValueError(
        f"condition {label!r} does not identify one registered attack for {question_id}"
    )


def _hdf5_task_id(path: Path) -> str:
    with h5py.File(path, "r") as hdf5:
        metadata = hdf5.get("/metadata")
        if not isinstance(metadata, h5py.Group):
            raise ValueError("HDF5 is missing /metadata")
        value = metadata.attrs.get("task_id")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        task_id = str(value or "").strip()
        if not task_id:
            raise ValueError("HDF5 metadata is missing task_id")
        return task_id


def build_auto_manifest(
    hdf5_paths: Iterable[str | Path], rubric_archive: str | Path, *,
    selection_by_sha256: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build trusted batch entries from HDF5 metadata and a Rule Package archive."""
    archive_path = Path(rubric_archive).expanduser().resolve()
    task_index = _rule_task_index(archive_path)
    attacks_by_question = _registered_attacks_by_question(archive_path)
    rows: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for position, value in enumerate(sorted(Path(path).resolve() for path in hdf5_paths), 1):
        base = {
            "question_id": f"UNMAPPED{position:06d}",
            "hdf5_path": str(value),
            "rubric_archive": str(archive_path),
        }
        try:
            task_id = _hdf5_task_id(value)
            question_id = task_index.get(task_id)
            if question_id is None:
                raise ValueError(f"task_id is not present in Rule Package archive: {task_id}")
            if selection_by_sha256 is not None:
                digest = sha256_file(value)
                declared = selection_by_sha256.get(digest)
                if declared is None:
                    raise ValueError(
                        f"HDF5 sha256 is not declared by selection_manifest.json: {digest}"
                    )
                if declared["task_id"] != task_id:
                    raise ValueError("selection manifest task_id does not match HDF5 metadata")
                condition = _declared_condition_id(
                    str(declared["condition"]), question_id,
                    attacks_by_question.get(question_id, ()),
                )
                base["declared_condition_id"] = condition
                base["declared_identity"] = {
                    key: value for key, value in declared.items()
                    if key in {"task_id", "system_id", "system_version", "seed", "policy_seed",
                               "attack_seed", "repeat_index", "run_id", "attempt_id", "purpose",
                               "scene_id", "block_id"} and value is not None
                }
                for source_key, target_key in (
                    ("system", "system_id"), ("scene", "scene_id"),
                    ("attempt", "attempt_id"),
                ):
                    if declared.get(source_key) is not None:
                        base["declared_identity"][target_key] = declared[source_key]
                base["declared_identity"]["condition_id"] = condition
                base["collection_declaration"] = {
                    key: declared.get(key)
                    for key in (
                        "file", "source_hdf5", "source_validation", "collection_status",
                        "score_status", "copy_verified", "sha256", "bytes",
                        "rubric_sha256",
                    ) if declared.get(key) is not None
                }
            else:
                identity = read_hdf5_identity(value, question_id=question_id)
                condition = (
                    identity.get("declared_condition_id")
                    or condition_from_variant(
                        identity.get("runner_variant"),
                        tuple({"attack_id": attack_id} for attack_id in attacks_by_question.get(question_id, ())),
                    )
                    or "unavailable"
                )
                if condition != "unavailable":
                    base["declared_condition_id"] = condition
            base.update({"question_id": question_id, "expected_task_id": task_id})
            inspected.append({"row": base, "condition_id": condition})
        except Exception as error:
            base["_input_error"] = str(error)
        rows.append(base)

    clean_rows = [item for item in inspected if item["condition_id"] == "clean"]
    for item in inspected:
        if item["condition_id"] in {"clean", "unavailable"}:
            continue
        row = item["row"]
        compatible: list[dict[str, Any]] = []
        for clean in clean_rows:
            if clean["row"]["question_id"] != row["question_id"]:
                continue
            try:
                resolve_clean_pair(
                    attack_path=Path(row["hdf5_path"]),
                    clean_path=Path(clean["row"]["hdf5_path"]),
                    question_id=row["question_id"],
                    clean_declared_condition_id="clean",
                    attack_declared_identity=row.get("declared_identity"),
                    clean_declared_identity=clean["row"].get("declared_identity"),
                )
            except ValueError:
                continue
            compatible.append(clean["row"])
        if len(compatible) == 1:
            row["matched_clean_hdf5_path"] = compatible[0]["hdf5_path"]
            row["matched_clean_declared_identity"] = compatible[0].get("declared_identity")
        elif len(compatible) > 1:
            row["_input_error"] = (
                "multiple compatible clean HDF5 files; explicit pairing is required"
            )
    return {"episodes": rows}


def _load_rules(
    entry: dict[str, Any], manifest_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, str]]:
    question_id = str(entry["question_id"])
    if "rubric_archive" in entry:
        archive_path = _resolve_input(str(entry["rubric_archive"]), manifest_dir)
        with zipfile.ZipFile(archive_path) as archive:
            rubric_raw = archive.read(f"tasks/{question_id}/rubric.json")
            package_raw = archive.read(f"tasks/{question_id}/rule_package.json")
            rubric = json.loads(rubric_raw)
            package = json.loads(package_raw)
            try:
                task_info = json.loads(archive.read(f"tasks/{question_id}/task_info.json"))
            except KeyError:
                task_info = None
        source_digests = {
            "rubric_source_sha256": hashlib.sha256(rubric_raw).hexdigest(),
            "rule_package_source_sha256": hashlib.sha256(package_raw).hexdigest(),
        }
        runner_digest = (
            task_info.get("runner_identity", {}).get("source_rubric_sha256")
            if isinstance(task_info, dict) else None
        )
        if runner_digest is not None:
            if not isinstance(runner_digest, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", runner_digest
            ):
                raise ValueError("task_info runner source_rubric_sha256 is invalid")
            source_digests["runner_rubric_source_sha256"] = runner_digest.lower()
        return rubric, package, task_info, source_digests
    rubric_path = _resolve_input(str(entry["rubric_path"]), manifest_dir)
    package_path = _resolve_input(str(entry["rule_package_path"]), manifest_dir)
    task_info = None
    if entry.get("task_info_path"):
        task_info_path = _resolve_input(str(entry["task_info_path"]), manifest_dir)
        task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
    rubric_raw = rubric_path.read_bytes()
    package_raw = package_path.read_bytes()
    source_digests = {
        "rubric_source_sha256": hashlib.sha256(rubric_raw).hexdigest(),
        "rule_package_source_sha256": hashlib.sha256(package_raw).hexdigest(),
    }
    runner_digest = (
        task_info.get("runner_identity", {}).get("source_rubric_sha256")
        if isinstance(task_info, dict) else None
    )
    if runner_digest is not None:
        if not isinstance(runner_digest, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", runner_digest
        ):
            raise ValueError("task_info runner source_rubric_sha256 is invalid")
        source_digests["runner_rubric_source_sha256"] = runner_digest.lower()
    return (
        json.loads(rubric_raw), json.loads(package_raw), task_info,
        source_digests,
    )


def merge_metadata(manifest, metadata_path):
    """One organizer sidecar for all entry modes, keyed by actual input bytes."""
    from v2.common.evidence_io import decode
    metadata_path=metadata_path.expanduser().resolve()
    sidecar=decode(metadata_path.read_bytes(),str(metadata_path))
    declarations={}
    for row in sidecar.get('episodes',[]):
        digest=row['hdf5_sha256']
        if digest in declarations:raise ValueError('duplicate metadata evidence SHA-256')
        declarations[digest]=row
    base=Path(manifest['_manifest_directory'])
    for row in manifest['episodes']:
        path=_resolve_input(row['hdf5_path'],base);row['hdf5_path']=str(path)
        declared=declarations.get(sha256_file(path))
        if declared is None:raise ValueError('input missing from trusted batch metadata')
        for key,value in declared.items():
            if key in ('hdf5_path','hdf5_sha256'):continue
            if key.endswith('_path') or key=='rubric_archive':value=str(_resolve_input(value,metadata_path.parent))
            if key in ('question_id','system_id','system_version') and row.get(key) is not None and row[key]!=value:
                raise ValueError('METADATA_IDENTITY_CONFLICT:'+key)
            row[key]=value
    manifest['result_metadata']={**manifest.get('result_metadata',{}),**sidecar.get('result_metadata',{})}
    return manifest


def generate_batch(manifest_path: str | Path, output_root: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["_manifest_directory"] = str(manifest_file.parent)
    if metadata_path:
        manifest = merge_metadata(manifest, Path(metadata_path))
    entries = manifest.get("episodes", [])
    if not isinstance(entries, list):
        raise ValueError("manifest.episodes must be an array")
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("output directory must be empty; existing judgements are immutable")
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for entry in entries:
        question_id = str(entry.get("question_id", ""))
        try:
            if entry.get("_input_error"):
                raise ValueError(str(entry["_input_error"]))
            safe_question = _safe_component(question_id)
            if safe_question != question_id:
                raise ValueError("question_id must already be a normalized path-safe identifier")
            hdf5_path = _resolve_input(str(entry["hdf5_path"]), manifest_file.parent)
            rubric, package, task_info, rule_source_digests = _load_rules(
                entry, manifest_file.parent
            )
            expected_task_id = entry.get("expected_task_id")
            if expected_task_id is not None and package.get("task_id") != expected_task_id:
                raise ValueError("manifest expected_task_id does not match Rule Package")
            expected_contract = None
            if entry.get("runner_judge_contract_path"):
                from v2.common.recorded_contract import load_trusted_contract
                expected_contract = load_trusted_contract(
                    _resolve_input(entry["runner_judge_contract_path"], manifest_file.parent),
                    entry.get("runner_judge_contract_sha256"))
            matched_clean_episode_id = None
            matched_clean_pairing = None
            if entry.get("matched_clean_hdf5_path"):
                clean_path = _resolve_input(
                    str(entry["matched_clean_hdf5_path"]), manifest_file.parent
                )
                clean_pair = resolve_clean_pair(
                    attack_path=hdf5_path,
                    clean_path=clean_path,
                    question_id=question_id,
                    registered_attacks=(
                        task_info.get("registered_attacks")
                        if isinstance(task_info, dict) else None
                    ),
                    clean_declared_condition_id="clean",
                    attack_declared_identity=entry.get("declared_identity"),
                    clean_declared_identity=entry.get("matched_clean_declared_identity"),
                )
                matched_clean_episode_id = clean_pair["episode_id"]
                matched_clean_pairing = clean_pair.get("_pairing_validation")
            full, summary = build_episode_delivery(
                question_id=question_id, hdf5_path=hdf5_path,
                rubric=rubric, rule_package=package,
                system_id=entry.get("system_id"),
                system_version=entry.get("system_version"),
                matched_clean_episode_id=matched_clean_episode_id,
                matched_clean_pairing=matched_clean_pairing,
                registered_attacks=(
                    task_info.get("registered_attacks")
                    if isinstance(task_info, dict) else None
                ),
                expected_safety_constraint_set_id=(
                    task_info.get("config", {}).get("constraint_set_id")
                    if isinstance(task_info, dict) else None
                ),
                declared_condition_id=entry.get("declared_condition_id"),
                declared_identity={
                    **({
                        key: entry.get(key) for key in (
                            "task_id", "system_id", "system_version", "seed", "policy_seed",
                            "attack_seed", "repeat_index", "run_id", "attempt_id", "purpose",
                            "scene_id", "block_id",
                        ) if entry.get(key) is not None
                    }),
                    **(entry.get("declared_identity") or {}),
                },
                expected_safety_config=(
                    task_info.get("config") if isinstance(task_info, dict) else None
                ),
                benchmark_contract_metadata=(
                    {
                        key: task_info.get(key)
                        for key in ("source", "readiness", "runner_identity", "_draft")
                        if task_info.get(key) is not None
                    }
                    if isinstance(task_info, dict) else None
                ),
                collection_declaration=entry.get("collection_declaration"),
                rule_source_digests=rule_source_digests,
                expected_runner_judge_contract=expected_contract,
            )
            if full["episode_id"] in seen_episode_ids:
                raise ValueError("duplicate episode_id in one manifest")
            seen_episode_ids.add(full["episode_id"])
            system_dir = (
                _safe_component(f"{full['system_id']}__{full['system_version']}")
                if full.get("system_id") and full.get("system_version")
                else "_unidentified_system"
            )
            episode_dir = root / "archive" / system_dir / full["question_id"] / "episodes" / full["episode_id"]
            save_final_result(episode_dir / "submission_result.json", full)
            save_summary(episode_dir / "submission_result.analysis_summary.json", summary)
            readiness = full.get("scoring_readiness", {})
            formal_readiness = readiness.get("formal_statistics", {})
            records.append({
                "question_id": question_id, "episode_id": full["episode_id"],
                "status": "ok",
                "validation_status": full.get("validation", {}).get("status", ""),
                "eligible_for_statistics": bool(
                    full.get("validation", {}).get("eligible_for_statistics", False)
                ),
                "scoring_readiness_statuses": {
                    key: value.get("status")
                    for key, value in readiness.items()
                    if isinstance(value, dict) and "status" in value
                },
                "formal_statistics_status": formal_readiness.get("status", "not_ready"),
                "formal_statistics_reason_codes": formal_readiness.get("reason_codes", []),
                "formal_statistics_remediation": formal_readiness.get("remediation", []),
                "output_directory": str(episode_dir),
            })
        except Exception as error:
            failure = {
                "question_id": question_id, "status": "technical_error",
                "error_type": type(error).__name__, "error": str(error),
            }
            details = getattr(error, "report", None)
            if isinstance(details, dict):
                failure["details"] = details
            records.append(failure)
    processed = [record for record in records if record["status"] == "ok"]
    formally_ready = [
        record for record in processed
        if record.get("formal_statistics_status") == "ready"
    ]
    reason_code_counts: dict[str, int] = {}
    remediation_counts: dict[str, int] = {}
    for record in processed:
        for reason in record.get("formal_statistics_reason_codes", []):
            reason_code_counts[str(reason)] = reason_code_counts.get(str(reason), 0) + 1
        for action in record.get("formal_statistics_remediation", []):
            remediation_counts[str(action)] = remediation_counts.get(str(action), 0) + 1
    from v2.common.final_results import export
    canonical = export(root, manifest, records)
    return {
        "canonical_result": canonical,
        "schema_version": "robolab-v2-episode-delivery-batch/v1",
        "success_count": sum(record["status"] == "ok" for record in records),
        "failure_count": sum(record["status"] != "ok" for record in records),
        "collection_completed_count": len(processed),
        "formal_statistics_ready_count": len(formally_ready),
        "formal_statistics_not_ready_count": len(processed) - len(formally_ready),
        "release_gate": {
            "status": "ready" if processed and len(formally_ready) == len(processed) else "not_ready",
            "criterion": "all_processed_episodes_formal_statistics_ready",
            "note": "processing success is independent from scoring readiness",
        },
        "formal_statistics_reason_code_counts": dict(sorted(reason_code_counts.items())),
        "remediation_counts": dict(sorted(remediation_counts.items())),
        "episodes": records,
    }


def create_delivery_zip(
    output_root: str | Path, zip_path: str | Path,
    report: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    target = Path(zip_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if report is None:
                paths = []
            else:
                paths = [
                    Path(record["output_directory"]) / name
                    for record in report.get("episodes", []) if record.get("status") == "ok"
                    for name in ("submission_result.json", "submission_result.analysis_summary.json")
                ]
            paths += list((root / "final_result").rglob("*"))
            paths += [root / "batch_report.json", root / "canonical_export_report.json"]
            for path in sorted(p for p in paths if p.is_file()):
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(temp_name, target)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--metadata", help="common batch metadata and SHA-256 keyed trusted contracts, for every input mode")
    parser.add_argument("--hdf5-archive")
    parser.add_argument("--hdf5-dir")
    parser.add_argument("--rubric-archive")
    parser.add_argument("--question-id")
    parser.add_argument("--hdf5")
    parser.add_argument("--rubric")
    parser.add_argument("--rule-package")
    parser.add_argument("--task-info")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--zip")
    args = parser.parse_args(argv)
    temporary_workspace: tempfile.TemporaryDirectory[str] | None = None
    remove_manifest = False
    if args.manifest:
        if any((
            args.hdf5_archive, args.hdf5_dir, args.rubric_archive,
            args.question_id, args.hdf5, args.rubric, args.rule_package, args.task_info,
        )):
            parser.error("--manifest cannot be combined with other input modes")
        manifest_path = Path(args.manifest)
    elif args.hdf5_archive or args.hdf5_dir:
        if args.hdf5_archive and args.hdf5_dir:
            parser.error("use only one of --hdf5-archive and --hdf5-dir")
        if not args.rubric_archive:
            parser.error("archive/directory mode requires --rubric-archive")
        if any((args.question_id, args.hdf5, args.rubric, args.rule_package, args.task_info)):
            parser.error("archive/directory mode cannot be combined with single-episode inputs")
        temporary_workspace = tempfile.TemporaryDirectory(prefix="episode-delivery-input-")
        temporary_root = Path(temporary_workspace.name)
        if args.hdf5_archive:
            selection_by_sha256 = load_archive_selection_manifest(args.hdf5_archive)
            hdf5_paths = extract_hdf5_archive(
                args.hdf5_archive, temporary_root / "hdf5"
            )
        else:
            hdf5_dir = Path(args.hdf5_dir).expanduser().resolve()
            if not hdf5_dir.is_dir():
                parser.error(f"HDF5 directory does not exist: {hdf5_dir}")
            hdf5_paths = sorted(
                path for path in hdf5_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}
            )
            if not hdf5_paths:
                parser.error("HDF5 directory contains no .hdf5 or .h5 files")
            selection_by_sha256 = load_directory_selection_manifest(hdf5_dir)
        manifest_path = temporary_root / "auto_manifest.json"
        auto = build_auto_manifest(hdf5_paths, args.rubric_archive, selection_by_sha256=(selection_by_sha256 or None))
        if args.hdf5_archive:
            with zipfile.ZipFile(args.hdf5_archive) as archive:
                members=[i.filename for i in sorted(archive.infolist(),key=lambda i:i.filename)
                         if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in ('.hdf5','.h5')]
            for row,member in zip(auto['episodes'],members,strict=True):
                row['persistent_evidence_source']={'kind':'zip_member','archive_path':str(Path(args.hdf5_archive).resolve()),'member':member}
        manifest_path.write_text(json.dumps(auto,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        if args.rubric_archive:
            parser.error("--rubric-archive requires --hdf5-archive or --hdf5-dir")
        if not all((args.question_id, args.hdf5, args.rubric, args.rule_package)):
            parser.error("single mode requires --question-id --hdf5 --rubric --rule-package")
        episode = {"question_id": args.question_id, "hdf5_path": args.hdf5,
                   "rubric_path": args.rubric, "rule_package_path": args.rule_package}
        if args.task_info:
            episode["task_info_path"] = args.task_info
        temp_manifest = {"episodes": [episode]}
        fd, temp_name = tempfile.mkstemp(prefix="episode-delivery-", suffix=".json")
        os.close(fd)
        manifest_path = Path(temp_name)
        manifest_path.write_text(json.dumps(temp_manifest), encoding="utf-8")
        remove_manifest = True
    try:
        report = generate_batch(manifest_path, args.output_root, args.metadata)
    finally:
        if remove_manifest:
            manifest_path.unlink(missing_ok=True)
        if temporary_workspace is not None:
            temporary_workspace.cleanup()
    save_final_result(Path(args.output_root) / "batch_report.json", report)
    if args.zip:
        create_delivery_zip(args.output_root, args.zip, report)
    console_report = {
        key: report[key] for key in (
            "schema_version", "success_count", "failure_count",
            "collection_completed_count", "formal_statistics_ready_count",
            "formal_statistics_not_ready_count", "release_gate",
            "formal_statistics_reason_code_counts", "remediation_counts",
        )
    }
    console_report["batch_report"] = str(
        Path(args.output_root).expanduser().resolve() / "batch_report.json"
    )
    if args.zip:
        console_report["delivery_zip"] = str(Path(args.zip).expanduser().resolve())
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    return 0 if report["failure_count"] == 0 and report["canonical_result"]["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
