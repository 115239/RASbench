# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Runner success/failure reports."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robolab_runner.common import load_json, sha256_file, utc_now, write_json_atomic
from robolab_runner.contract import POLICY_PROTOCOL_NAME, V4TaskContract
from robolab_runner.errors import RunnerError
from robolab_runner.validator import ValidationReport

LEGACY_RECORD_SCHEMA = "robolab-runner-record/v1"
RECORD_SCHEMA = "robolab-runner-record/v2"
LEGACY_FAILURE_SCHEMA = "robolab-runner-failure/v1"
FAILURE_SCHEMA = "robolab-runner-failure/v2"
_PORTABLE_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def build_record(
    submission_path: str | Path,
    contract: V4TaskContract,
    *,
    runner_version: str,
    endpoint_scope: str,
    policy_id: str,
    policy_version: str,
    steps: int,
    query_durations_s: list[float],
    video_path: str | Path | None = None,
    video_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    submission = Path(submission_path).expanduser().resolve()
    if submission.name != "submission.hdf5":
        raise ValueError("record report only binds submission.hdf5")
    durations = [float(value) for value in query_durations_s]
    if not durations or any(not math.isfinite(value) or value < 0 for value in durations):
        raise ValueError("record report requires finite non-negative query durations")
    record = {
        "record_schema": RECORD_SCHEMA if contract.uses_full_capture else LEGACY_RECORD_SCHEMA,
        "generated_at": utc_now(),
        "submission": {
            "file": "submission.hdf5",
            "sha256": sha256_file(submission),
            "size_bytes": submission.stat().st_size,
        },
        "contract": {"contract_id": contract.contract_id, "sha256": contract.digest},
        "versions": {
            "runner": runner_version,
            "robolab": contract.versions["robolab"],
            "isaaclab": contract.versions["isaaclab"],
            "isaacsim": contract.versions["isaacsim"],
            "policy_protocol": f"{POLICY_PROTOCOL_NAME}/{contract.policy_protocol_version}",
        },
        "endpoint_scope": endpoint_scope,
        "policy": {"policy_id": policy_id, "policy_version": policy_version},
        "episode": {"steps": int(steps), "queries": len(durations)},
        "timing": {
            "inference_total_s": float(sum(durations)),
            "inference_mean_s": float(sum(durations) / len(durations)),
            "inference_max_s": float(max(durations)),
        },
    }
    if contract.uses_full_capture:
        video = Path(video_path).expanduser().resolve() if video_path is not None else None
        if video is None or video.name != "episode.mp4" or not video.is_file():
            raise ValueError("V4.1 record report must bind episode.mp4")
        if not isinstance(video_validation, dict) or video_validation.get("valid") is not True:
            raise ValueError("V4.1 record report requires a valid video report")
        record["format_revision"] = contract.format_revision
        record["video"] = {
            "file": "episode.mp4",
            "sha256": sha256_file(video),
            "size_bytes": video.stat().st_size,
            "camera": video_validation["camera"],
            "codec": video_validation["codec"],
            "pixel_format": video_validation["pixel_format"],
            "width": int(video_validation["width"]),
            "height": int(video_validation["height"]),
            "fps": float(video_validation["fps"]),
            "frames": int(video_validation["frames"]),
            "duration_s": float(video_validation["duration_s"]),
        }
    return record


def write_failure(
    output_dir: str | Path,
    contract: V4TaskContract,
    *,
    stage: str,
    step: int | None,
    code: str,
    endpoint_scope: str,
    partial_exists: bool,
) -> Path:
    value = {
        "failure_schema": FAILURE_SCHEMA if contract.uses_full_capture else LEGACY_FAILURE_SCHEMA,
        "generated_at": utc_now(),
        "contract": {"contract_id": contract.contract_id, "sha256": contract.digest},
        "stage": stage,
        "step": step,
        "error_code": code,
        "message": "recording failed before final publication",
        "endpoint_scope": endpoint_scope,
        "partial_file": "submission.hdf5.partial" if partial_exists else None,
    }
    if contract.uses_full_capture:
        value["format_revision"] = contract.format_revision
        value["partial_files"] = [
            name
            for name in ("submission.hdf5.partial", "episode.mp4.partial")
            if (Path(output_dir).expanduser().resolve() / name).exists()
        ]
    return write_json_atomic(Path(output_dir).expanduser().resolve() / "failure.json", value)


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError(f"{path} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], path: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise RunnerError(f"{path} has missing or unexpected fields")


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def validate_record(
    submission_dir: str | Path,
    contract: V4TaskContract,
    validation: ValidationReport,
) -> dict[str, Any]:
    root = Path(submission_dir).expanduser().resolve()
    record_path = root / "record.json"
    submission_path = root / "submission.hdf5"
    record = load_json(record_path)
    expected_schema = RECORD_SCHEMA if contract.uses_full_capture else LEGACY_RECORD_SCHEMA
    if record.get("record_schema") != expected_schema:
        raise RunnerError(f"unsupported record schema in {record_path}")
    expected_top_level = {
        "record_schema",
        "generated_at",
        "submission",
        "contract",
        "versions",
        "endpoint_scope",
        "policy",
        "episode",
        "timing",
    }
    if contract.uses_full_capture:
        expected_top_level.update({"format_revision", "video"})
    if set(record) != expected_top_level:
        raise RunnerError("record.json has missing or unexpected top-level fields")
    if contract.uses_full_capture and record.get("format_revision") != contract.format_revision:
        raise RunnerError("record format_revision differs from the contract")
    generated_at = record.get("generated_at")
    try:
        generated_time = datetime.fromisoformat(generated_at) if isinstance(generated_at, str) else None
    except ValueError:
        generated_time = None
    if generated_time is None or generated_time.tzinfo is None:
        raise RunnerError("record.generated_at must be an ISO-8601 timestamp with timezone")
    submission = _require_dict(record.get("submission"), "record.submission")
    _require_exact_keys(submission, "record.submission", {"file", "sha256", "size_bytes"})
    if submission.get("file") != "submission.hdf5":
        raise RunnerError("record submission file must remain submission.hdf5")
    if Path(str(submission.get("file"))).is_absolute() or ".." in Path(str(submission.get("file"))).parts:
        raise RunnerError("record submission filename is not portable")
    if submission.get("sha256") != sha256_file(submission_path):
        raise RunnerError("record submission SHA-256 differs from current bytes")
    if submission.get("size_bytes") != submission_path.stat().st_size:
        raise RunnerError("record submission size differs from current bytes")
    if contract.uses_full_capture:
        video_path = root / "episode.mp4"
        video = _require_dict(record.get("video"), "record.video")
        _require_exact_keys(
            video,
            "record.video",
            {
                "file",
                "sha256",
                "size_bytes",
                "camera",
                "codec",
                "pixel_format",
                "width",
                "height",
                "fps",
                "frames",
                "duration_s",
            },
        )
        if video.get("file") != "episode.mp4" or not video_path.is_file():
            raise RunnerError("record video file must remain episode.mp4")
        if video.get("sha256") != sha256_file(video_path) or video.get("size_bytes") != video_path.stat().st_size:
            raise RunnerError("record video hash or size differs from current bytes")
        with h5py.File(submission_path, "r") as hdf5_file:
            rgb_samples = int(
                hdf5_file[f"data/demo_0/sensors/camera/{contract.runner_capture.video.camera}/rgb"].shape[0]
            )
        from robolab_runner.validator import validate_episode_video

        video_validation = validate_episode_video(video_path, contract, expected_frames=rgb_samples)
        expected_video = {
            "file": "episode.mp4",
            "sha256": video_validation["sha256"],
            "size_bytes": video_validation["size_bytes"],
            "camera": video_validation["camera"],
            "codec": video_validation["codec"],
            "pixel_format": video_validation["pixel_format"],
            "width": video_validation["width"],
            "height": video_validation["height"],
            "fps": video_validation["fps"],
            "frames": video_validation["frames"],
            "duration_s": video_validation["duration_s"],
        }
        if not video_validation["valid"] or video != expected_video:
            raise RunnerError("record video metadata differs from the validated primary camera video")
    contract_entry = _require_dict(record.get("contract"), "record.contract")
    _require_exact_keys(contract_entry, "record.contract", {"contract_id", "sha256"})
    if contract_entry != {"contract_id": contract.contract_id, "sha256": contract.digest}:
        raise RunnerError("record contract identity differs from official contract")
    versions = _require_dict(record.get("versions"), "record.versions")
    _require_exact_keys(
        versions,
        "record.versions",
        {"runner", "robolab", "isaaclab", "isaacsim", "policy_protocol"},
    )
    scope = record.get("endpoint_scope")
    if scope not in {"loopback", "remote"}:
        raise RunnerError("record endpoint_scope must be loopback or remote")
    policy = _require_dict(record.get("policy"), "record.policy")
    _require_exact_keys(policy, "record.policy", {"policy_id", "policy_version"})
    if not all(
        isinstance(policy.get(name), str) and _PORTABLE_IDENTITY_RE.fullmatch(policy[name])
        for name in ("policy_id", "policy_version")
    ):
        raise RunnerError("record policy identity is invalid")
    episode = _require_dict(record.get("episode"), "record.episode")
    _require_exact_keys(episode, "record.episode", {"steps", "queries"})
    if any(isinstance(episode.get(name), bool) or not isinstance(episode.get(name), int) for name in episode):
        raise RunnerError("record episode counts must be integers")
    if episode.get("steps") != validation.num_samples or episode.get("queries") != validation.query_count:
        raise RunnerError("record episode counts differ from submission HDF5")
    timing = _require_dict(record.get("timing"), "record.timing")
    _require_exact_keys(
        timing,
        "record.timing",
        {"inference_total_s", "inference_mean_s", "inference_max_s"},
    )
    with h5py.File(submission_path, "r") as hdf5_file:
        if contract.uses_full_capture:
            network = hdf5_file["data/demo_0/policy/network"]
            messages = [_decode(value) for value in network["message_type"][:]]
            statuses = [_decode(value) for value in network["status"][:]]
            round_trips = np.asarray(network["round_trip_s"][:], dtype=np.float64)
            durations = np.asarray(
                [
                    duration
                    for message, status, duration in zip(messages, statuses, round_trips)
                    if message == "infer" and status == "success"
                ],
                dtype=np.float64,
            )
        else:
            durations = np.asarray(hdf5_file["data/demo_0/policy/inference_duration_s"][:], dtype=np.float64)
        metadata = hdf5_file["metadata"]
        metadata_policy_id = _decode(metadata.attrs["policy_id"])
        metadata_policy_version = _decode(metadata.attrs["policy_version"])
        metadata_scope = _decode(metadata.attrs["policy_endpoint_scope"])
        metadata_runner_version = _decode(metadata.attrs["runner_version"])
    expected_versions = {
        "runner": metadata_runner_version,
        "robolab": contract.versions["robolab"],
        "isaaclab": contract.versions["isaaclab"],
        "isaacsim": contract.versions["isaacsim"],
        "policy_protocol": f"{POLICY_PROTOCOL_NAME}/{contract.policy_protocol_version}",
    }
    if versions != expected_versions:
        raise RunnerError("record versions differ from HDF5 metadata or official contract")
    expected_timing = {
        "inference_total_s": float(durations.sum()),
        "inference_mean_s": float(durations.mean()),
        "inference_max_s": float(durations.max()),
    }
    for name, expected in expected_timing.items():
        value = timing.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RunnerError(f"record timing {name} differs from HDF5 audit")
    if policy != {"policy_id": metadata_policy_id, "policy_version": metadata_policy_version}:
        raise RunnerError("record policy identity differs from HDF5 metadata")
    if scope != metadata_scope:
        raise RunnerError("record endpoint scope differs from HDF5 metadata")
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
    if str(root) in serialized or str(submission_path) in serialized or "://" in serialized:
        raise RunnerError("record.json contains a path or endpoint instead of portable fields")
    return record


__all__ = [
    "FAILURE_SCHEMA",
    "LEGACY_FAILURE_SCHEMA",
    "LEGACY_RECORD_SCHEMA",
    "RECORD_SCHEMA",
    "build_record",
    "validate_record",
    "write_failure",
]
