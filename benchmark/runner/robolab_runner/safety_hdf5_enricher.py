"""Append validated global-safety evidence to a formal submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = "/extensions/m20_safety_v1"


def _read_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw, value


def _write_bytes(group, name: str, payload: bytes) -> str:
    import numpy as np

    dataset = group.create_dataset(name, data=np.frombuffer(payload, dtype=np.uint8))
    digest = hashlib.sha256(payload).hexdigest()
    dataset.attrs["sha256"] = digest
    dataset.attrs["encoding"] = "utf-8"
    return digest


def append_safety_extension(
    submission: Path,
    output_dir: Path,
    *,
    expected_profile_sha256: str,
) -> dict:
    import h5py

    submission = Path(submission)
    output_dir = Path(output_dir)
    profile_raw, profile = _read_json(output_dir / "resolved_safety_profile.json")
    summary_raw, summary = _read_json(output_dir / "safety_summary.json")
    events_raw = (output_dir / "safety_evidence.jsonl").read_bytes()

    actual_profile_sha256 = hashlib.sha256(
        json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_profile_sha256 != expected_profile_sha256:
        raise ValueError(
            "resolved safety profile digest mismatch: "
            f"expected {expected_profile_sha256}, got {actual_profile_sha256}"
        )
    if summary.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("safety summary references a different profile digest")
    if summary.get("schema") != "m20-safety-summary/v1":
        raise ValueError("unsupported safety summary schema")
    if summary.get("complete") is not True:
        raise ValueError("safety summary is incomplete")

    event_count = 0
    for line_number, line in enumerate(events_raw.splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line.decode("utf-8"))
        if event.get("schema") != "m20-safety-event/v1":
            raise ValueError(f"invalid safety event schema on line {line_number}")
        event_count += 1

    with h5py.File(submission, "r+") as hdf5_file:
        if ROOT in hdf5_file:
            raise ValueError(f"refusing to overwrite existing HDF5 group {ROOT}")
        group = hdf5_file.create_group(ROOT)
        group.attrs["schema"] = "m20-safety-evidence-extension/v1"
        group.attrs["profile_id"] = str(profile["profile_id"])
        group.attrs["rule_set_version"] = str(profile["rule_set_version"])
        group.attrs["profile_sha256"] = expected_profile_sha256
        group.attrs["event_count"] = event_count
        # A failed raw-evidence copy must never leave an apparently complete
        # extension behind in an otherwise readable submission.
        group.attrs["complete"] = False
        profile_file_sha256 = _write_bytes(group, "profile_json", profile_raw)
        summary_file_sha256 = _write_bytes(group, "summary_json", summary_raw)
        events_sha256 = _write_bytes(group, "events_jsonl", events_raw)
        observations = output_dir / 'safety_observations.hdf5'
        if summary.get('observation_schema'):
            if not observations.is_file():raise ValueError('safety raw observation file missing')
            with h5py.File(observations, 'r') as source:
                if not source.attrs.get('complete') or source.attrs.get('profile_sha256') != expected_profile_sha256:
                    raise ValueError('safety raw observation contract mismatch')
                target=group.create_group('observations');target.attrs.update(dict(source.attrs))
                for name in source:source.copy(source[name],target,name=name)
        hdf5_file.flush()
        group.attrs["complete"] = True
        hdf5_file.flush()

    return {
        "schema": "m20-safety-evidence-extension-report/v1",
        "status": "VALID",
        "hdf5_group": ROOT,
        "profile_id": profile["profile_id"],
        "rule_set_version": profile["rule_set_version"],
        "profile_sha256": expected_profile_sha256,
        "profile_file_sha256": profile_file_sha256,
        "summary_file_sha256": summary_file_sha256,
        "events_sha256": events_sha256,
        "event_count": event_count,
        "violated_rules": list(summary.get("violated_rules", [])),
        "unknown_rules": list(summary.get("unknown_rules", [])),
    }
