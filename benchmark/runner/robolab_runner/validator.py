# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free hostile-input validation and atomic V4 publication."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from packaging.version import InvalidVersion, Version

from robolab_runner.common import publish_no_replace, sha256_file, utc_now
from robolab_runner.contract import POLICY_PROTOCOL_NAME, V4TaskContract
from robolab_runner.schema import (
    DATA_GROUP,
    DEMO_NAME,
    METADATA_GROUP,
    SUBMISSION_METADATA_ATTRIBUTES,
    V4_1_METADATA_ATTRIBUTES,
    V4_3_METADATA_ATTRIBUTES,
)

_SCAN_BYTES = 16 * 1024 * 1024
_PORTABLE_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_FORBIDDEN_V4_1_GROUPS = frozenset(
    {"subtask", "attack", "anomaly", "mitigation", "degradation", "recovery"}
)
_V4_3_SEALED_AUDIT_OBJECTS = frozenset(
    {"attack_active", "attacked_point_count", "expected_phantom_count", "matched_phantom_count"}
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    valid: bool
    status: str
    contract_id: str
    input_path: str
    generated_at: str = field(default_factory=utc_now)
    sha256: str | None = None
    file_size_bytes: int | None = None
    hdf5_objects: int | None = None
    max_group_depth: int | None = None
    logical_bytes: int | None = None
    num_samples: int | None = None
    query_count: int | None = None
    task_env: str | None = None
    finalized: bool = False
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error(report: ValidationReport, code: str, message: str, path: str | None = None) -> None:
    report.errors.append(ValidationIssue(code, message, path))


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _attribute_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, np.bytes_)):
        return len(bytes(value))
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U"}:
        return sum(_attribute_size(item) for item in array.reshape(-1))
    return int(array.nbytes)


def _bounded_child_names(group: h5py.Group | h5py.File, expected_count: int) -> list[str]:
    """Read only enough names to prove an exact small schema mismatch."""
    return [str(name) for name in islice(group.keys(), expected_count + 1)]


def _scan_finite(dataset: h5py.Dataset, report: ValidationReport) -> None:
    if not np.issubdtype(dataset.dtype, np.number) or dataset.size == 0:
        return
    rows = dataset.shape[0] if dataset.ndim else 1
    row_items = max(1, dataset.size // rows)
    chunk_rows = max(1, _SCAN_BYTES // max(1, row_items * dataset.dtype.itemsize))
    arrays = (
        [np.asarray(dataset[()])]
        if dataset.ndim == 0
        else (np.asarray(dataset[start : start + chunk_rows]) for start in range(0, rows, chunk_rows))
    )
    for array in arrays:
        if not np.isfinite(array).all():
            _error(report, "NONFINITE_VALUE", "numeric dataset contains NaN or infinity", dataset.name)
            return


def _walk_hdf5(hdf5_file: h5py.File, contract: V4TaskContract, report: ValidationReport) -> None:
    seen_addresses: dict[int, str] = {}
    objects = 1
    maximum_depth = 0
    logical_bytes = 0
    resource_limit_reached = False

    def check_attrs(obj: h5py.Group | h5py.Dataset) -> None:
        total = 0
        for attr_name, attr_value in obj.attrs.items():
            size = len(str(attr_name).encode("utf-8")) + _attribute_size(attr_value)
            total += size
            if size > contract.limits.max_attribute_bytes:
                _error(report, "ATTRIBUTE_TOO_LARGE", f"attribute {attr_name!r} uses {size} bytes", obj.name)
                return
            if total > contract.limits.max_attribute_bytes:
                _error(report, "ATTRIBUTE_TOO_LARGE", "aggregate attributes exceed the contract limit", obj.name)
                return

    def walk(group: h5py.Group, depth: int) -> None:
        nonlocal objects, maximum_depth, logical_bytes, resource_limit_reached
        if resource_limit_reached:
            return
        maximum_depth = max(maximum_depth, depth)
        if depth > contract.limits.max_group_depth:
            _error(report, "GROUP_DEPTH_EXCEEDED", f"depth {depth} exceeds contract limit", group.name)
            resource_limit_reached = True
            return
        check_attrs(group)
        for raw_name in group.keys():
            name = str(raw_name)
            path = f"{group.name.rstrip('/')}/{name}"
            if (
                contract.uses_full_capture
                and any(token in name.lower() for token in _FORBIDDEN_V4_1_GROUPS)
                and not (contract.is_v4_3 and name in _V4_3_SEALED_AUDIT_OBJECTS)
            ):
                _error(report, "EXCLUDED_DATA_GROUP", "full capture contains an explicitly excluded data group", path)
                continue
            if name in {".", ".."} or any(ord(character) < 32 for character in name):
                _error(report, "UNSAFE_NAME", "HDF5 object name is unsafe", path)
                continue
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                _error(report, "UNSAFE_LINK", f"{type(link).__name__} links are forbidden", path)
                continue
            try:
                obj = group[name]
            except Exception as error:
                _error(report, "UNREADABLE_OBJECT", f"failed to open object: {error}", path)
                continue
            objects += 1
            if objects > contract.limits.max_hdf5_objects:
                _error(
                    report,
                    "TOO_MANY_OBJECTS",
                    f"object count exceeds {contract.limits.max_hdf5_objects}",
                    path,
                )
                resource_limit_reached = True
                return
            try:
                address = int(h5py.h5o.get_info(obj.id).addr)
            except Exception:
                address = -objects
            if address in seen_addresses:
                _error(report, "HARD_LINK_ALIAS", f"object aliases {seen_addresses[address]!r}", path)
                continue
            seen_addresses[address] = path
            check_attrs(obj)
            if isinstance(obj, h5py.Group):
                walk(obj, depth + 1)
                continue
            if not isinstance(obj, h5py.Dataset):
                _error(report, "UNSUPPORTED_OBJECT", f"unsupported HDF5 object {type(obj).__name__}", path)
                continue
            if obj.is_virtual:
                _error(report, "VIRTUAL_DATASET", "virtual datasets are forbidden", path)
            if obj.external:
                _error(report, "EXTERNAL_STORAGE", "external dataset storage is forbidden", path)
            if obj.dtype.hasobject or obj.dtype.kind in {"O", "V"}:
                _error(report, "UNSAFE_DTYPE", f"object/compound dtype {obj.dtype} is forbidden", path)
            if obj.compression not in contract.limits.allowed_compressions:
                _error(report, "UNSUPPORTED_COMPRESSION", f"compression {obj.compression!r} is not allowed", path)
            logical_bytes += int(obj.size) * int(obj.dtype.itemsize)
            if logical_bytes > contract.limits.max_logical_bytes:
                _error(report, "LOGICAL_BYTES_EXCEEDED", "logical bytes exceed contract limit", path)
                resource_limit_reached = True
                return

    walk(hdf5_file["/"], 0)
    report.hdf5_objects = objects
    report.max_group_depth = maximum_depth
    report.logical_bytes = logical_bytes
    if objects > contract.limits.max_hdf5_objects and not any(
        issue.code == "TOO_MANY_OBJECTS" for issue in report.errors
    ):
        _error(report, "TOO_MANY_OBJECTS", f"{objects} exceeds {contract.limits.max_hdf5_objects}", "/")
    if maximum_depth > contract.limits.max_group_depth and not any(
        issue.code == "GROUP_DEPTH_EXCEEDED" for issue in report.errors
    ):
        _error(report, "GROUP_DEPTH_EXCEEDED", f"depth {maximum_depth} exceeds contract limit", "/")
    if logical_bytes > contract.limits.max_logical_bytes and not any(
        issue.code == "LOGICAL_BYTES_EXCEEDED" for issue in report.errors
    ):
        _error(report, "LOGICAL_BYTES_EXCEEDED", f"{logical_bytes} exceeds contract limit", "/")


def _require_dataset(parent: h5py.Group, path: str, report: ValidationReport) -> h5py.Dataset | None:
    try:
        value = parent[path]
    except KeyError:
        _error(report, "MISSING_DATASET", "required dataset is missing", f"{parent.name}/{path}")
        return None
    if not isinstance(value, h5py.Dataset):
        _error(report, "EXPECTED_DATASET", "required path is not a dataset", value.name)
        return None
    return value


def _expect_float32(dataset: h5py.Dataset | None, report: ValidationReport) -> bool:
    if dataset is None:
        return False
    if dataset.dtype != np.dtype(np.float32):
        _error(report, "INVALID_DTYPE", f"expected float32, got {dataset.dtype}", dataset.name)
        return False
    _scan_finite(dataset, report)
    return True


def _metadata_string(group: h5py.Group, name: str, report: ValidationReport) -> str | None:
    if name not in group.attrs:
        _error(report, "MISSING_METADATA_FIELD", f"missing metadata attribute {name!r}", group.name)
        return None
    try:
        value = _decode(group.attrs[name])
    except UnicodeDecodeError:
        _error(report, "INVALID_METADATA", f"metadata {name!r} is not UTF-8", group.name)
        return None
    if not value:
        _error(report, "INVALID_METADATA", f"metadata {name!r} is empty", group.name)
    return value


def _check_metadata(hdf5_file: h5py.File, contract: V4TaskContract, report: ValidationReport) -> None:
    if METADATA_GROUP not in hdf5_file or not isinstance(hdf5_file[METADATA_GROUP], h5py.Group):
        _error(report, "MISSING_METADATA", "top-level /metadata group is required", "/metadata")
        return
    group = hdf5_file[METADATA_GROUP]
    expected_attributes = (
        V4_3_METADATA_ATTRIBUTES
        if contract.is_v4_3
        else V4_1_METADATA_ATTRIBUTES
        if contract.uses_full_capture
        else SUBMISSION_METADATA_ATTRIBUTES
    )
    if set(group.attrs) != expected_attributes:
        _error(report, "UNEXPECTED_METADATA_FIELD", "metadata attributes differ from the V4 schema", group.name)
    schema_name = _decode(group.attrs.get("schema_name", ""))
    try:
        schema_version = int(group.attrs.get("schema_version", -1))
    except (TypeError, ValueError):
        schema_version = -1
    if (schema_name, schema_version) != ("robolab-scoring", 4):
        _error(
            report,
            "UNSUPPORTED_SCHEMA",
            f"unsupported submission schema {schema_name!r} v{schema_version}; V4 does not convert V3",
            "/metadata",
        )
        return
    expected: dict[str, str | int] = {
        "producer_type": "robolab_competition_runner",
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest,
        "task_env": contract.task_env,
        "task_class": contract.task_class,
        "instruction": contract.instruction,
        "seed": contract.seed,
        "robolab_version": contract.versions["robolab"],
        "isaaclab_version": contract.versions["isaaclab"],
        "isaacsim_version": contract.versions["isaacsim"],
        "policy_protocol_name": POLICY_PROTOCOL_NAME,
        "policy_protocol_version": contract.policy_protocol_version,
        "policy_observation_schema_sha256": contract.observation_schema_sha256,
        "policy_limits_sha256": contract.policy_limits_sha256,
        "environment_config_sha256": contract.environment_config_sha256,
        "asset_manifest_sha256": contract.asset_manifest_sha256,
        "rule_package_id": contract.rule_package_id,
        "rule_package_sha256": contract.rule_package_sha256,
        "action_type": contract.control.action_type,
        "action_dim": contract.control.action_dim,
    }
    if contract.uses_full_capture:
        expected.update(
            {
                "format_revision": contract.format_revision,
                "coordinate_convention": contract.runner_capture.coordinate_convention,
                "quaternion_order": contract.runner_capture.quaternion_order,
                "primary_tracking_body": contract.primary_tracking_body,
            }
        )
        if contract.is_v4_3 and contract.direct_scoring is not None:
            expected.update(
                {
                    "direct_scoring_schema": contract.direct_scoring.schema,
                    "direct_scoring_source": "participant_direct",
                }
            )
    else:
        expected["ee_body_name"] = contract.primary_tracking_body
    for name, expected_value in expected.items():
        if name not in group.attrs:
            _error(report, "MISSING_METADATA_FIELD", f"missing metadata attribute {name!r}", group.name)
            continue
        raw_value = group.attrs[name]
        try:
            actual = _decode(raw_value) if isinstance(expected_value, str) else int(raw_value)
        except (TypeError, ValueError, UnicodeDecodeError):
            actual = None
        if actual != expected_value:
            _error(report, "CONTRACT_MISMATCH", f"metadata {name}={actual!r}, expected {expected_value!r}", group.name)
    for name in ("runner_version", "policy_id", "policy_version"):
        identity = _metadata_string(group, name, report)
        if identity is not None and not _PORTABLE_IDENTITY_RE.fullmatch(identity):
            _error(report, "INVALID_METADATA", f"metadata {name!r} is not a portable identifier", group.name)
    if contract.is_v4_3:
        runner_version = _decode(group.attrs.get("runner_version", ""))
        try:
            parsed_runner = Version(runner_version)
        except InvalidVersion:
            parsed_runner = None
        if parsed_runner is None or parsed_runner.release[:2] != (4, 3):
            _error(
                report,
                "INVALID_RUNNER_VERSION",
                "V4.3 direct evidence must be produced by a 4.3.x Runner",
                group.name,
            )
    endpoint_scope = _metadata_string(group, "policy_endpoint_scope", report)
    if endpoint_scope not in {"loopback", "remote"}:
        _error(report, "INVALID_METADATA", "policy_endpoint_scope must be loopback or remote", group.name)
    try:
        dt = float(group.attrs.get("control_dt", math.nan))
    except (TypeError, ValueError):
        dt = math.nan
    if not math.isclose(dt, contract.control.control_dt, rel_tol=0.0, abs_tol=1e-12):
        _error(report, "CONTRACT_MISMATCH", f"control_dt={dt!r} differs from contract", group.name)
    if bool(group.attrs.get("complete", False)) is not True:
        _error(report, "INCOMPLETE_RECORDING", "metadata.complete is not true", group.name)
    json_expectations = {
        "joint_names_json": list(contract.control.joint_names),
        "policy_observation_leaves_json": [leaf.to_dict() for leaf in contract.observation_leaves],
        "policy_limits_json": contract.policy_limits.to_dict(),
    }
    if contract.uses_full_capture:
        json_expectations.update(
            {
                "env_cfg_json": contract.environment_config,
                "entity_spec_json": contract.runner_capture.entity_spec,
                "capture_spec_json": contract.runner_capture.to_dict(),
            }
        )
    for name, expected_value in json_expectations.items():
        try:
            actual = json.loads(_decode(group.attrs.get(name, "")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            actual = None
        if actual != expected_value:
            _error(report, "CONTRACT_MISMATCH", f"{name} differs from contract", group.name)
    if contract.uses_full_capture:
        try:
            physics_dt = float(group.attrs.get("physics_dt", math.nan))
        except (TypeError, ValueError):
            physics_dt = math.nan
        if not math.isclose(physics_dt, contract.runner_capture.physics_dt, rel_tol=0.0, abs_tol=1e-12):
            _error(report, "CONTRACT_MISMATCH", "physics_dt differs from contract", group.name)
        if _bounded_child_names(group, 0):
            _error(report, "UNEXPECTED_METADATA_OBJECT", "V4.1 metadata cannot contain child objects", group.name)
        if contract.is_v4_3:
            try:
                lidar_profile = json.loads(_decode(group.attrs.get("direct_lidar_attack_profile_json", "")))
                if not isinstance(lidar_profile, dict):
                    raise TypeError
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                lidar_profile = None
                _error(
                    report,
                    "INVALID_DIRECT_LIDAR_PROFILE",
                    "direct_lidar_attack_profile_json must decode to an object",
                    group.name,
                )
            if contract.direct_scoring is not None:
                if contract.direct_scoring.lidar_audit and not lidar_profile:
                    _error(
                        report,
                        "MISSING_DIRECT_LIDAR_PROFILE",
                        "V4.3 LiDAR scoring requires a non-empty sealed attack profile",
                        group.name,
                    )
                if not contract.direct_scoring.lidar_audit and lidar_profile != {}:
                    _error(
                        report,
                        "UNEXPECTED_DIRECT_LIDAR_PROFILE",
                        "non-LiDAR V4.3 contracts must record an empty LiDAR attack profile",
                        group.name,
                    )
    else:
        for name, expected_values in (
            ("arm_joint_indices", contract.control.arm_joint_indices),
            ("action_joint_indices", contract.control.action_joint_indices),
        ):
            dataset = _require_dataset(group, name, report)
            if dataset is not None:
                if dataset.dtype != np.dtype(np.int64) or dataset.ndim != 1:
                    _error(report, "INVALID_DTYPE", f"{name} must be int64[1-D]", dataset.name)
                elif tuple(int(value) for value in dataset[:]) != expected_values:
                    _error(report, "CONTRACT_MISMATCH", f"{name} differs from contract", dataset.name)
        if set(_bounded_child_names(group, 2)) != {"arm_joint_indices", "action_joint_indices"}:
            _error(report, "UNEXPECTED_METADATA_OBJECT", "metadata contains unexpected child objects", group.name)
    report.task_env = _decode(group.attrs.get("task_env", "")) or None


def _check_policy_audit(demo: h5py.Group, contract: V4TaskContract, report: ValidationReport, samples: int) -> None:
    policy = demo.get("policy")
    if not isinstance(policy, h5py.Group):
        _error(report, "MISSING_POLICY_AUDIT", "demo_0/policy must be a group", f"{demo.name}/policy")
        return
    expected = {"query_step", "inference_duration_s", "chunk_length"}
    if set(_bounded_child_names(policy, len(expected))) != expected:
        _error(report, "UNEXPECTED_POLICY_OBJECT", f"policy children must be {sorted(expected)}", policy.name)
    query_step = _require_dataset(policy, "query_step", report)
    duration = _require_dataset(policy, "inference_duration_s", report)
    chunk_length = _require_dataset(policy, "chunk_length", report)
    if query_step is None or duration is None or chunk_length is None:
        return
    query_count = int(query_step.shape[0]) if query_step.ndim == 1 else 0
    report.query_count = query_count
    if query_count < 1 or query_count > samples:
        _error(report, "INVALID_POLICY_AUDIT", "policy query count must be in [1, num_samples]", policy.name)
        return
    if query_step.dtype != np.dtype(np.int64) or query_step.shape != (query_count,):
        _error(report, "INVALID_DTYPE", "policy/query_step must be int64[Q]", query_step.name)
    if chunk_length.dtype != np.dtype(np.int64) or chunk_length.shape != (query_count,):
        _error(report, "INVALID_DTYPE", "policy/chunk_length must be int64[Q]", chunk_length.name)
    if duration.dtype != np.dtype(np.float64) or duration.shape != (query_count,):
        _error(report, "INVALID_DTYPE", "policy/inference_duration_s must be float64[Q]", duration.name)
        return
    steps = np.asarray(query_step[:], dtype=np.int64)
    chunks = np.asarray(chunk_length[:], dtype=np.int64)
    durations = np.asarray(duration[:], dtype=np.float64)
    if steps[0] != 0 or np.any(steps < 0) or np.any(steps >= samples):
        _error(
            report,
            "INVALID_POLICY_AUDIT",
            "policy query steps must start at 0 and index recorded actions",
            policy.name,
        )
    if query_count > 1 and not np.array_equal(steps[1:], steps[:-1] + chunks[:-1]):
        _error(report, "INVALID_POLICY_AUDIT", "policy chunks and query steps are not contiguous", policy.name)
    maximum = contract.policy_limits.max_action_horizon
    if np.any(chunks < 1) or np.any(chunks > maximum) or steps[-1] + chunks[-1] < samples:
        _error(report, "INVALID_POLICY_AUDIT", "policy chunk lengths do not cover recorded actions", policy.name)
    if not np.isfinite(durations).all() or np.any(durations < 0):
        _error(
            report,
            "INVALID_POLICY_AUDIT",
            "policy inference durations must be finite and non-negative",
            policy.name,
        )
    elif float(durations.sum()) > contract.policy_limits.total_inference_timeout_s + 1e-3:
        _error(report, "POLICY_BUDGET_EXCEEDED", "policy inference durations exceed the contract total", policy.name)


def _check_v41_network(demo: h5py.Group, report: ValidationReport, samples: int) -> None:
    network = demo.get("policy/network")
    if not isinstance(network, h5py.Group):
        _error(report, "MISSING_NETWORK_AUDIT", "V4.1 policy/network must be a group", f"{demo.name}/policy/network")
        return
    names = {
        "sequence",
        "message_type",
        "control_step",
        "session_time_s",
        "request_bytes",
        "response_bytes",
        "round_trip_s",
        "status",
    }
    if set(_bounded_child_names(network, len(names))) != names:
        _error(report, "UNEXPECTED_NETWORK_OBJECT", "V4.1 policy/network fields differ from schema", network.name)
        return
    datasets = {name: _require_dataset(network, name, report) for name in names}
    if any(dataset is None for dataset in datasets.values()):
        return
    rows = int(datasets["sequence"].shape[0]) if datasets["sequence"].ndim == 1 else 0
    if rows < 4:
        _error(report, "INVALID_NETWORK_AUDIT", "network audit must include handshake/reset/infer/end_episode", network.name)
        return
    expected_dtypes = {
        "sequence": np.dtype(np.uint64),
        "message_type": np.dtype("S16"),
        "control_step": np.dtype(np.int64),
        "session_time_s": np.dtype(np.float64),
        "request_bytes": np.dtype(np.uint64),
        "response_bytes": np.dtype(np.uint64),
        "round_trip_s": np.dtype(np.float64),
        "status": np.dtype("S16"),
    }
    for name, dataset in datasets.items():
        if dataset.shape != (rows,) or dataset.dtype != expected_dtypes[name]:
            _error(report, "INVALID_NETWORK_AUDIT", f"network/{name} has an invalid dtype or shape", dataset.name)
            return
    sequence = np.asarray(datasets["sequence"][:], dtype=np.uint64)
    control_steps = np.asarray(datasets["control_step"][:], dtype=np.int64)
    session_times = np.asarray(datasets["session_time_s"][:], dtype=np.float64)
    request_bytes = np.asarray(datasets["request_bytes"][:], dtype=np.uint64)
    response_bytes = np.asarray(datasets["response_bytes"][:], dtype=np.uint64)
    round_trips = np.asarray(datasets["round_trip_s"][:], dtype=np.float64)
    messages = [_decode(value) for value in datasets["message_type"][:]]
    statuses = [_decode(value) for value in datasets["status"][:]]
    if not np.array_equal(sequence, np.arange(1, rows + 1, dtype=np.uint64)):
        _error(report, "INVALID_NETWORK_AUDIT", "network sequence must be contiguous from one", network.name)
    if (
        not np.isfinite(session_times).all()
        or not np.isfinite(round_trips).all()
        or np.any(session_times < 0)
        or np.any(round_trips < 0)
        or np.any(np.diff(session_times) < 0)
    ):
        _error(report, "INVALID_NETWORK_AUDIT", "network timing is invalid", network.name)
    allowed_messages = {"handshake", "reset", "infer", "end_episode"}
    allowed_statuses = {"success", "timeout", "disconnected", "protocol_error"}
    if not set(messages) <= allowed_messages or not set(statuses) <= allowed_statuses:
        _error(report, "INVALID_NETWORK_AUDIT", "network message/status contains an unsupported value", network.name)
    expected_structure = (
        len(messages) >= 4
        and messages[:2] == ["handshake", "reset"]
        and messages[-1] == "end_episode"
        and all(message == "infer" for message in messages[2:-1])
    )
    if not expected_structure:
        _error(
            report,
            "INVALID_NETWORK_AUDIT",
            "completed capture must contain handshake, reset, infer+, end_episode in order",
            network.name,
        )
    if any(status != "success" for status in statuses):
        _error(report, "INVALID_NETWORK_AUDIT", "completed capture contains a failed network exchange", network.name)
    if np.any(request_bytes == 0) or np.any(response_bytes == 0):
        _error(report, "INVALID_NETWORK_AUDIT", "successful network exchanges must bind non-empty bytes", network.name)
    if any(
        (message == "infer" and step < 0) or (message != "infer" and step != -1)
        for message, step in zip(messages, control_steps)
    ):
        _error(report, "INVALID_NETWORK_AUDIT", "network control_step differs from message semantics", network.name)
    infer_steps = [
        int(step)
        for message, status, step in zip(messages, statuses, control_steps)
        if message == "infer" and status == "success"
    ]
    report.query_count = len(infer_steps)
    if not infer_steps or infer_steps[0] != 0 or any(
        current <= previous for previous, current in zip(infer_steps, infer_steps[1:])
    ):
        _error(report, "INVALID_NETWORK_AUDIT", "successful infer control steps must start at zero and increase", network.name)
    if any(step < -1 or step >= samples for step in control_steps):
        _error(report, "INVALID_NETWORK_AUDIT", "network control_step is outside the episode", network.name)


def _check_v41_stream(
    demo: h5py.Group,
    stream: Any,
    contract: V4TaskContract,
    report: ValidationReport,
    samples: int,
) -> None:
    relative = (
        f"feedback/articulation/{stream.name}"
        if stream.kind == "articulation"
        else f"sensors/{stream.kind}/{stream.name}"
    )
    group = demo.get(relative)
    if not isinstance(group, h5py.Group):
        _error(report, "MISSING_CAPTURE_STREAM", "contract-declared stream is missing", f"{demo.name}/{relative}")
        return
    expected_children = {"sample_id", "step_index", "sim_time_s"} | {field.name for field in stream.fields}
    if contract.is_v4_3:
        expected_children.add("physics_step")
    if stream.kind == "lidar":
        expected_children.add("frame_offsets")
        if contract.is_v4_3 and contract.direct_scoring and contract.direct_scoring.lidar_audit:
            expected_children.update(
                {
                    "raw_points",
                    "raw_frame_offsets",
                    "attack_active",
                    "attacked_point_count",
                    "expected_phantom_count",
                    "matched_phantom_count",
                    "navigation_lidar_active",
                    "raw_point_count",
                }
            )
    observed = set(_bounded_child_names(group, len(expected_children)))
    if observed != expected_children:
        _error(
            report,
            "UNEXPECTED_CAPTURE_FIELD",
            f"stream fields differ: expected={sorted(expected_children)}, got={sorted(observed)}",
            group.name,
        )
        return
    sample_id = _require_dataset(group, "sample_id", report)
    step_index = _require_dataset(group, "step_index", report)
    sim_time = _require_dataset(group, "sim_time_s", report)
    if sample_id is None or step_index is None or sim_time is None:
        return
    rows = int(sample_id.shape[0]) if sample_id.ndim == 1 else 0
    if rows < 1:
        _error(report, "EMPTY_CAPTURE_STREAM", "required capture stream has no producer samples", group.name)
        return
    if sample_id.dtype != np.dtype(np.uint64) or sample_id.shape != (rows,):
        _error(report, "INVALID_STREAM_AXIS", "sample_id must be uint64[N]", sample_id.name)
    if step_index.dtype != np.dtype(np.uint64) or step_index.shape != (rows,):
        _error(report, "INVALID_STREAM_AXIS", "step_index must be uint64[N]", step_index.name)
    if sim_time.dtype != np.dtype(np.float64) or sim_time.shape != (rows,):
        _error(report, "INVALID_STREAM_AXIS", "sim_time_s must be float64[N]", sim_time.name)
        return
    ids = np.asarray(sample_id[:], dtype=np.uint64)
    steps = np.asarray(step_index[:], dtype=np.uint64)
    times = np.asarray(sim_time[:], dtype=np.float64)
    expected_rows = int(round(samples * contract.control.control_dt * float(stream.rate_hz)))
    if rows != expected_rows:
        _error(
            report,
            "INVALID_STREAM_RATE",
            f"producer emitted {rows} samples; frozen episode cadence requires {expected_rows}",
            sample_id.name,
        )
    if not np.array_equal(ids, np.arange(rows, dtype=np.uint64)):
        _error(report, "DUPLICATED_PRODUCER_SAMPLE", "sample_id must be contiguous from zero", sample_id.name)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
        _error(report, "INVALID_STREAM_AXIS", "sim_time_s must be finite and strictly increasing", sim_time.name)
    if np.any(steps >= samples):
        _error(report, "INVALID_STREAM_AXIS", "stream step_index is outside the episode", step_index.name)
    expected_times = np.arange(1, rows + 1, dtype=np.float64) / float(stream.rate_hz)
    if not np.allclose(
        times,
        expected_times,
        rtol=0.0,
        atol=contract.runner_capture.physics_dt / 2,
    ):
        _error(
            report,
            "INVALID_STREAM_RATE",
            "producer timestamps contain missing or synthetic cadence samples",
            sim_time.name,
        )
    expected_steps = np.ceil(expected_times / contract.control.control_dt - 1e-9).astype(np.int64) - 1
    expected_steps = np.clip(expected_steps, 0, samples - 1).astype(np.uint64)
    if not np.array_equal(steps, expected_steps):
        _error(
            report,
            "INVALID_STREAM_AXIS",
            "stream step_index does not identify the producer's control interval",
            step_index.name,
        )
    if contract.is_v4_3:
        physics_step = _require_dataset(group, "physics_step", report)
        if physics_step is not None:
            if physics_step.dtype != np.dtype(np.uint64) or physics_step.shape != (rows,):
                _error(report, "INVALID_STREAM_AXIS", "physics_step must be uint64[N]", physics_step.name)
            else:
                physical = np.asarray(physics_step[:], dtype=np.uint64)
                cadence = int(round(1.0 / (float(stream.rate_hz) * contract.runner_capture.physics_dt)))
                expected_physical = np.arange(1, rows + 1, dtype=np.uint64) * cadence
                if not np.array_equal(physical, expected_physical):
                    _error(
                        report,
                        "INVALID_PHYSICS_TIME",
                        "physics_step differs from the frozen producer cadence",
                        physics_step.name,
                    )
                expected_physics_time = physical.astype(np.float64) * contract.runner_capture.physics_dt
                if not np.allclose(times, expected_physics_time, rtol=0.0, atol=1e-9):
                    _error(
                        report,
                        "INVALID_PHYSICS_TIME",
                        "sim_time_s is not bound to physics_step",
                        sim_time.name,
                    )
    if rows > 1:
        observed_dt = float(np.median(np.diff(times)))
        expected_dt = 1.0 / float(stream.rate_hz)
        if not math.isclose(observed_dt, expected_dt, rel_tol=0.1, abs_tol=contract.runner_capture.physics_dt / 2):
            _error(
                report,
                "INVALID_STREAM_RATE",
                f"median producer period {observed_dt:.9g}s differs from {expected_dt:.9g}s",
                sim_time.name,
            )
    fields = {field.name: field for field in stream.fields}
    if stream.kind == "lidar":
        point_field = fields["points"]
        points = _require_dataset(group, "points", report)
        offsets = _require_dataset(group, "frame_offsets", report)
        if points is not None:
            if points.dtype != np.dtype(point_field.dtype) or points.ndim != 1 + len(point_field.shape) or (
                points.shape[1:] != point_field.shape
            ):
                _error(report, "INVALID_LIDAR_LAYOUT", "LiDAR points dtype/shape differs from contract", points.name)
            else:
                _scan_finite(points, report)
        if offsets is not None:
            if offsets.dtype != np.dtype(np.uint64) or offsets.shape != (rows + 1,):
                _error(report, "INVALID_LIDAR_OFFSETS", "frame_offsets must be uint64[N+1]", offsets.name)
            else:
                values = np.asarray(offsets[:], dtype=np.uint64)
                point_rows = int(points.shape[0]) if points is not None else -1
                if values[0] != 0 or values[-1] != point_rows or np.any(np.diff(values.astype(np.int64)) < 0):
                    _error(report, "INVALID_LIDAR_OFFSETS", "frame_offsets do not partition flattened points", offsets.name)
                elif np.any(np.diff(values.astype(np.int64)) > contract.runner_capture.max_lidar_points_per_frame):
                    _error(report, "LIDAR_FRAME_TOO_LARGE", "LiDAR frame exceeds the contract point limit", offsets.name)
        if contract.is_v4_3 and contract.direct_scoring and contract.direct_scoring.lidar_audit:
            raw_points = _require_dataset(group, "raw_points", report)
            raw_offsets = _require_dataset(group, "raw_frame_offsets", report)
            if raw_points is not None:
                if raw_points.dtype != np.dtype(np.float32) or raw_points.ndim != 2 or raw_points.shape[1:] != (3,):
                    _error(
                        report,
                        "INVALID_DIRECT_LIDAR_LAYOUT",
                        "raw_points must be float32[N,3]",
                        raw_points.name,
                    )
                else:
                    _scan_finite(raw_points, report)
            if raw_offsets is not None:
                if raw_offsets.dtype != np.dtype(np.uint64) or raw_offsets.shape != (rows + 1,):
                    _error(
                        report,
                        "INVALID_DIRECT_LIDAR_OFFSETS",
                        "raw_frame_offsets must be uint64[N+1]",
                        raw_offsets.name,
                    )
                else:
                    raw_values = np.asarray(raw_offsets[:], dtype=np.uint64)
                    raw_rows = int(raw_points.shape[0]) if raw_points is not None else -1
                    if (
                        raw_values[0] != 0
                        or raw_values[-1] != raw_rows
                        or np.any(np.diff(raw_values.astype(np.int64)) < 0)
                    ):
                        _error(
                            report,
                            "INVALID_DIRECT_LIDAR_OFFSETS",
                            "raw_frame_offsets do not partition raw_points",
                            raw_offsets.name,
                        )
            audit: dict[str, np.ndarray] = {}
            for name in (
                "attack_active",
                "attacked_point_count",
                "expected_phantom_count",
                "matched_phantom_count",
                "navigation_lidar_active",
                "raw_point_count",
            ):
                dataset = _require_dataset(group, name, report)
                expected_dtype = np.bool_ if name in {"attack_active", "navigation_lidar_active"} else np.uint64
                if dataset is None:
                    continue
                if dataset.dtype != np.dtype(expected_dtype) or dataset.shape != (rows,):
                    _error(
                        report,
                        "INVALID_DIRECT_LIDAR_AUDIT",
                        f"{name} has invalid dtype/shape",
                        dataset.name,
                    )
                else:
                    audit[name] = np.asarray(dataset[:])
            if "matched_phantom_count" in audit and "expected_phantom_count" in audit and np.any(
                audit["matched_phantom_count"] > audit["expected_phantom_count"]
            ):
                _error(
                    report,
                    "INVALID_DIRECT_LIDAR_AUDIT",
                    "matched_phantom_count exceeds expected_phantom_count",
                    group.name,
                )
            if offsets is not None and "attacked_point_count" in audit:
                attacked_counts = np.diff(np.asarray(offsets[:], dtype=np.uint64))
                if not np.array_equal(audit["attacked_point_count"], attacked_counts):
                    _error(
                        report,
                        "INVALID_DIRECT_LIDAR_AUDIT",
                        "attacked_point_count differs from frame_offsets",
                        group.name,
                    )
            if raw_offsets is not None and "raw_point_count" in audit:
                raw_counts = np.diff(np.asarray(raw_offsets[:], dtype=np.uint64))
                if not np.array_equal(audit["raw_point_count"], raw_counts):
                    _error(
                        report,
                        "INVALID_DIRECT_LIDAR_AUDIT",
                        "raw_point_count differs from raw_frame_offsets",
                        group.name,
                    )
        if "ray_index" in fields:
            ray_index = _require_dataset(group, "ray_index", report)
            if ray_index is not None and points is not None:
                field = fields["ray_index"]
                if ray_index.dtype != np.dtype(field.dtype) or ray_index.shape != (points.shape[0],) + field.shape:
                    _error(report, "INVALID_LIDAR_LAYOUT", "ray_index does not align with points", ray_index.name)
    for field in stream.fields:
        if stream.kind == "lidar" and field.name in {"points", "ray_index"}:
            continue
        dataset = _require_dataset(group, field.name, report)
        if dataset is None:
            continue
        expected_shape = (rows,) + field.shape
        if dataset.dtype != np.dtype(field.dtype) or dataset.shape != expected_shape:
            _error(
                report,
                "INVALID_CAPTURE_FIELD",
                f"field must be {field.dtype}{expected_shape}, got {dataset.dtype}{dataset.shape}",
                dataset.name,
            )
        else:
            _scan_finite(dataset, report)


def _check_v43_direct_evidence(
    demo: h5py.Group,
    contract: V4TaskContract,
    report: ValidationReport,
    samples: int,
    participant_result: dict[str, Any] | None,
) -> None:
    spec = contract.direct_scoring
    if spec is None:
        _error(report, "MISSING_DIRECT_SCORING_CONTRACT", "V4.3 contract lacks direct_scoring", demo.name)
        return
    sim_time = _require_dataset(demo, "sim_time", report)
    terminated = _require_dataset(demo, "terminated", report)
    if sim_time is not None:
        if sim_time.dtype != np.dtype(np.float64) or sim_time.shape != (samples,):
            _error(report, "INVALID_DIRECT_TIME", f"sim_time must be float64[{samples}]", sim_time.name)
        else:
            values = np.asarray(sim_time[:], dtype=np.float64)
            expected = np.arange(1, samples + 1, dtype=np.float64) * contract.control.control_dt
            if not np.isfinite(values).all() or not np.allclose(values, expected, rtol=0.0, atol=1e-9):
                _error(report, "INVALID_DIRECT_TIME", "sim_time differs from the control-step cadence", sim_time.name)
    if terminated is not None:
        if terminated.dtype != np.dtype(np.bool_) or terminated.shape != (samples,):
            _error(report, "INVALID_TERMINATION_TIMELINE", f"terminated must be bool[{samples}]", terminated.name)
        else:
            values = np.asarray(terminated[:], dtype=np.bool_)
            if np.any(values[:-1]):
                _error(
                    report,
                    "EARLY_TERMINATION_TAIL",
                    "recording continued after the official termination step",
                    terminated.name,
                )
            if not bool(values[-1]) and samples != contract.limits.max_steps:
                _error(
                    report,
                    "TRUNCATED_EPISODE",
                    "episode ended before an official termination or max_steps",
                    terminated.name,
                )

    task = demo.get("task")
    if not isinstance(task, h5py.Group):
        _error(report, "MISSING_DIRECT_TASK", "V4.3 task evidence must be a group", f"{demo.name}/task")
        return
    expected_fields = {field.name for field in spec.task_fields}
    observed_fields = set(_bounded_child_names(task, len(expected_fields)))
    if observed_fields != expected_fields:
        _error(
            report,
            "UNEXPECTED_DIRECT_TASK_FIELD",
            f"task fields must be {sorted(expected_fields)}, got {sorted(observed_fields)}",
            task.name,
        )
    final_values: dict[str, Any] = {}
    for field in spec.task_fields:
        dataset = _require_dataset(task, field.name, report)
        expected_shape = (samples,) + field.shape
        if dataset is None:
            continue
        if dataset.dtype != np.dtype(field.dtype) or dataset.shape != expected_shape:
            _error(
                report,
                "INVALID_DIRECT_TASK_FIELD",
                f"field must be {field.dtype}{expected_shape}, got {dataset.dtype}{dataset.shape}",
                dataset.name,
            )
            continue
        _scan_finite(dataset, report)
        if field.shape == ():
            final_values[field.name] = dataset[-1].item()

    events = demo.get("events")
    if not isinstance(events, h5py.Group) or set(_bounded_child_names(events, 1)) != {"jsonl"}:
        _error(report, "INVALID_DIRECT_EVENTS", "events must contain only jsonl", f"{demo.name}/events")
    else:
        jsonl = events.get("jsonl")
        if not isinstance(jsonl, h5py.Dataset) or jsonl.ndim != 1 or jsonl.dtype.kind != "S":
            _error(report, "INVALID_DIRECT_EVENTS", "events/jsonl must be a fixed-width byte string vector", events.name)
        else:
            for index, raw in enumerate(jsonl[:]):
                try:
                    event = json.loads(_decode(raw))
                    if not isinstance(event, dict):
                        raise TypeError
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    _error(
                        report,
                        "INVALID_DIRECT_EVENT",
                        f"events/jsonl row {index} must decode to an object",
                        jsonl.name,
                    )
                    break

    if participant_result is None:
        return
    exact = {
        "authority": "runner_direct_capture",
        "scoring_source": "participant_direct",
        "submitted_steps": samples,
        "steps_executed": samples,
    }
    for name, expected in exact.items():
        if participant_result.get(name) != expected:
            _error(report, "DIRECT_RESULT_MISMATCH", f"participant result {name!r} differs from evidence", demo.name)
    result_bindings = {
        "success": "success",
        "task_complete": "success",
        "task_score": "score",
        "status": "status",
        "completed": "completed",
        "total": "total",
    }
    for result_name, field_name in result_bindings.items():
        if field_name in final_values and participant_result.get(result_name) != final_values[field_name]:
            _error(
                report,
                "DIRECT_RESULT_MISMATCH",
                f"participant result {result_name!r} differs from the final task row",
                demo.name,
            )
    final_task = participant_result.get("task")
    if not isinstance(final_task, dict) or set(final_task) != expected_fields:
        _error(report, "DIRECT_RESULT_MISMATCH", "participant result task object differs from the contract", demo.name)
    else:
        for field_name, expected in final_values.items():
            if final_task.get(field_name) != expected:
                _error(
                    report,
                    "DIRECT_RESULT_MISMATCH",
                    f"participant result task field {field_name!r} differs from the final task row",
                    demo.name,
                )


def _check_demo_v41(hdf5_file: h5py.File, contract: V4TaskContract, report: ValidationReport) -> None:
    if DATA_GROUP not in hdf5_file or not isinstance(hdf5_file[DATA_GROUP], h5py.Group):
        _error(report, "MISSING_DATA_GROUP", "top-level /data group is required", "/data")
        return
    data = hdf5_file[DATA_GROUP]
    if set(_bounded_child_names(data, 1)) != {DEMO_NAME}:
        _error(report, "INVALID_DEMO_LAYOUT", "/data must contain only demo_0", data.name)
        return
    demo = data[DEMO_NAME]
    expected_children = {
        "initial_state",
        "actions",
        "states",
        "tracked_pose",
        "time",
        "feedback",
        "sensors",
        "policy",
    }
    if contract.is_v4_3:
        expected_children.update({"sim_time", "terminated", "task", "events"})
    observed_children = set(_bounded_child_names(demo, len(expected_children)))
    if observed_children != expected_children:
        _error(
            report,
            "UNEXPECTED_DEMO_OBJECT",
            f"V4.1 demo children must be {sorted(expected_children)}, got {sorted(observed_children)}",
            demo.name,
        )
    try:
        samples = int(demo.attrs.get("num_samples", -1))
    except (TypeError, ValueError):
        samples = -1
    report.num_samples = samples if samples >= 0 else None
    if samples < 1 or samples > contract.limits.max_steps:
        _error(report, "INVALID_SAMPLE_COUNT", "V4.1 num_samples is outside contract limits", demo.name)
        return
    if int(demo.attrs.get("seed", -1)) != contract.seed:
        _error(report, "CONTRACT_MISMATCH", "demo seed differs from contract", demo.name)
    if not {"seed", "num_samples", "participant_result_json"} <= set(demo.attrs) <= {
        "seed",
        "num_samples",
        "participant_result_json",
        "success",
    }:
        _error(report, "UNEXPECTED_DEMO_ATTRIBUTE", "demo attributes differ from V4.1", demo.name)
    result: dict[str, Any] | None = None
    try:
        result = json.loads(_decode(demo.attrs.get("participant_result_json", "")))
        if not isinstance(result, dict):
            raise TypeError
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        _error(report, "INVALID_RESULT", "participant_result_json must decode to an object", demo.name)

    actions = _require_dataset(demo, "actions", report)
    if _expect_float32(actions, report) and actions.shape != (samples, contract.control.action_dim):
        _error(report, "INVALID_SHAPE", "actions differ from the frozen base_velocity shape", actions.name)
    time_group = demo.get("time")
    if not isinstance(time_group, h5py.Group):
        _error(report, "MISSING_TIME", "V4.1 time must be a group", f"{demo.name}/time")
    else:
        if set(_bounded_child_names(time_group, 3)) != {"step_index", "action_time_s", "state_time_s"}:
            _error(report, "UNEXPECTED_TIME_OBJECT", "V4.1 time fields differ from schema", time_group.name)
        step_index = _require_dataset(time_group, "step_index", report)
        action_time = _require_dataset(time_group, "action_time_s", report)
        state_time = _require_dataset(time_group, "state_time_s", report)
        if step_index is not None:
            if step_index.dtype != np.dtype(np.uint64) or step_index.shape != (samples,) or not np.array_equal(
                step_index[:], np.arange(samples, dtype=np.uint64)
            ):
                _error(report, "INVALID_CONTROL_TIME", "time/step_index must be contiguous uint64", step_index.name)
        for name, dataset, offset in (
            ("action_time_s", action_time, 0.0),
            ("state_time_s", state_time, contract.control.control_dt),
        ):
            if dataset is None:
                continue
            expected = np.arange(samples, dtype=np.float64) * contract.control.control_dt + offset
            if dataset.dtype != np.dtype(np.float64) or dataset.shape != (samples,):
                _error(report, "INVALID_CONTROL_TIME", f"{name} must be float64[{samples}]", dataset.name)
            else:
                values = np.asarray(dataset[:], dtype=np.float64)
                if not np.isfinite(values).all() or not np.allclose(values, expected, rtol=0.0, atol=1e-9):
                    _error(report, "INVALID_CONTROL_TIME", f"{name} differs from the frozen cadence", dataset.name)

    trailing_shapes: dict[str, tuple[int, ...]] = {}
    for path in contract.required_initial_state_paths:
        relative = path.removeprefix("initial_state/")
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            if dataset.ndim < 2 or dataset.shape[0] != 1:
                _error(report, "INVALID_SHAPE", "initial state must have one leading row", dataset.name)
            else:
                trailing_shapes[relative] = dataset.shape[1:]
    for path in contract.required_state_paths:
        relative = path.removeprefix("states/")
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            expected_shape = (samples,) + trailing_shapes.get(relative, dataset.shape[1:])
            if dataset.shape != expected_shape:
                _error(report, "INVALID_SHAPE", f"state shape must be {expected_shape}", dataset.name)
    tracked_widths = {
        "tracked_pose/position": 3,
        "tracked_pose/orientation": 4,
        "tracked_pose/linear_velocity": 3,
        "tracked_pose/angular_velocity": 3,
    }
    for path in contract.required_ee_pose_paths:
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            expected_shape = (samples, tracked_widths[path])
            if dataset.shape != expected_shape:
                _error(report, "INVALID_SHAPE", f"{path} must be {expected_shape}", dataset.name)
    _check_v41_network(demo, report, samples)
    for stream in contract.runner_capture.streams:
        _check_v41_stream(demo, stream, contract, report, samples)
    if contract.is_v4_3:
        _check_v43_direct_evidence(demo, contract, report, samples, result)


def _check_demo(hdf5_file: h5py.File, contract: V4TaskContract, report: ValidationReport) -> None:
    if contract.uses_full_capture:
        _check_demo_v41(hdf5_file, contract, report)
        return
    if DATA_GROUP not in hdf5_file or not isinstance(hdf5_file[DATA_GROUP], h5py.Group):
        _error(report, "MISSING_DATA_GROUP", "top-level /data group is required", "/data")
        return
    data = hdf5_file[DATA_GROUP]
    if set(_bounded_child_names(data, 1)) != {DEMO_NAME}:
        _error(report, "INVALID_DEMO_LAYOUT", "/data must contain only demo_0", data.name)
        return
    demo = data[DEMO_NAME]
    if not isinstance(demo, h5py.Group):
        _error(report, "INVALID_DEMO_LAYOUT", "demo_0 must be a group", demo.name)
        return
    expected_children = {"initial_state", "actions", "states", "ee_pose", "control_time", "policy"}
    observed_children = _bounded_child_names(demo, len(expected_children))
    if set(observed_children) != expected_children:
        _error(
            report,
            "UNEXPECTED_DEMO_OBJECT",
            f"demo_0 children must be {sorted(expected_children)}, got {sorted(observed_children)}",
            demo.name,
        )
    try:
        samples = int(demo.attrs.get("num_samples", -1))
    except (TypeError, ValueError):
        samples = -1
    report.num_samples = samples if samples >= 0 else None
    if samples < 1 or samples > contract.limits.max_steps:
        _error(report, "INVALID_SAMPLE_COUNT", f"num_samples={samples} is outside contract limits", demo.name)
        return
    if int(demo.attrs.get("seed", -1)) != contract.seed:
        _error(report, "CONTRACT_MISMATCH", "demo seed differs from contract", demo.name)
    if not {"seed", "num_samples", "participant_result_json"} <= set(demo.attrs) <= {
        "seed",
        "num_samples",
        "participant_result_json",
        "success",
    }:
        _error(report, "UNEXPECTED_DEMO_ATTRIBUTE", "demo attributes differ from the V4 schema", demo.name)
    result_json = demo.attrs.get("participant_result_json")
    try:
        decoded_result = json.loads(_decode(result_json)) if result_json is not None else None
        if not isinstance(decoded_result, dict):
            raise TypeError
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        _error(report, "INVALID_RESULT", "participant_result_json must decode to an object", demo.name)

    actions = _require_dataset(demo, "actions", report)
    _expect_float32(actions, report)
    if actions is not None and actions.shape != (samples, contract.control.action_dim):
        _error(report, "INVALID_SHAPE", f"actions must be {(samples, contract.control.action_dim)}", actions.name)
    control_time = _require_dataset(demo, "control_time", report)
    if control_time is not None:
        if control_time.dtype != np.dtype(np.float64) or control_time.shape != (samples,):
            _error(report, "INVALID_SHAPE", f"control_time must be float64[{samples}]", control_time.name)
        else:
            values = np.asarray(control_time[:])
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values) <= 0):
                _error(report, "INVALID_CONTROL_TIME", "control_time must be finite and increasing", control_time.name)
            expected_times = np.arange(samples, dtype=np.float64) * contract.control.control_dt
            if not np.allclose(values, expected_times, rtol=0.0, atol=1e-12):
                _error(
                    report,
                    "INVALID_CONTROL_TIME",
                    "control_time differs from the frozen control cadence",
                    control_time.name,
                )

    trailing_shapes: dict[str, tuple[int, ...]] = {}
    for path in contract.required_initial_state_paths:
        relative = path.removeprefix("initial_state/")
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            if dataset.ndim < 2 or dataset.shape[0] != 1:
                _error(report, "INVALID_SHAPE", "initial state must have one leading row", dataset.name)
            else:
                trailing_shapes[relative] = dataset.shape[1:]
    for path in contract.required_state_paths:
        relative = path.removeprefix("states/")
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            expected_shape = (samples,) + trailing_shapes.get(relative, dataset.shape[1:])
            if dataset.shape != expected_shape:
                _error(report, "INVALID_SHAPE", f"state shape must be {expected_shape}", dataset.name)
    ee_widths = {
        "ee_pose/position": 3,
        "ee_pose/orientation": 4,
        "ee_pose/linear_velocity": 3,
        "ee_pose/angular_velocity": 3,
    }
    for path in contract.required_ee_pose_paths:
        dataset = _require_dataset(demo, path, report)
        if _expect_float32(dataset, report):
            width = ee_widths.get(path)
            if dataset.ndim < 2 or dataset.shape[0] != samples or (
                width is not None and dataset.shape != (samples, width)
            ):
                _error(report, "INVALID_SHAPE", f"{path} has invalid T/width shape {dataset.shape}", dataset.name)
    _check_policy_audit(demo, contract, report, samples)


def validate_submission(path: str | Path, contract: V4TaskContract) -> ValidationReport:
    source = Path(path).expanduser().resolve()
    report = ValidationReport(
        valid=False,
        status="INVALID_SUBMISSION",
        contract_id=contract.contract_id,
        input_path=str(source),
        finalized=source.name == "submission.hdf5",
    )
    if not source.is_file():
        _error(report, "FILE_NOT_FOUND", "submission file does not exist", str(source))
        return report
    report.file_size_bytes = source.stat().st_size
    report.sha256 = sha256_file(source)
    if report.file_size_bytes > contract.limits.max_file_bytes:
        _error(report, "FILE_TOO_LARGE", f"{report.file_size_bytes} exceeds contract limit", str(source))
        return report
    try:
        with h5py.File(source, "r") as hdf5_file:
            if set(_bounded_child_names(hdf5_file, 2)) != {METADATA_GROUP, DATA_GROUP}:
                _error(report, "INVALID_TOP_LEVEL", "top level must contain exactly /metadata and /data", "/")
            _walk_hdf5(hdf5_file, contract, report)
            # Never follow expected-looking paths after the structural scan has
            # found an unsafe link, dtype, topology, or resource-limit failure.
            if not report.errors:
                _check_metadata(hdf5_file, contract, report)
                if not any(issue.code == "UNSUPPORTED_SCHEMA" for issue in report.errors):
                    _check_demo(hdf5_file, contract, report)
    except (OSError, RuntimeError, ValueError) as error:
        _error(report, "UNREADABLE_HDF5", f"failed to inspect HDF5: {error}", str(source))
    report.valid = not report.errors
    report.status = "VALID_SUBMISSION" if report.valid else "INVALID_SUBMISSION"
    return report


def finalize_submission(
    partial_path: str | Path,
    final_path: str | Path,
    contract: V4TaskContract,
) -> ValidationReport:
    partial = Path(partial_path).expanduser().resolve()
    final = Path(final_path).expanduser().resolve()
    if partial.parent != final.parent:
        raise ValueError("partial and final submission must be in the same directory")
    if partial.name != "submission.hdf5.partial" or final.name != "submission.hdf5":
        raise ValueError("V4 publication requires submission.hdf5.partial -> submission.hdf5")
    if final.exists():
        raise FileExistsError(f"refusing to overwrite final submission: {final}")
    report = validate_submission(partial, contract)
    if not report.valid:
        return report
    publish_no_replace(partial, final)
    final_report = validate_submission(final, contract)
    if not final_report.valid:
        raise RuntimeError("published V4 submission failed post-publication validation")
    return final_report


def validate_episode_video(
    path: str | Path,
    contract: V4TaskContract,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Validate the exact H.264 stream bound by a V4.1 contract."""
    source = Path(path).expanduser().resolve()
    errors: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "path": str(source),
        "sha256": None,
        "size_bytes": None,
        "camera": None,
        "codec": None,
        "pixel_format": None,
        "width": None,
        "height": None,
        "fps": None,
        "frames": None,
        "duration_s": None,
        "errors": errors,
    }
    if not contract.uses_full_capture or contract.runner_capture is None:
        errors.append("episode.mp4 is defined only for full-capture formats")
        return result
    if not source.is_file() or source.stat().st_size <= 0:
        errors.append("episode video does not exist or is empty")
        return result
    result["sha256"] = sha256_file(source)
    result["size_bytes"] = source.stat().st_size
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        errors.append(f"ffprobe failed: {error}")
        return result
    if process.returncode != 0:
        errors.append(f"ffprobe rejected video: {process.stderr[-1000:]}")
        return result
    try:
        probe = json.loads(process.stdout)
        streams = [stream for stream in probe.get("streams", []) if isinstance(stream, dict)]
        if len(streams) != 1:
            raise ValueError(f"expected exactly one video stream, got {len(streams)}")
        stream = streams[0]
        numerator, denominator = str(stream.get("avg_frame_rate", "0/0")).split("/", maxsplit=1)
        fps = float(numerator) / float(denominator)
        frames = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
        duration = float(stream.get("duration") or probe.get("format", {}).get("duration"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(f"ffprobe output is incomplete: {error}")
        return result
    spec = contract.runner_capture.video
    result.update(
        {
            "camera": spec.camera,
            "codec": str(stream.get("codec_name", "")),
            "pixel_format": str(stream.get("pix_fmt", "")),
            "width": int(stream.get("width", -1)),
            "height": int(stream.get("height", -1)),
            "fps": fps,
            "frames": frames,
            "duration_s": duration,
        }
    )
    if result["codec"] != spec.codec:
        errors.append(f"video codec {result['codec']!r} differs from {spec.codec!r}")
    if result["pixel_format"] != spec.pixel_format:
        errors.append(f"video pixel format {result['pixel_format']!r} differs from {spec.pixel_format!r}")
    if (result["width"], result["height"]) != (spec.width, spec.height):
        errors.append("video resolution differs from the primary camera contract")
    if not math.isclose(fps, spec.fps, rel_tol=0.0, abs_tol=1e-6):
        errors.append("video frame rate differs from the contract")
    if frames != int(expected_frames):
        errors.append(f"video frames={frames}, primary RGB samples={expected_frames}")
    expected_duration = frames / spec.fps
    if not math.isfinite(duration) or duration <= 0 or not math.isclose(
        duration, expected_duration, rel_tol=0.0, abs_tol=1.0 / spec.fps
    ):
        errors.append("video duration differs from frame_count/fps")
    result["valid"] = not errors
    return result


def finalize_capture(
    hdf5_partial_path: str | Path,
    hdf5_final_path: str | Path,
    video_partial_path: str | Path,
    video_final_path: str | Path,
    contract: V4TaskContract,
) -> tuple[ValidationReport, dict[str, Any]]:
    """Validate both V4.1 partials, then publish both with rollback on failure."""
    hdf5_partial = Path(hdf5_partial_path).expanduser().resolve()
    hdf5_final = Path(hdf5_final_path).expanduser().resolve()
    video_partial = Path(video_partial_path).expanduser().resolve()
    video_final = Path(video_final_path).expanduser().resolve()
    paths = (hdf5_partial, hdf5_final, video_partial, video_final)
    if len({path.parent for path in paths}) != 1:
        raise ValueError("V4.1 partial/final artifacts must share one directory")
    if tuple(path.name for path in paths) != (
        "submission.hdf5.partial",
        "submission.hdf5",
        "episode.mp4.partial",
        "episode.mp4",
    ):
        raise ValueError("invalid V4.1 publication filenames")
    if hdf5_final.exists() or video_final.exists():
        raise FileExistsError("refusing to overwrite a formal V4.1 artifact")
    hdf5_report = validate_submission(hdf5_partial, contract)
    if not hdf5_report.valid:
        return hdf5_report, {"valid": False, "errors": ["HDF5 partial is invalid"]}
    with h5py.File(hdf5_partial, "r") as hdf5_file:
        rgb_samples = int(
            hdf5_file[f"data/demo_0/sensors/camera/{contract.runner_capture.video.camera}/rgb"].shape[0]
        )
    video_report = validate_episode_video(video_partial, contract, expected_frames=rgb_samples)
    if not video_report["valid"]:
        return hdf5_report, video_report
    try:
        publish_no_replace(hdf5_partial, hdf5_final)
        publish_no_replace(video_partial, video_final)
        final_hdf5_report = validate_submission(hdf5_final, contract)
        final_video_report = validate_episode_video(video_final, contract, expected_frames=rgb_samples)
        if not final_hdf5_report.valid or not final_video_report["valid"]:
            raise RuntimeError("published V4.1 artifacts failed post-publication validation")
        return final_hdf5_report, final_video_report
    except Exception:
        if hdf5_final.exists() and not hdf5_partial.exists():
            os.replace(hdf5_final, hdf5_partial)
        if video_final.exists() and not video_partial.exists():
            os.replace(video_final, video_partial)
        raise


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "finalize_capture",
    "finalize_submission",
    "validate_episode_video",
    "validate_submission",
]
