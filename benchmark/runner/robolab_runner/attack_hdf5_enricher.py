"""Build a lossless attack-evidence extension without modifying submission.hdf5."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


DIMENSIONS = {
    "image": "图像攻击",
    "state_feedback": "状态反馈攻击",
    "prompt": "语言指令攻击",
    "vla_output_control": "VLA输出与控制攻击",
    "lidar": "LiDAR攻击",
    "imu": "IMU攻击",
}

STAGE_DIMENSION = {
    "request_image": "image",
    "request_state": "state_feedback",
    "state_feedback": "state_feedback",
    "request_prompt": "prompt",
    "response_action": "vla_output_control",
    "control_base": "vla_output_control",
    "control_joint": "vla_output_control",
    "control_gripper": "vla_output_control",
    "control_ee_target": "vla_output_control",
    "immediate_robot_control": "vla_output_control",
    "immediate_gripper": "vla_output_control",
    "immediate_visual_material": "image",
    "sensor_lidar": "lidar",
    "immediate_lidar": "lidar",
    "sensor_imu": "imu",
    "immediate_imu": "imu",
    "sensor_force_torque": "state_feedback",
    "sensor_wheel_encoder": "state_feedback",
    "sensor_joint_encoder": "state_feedback",
}

Q17_FORCE_FEEDBACK_ATTACK_ID = "q17_gripper_force_torque_positive_feedback"
Q17_FORCE_FEEDBACK_CONTEXT_KEY = "q17_force_feedback_bridge"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_digest(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, np.ndarray):
        payload = np.ascontiguousarray(value).tobytes()
    elif isinstance(value, bytes):
        payload = value
    else:
        payload = np.asarray(value).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _original_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                result["/" + name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "compression": obj.compression,
                    "sha256": _dataset_digest(obj),
                }
        h5.visititems(visitor)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.append(json.loads(line))
    return result


ATTACK_MARKER_EVENTS = frozenset({"condition_triggered", "effect_first_applied"})


def _attack_marker_rows(
    events: list[dict[str, Any]],
    specs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and return the exact attack marker field-contract rows."""
    configured_ids = {str(spec["id"]) for spec in specs}
    rows: list[dict[str, Any]] = []
    conditions: dict[str, tuple[str, float, int]] = {}
    effects: set[str] = set()
    for event in events:
        if event.get("event") not in ATTACK_MARKER_EVENTS:
            continue
        attack_id = event.get("attack_id")
        instance = event.get("attack_instance_id")
        time_s = event.get("time_s")
        physics_step = event.get("physics_step")
        marker_only = event.get("marker_only")
        effect_applied = event.get("effect_applied")
        observation_gap = event.get("first_confirmed_after_observation_gap")
        evidence = event.get("evidence")
        if not isinstance(attack_id, str) or attack_id not in configured_ids:
            raise ValueError("attack marker references an unconfigured attack")
        if not isinstance(instance, str) or not instance.strip():
            raise ValueError("attack marker is missing attack_instance_id")
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not np.isfinite(time_s)
            or float(time_s) < 0.0
        ):
            raise ValueError(f"invalid attack marker time: {attack_id}")
        if (
            isinstance(physics_step, bool)
            or not isinstance(physics_step, int)
            or physics_step < 0
        ):
            raise ValueError(f"invalid attack marker physics step: {attack_id}")
        if not all(type(value) is bool for value in (marker_only, effect_applied, observation_gap)):
            raise ValueError(f"invalid attack marker flags: {attack_id}")
        if marker_only and effect_applied:
            raise ValueError(f"attack marker cannot be marker-only and applied: {attack_id}")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"attack marker evidence must be an object: {attack_id}")
        if event["event"] == "condition_triggered":
            if not marker_only or effect_applied or evidence.get("event") != "condition_triggered":
                raise ValueError(f"invalid condition marker semantics: {attack_id}")
            if instance in conditions:
                raise ValueError(f"duplicate condition marker instance: {instance}")
            conditions[instance] = (attack_id, float(time_s), int(physics_step))
        else:
            if marker_only or not effect_applied or evidence.get("event") != "effect_first_applied":
                raise ValueError(f"invalid effect marker semantics: {attack_id}")
            if instance in effects:
                raise ValueError(f"duplicate effect marker instance: {instance}")
            if instance not in conditions:
                raise ValueError(f"effect marker has no condition marker: {instance}")
            condition_id, condition_time, condition_step = conditions[instance]
            if (
                condition_id != attack_id
                or float(time_s) < condition_time
                or int(physics_step) < condition_step
            ):
                raise ValueError(f"effect marker precedes its condition: {instance}")
            effects.add(instance)
        rows.append({
            "attack_id": attack_id,
            "attack_instance_id": instance,
            "time_s": float(time_s),
            "physics_step": int(physics_step),
            "marker_only": marker_only,
            "effect_applied": effect_applied,
            "first_confirmed_after_observation_gap": observation_gap,
            "evidence": dict(evidence),
        })
    return rows


def _write_json_rows(group: h5py.Group, rows: list[Mapping[str, Any]]) -> None:
    """Use the handoff document's offset-indexed-json/v1 byte layout."""
    encoded = [
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        for row in rows
    ]
    payload = b"".join(encoded)
    offsets = np.concatenate((
        np.asarray([0], dtype=np.uint64),
        np.cumsum([len(value) for value in encoded], dtype=np.uint64),
    ))
    group.create_dataset(
        "utf8",
        data=np.frombuffer(payload, dtype=np.uint8),
        compression="gzip" if payload else None,
    )
    group.create_dataset(
        "offsets",
        data=offsets,
        compression="gzip" if len(offsets) > 1 else None,
    )
    group.attrs["encoding"] = "utf-8"
    group.attrs["layout"] = "offset-indexed-json/v1"


def _put(group: h5py.Group, name: str, value: Any, *, compress: bool = True) -> h5py.Dataset:
    if name in group:
        del group[name]
    if isinstance(value, str):
        return group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    if isinstance(value, (list, tuple)) and (not value or all(isinstance(item, str) for item in value)):
        array = np.asarray(value, dtype=h5py.string_dtype("utf-8"))
        return group.create_dataset(name, data=array)
    array = np.asarray(value)
    if array.dtype.kind == "U":
        if name == "object_ids":
            # Exact RTX IDs are 32 ASCII hex characters (with short sentinels
            # such as "-1" for synthetic returns).  A fixed byte dtype avoids
            # the enormous per-element overhead of HDF5 variable-length text.
            encoded = array.astype("S32")
            compression = "gzip" if encoded.size else None
            return group.create_dataset(name, data=encoded, compression=compression)
        dtype = h5py.string_dtype("utf-8")
        return group.create_dataset(name, data=array.astype(object), dtype=dtype)
    compression = "gzip" if compress and array.ndim > 0 and array.size > 0 else None
    return group.create_dataset(name, data=array, compression=compression)


def _safe_name(value: str) -> str:
    return value.replace("/", "%2F").replace("\\", "%5C")


def _event_time(event: Mapping[str, Any]) -> float | None:
    context = event.get("runtime_context", {})
    value = context.get("time_s") if isinstance(context, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _event_step(event: Mapping[str, Any]) -> int | None:
    context = event.get("runtime_context", {})
    value = context.get("physics_step") if isinstance(context, Mapping) else None
    return int(value) if isinstance(value, (int, float)) else None


def _immediate_event_time(event: Mapping[str, Any]) -> float | None:
    """Read the simulation time carried by an immediate-attack audit event."""

    value = _event_time(event)
    if value is not None:
        return value
    value = event.get("activated_at")
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    snapshot = event.get("trigger_snapshot", {})
    simulation = snapshot.get("simulation", {}) if isinstance(snapshot, Mapping) else {}
    value = simulation.get("time_s") if isinstance(simulation, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _immediate_event_step(event: Mapping[str, Any]) -> int | None:
    """Read the physics step carried by an immediate-attack audit event."""

    value = _event_step(event)
    if value is not None:
        return value
    for key in ("activated_physics_step", "restored_physics_step", "physics_step"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    snapshot = event.get("trigger_snapshot", {})
    simulation = snapshot.get("simulation", {}) if isinstance(snapshot, Mapping) else {}
    value = simulation.get("physics_step") if isinstance(simulation, Mapping) else None
    return int(value) if isinstance(value, (int, float)) else None


def _trigger_physics_step(
    condition_event: Mapping[str, Any] | None,
) -> int | None:
    """Return the exact task-relative step recorded at trigger confirmation.

    ``condition_triggered`` is emitted by the orchestrator from
    ``simulation.task_physics_step``.  Read that persisted value directly;
    do not infer it from floating-point time or from the first effect
    application, which can occur later.
    """

    if condition_event is None:
        return None
    value = condition_event.get("physics_step")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _stack(values: Iterable[Any]) -> np.ndarray | None:
    arrays = [np.asarray(value) for value in values]
    if not arrays:
        return None
    if any(array.shape != arrays[0].shape for array in arrays):
        return None
    return np.stack(arrays)


def _canonical_payload_json(value: Any) -> str:
    """Serialize one persisted audit payload without inventing missing values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _application_payload_changed(event: Mapping[str, Any]) -> bool:
    """Use persisted hashes when available, otherwise compare exact JSON values."""

    before = event.get("before")
    after = event.get("after")
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_sha256 = before.get("sha256")
        after_sha256 = after.get("sha256")
        if isinstance(before_sha256, str) and isinstance(after_sha256, str):
            return before_sha256 != after_sha256
    return _canonical_payload_json(before) != _canonical_payload_json(after)



def _validate_application_audit(
    attack_id: str,
    applications: list[dict[str, Any]],
    *,
    triggered: bool,
    applied: bool,
    reported_count: Any,
) -> None:
    """Reject incomplete evidence before publishing a completed submission."""
    def fail(reason: str) -> None:
        raise ValueError(f"ATTACK_APPLICATION_EVIDENCE_INCOMPLETE:{attack_id}:{reason}")

    if applications and not triggered:
        fail("application rows exist without an activation event")
    if applied and not applications:
        fail("payload applied but no payload_intercepted audit rows")
    if reported_count is not None and reported_count != len(applications):
        fail("reported application_count differs from persisted audit row count")
    for index, event in enumerate(applications, 1):
        if event.get("application") != index:
            fail("application indices must be contiguous and start at one")
        if "before" not in event or "after" not in event:
            fail("application row is missing before/after payloads")
        if _event_time(event) is None or _event_step(event) is None:
            fail("application row is missing finite runtime time/physics step")


def _write_application_payloads(
    attack_group: h5py.Group,
    applications: list[dict[str, Any]],
    availability: dict[str, bool],
) -> None:
    """Persist exact per-application before/after payloads from Runner audit rows.

    The VLA request window is useful causal context, but it is not always the
    instant at which a state/sensor handler ran.  These datasets retain that
    exact interception boundary so an already-settled later request cannot
    mask a real mutation (for example gripper state 0.608 -> 1.0).
    """

    group = attack_group.create_group("applications")
    availability["application_payloads"] = bool(applications)
    if not applications:
        group.attrs["available"] = False
        group.attrs["reason"] = "no payload_intercepted audit rows were recorded"
        return

    before_values = [event.get("before") for event in applications]
    after_values = [event.get("after") for event in applications]
    changed = np.asarray(
        [_application_payload_changed(event) for event in applications],
        dtype=np.bool_,
    )
    _put(
        group,
        "application_index",
        [int(event.get("application", index + 1)) for index, event in enumerate(applications)],
    )
    _put(
        group,
        "physics_step",
        [-1 if _event_step(event) is None else int(_event_step(event)) for event in applications],
    )
    _put(
        group,
        "timestamp_s",
        [np.nan if _event_time(event) is None else float(_event_time(event)) for event in applications],
    )
    _put(group, "payload_changed", changed)
    _put(group, "original_json", [_canonical_payload_json(value) for value in before_values])
    _put(group, "attacked_json", [_canonical_payload_json(value) for value in after_values])

    before_hashes = [
        str(value.get("sha256", "")) if isinstance(value, Mapping) else ""
        for value in before_values
    ]
    after_hashes = [
        str(value.get("sha256", "")) if isinstance(value, Mapping) else ""
        for value in after_values
    ]
    _put(group, "original_sha256", before_hashes)
    _put(group, "attacked_sha256", after_hashes)

    # Numeric ndarray payloads additionally get directly comparable tensors.
    if all(
        isinstance(value, Mapping) and "values" in value
        for value in before_values + after_values
    ):
        original = _stack(value["values"] for value in before_values)
        attacked = _stack(value["values"] for value in after_values)
        if (
            original is not None
            and attacked is not None
            and original.shape == attacked.shape
            and original.dtype.kind in "biufc"
            and attacked.dtype.kind in "biufc"
        ):
            _put(group, "original_values", original)
            _put(group, "attacked_values", attacked)
            group.attrs["numeric_values_available"] = True
        else:
            group.attrs["numeric_values_available"] = False
    else:
        group.attrs["numeric_values_available"] = False


def _write_immediate_effect_evidence(
    attack_group: h5py.Group,
    execution: Mapping[str, Any] | None,
    lifecycle_events: list[dict[str, Any]],
    availability: dict[str, bool],
) -> None:
    """Persist immediate-handler evidence without pretending it is an interceptor payload."""

    group = attack_group.create_group("immediate_effect")
    availability["immediate_execution"] = execution is not None
    availability["immediate_effect_lifecycle"] = bool(lifecycle_events)
    if execution is None:
        group.attrs["available"] = False
        group.attrs["reason"] = "no immediate_executed audit row was recorded"
        return

    execution_time = _immediate_event_time(execution)
    execution_step = _immediate_event_step(execution)
    group.attrs["available"] = True
    _put(group, "execution_event_json", _canonical_payload_json(execution), compress=False)
    _put(
        group,
        "execution_time_s",
        np.nan if execution_time is None else execution_time,
        compress=False,
    )
    _put(
        group,
        "execution_physics_step",
        -1 if execution_step is None else execution_step,
        compress=False,
    )
    _put(
        group,
        "lifecycle_event_name",
        [str(event.get("event", "")) for event in lifecycle_events],
    )
    _put(
        group,
        "lifecycle_event_json",
        [_canonical_payload_json(event) for event in lifecycle_events],
    )


POLICY_EXECUTION_FIELDS = {
    "action": "policy_execution_action",
    "raw_action": "policy_execution_raw_action",
    "request_id": "policy_execution_request_id",
    "vla_chunk_id": "policy_execution_vla_chunk_id",
    "action_index_in_chunk": "policy_execution_action_index_in_chunk",
    "execution_index": "policy_execution_index",
    "start_physics_step": "policy_execution_start_physics_step",
    "end_physics_step": "policy_execution_end_physics_step",
    "start_time_s": "policy_execution_start_time_s",
    "end_time_s": "policy_execution_end_time_s",
    "control_source": "policy_execution_control_source",
}


def _policy_execution_stream(
    episode: Mapping[str, Any],
) -> dict[str, np.ndarray] | None:
    """Return the Runner-recorded causal execution stream, never a guessed join."""

    if not all(dataset in episode for dataset in POLICY_EXECUTION_FIELDS.values()):
        return None
    result = {
        name: np.asarray(episode[dataset])
        for name, dataset in POLICY_EXECUTION_FIELDS.items()
    }
    action = result["action"]
    if action.ndim != 2 or action.shape[1:] != (10,):
        raise ValueError(
            f"policy execution action must have shape [E,10], got {action.shape}"
        )
    count = int(action.shape[0])
    expected_shapes = {
        "action": (count, 10),
        "raw_action": (count, 10),
        **{
            name: (count,)
            for name in POLICY_EXECUTION_FIELDS
            if name not in {"action", "raw_action"}
        },
    }
    invalid = {
        name: {"actual": value.shape, "expected": expected_shapes[name]}
        for name, value in result.items()
        if value.shape != expected_shapes[name]
    }
    if invalid:
        raise ValueError(f"policy execution stream shapes differ: {invalid}")
    if count and not np.array_equal(
        result["execution_index"], np.arange(count, dtype=np.uint64)
    ):
        raise ValueError("policy execution_index must be exactly 0..E-1")
    if any(
        not np.isfinite(result[name].astype(np.float64, copy=False)).all()
        for name in ("action", "raw_action", "start_time_s", "end_time_s")
    ):
        raise ValueError("policy execution stream contains NaN or infinity")
    if np.any(result["end_physics_step"] < result["start_physics_step"]) or np.any(
        result["end_time_s"] < result["start_time_s"]
    ):
        raise ValueError("policy execution end precedes start")
    return result


def _hash_images(images: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images]


def _load_vla_requests(root: Path, start_s: float, end_s: float) -> list[dict[str, Any]]:
    return [
        item
        for item in _load_all_vla_requests(root)
        if start_s <= float(np.asarray(item["time_s"]).item()) <= end_s
    ]


def _load_all_vla_requests(root: Path) -> list[dict[str, Any]]:
    """Load every persisted VLA transaction in causal request order."""

    result = []
    for path in sorted((root / "attack_evidence" / "raw_payloads" / "vla_requests").glob("request_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            time_s = float(np.asarray(archive["time_s"]).item())
            if not np.isfinite(time_s):
                raise ValueError(f"VLA request has non-finite time: {path}")
            result.append({key: np.array(archive[key]) for key in archive.files})
    result.sort(key=lambda item: int(np.asarray(item["request_id"]).item()))
    request_ids = [int(np.asarray(item["request_id"]).item()) for item in result]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("VLA request stream contains duplicate request_id values")
    return result


def _flatten_vla_action_chunks(
    requests: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    offsets = []
    counts = []
    cursor = 0
    for request in requests:
        chunk = np.asarray(request[key], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1:] != (10,):
            request_id = int(np.asarray(request["request_id"]).item())
            raise ValueError(
                f"VLA {key} for request {request_id} must have shape [H,10], "
                f"got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError(f"VLA {key} contains NaN or infinity")
        offsets.append(cursor)
        counts.append(int(chunk.shape[0]))
        chunks.append(chunk)
        cursor += int(chunk.shape[0])
    flattened = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, 10), dtype=np.float32)
    )
    return (
        flattened,
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(counts, dtype=np.uint64),
    )


def _write_full_vla_trace(
    demo_group: h5py.Group,
    output_dir: Path,
    episode: Mapping[str, Any],
) -> dict[str, int | bool]:
    """Append the complete real VLA response and execution streams.

    The full trace stores the policy's original responses and their causal
    execution mapping.  Attack-modified responses remain solely in the existing
    per-attack evidence groups, avoiding a second source of truth.

    The formal Judge datasets are left byte-for-byte unchanged.  This trace is
    deliberately kept outside ``/data/demo_0/attacks`` because the frozen
    attack-evidence protocol rejects unknown dimension groups there.  The
    extension namespace is additive and ignored by legacy Judges.
    """

    requests = _load_all_vla_requests(output_dir)
    execution = _policy_execution_stream(episode)
    if not requests and execution is None:
        return {
            "available": False,
            "request_count": 0,
            "response_action_count": 0,
            "execution_count": 0,
        }
    if requests and execution is None:
        raise ValueError("complete VLA trace requires the policy execution stream")

    extensions = demo_group.require_group("extensions")
    if "vla_full_trace" in extensions:
        raise ValueError(
            "formal submission unexpectedly already contains "
            "/data/demo_0/extensions/vla_full_trace"
        )
    trace = extensions.create_group("vla_full_trace")
    trace.attrs["schema"] = "robolab-m20-vla-full-trace/v1"
    trace.attrs["complete"] = False
    request_group = trace.create_group("requests")
    request_ids = np.asarray(
        [int(np.asarray(item["request_id"]).item()) for item in requests],
        dtype=np.int64,
    )
    physics_steps = np.asarray(
        [int(np.asarray(item["physics_step"]).item()) for item in requests],
        dtype=np.int64,
    )
    timestamps = np.asarray(
        [float(np.asarray(item["time_s"]).item()) for item in requests],
        dtype=np.float64,
    )
    _put(request_group, "request_id", request_ids)
    _put(request_group, "physics_step", physics_steps)
    _put(request_group, "timestamp_s", timestamps)

    raw_actions, raw_offsets, raw_counts = _flatten_vla_action_chunks(
        requests, "vla_raw_actions"
    )
    _put(request_group, "action_offset", raw_offsets)
    _put(request_group, "action_count", raw_counts)
    _put(request_group, "raw_actions", raw_actions)
    _put(
        request_group,
        "raw_response_json",
        [str(np.asarray(item["raw_response_json"]).item()) for item in requests],
    )

    execution_group = trace.create_group("execution")
    assert execution is not None
    for name, values in execution.items():
        _put(execution_group, name, values)

    trace.attrs["request_count"] = np.uint64(len(requests))
    trace.attrs["response_action_count"] = np.uint64(raw_actions.shape[0])
    trace.attrs["execution_count"] = np.uint64(execution["action"].shape[0])
    trace.attrs["action_dim"] = np.uint64(10)
    trace.attrs["complete"] = True
    return {
        "available": bool(requests),
        "request_count": len(requests),
        "response_action_count": int(raw_actions.shape[0]),
        "execution_count": int(execution["action"].shape[0]),
    }


VLA_CHANGE_FIELDS = {
    "request_image": (
        ("original_image", "attacked_image"),
        ("original_m20_camera_0_rgb", "forwarded_m20_camera_0_rgb"),
        ("original_m20_camera_1_rgb", "forwarded_m20_camera_1_rgb"),
        ("original_piper_wrist_rgb", "forwarded_piper_wrist_rgb"),
    ),
    "request_state": (("original_state", "attacked_state"),),
    "request_prompt": (("original_prompt", "attacked_prompt"),),
    "response_action": (("vla_raw_actions", "vla_attacked_actions"),),
}


def _load_changed_vla_request_boundaries(
    root: Path,
    stage: str,
) -> list[dict[str, float | int]]:
    """Recover exact VLA application boundaries from persisted payload diffs.

    This is a fail-safe for legacy audit rows that recorded the payload key but
    omitted ``physics_step``/``time_s``.  It never invents a window: a request
    is included only when the stored original and forwarded payloads differ for
    the attack stage.
    """

    field_pairs = VLA_CHANGE_FIELDS.get(stage)
    if not field_pairs:
        return []
    result: list[dict[str, float | int]] = []
    directory = root / "attack_evidence" / "raw_payloads" / "vla_requests"
    for path in sorted(directory.glob("request_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            changed = any(
                before in archive
                and after in archive
                and not np.array_equal(archive[before], archive[after])
                for before, after in field_pairs
            )
            if not changed:
                continue
            time_s = float(np.asarray(archive["time_s"]).item())
            if not np.isfinite(time_s):
                continue
            result.append(
                {
                    "request_id": int(np.asarray(archive["request_id"]).item()),
                    "physics_step": int(np.asarray(archive["physics_step"]).item()),
                    "time_s": time_s,
                }
            )
    return result


def _load_changed_lidar(root: Path, start_s: float, end_s: float) -> list[dict[str, Any]]:
    result = []
    directory = root / "attack_evidence" / "raw_payloads" / "sensor_lidar"
    for path in sorted(directory.glob("sample_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            time_s = float(np.asarray(archive["time_s"]).item())
            if np.isfinite(time_s) and start_s <= time_s <= end_s:
                result.append({key: np.array(archive[key]) for key in archive.files})
    return result


def _filter_raw_records(root: Path, stage: str, start_s: float, end_s: float) -> list[dict[str, Any]]:
    records = _read_jsonl(root / "attack_evidence" / "raw_payloads" / f"{stage}.jsonl")
    result = []
    for record in records:
        context = record.get("runtime_context", {})
        value = context.get("time_s") if isinstance(context, Mapping) else None
        if isinstance(value, (int, float)) and start_s <= float(value) <= end_s:
            result.append(record)
    return result


def _write_availability(group: h5py.Group, availability: Mapping[str, bool]) -> None:
    target = group.create_group("availability")
    for key, value in sorted(availability.items()):
        _put(target, key, np.asarray(bool(value)), compress=False)


def _write_images(
    attack_group: h5py.Group,
    episode: Mapping[str, Any],
    requests: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    availability: dict[str, bool],
) -> None:
    images_group = attack_group.create_group("images")
    times = np.asarray(episode["timestamp_s"], dtype=np.float64)
    mask = (times >= start_s) & (times <= end_s)
    indices = np.flatnonzero(mask)
    simulation = images_group.create_group("simulation")
    simulation_fields = {
        "m20_camera_0_rgb": "m20_camera_0_rgb" if "m20_camera_0_rgb" in episode else "image",
        "m20_camera_1_rgb": "m20_camera_1_rgb" if "m20_camera_1_rgb" in episode else None,
        "piper_wrist_rgb": "piper_wrist_rgb" if "piper_wrist_rgb" in episode else "wrist_image",
    }
    if indices.size:
        for output_name, episode_name in simulation_fields.items():
            if episode_name is not None:
                _put(simulation, output_name, np.asarray(episode[episode_name])[indices])
        # Backward-compatible names are hard links, not duplicate image data.
        simulation["main"] = simulation["m20_camera_0_rgb"]
        simulation["wrist"] = simulation["piper_wrist_rgb"]
        _put(simulation, "timestamp_s", times[indices])
        _put(simulation, "frame_index", indices.astype(np.uint64))
        availability["simulation_images"] = True
    else:
        availability["simulation_images"] = False
        simulation.attrs["available"] = False

    vla = images_group.create_group("vla_input")
    if not requests:
        availability["vla_input_pixels"] = False
        vla.attrs["available"] = False
        return
    original = vla.create_group("original")
    forwarded = vla.create_group("forwarded")
    camera_npz_fields = {
        "m20_camera_0_rgb": ("original_m20_camera_0_rgb", "forwarded_m20_camera_0_rgb"),
        "m20_camera_1_rgb": ("original_m20_camera_1_rgb", "forwarded_m20_camera_1_rgb"),
        "piper_wrist_rgb": ("original_piper_wrist_rgb", "forwarded_piper_wrist_rgb"),
    }
    camera_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (original_key, forwarded_key) in camera_npz_fields.items():
        if original_key not in requests[0] or forwarded_key not in requests[0]:
            continue
        clean = np.stack([item[original_key] for item in requests])
        sent = np.stack([item[forwarded_key] for item in requests])
        _put(original, name, clean)
        _put(forwarded, name, sent)
        camera_pairs[name] = (clean, sent)
    # Legacy names remain available as HDF5 hard links.
    if "m20_camera_0_rgb" in original:
        original["main"] = original["m20_camera_0_rgb"]
        forwarded["main"] = forwarded["m20_camera_0_rgb"]
    if "piper_wrist_rgb" in original:
        original["wrist"] = original["piper_wrist_rgb"]
        forwarded["wrist"] = forwarded["piper_wrist_rgb"]
    vla["attacked"] = forwarded
    _put(vla, "request_id", [int(item["request_id"].item()) for item in requests])
    _put(vla, "timestamp_s", [float(item["time_s"].item()) for item in requests])
    _put(vla, "physics_step", [int(item["physics_step"].item()) for item in requests])
    change_group = vla.create_group("change_metrics")
    for name, (clean, sent) in camera_pairs.items():
        target = change_group.create_group(name)
        original_hashes = _hash_images(clean)
        forwarded_hashes = _hash_images(sent)
        changed_pixels = np.count_nonzero(clean != sent, axis=(1, 2, 3)).astype(np.uint64)
        _put(target, "original_sha256", original_hashes)
        _put(target, "forwarded_sha256", forwarded_hashes)
        _put(target, "image_changed", np.asarray(original_hashes) != np.asarray(forwarded_hashes))
        _put(target, "changed_pixel_count", changed_pixels)
        _put(target, "changed_ratio", changed_pixels / np.prod(clean.shape[1:]))
    availability["vla_input_pixels"] = len(camera_pairs) == 3
    availability["vla_input_three_cameras"] = len(camera_pairs) == 3


def _write_vla_data(
    attack_group: h5py.Group,
    episode: Mapping[str, Any],
    requests: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    action_semantics: list[str],
    availability: dict[str, bool],
) -> None:
    vla = attack_group.create_group("vla")
    input_group = vla.create_group("input")
    input_original = input_group.create_group("original")
    input_forwarded = input_group.create_group("forwarded")
    output_group = vla.create_group("output")
    output_raw = output_group.create_group("raw")
    output_returned = output_group.create_group("returned")
    state_group = attack_group.create_group("observation_state")
    prompt_group = attack_group.create_group("prompt")
    actions_group = attack_group.create_group("actions")
    if requests:
        request_ids = [int(item["request_id"].item()) for item in requests]
        request_times = [float(item["time_s"].item()) for item in requests]
        original_state = np.stack([item["original_state"] for item in requests]).astype(np.float32)
        attacked_state = np.stack([item["attacked_state"] for item in requests]).astype(np.float32)
        _put(input_original, "state", original_state)
        _put(input_forwarded, "state", attacked_state)
        state_group["original"] = input_original["state"]
        state_group["attacked"] = input_forwarded["state"]
        _put(state_group, "request_id", request_ids)
        _put(state_group, "timestamp_s", request_times)
        _put(state_group, "state_changed", np.any(original_state != attacked_state, axis=1))
        availability["observation_state"] = True

        original_prompt = [str(item["original_prompt"].item()) for item in requests]
        attacked_prompt = [str(item["attacked_prompt"].item()) for item in requests]
        _put(input_original, "prompt", original_prompt)
        _put(input_forwarded, "prompt", attacked_prompt)
        prompt_group["original"] = input_original["prompt"]
        prompt_group["attacked"] = input_forwarded["prompt"]
        _put(prompt_group, "request_id", request_ids)
        _put(prompt_group, "timestamp_s", request_times)
        _put(prompt_group, "prompt_changed", np.asarray(original_prompt) != np.asarray(attacked_prompt))
        availability["prompt"] = True

        raw_actions = _stack(item["vla_raw_actions"] for item in requests)
        attacked_actions = _stack(item["vla_attacked_actions"] for item in requests)
        if raw_actions is not None and attacked_actions is not None:
            _put(output_raw, "actions", raw_actions.astype(np.float32))
            _put(output_returned, "actions", attacked_actions.astype(np.float32))
            actions_group["vla_raw"] = output_raw["actions"]
            actions_group["vla_attacked"] = output_returned["actions"]
            _put(actions_group, "request_id", request_ids)
            _put(actions_group, "action_changed", np.any(raw_actions != attacked_actions, axis=(1, 2)))
            availability["vla_actions"] = True
        else:
            availability["vla_actions"] = False
            actions_group.attrs["variable_horizon_not_stacked"] = True
        for name in (
            "original_request_metadata_json",
            "forwarded_request_metadata_json",
        ):
            if name in requests[0]:
                _put(input_group, name, [str(item[name].item()) for item in requests])
        if "raw_response_json" in requests[0]:
            _put(output_raw, "response_json", [str(item["raw_response_json"].item()) for item in requests])
        if "returned_response_json" in requests[0]:
            _put(output_returned, "response_json", [str(item["returned_response_json"].item()) for item in requests])
        _put(input_group, "request_id", request_ids)
        _put(input_group, "timestamp_s", request_times)
        _put(input_group, "physics_step", [int(item["physics_step"].item()) for item in requests])
        # Link exact three-camera pixels recorded by _write_images.
        image_original = attack_group["images/vla_input/original"]
        image_forwarded = attack_group["images/vla_input/forwarded"]
        for camera_name in ("m20_camera_0_rgb", "m20_camera_1_rgb", "piper_wrist_rgb"):
            if camera_name in image_original:
                input_original[camera_name] = image_original[camera_name]
                input_forwarded[camera_name] = image_forwarded[camera_name]
        availability["vla_complete_input"] = all(
            name in input_original and name in input_forwarded
            for name in ("m20_camera_0_rgb", "m20_camera_1_rgb", "piper_wrist_rgb", "state", "prompt")
        )
        availability["vla_complete_output"] = raw_actions is not None and attacked_actions is not None
    else:
        availability.update({
            "observation_state": False,
            "prompt": False,
            "vla_actions": False,
            "vla_complete_input": False,
            "vla_complete_output": False,
        })
        state_group.attrs["available"] = False
        prompt_group.attrs["available"] = False

    times = np.asarray(episode["timestamp_s"], dtype=np.float64)
    mask = (times >= start_s) & (times <= end_s)
    sampled_control = actions_group.create_group("control_stream_in_attack_window")
    _put(
        sampled_control,
        "action",
        np.asarray(episode["action"], dtype=np.float32)[mask],
    )
    _put(sampled_control, "timestamp_s", times[mask])

    execution = _policy_execution_stream(episode)
    if execution is not None and requests:
        request_ids = np.asarray(
            [int(item["request_id"].item()) for item in requests], dtype=np.int64
        )
        relevant = np.isin(execution["request_id"], request_ids)
        selected = {name: value[relevant] for name, value in execution.items()}
        _put(actions_group, "executed", selected["action"].astype(np.float32))
        _put(
            actions_group,
            "executed_raw_policy_action",
            selected["raw_action"].astype(np.float32),
        )
        _put(actions_group, "executed_request_id", selected["request_id"].astype(np.int64))
        _put(actions_group, "vla_chunk_id", selected["vla_chunk_id"].astype(np.int64))
        _put(
            actions_group,
            "action_index_in_chunk",
            selected["action_index_in_chunk"].astype(np.int64),
        )
        _put(actions_group, "execution_index", selected["execution_index"].astype(np.uint64))
        _put(
            actions_group,
            "execution_start_physics_step",
            selected["start_physics_step"].astype(np.uint64),
        )
        _put(
            actions_group,
            "execution_end_physics_step",
            selected["end_physics_step"].astype(np.uint64),
        )
        _put(
            actions_group,
            "execution_start_time_s",
            selected["start_time_s"].astype(np.float64),
        )
        _put(
            actions_group,
            "execution_end_time_s",
            selected["end_time_s"].astype(np.float64),
        )
        _put(actions_group, "control_source", [str(value) for value in selected["control_source"]])
        _put(actions_group, "execution_status", ["executed"] * int(relevant.sum()))
        within_attack_window = (
            (selected["end_time_s"] >= start_s)
            & (selected["start_time_s"] <= end_s)
        )
        _put(actions_group, "within_attack_window", within_attack_window)

        request_execution = actions_group.create_group("request_execution")
        executed_counts = np.asarray(
            [np.count_nonzero(selected["request_id"] == request_id) for request_id in request_ids],
            dtype=np.uint64,
        )
        _put(request_execution, "request_id", request_ids)
        _put(request_execution, "executed_action_count", executed_counts)
        _put(
            request_execution,
            "status",
            ["executed" if count else "not_executed" for count in executed_counts],
        )
        availability["executed_actions"] = bool(relevant.any())
        availability["executed_request_mapping"] = True
        actions_group.attrs["execution_selection"] = (
            "all high-level actions causally generated by the recorded VLA requests; "
            "may extend beyond the attack time window"
        )
    else:
        if np.any(mask):
            _put(
                actions_group,
                "executed",
                np.asarray(episode["action"], dtype=np.float32)[mask],
            )
            _put(actions_group, "timestamp_s", times[mask])
            availability["executed_actions"] = True
        else:
            availability["executed_actions"] = False
        availability["executed_request_mapping"] = False
        actions_group.attrs["executed_request_mapping_available"] = False
        actions_group.attrs["executed_request_mapping_reason"] = (
            "no persisted causal policy execution stream or no VLA request boundary"
        )
    if action_semantics:
        _put(actions_group, "action_semantics", action_semantics)


def _write_lidar(
    attack_group: h5py.Group,
    root: Path,
    samples: list[dict[str, Any]],
    availability: dict[str, bool],
) -> None:
    group = attack_group.create_group("lidar")
    if not samples:
        group.attrs["available"] = False
        group.attrs["reason"] = "no changed LiDAR payload was captured during this attack"
        availability["lidar_points"] = False
        availability["lidar_object_ids"] = False
        return
    original_points = [sample["original_points"] for sample in samples]
    attacked_points = [sample["attacked_points"] for sample in samples]
    original_offsets = np.concatenate(([0], np.cumsum([len(value) for value in original_points]))).astype(np.uint64)
    attacked_offsets = np.concatenate(([0], np.cumsum([len(value) for value in attacked_points]))).astype(np.uint64)
    original = group.create_group("original")
    attacked = group.create_group("attacked")
    _put(original, "points", np.concatenate(original_points).astype(np.float32))
    _put(original, "frame_offsets", original_offsets)
    _put(attacked, "points", np.concatenate(attacked_points).astype(np.float32))
    _put(attacked, "frame_offsets", attacked_offsets)
    _put(group, "timestamp_s", [float(sample["time_s"].item()) for sample in samples])
    _put(group, "physics_step", [int(sample["physics_step"].item()) for sample in samples])
    availability["lidar_points"] = True
    have_ids = all(
        "original_object_ids" in sample and "attacked_object_ids" in sample
        for sample in samples
    )
    if have_ids:
        _put(original, "object_ids", np.concatenate([sample["original_object_ids"] for sample in samples]))
        _put(attacked, "object_ids", np.concatenate([sample["attacked_object_ids"] for sample in samples]))
        map_path = root / "attack_evidence" / "raw_payloads" / "lidar_stable_id_map.json"
        if map_path.is_file():
            stable_id_map = json.loads(map_path.read_text(encoding="utf-8"))
            mapping = group.create_group("stable_id_map")
            identifiers = sorted(stable_id_map)
            _put(mapping, "object_id", identifiers)
            _put(mapping, "prim_path", [str(stable_id_map[value]) for value in identifiers])
            availability["lidar_stable_id_map"] = True
        else:
            availability["lidar_stable_id_map"] = False
    else:
        group.attrs["object_ids_available"] = False
        group.attrs["object_ids_reason"] = "RTX LiDAR frame did not provide object_ids"
    availability["lidar_object_ids"] = have_ids


def _write_imu(
    attack_group: h5py.Group,
    records: list[dict[str, Any]],
    availability: dict[str, bool],
) -> None:
    group = attack_group.create_group("imu")
    original = group.create_group("original")
    attacked = group.create_group("attacked")
    fields = ("angular_velocity", "linear_acceleration", "projected_gravity")
    for field in fields:
        values_original = [record.get("original", {}).get(field) for record in records]
        values_attacked = [record.get("attacked", {}).get(field) for record in records]
        available = bool(records) and all(value is not None for value in values_original + values_attacked)
        availability[f"imu_{field}"] = available
        if available:
            _put(original, field, np.asarray(values_original, dtype=np.float32))
            _put(attacked, field, np.asarray(values_attacked, dtype=np.float32))
    if records:
        _put(group, "timestamp_s", [record["runtime_context"]["time_s"] for record in records])
        _put(group, "physics_step", [record["runtime_context"]["physics_step"] for record in records])
        _put(group, "imu_changed", [bool(record["changed"]) for record in records])
    if not availability["imu_linear_acceleration"]:
        group.attrs["linear_acceleration_available"] = False
        group.attrs["linear_acceleration_reason"] = (
            "runtime exposes angular_velocity and projected_gravity only; no value was fabricated"
        )


def _write_force_torque(
    attack_group: h5py.Group,
    records: list[dict[str, Any]],
    availability: dict[str, bool],
) -> None:
    """Persist exact pre/post Piper TCP wrench samples for a sensor attack."""

    group = attack_group.create_group("force_torque")
    original = group.create_group("original")
    attacked = group.create_group("attacked")
    numeric_fields = (
        ("force_xyz_n", (3,)),
        ("torque_xyz_nm", (3,)),
        ("grip_force_n", ()),
    )
    for field, trailing_shape in numeric_fields:
        values_original = [record.get("original", {}).get(field) for record in records]
        values_attacked = [record.get("attacked", {}).get(field) for record in records]
        available = bool(records) and all(
            value is not None for value in values_original + values_attacked
        )
        if available:
            clean = np.asarray(values_original, dtype=np.float32)
            changed = np.asarray(values_attacked, dtype=np.float32)
            expected_shape = (len(records), *trailing_shape)
            available = bool(
                clean.shape == expected_shape
                and changed.shape == expected_shape
                and np.isfinite(clean).all()
                and np.isfinite(changed).all()
            )
        availability[f"force_torque_{field}"] = available
        if available:
            _put(original, field, clean)
            _put(attacked, field, changed)

    metadata_fields = (
        "sensor_id",
        "frame_id",
        "source",
        "torque_estimation",
    )
    for field in metadata_fields:
        values = [record.get("original", {}).get(field) for record in records]
        available = bool(records) and all(value is not None for value in values)
        availability[f"force_torque_{field}"] = available
        if available:
            _put(group, field, [str(value) for value in values])
    if records:
        _put(
            group,
            "timestamp_s",
            [record["runtime_context"]["time_s"] for record in records],
        )
        _put(
            group,
            "physics_step",
            [record["runtime_context"]["physics_step"] for record in records],
        )
        _put(group, "changed", [bool(record["changed"]) for record in records])
    else:
        group.attrs["available"] = False
        group.attrs["reason"] = (
            "no force/torque payload was captured during this attack"
        )
    availability["force_torque"] = all(
        availability.get(f"force_torque_{field}", False)
        for field, _shape in numeric_fields
    )


def _mapping_matrix(records: list[dict[str, Any]], side: str, field: str) -> tuple[list[str], np.ndarray] | None:
    mappings = [record.get(side, {}).get(field) for record in records]
    if not mappings or any(not isinstance(value, Mapping) for value in mappings):
        return None
    names = sorted(set().union(*(value.keys() for value in mappings)))
    if any(set(value.keys()) != set(names) for value in mappings):
        return None
    return names, np.asarray([[value[name] for name in names] for value in mappings], dtype=np.float32)


def _write_controls(
    attack_group: h5py.Group,
    root: Path,
    start_s: float,
    end_s: float,
    availability: dict[str, bool],
) -> None:
    group = attack_group.create_group("control_targets")
    original = group.create_group("original")
    attacked = group.create_group("attacked")
    base = _filter_raw_records(root, "control_base", start_s, end_s)
    if base:
        _put(original, "base_command", np.asarray([record["original"] for record in base], dtype=np.float32))
        _put(attacked, "base_command", np.asarray([record["attacked"] for record in base], dtype=np.float32))
        _put(group, "base_timestamp_s", [record["runtime_context"]["time_s"] for record in base])
        _put(group, "base_physics_step", [record["runtime_context"]["physics_step"] for record in base])
    availability["control_base"] = bool(base)

    joint = _filter_raw_records(root, "control_joint", start_s, end_s)
    for field, output_name in (("positions", "joint_positions"), ("wheel_velocities", "wheel_velocities")):
        clean = _mapping_matrix(joint, "original", field)
        changed = _mapping_matrix(joint, "attacked", field)
        available = clean is not None and changed is not None and clean[0] == changed[0]
        availability[f"control_{output_name}"] = available
        if available:
            _put(group, f"{output_name}_names", clean[0])
            _put(original, output_name, clean[1])
            _put(attacked, output_name, changed[1])
    if joint:
        _put(group, "joint_timestamp_s", [record["runtime_context"]["time_s"] for record in joint])
        _put(group, "joint_physics_step", [record["runtime_context"]["physics_step"] for record in joint])
    gripper = _filter_raw_records(root, "control_gripper", start_s, end_s)
    availability["control_gripper"] = bool(gripper)
    availability["control_immediate"] = False
    if not gripper:
        group.attrs["gripper_command_available"] = False
    group.attrs["immediate_command_available"] = False


def _q17_force_feedback_causality_records(
    events: list[dict[str, Any]],
    episode: Mapping[str, Any],
    attack_id: str,
) -> list[dict[str, Any]]:
    """Join the Q17 force-feedback input to the exact VLA request and actions.

    A ``vla_transaction`` row is emitted only after the external Policy call
    returns, so it is concrete consumption proof.  Execution proof comes only
    from the Runner's existing request-id keyed policy execution stream.
    """

    if attack_id != Q17_FORCE_FEEDBACK_ATTACK_ID:
        return []
    execution = _policy_execution_stream(episode)
    records: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "vla_transaction":
            continue
        context = event.get("runtime_context", {})
        if not isinstance(context, Mapping):
            continue
        bridge = context.get(Q17_FORCE_FEEDBACK_CONTEXT_KEY)
        if not isinstance(bridge, Mapping):
            continue
        attack_ids = [str(value) for value in bridge.get("attack_ids", [])]
        if attack_id not in attack_ids:
            continue
        request_id = event.get("request_id")
        if not isinstance(request_id, (int, float)):
            raise ValueError("Q17 force-feedback bridge is missing request_id")
        request_id = int(request_id)
        input_changed = bool(bridge.get("input_changed", False))
        base_detail = {
            "bridge_schema": str(bridge.get("schema", "")),
            "state_index": int(bridge.get("state_index", 24)),
            "clean_state_value": float(bridge.get("clean_state_value", np.nan)),
            "forwarded_state_value": float(
                bridge.get("forwarded_state_value", np.nan)
            ),
            "sensor_payload_changed": bool(
                bridge.get("sensor_payload_changed", False)
            ),
            "clean_grip_force_n": float(
                bridge.get("clean_grip_force_n", np.nan)
            ),
            "attacked_grip_force_n": float(
                bridge.get("attacked_grip_force_n", np.nan)
            ),
            "policy_call_returned": True,
            "judge_uses_attacked_feedback": bool(
                bridge.get("judge_uses_attacked_feedback", False)
            ),
        }
        records.append(
            {
                "status": "consumed" if input_changed else "masked",
                "consumer": (
                    "external_vla_policy"
                    if input_changed
                    else "vla_input_equivalence_gate"
                ),
                "stage": "sensor_force_torque",
                "physics_step": int(context.get("physics_step", -1)),
                "time_s": float(context.get("time_s", np.nan)),
                "request_id": request_id,
                "vla_chunk_id": -1,
                "action_index_in_chunk": -1,
                "input_changed": input_changed,
                "downstream_changed": None,
                "executed_changed": None,
                "detail": base_detail,
            }
        )
        if not input_changed:
            continue
        executed_indices = (
            np.flatnonzero(execution["request_id"] == request_id)
            if execution is not None
            else np.asarray([], dtype=np.int64)
        )
        if not executed_indices.size:
            records.append(
                {
                    "status": "discarded",
                    "consumer": "request_execution_join",
                    "stage": "sensor_force_torque",
                    "physics_step": int(context.get("physics_step", -1)),
                    "time_s": float(context.get("time_s", np.nan)),
                    "request_id": request_id,
                    "vla_chunk_id": -1,
                    "action_index_in_chunk": -1,
                    "input_changed": True,
                    "downstream_changed": None,
                    "executed_changed": False,
                    "detail": {
                        **base_detail,
                        "reason": "no_executed_action_mapped_to_request",
                    },
                }
            )
            continue
        for index in executed_indices:
            row = int(index)
            records.append(
                {
                    "status": "executed",
                    "consumer": "m20_policy_action_pipeline",
                    "stage": "sensor_force_torque",
                    "physics_step": int(execution["start_physics_step"][row]),
                    "time_s": float(execution["start_time_s"][row]),
                    "request_id": request_id,
                    "vla_chunk_id": int(execution["vla_chunk_id"][row]),
                    "action_index_in_chunk": int(
                        execution["action_index_in_chunk"][row]
                    ),
                    "input_changed": True,
                    "downstream_changed": None,
                    # No clean counterfactual Policy call exists, so do not
                    # claim that the attack changed the executed command.
                    "executed_changed": None,
                    "detail": {
                        **base_detail,
                        "execution_index": int(execution["execution_index"][row]),
                        "control_source": str(execution["control_source"][row]),
                    },
                }
            )
    return records


def _q17_nullable_bool(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [-1 if value is None else int(bool(value)) for value in values],
        dtype=np.int8,
    )


def _write_q17_force_feedback_causality(
    attack_group: h5py.Group,
    records: list[dict[str, Any]],
    *,
    applied: bool,
) -> dict[str, Any]:
    group = attack_group.create_group("causality")
    _put(group, "event_count", len(records), compress=False)
    for name in ("status", "consumer", "stage"):
        _put(group, name, [str(value.get(name, "")) for value in records])
    for name in (
        "physics_step",
        "request_id",
        "vla_chunk_id",
        "action_index_in_chunk",
    ):
        _put(group, name, np.asarray(
            [int(value.get(name, -1)) for value in records], dtype=np.int64
        ))
    _put(
        group,
        "time_s",
        np.asarray([float(value.get("time_s", np.nan)) for value in records],
                   dtype=np.float64),
    )
    for name in ("input_changed", "downstream_changed", "executed_changed"):
        _put(
            group,
            name,
            _q17_nullable_bool(value.get(name) for value in records),
        )
    _put(
        group,
        "detail_json",
        [
            json.dumps(value.get("detail", {}), ensure_ascii=False, sort_keys=True)
            for value in records
        ],
    )

    consumed = any(value.get("status") == "consumed" for value in records)
    executed = any(value.get("status") == "executed" for value in records)
    blocked = any(value.get("status") in {"masked", "discarded"} for value in records)
    if executed:
        status = "executed_without_proven_command_change"
    elif consumed:
        status = "consumed_without_proven_downstream_change"
    elif applied and blocked:
        status = "mutated_but_blocked_before_execution"
    elif applied:
        status = "mutated_not_consumed"
    else:
        status = "no_mutated_payload"
    consumed_request_ids = {
        int(value["request_id"])
        for value in records
        if value.get("status") == "consumed"
    }
    executed_request_ids = {
        int(value["request_id"])
        for value in records
        if value.get("status") == "executed"
    }
    for name, value in {
        "causality_status": status,
        "consumer_recorded": consumed,
        "execution_recorded": executed,
        "consumed_request_count": len(consumed_request_ids),
        "executed_request_count": len(executed_request_ids),
        "mapped_executed_action_count": sum(
            value.get("status") == "executed" for value in records
        ),
        "executed_command_change_proven": False,
    }.items():
        _put(group, name, value, compress=False)
    return {
        "status": status,
        "consumed": consumed,
        "executed": executed,
        "consumed_request_count": len(consumed_request_ids),
        "executed_request_count": len(executed_request_ids),
    }


def _write_attack_instance(
    parent: h5py.Group,
    root: Path,
    episode: Mapping[str, Any],
    spec: Mapping[str, Any],
    all_specs: list[Mapping[str, Any]],
    profile_path: Path,
    events: list[dict[str, Any]],
    action_semantics: list[str],
) -> dict[str, Any]:
    attack_id = str(spec["id"])
    stage = str(spec.get("stage"))
    if stage not in STAGE_DIMENSION:
        raise ValueError(f"Unregistered attack evidence stage: {stage}")
    dimension = STAGE_DIMENSION[stage]
    group = parent[dimension].create_group(_safe_name(attack_id))
    mode = str(spec.get("mode", ""))
    condition_event = next((event for event in events if event.get("event") == "condition_triggered" and event.get("attack_id") == attack_id), None)
    intercept_activated = next((event for event in events if event.get("event") == "intercept_activated" and event.get("attack_id") == attack_id), None)
    intercept_deactivated = next((event for event in events if event.get("event") == "intercept_deactivated" and event.get("attack_id") == attack_id), None)
    immediate_execution = next((event for event in events if event.get("event") == "immediate_executed" and event.get("attack_id") == attack_id), None)
    lifecycle_events = [
        event
        for event in events
        if event.get("attack_id") == attack_id
        and str(event.get("event", "")).endswith(
            ("_installed", "_restored", "_removed", "_expired", "_completed")
        )
    ]
    effect_started = next(
        (event for event in lifecycle_events if str(event.get("event", "")).endswith("_installed")),
        None,
    )
    effect_finished = next(
        (
            event
            for event in lifecycle_events
            if str(event.get("event", "")).endswith(
                ("_restored", "_removed", "_expired", "_completed")
            )
        ),
        None,
    )
    activated = intercept_activated or immediate_execution
    deactivated = intercept_deactivated or effect_finished
    applications = [event for event in events if event.get("event") == "payload_intercepted" and event.get("attack_id") == attack_id]
    changed_applications = [
        event for event in applications if _application_payload_changed(event)
    ]
    condition_triggered = condition_event is not None
    effect_activated = activated is not None
    triggered = condition_triggered
    immediate_applied = mode == "immediate" and immediate_execution is not None
    applied = bool(applications) or immediate_applied
    activation_value = (
        condition_event.get("time_s")
        if condition_event is not None
        else _immediate_event_time(immediate_execution)
        if immediate_execution is not None
        else activated.get("activated_at") if activated else None
    )
    activation_s = float(activation_value) if activation_value is not None else np.nan
    application_times = [value for value in (_event_time(event) for event in applications) if value is not None]
    timing_source = (
        "condition_marker"
        if condition_event is not None and not effect_activated
        else "immediate_attack_audit"
        if mode == "immediate"
        else "audit_runtime_context"
    )
    boundary_requests = _load_changed_vla_request_boundaries(root, str(spec.get("stage", "")))
    if boundary_requests:
        application_times = [float(item["time_s"]) for item in boundary_requests]
        timing_source = "persisted_vla_payload_diff"
    force_torque_boundaries: list[dict[str, Any]] = []
    if str(spec.get("stage", "")) == "sensor_force_torque":
        force_torque_boundaries = [
            record
            for record in _read_jsonl(
                root
                / "attack_evidence"
                / "raw_payloads"
                / "sensor_force_torque.jsonl"
            )
            if bool(record.get("changed"))
            and _event_time(record) is not None
        ]
        applied = triggered and bool(force_torque_boundaries)
        if force_torque_boundaries:
            application_times = [
                float(_event_time(record)) for record in force_torque_boundaries
            ]
            timing_source = "persisted_force_torque_payload_diff"
    if mode != "immediate":
        _validate_application_audit(
            attack_id, applications, triggered=effect_activated, applied=applied,
            reported_count=(intercept_deactivated.get("application_count") if intercept_deactivated else None),
        )
    start_s = min(application_times) if application_times else activation_s
    application_end_s = max(application_times) if application_times else activation_s
    if mode == "immediate" and effect_finished is not None and np.isfinite(activation_s):
        elapsed_s = effect_finished.get("elapsed_s")
        reported_deactivation_s = (
            activation_s + float(elapsed_s)
            if isinstance(elapsed_s, (int, float)) and np.isfinite(elapsed_s)
            else activation_s
        )
    else:
        reported_deactivation_s = (
            activation_s + float(intercept_deactivated.get("elapsed_attack_s"))
            if intercept_deactivated and np.isfinite(activation_s) and intercept_deactivated.get("elapsed_attack_s") is not None
            else application_end_s
        )
    deactivation_s = max(application_end_s, reported_deactivation_s)
    end_s = deactivation_s if mode == "immediate" and effect_finished is not None else application_end_s
    effective_duration_s = max(0.0, deactivation_s - start_s)
    completed = bool(intercept_deactivated or effect_finished)
    if mode == "immediate" and immediate_execution is not None and not spec.get("duration_s"):
        completed = True
    metadata = group.create_group("metadata")
    for name, value in {
        "attack_id": attack_id,
        "attack_set_id": str(root.name),
        "dimension_code": dimension,
        "dimension_name_zh": DIMENSIONS[dimension],
        "handler": str(spec.get("handler", "")),
        "stage": str(spec.get("stage", "")),
        "mode": mode,
        "profile_path": str(profile_path),
        "trigger_description": json.dumps(spec.get("trigger", {}), ensure_ascii=False),
        "application_count": 1 if immediate_applied else (int(intercept_deactivated.get("application_count", len(applications))) if intercept_deactivated else len(applications)),
        "execution_count": 1 if immediate_execution is not None else 0,
        "status": "applied" if applied else ("activated_no_payload" if effect_activated else "condition_triggered_marker_only" if condition_triggered else "not_triggered"),
        "triggered": triggered,
        "condition_triggered": condition_triggered,
        "effect_activated": effect_activated,
        "attack_instance_id": str(condition_event.get("attack_instance_id", "")) if condition_event else "",
        "payload_recorded": bool(applications) or bool(force_torque_boundaries),
        "effect_evidence_recorded": bool(lifecycle_events),
        "changed_application_count": (
            1
            if immediate_applied and effect_started is not None
            else len(force_torque_boundaries)
            if str(spec.get("stage", "")) == "sensor_force_torque"
            else len(changed_applications)
        ),
        "max_applications": int(spec.get("max_applications", 0)),
        "stop_reason": (
            str(intercept_deactivated.get("stop_reason", "not_deactivated"))
            if intercept_deactivated
            else "effect_restored"
            if effect_finished is not None
            else "immediate_completed"
            if completed
            else "not_deactivated"
        ),
        "related_attack_ids": [
            str(item["id"]) for item in all_specs if str(item["id"]) != attack_id
        ],
    }.items():
        _put(metadata, name, value, compress=False)

    q17_causality = None
    if attack_id == Q17_FORCE_FEEDBACK_ATTACK_ID:
        q17_records = _q17_force_feedback_causality_records(
            events, episode, attack_id
        )
        q17_causality = _write_q17_force_feedback_causality(
            group, q17_records, applied=applied
        )
        for name, value in {
            "causality_status": q17_causality["status"],
            "consumer_recorded": q17_causality["consumed"],
            "execution_recorded": q17_causality["executed"],
        }.items():
            _put(metadata, name, value, compress=False)

    timing = group.create_group("timing")
    if mode == "immediate":
        first_event = effect_started or immediate_execution or condition_event
        last_event = effect_finished or effect_started or immediate_execution or condition_event
        start_physics_step = _immediate_event_step(first_event) if first_event else None
        end_physics_step = _immediate_event_step(last_event) if last_event else None
    else:
        first_event = applications[0] if applications else activated or condition_event
        last_event = applications[-1] if applications else deactivated or activated or condition_event
        start_physics_step = (
            int(boundary_requests[0]["physics_step"])
            if boundary_requests
            else _event_step(force_torque_boundaries[0])
            if force_torque_boundaries
            else _event_step(applications[0]) if applications
            else int(condition_event["physics_step"]) if condition_event else None
        )
        end_physics_step = (
            int(boundary_requests[-1]["physics_step"])
            if boundary_requests
            else _event_step(force_torque_boundaries[-1])
            if force_torque_boundaries
            else _event_step(applications[-1]) if applications
            else int(condition_event["physics_step"]) if condition_event else None
        )
    timing_values = {
        "trigger_time_s": activation_s,
        "trigger_physics_step": _trigger_physics_step(condition_event),
        "activation_time_s": activation_s,
        "first_application_time_s": start_s,
        "last_application_time_s": end_s,
        "deactivation_time_s": deactivation_s,
        "start_time_s": start_s,
        "end_time_s": end_s,
        "configured_duration_s": float(spec.get("duration_s", 0.0)),
        "effective_duration_s": effective_duration_s,
        "start_time_utc": str(first_event.get("timestamp_utc", "")) if first_event else "",
        "end_time_utc": str(last_event.get("timestamp_utc", "")) if last_event else "",
        "start_physics_step": start_physics_step,
        "end_physics_step": end_physics_step,
        "timing_source": timing_source,
    }
    for name, value in timing_values.items():
        _put(timing, name, -1 if value is None else value, compress=False)

    # Keep an explicit unavailable group even when the attack never fired.
    # Never fabricate application rows from a counter or a different episode.
    availability: dict[str, bool] = {}
    _write_application_payloads(group, applications, availability)
    if mode == "immediate":
        _write_immediate_effect_evidence(
            group, immediate_execution, lifecycle_events, availability
        )
    if q17_causality is not None:
        availability["q17_policy_consumption"] = q17_causality["consumed"]
        availability["q17_execution_mapping"] = q17_causality["executed"]
    if not triggered:
        _write_availability(metadata, availability)
        return {
            "attack_id": attack_id,
            "dimension": dimension,
            "triggered": False,
            "effect_activated": effect_activated,
            "applied": False,
            "completed": False,
            **(
                {"causality_status": q17_causality["status"]}
                if q17_causality is not None
                else {}
            ),
            "start_time_s": start_s,
            "end_time_s": end_s,
            "availability": availability,
        }
    requests = _load_vla_requests(root, start_s, end_s) if np.isfinite(start_s) and np.isfinite(end_s) else []
    _write_images(group, episode, requests, start_s, end_s, availability)
    _write_vla_data(group, episode, requests, start_s, end_s, action_semantics, availability)
    lidar_samples = _load_changed_lidar(root, start_s, end_s)
    _write_lidar(group, root, lidar_samples, availability)
    imu_records = _filter_raw_records(root, "sensor_imu", start_s, end_s)
    _write_imu(group, imu_records, availability)
    if str(spec.get("stage", "")) == "sensor_force_torque":
        force_torque_records = _filter_raw_records(
            root, "sensor_force_torque", start_s, end_s
        )
        _write_force_torque(group, force_torque_records, availability)
    _write_controls(group, root, start_s, end_s, availability)
    _write_availability(metadata, availability)
    return {
        "attack_id": attack_id,
        "dimension": dimension,
        "triggered": triggered,
        "effect_activated": effect_activated,
        "applied": applied,
        "completed": completed,
        **(
            {"causality_status": q17_causality["status"]}
            if q17_causality is not None
            else {}
        ),
        "start_time_s": start_s,
        "end_time_s": end_s,
        "availability": availability,
    }


def write_enriched_submission(
    output_dir: Path,
    attack_profile_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Copy the formal submission, append attack evidence, and verify originals."""
    output_dir = output_dir.resolve()
    source = output_dir / "submission.hdf5"
    destination = output_dir / "submission_enriched.hdf5"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    profile = json.loads(attack_profile_path.read_text(encoding="utf-8"))
    specs = [spec for spec in profile.get("attack", {}).get("attacks", []) if spec.get("enabled", True)]
    events = _read_jsonl(output_dir / "attack_evidence" / "unified_attack_audit.jsonl")
    marker_rows = _attack_marker_rows(events, specs)
    server_metadata = next((event.get("metadata", {}) for event in events if event.get("event") == "server_metadata"), {})
    action_semantics = [str(value) for value in server_metadata.get("action_semantics", [])]
    original_sha256 = _sha256_file(source)
    original_manifest = _original_manifest(source)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        with np.load(output_dir / "episode.npz", allow_pickle=False) as episode:
            with h5py.File(temporary, "r+") as h5:
                h5["/metadata"].attrs.modify("complete", np.bool_(False))
                demo = h5["/data/demo_0"]
                if "attacks" in demo:
                    raise ValueError("formal submission unexpectedly already contains /data/demo_0/attacks")
                attacks = demo.create_group("attacks")
                attacks.attrs["field_contract"] = "attack-safety-timing/v1"
                for code, name_zh in DIMENSIONS.items():
                    dimension_group = attacks.create_group(code)
                    dimension_group.attrs["dimension_code"] = code
                    dimension_group.attrs["dimension_name_zh"] = name_zh
                _write_json_rows(attacks.create_group("attack_markers"), marker_rows)
                vla_full_trace = _write_full_vla_trace(demo, output_dir, episode)
                summaries = [
                    _write_attack_instance(
                        attacks,
                        output_dir,
                        episode,
                        spec,
                        specs,
                        attack_profile_path,
                        events,
                        action_semantics,
                    )
                    for spec in specs
                ]
                summary = attacks.create_group("summary")
                triggered_summaries = [item for item in summaries if item["triggered"]]
                applied_summaries = [item for item in summaries if item["applied"]]
                dimensions = sorted({item["dimension"] for item in triggered_summaries})
                starts = [item["start_time_s"] for item in summaries if np.isfinite(item["start_time_s"])]
                ends = [item["end_time_s"] for item in summaries if np.isfinite(item["end_time_s"])]
                summary_values = {
                    "attack_schema_name": "robolab-m20-attack-evidence",
                    "attack_schema_version": "4",
                    "attack_set_id": str(profile.get("question_id", output_dir.name)),
                    "attack_count": len(triggered_summaries),
                    "configured_attack_count": len(summaries),
                    "triggered_attack_count": len(triggered_summaries),
                    "applied_attack_count": len(applied_summaries),
                    "actual_attacks_applied": sum(
                        1 for row in marker_rows if row["effect_applied"]
                    ),
                    "condition_marker_count": sum(
                        1
                        for row in marker_rows
                        if row["evidence"].get("event") == "condition_triggered"
                    ),
                    "effect_marker_count": sum(
                        1 for row in marker_rows if row["effect_applied"]
                    ),
                    "dimension_count": len(dimensions),
                    "dimension_codes": dimensions,
                    "dimension_names_zh": [DIMENSIONS[code] for code in dimensions],
                    "attack_ids": [item["attack_id"] for item in summaries],
                    "triggered_attack_ids": [item["attack_id"] for item in triggered_summaries],
                    "not_triggered_attack_ids": [item["attack_id"] for item in summaries if not item["triggered"]],
                    "multi_dimension": len(dimensions) > 1,
                    "overall_start_time_s": min(starts) if starts else np.nan,
                    "overall_end_time_s": max(ends) if ends else np.nan,
                    "overall_duration_s": max(ends) - min(starts) if starts and ends else np.nan,
                    "all_attacks_completed": all(item["completed"] for item in summaries),
                    "original_submission_sha256": original_sha256,
                }
                for name, value in summary_values.items():
                    _put(summary, name, value, compress=False)
                h5["/metadata"].attrs.modify("complete", np.bool_(True))
                h5.flush()
        enriched_manifest = _original_manifest(temporary)
        changed = {
            path: {"before": metadata, "after": enriched_manifest.get(path)}
            for path, metadata in original_manifest.items()
            if enriched_manifest.get(path) != metadata
        }
        if changed:
            raise RuntimeError(f"original submission datasets changed: {sorted(changed)}")
        os.replace(temporary, destination)
        report = {
            "schema": "robolab-m20-enriched-hdf5-validation/v1",
            "destination": str(destination),
            "source": str(source),
            "source_sha256": original_sha256,
            "original_dataset_count": len(original_manifest),
            "original_datasets_unchanged": True,
            "vla_full_trace": vla_full_trace,
            "attacks": summaries,
            "attack_markers": marker_rows,
        }
        report_path = output_dir / "submission_enriched_validation.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return destination, report
    finally:
        temporary.unlink(missing_ok=True)
