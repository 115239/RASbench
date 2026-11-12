"""Formal RoboLab V4 action-scoring HDF5 writer for M20 episodes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from robolab_action_schema import (
    DATA_ROOT,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TRACKED_ANGULAR_VELOCITY,
    TRACKED_LINEAR_VELOCITY,
    TRACKED_ORIENTATION,
    TRACKED_POSITION,
    articulation_feedback_path,
    articulation_state_path,
    contact_field_path,
    rigid_root_pose_path,
    rigid_root_velocity_path,
    task_reference_pose_path,
    validate_submission_structure,
)


def _put(h5: h5py.File, path: str, value: Any, dtype: Any) -> None:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        raise ValueError(f"empty V4 dataset: {path}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"non-finite V4 dataset: {path}")
    h5.create_dataset(path, data=array, compression="gzip" if array.ndim > 0 else None)


def _require_shape(name: str, value: Any, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(expected is not None and array.shape[i] != expected for i, expected in enumerate(shape)):
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_pose(name: str, value: Any, *, time_series: bool) -> np.ndarray:
    array = _require_shape(name, value, (None, 7) if time_series else (7,))
    quaternions = array[:, 3:] if time_series else array[None, 3:]
    if np.any(np.linalg.norm(quaternions, axis=1) <= 1e-12):
        raise ValueError(f"{name} contains a zero quaternion")
    return array


def validate_record(record: Mapping[str, Any]) -> int:
    actions = _require_shape("actions", record["actions"], (None, None))
    if actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError("actions must be non-empty [T,A]")
    count = actions.shape[0]
    for name, trailing in {
        "tracked_position": (3,), "tracked_orientation": (4,),
        "tracked_linear_velocity": (3,), "tracked_angular_velocity": (3,),
        "robot_root_pose": (7,), "robot_root_velocity": (6,),
        "robot_joint_position": (None,), "robot_joint_velocity": (None,),
        "object_root_pose": (7,), "object_root_velocity": (6,),
    }.items():
        array = np.asarray(record[name])
        if array.shape[0] != count or array.ndim != len(trailing) + 1:
            raise ValueError(f"{name} does not share T={count}: {array.shape}")
        for index, expected in enumerate(trailing, start=1):
            if expected is not None and array.shape[index] != expected:
                raise ValueError(f"{name} has invalid trailing shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
    _validate_pose("robot_root_pose", record["robot_root_pose"], time_series=True)
    _validate_pose("object_root_pose", record["object_root_pose"], time_series=True)
    _validate_pose("initial_robot_root_pose", record["initial_robot_root_pose"], time_series=False)
    _validate_pose("initial_object_root_pose", record["initial_object_root_pose"], time_series=False)
    _validate_pose("task_reference_pose", record["task_reference_pose"], time_series=False)
    body_pose = _require_shape("robot_body_pose", record["robot_body_pose"], (count, None, 7))
    _require_shape("robot_body_velocity", record["robot_body_velocity"], (count, body_pose.shape[1], 6))
    if np.any(np.linalg.norm(body_pose[..., 3:], axis=2) <= 1e-12):
        raise ValueError("robot_body_pose contains a zero quaternion")
    joint_count = np.asarray(record["robot_joint_position"]).shape[1]
    if len(record["joint_paths"]) != joint_count:
        raise ValueError("joint path ordering length differs from J")
    if len(record["body_paths"]) != body_pose.shape[1]:
        raise ValueError("body path ordering length differs from B")
    contact_forces = _require_shape("contact_net_forces_w", record["contact_net_forces_w"], (None, None, 3))
    contact_count = contact_forces.shape[0]
    if contact_count < 1 or len(record["contact_body_paths"]) != contact_forces.shape[1]:
        raise ValueError("contact stream/body ordering is empty or inconsistent")
    contact_steps = _require_shape("contact_step_index", record["contact_step_index"], (contact_count,))
    contact_time = _require_shape("contact_time_s", record["contact_time_s"], (contact_count,))
    if contact_count > 1 and (not np.all(np.diff(contact_steps) > 0) or not np.all(np.diff(contact_time) > 0)):
        raise ValueError("contact step/time axes must be strictly increasing")
    action_time = np.asarray(record["action_time_s"], dtype=np.float64)
    state_time = np.asarray(record["state_time_s"], dtype=np.float64)
    if action_time.shape != (count,) or state_time.shape != (count,):
        raise ValueError("time arrays must have shape [T]")
    if count > 1 and (not np.all(np.diff(action_time) > 0) or not np.all(np.diff(state_time) > 0)):
        raise ValueError("V4 action and state times must be strictly increasing")
    if np.any(state_time < action_time):
        raise ValueError("state_time_s must not precede action_time_s")
    return count


def record_from_episode(episode: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    robot_pose = np.asarray(episode["scoring_robot_root_pose"], dtype=np.float32)
    robot_velocity = np.asarray(episode["scoring_robot_root_velocity"], dtype=np.float32)
    return {
        "task_id": str(task["task_id"]),
        "instruction": str(task["instruction"]),
        "articulation_prim": str(np.asarray(episode["scoring_articulation_prim"]).item()),
        "object_prim": str(np.asarray(episode["scoring_object_prim"]).item()),
        "target_prim": str(np.asarray(episode["scoring_target_prim"]).item()),
        "joint_paths": [str(x) for x in episode["scoring_joint_paths"].tolist()],
        "body_paths": [str(x) for x in episode["scoring_body_paths"].tolist()],
        "actions": episode["scoring_actions"],
        "step_index": np.arange(len(episode["scoring_actions"]), dtype=np.uint64),
        "action_time_s": episode["scoring_action_time_s"],
        "state_time_s": episode["scoring_state_time_s"],
        "tracked_position": robot_pose[:, :3],
        "tracked_orientation": robot_pose[:, 3:],
        "tracked_linear_velocity": robot_velocity[:, :3],
        "tracked_angular_velocity": robot_velocity[:, 3:],
        "initial_robot_root_pose": episode["scoring_initial_robot_root_pose"],
        "initial_robot_root_velocity": episode["scoring_initial_robot_root_velocity"],
        "initial_robot_joint_position": episode["scoring_initial_robot_joint_position"],
        "initial_robot_joint_velocity": episode["scoring_initial_robot_joint_velocity"],
        "robot_root_pose": robot_pose,
        "robot_root_velocity": robot_velocity,
        "robot_joint_position": episode["scoring_robot_joint_position"],
        "robot_joint_velocity": episode["scoring_robot_joint_velocity"],
        "robot_body_pose": episode["scoring_robot_body_pose"],
        "robot_body_velocity": episode["scoring_robot_body_velocity"],
        "initial_object_root_pose": episode["scoring_initial_object_root_pose"],
        "initial_object_root_velocity": episode["scoring_initial_object_root_velocity"],
        "object_root_pose": episode["scoring_object_root_pose"],
        "object_root_velocity": episode["scoring_object_root_velocity"],
        "task_reference_pose": episode["scoring_task_reference_pose"],
        "contact_source_prim": str(np.asarray(episode["scoring_contact_source_prim"]).item()),
        "contact_target_prim": str(np.asarray(episode["scoring_contact_target_prim"]).item()),
        "contact_body_paths": [str(x) for x in episode["scoring_contact_body_paths"].tolist()],
        "contact_net_forces_w": episode["scoring_contact_net_forces_w"],
        "contact_step_index": episode["scoring_contact_step_index"],
        "contact_time_s": episode["scoring_contact_time_s"],
    }


def write_submission_atomically(destination: Path, record: Mapping[str, Any]) -> None:
    count = validate_record(record)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as h5:
            metadata = h5.create_group("metadata")
            metadata.attrs.update({
                "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
                "task_id": record["task_id"], "instruction": record["instruction"], "complete": False,
            })
            _put(h5, f"{DATA_ROOT}/actions", record["actions"], np.float32)
            _put(h5, f"{DATA_ROOT}/time/step_index", record["step_index"], np.uint64)
            _put(h5, f"{DATA_ROOT}/time/action_time_s", record["action_time_s"], np.float64)
            _put(h5, f"{DATA_ROOT}/time/state_time_s", record["state_time_s"], np.float64)
            for path, key in ((TRACKED_POSITION, "tracked_position"), (TRACKED_ORIENTATION, "tracked_orientation"),
                              (TRACKED_LINEAR_VELOCITY, "tracked_linear_velocity"),
                              (TRACKED_ANGULAR_VELOCITY, "tracked_angular_velocity")):
                _put(h5, path, record[key], np.float32)
            articulation = record["articulation_prim"]
            for field in ("root_pose", "root_velocity", "joint_position", "joint_velocity"):
                _put(h5, articulation_state_path(articulation, field, initial=True), record[f"initial_robot_{field}"], np.float32)
                _put(h5, articulation_state_path(articulation, field), record[f"robot_{field}"], np.float32)
            for field in ("joint_position", "joint_velocity", "body_pose", "body_velocity"):
                _put(h5, articulation_feedback_path(articulation, field), record[f"robot_{field}"], np.float32)
            _put(h5, articulation_feedback_path(articulation, "step_index"), record["step_index"], np.uint64)
            _put(h5, articulation_feedback_path(articulation, "sim_time_s"), record["state_time_s"], np.float64)
            object_prim = record["object_prim"]
            _put(h5, rigid_root_pose_path(object_prim, initial=True), record["initial_object_root_pose"], np.float32)
            _put(h5, rigid_root_velocity_path(object_prim, initial=True), record["initial_object_root_velocity"], np.float32)
            _put(h5, rigid_root_pose_path(object_prim), record["object_root_pose"], np.float32)
            _put(h5, rigid_root_velocity_path(object_prim), record["object_root_velocity"], np.float32)
            _put(h5, task_reference_pose_path(record["target_prim"]), record["task_reference_pose"], np.float32)
            contact_source = record["contact_source_prim"]
            contact_target = record["contact_target_prim"]
            _put(h5, contact_field_path(contact_source, contact_target, "net_forces_w"), record["contact_net_forces_w"], np.float32)
            _put(h5, contact_field_path(contact_source, contact_target, "step_index"), record["contact_step_index"], np.uint64)
            _put(h5, contact_field_path(contact_source, contact_target, "sim_time_s"), record["contact_time_s"], np.float64)
            validate_submission_structure(h5)
            if h5[f"{DATA_ROOT}/actions"].shape[0] != count:
                raise ValueError("written actions length changed")
            metadata.attrs.modify("complete", True)
            h5.flush()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
