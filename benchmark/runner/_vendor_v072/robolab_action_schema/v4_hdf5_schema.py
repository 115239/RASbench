"""Versioned path, identity, ordering, and shape rules for action-scoring HDF5."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unicodedata import normalize
from urllib.parse import quote, unquote

import h5py
import numpy as np

from .robot_profiles import FORMAL_ROBOT_PROFILES

SCHEMA_NAME = "robolab-action-hdf5"
SCHEMA_VERSION = 1
SCHEMA_IMPLEMENTATION_VERSION = "0.7.2"
DATA_ROOT = "/data/demo_0"
FORBIDDEN_RUNTIME_PRIM_PREFIX = "/World/envs/env_0"
FORMAL_ROBOT_ARTICULATION_PRIMS = {
    role: profile.articulation_prim
    for role, profile in FORMAL_ROBOT_PROFILES.items()
}

TRACKED_POSITION = f"{DATA_ROOT}/tracked_pose/position"
TRACKED_ORIENTATION = f"{DATA_ROOT}/tracked_pose/orientation"
TRACKED_LINEAR_VELOCITY = f"{DATA_ROOT}/tracked_pose/linear_velocity"
TRACKED_ANGULAR_VELOCITY = f"{DATA_ROOT}/tracked_pose/angular_velocity"

_RIGID_FIELDS = frozenset({"root_pose", "root_velocity"})
_ARTICULATION_STATE_FIELDS = frozenset({"root_pose", "root_velocity", "joint_position", "joint_velocity"})
_ARTICULATION_FEEDBACK_FIELDS = frozenset(
    {
        "joint_position",
        "joint_velocity",
        "joint_position_target",
        "joint_velocity_target",
        "joint_effort_target",
        "applied_torque",
        "computed_torque",
        "body_pose",
        "body_velocity",
        "step_index",
        "sim_time_s",
    }
)
_CONTACT_FIELDS = frozenset({"net_forces_w", "step_index", "sim_time_s"})


class SchemaValidationError(ValueError):
    """Raised when a path input or HDF5 submission violates the shared schema."""


def _absolute_prim_path(prim_path: str) -> str:
    if not isinstance(prim_path, str) or not prim_path.startswith("/") or prim_path == "/":
        raise SchemaValidationError("prim_path must be a non-root absolute USD Prim path")
    parts = prim_path.split("/")[1:]
    if (
        any(not part or part in {".", ".."} for part in parts)
        or normalize("NFC", prim_path) != prim_path
    ):
        raise SchemaValidationError("prim_path must be normalized")
    return prim_path


def entity_id(prim_path: str) -> str:
    """Encode one absolute USD Prim path as a reversible HDF5 group name."""

    return quote(_absolute_prim_path(prim_path), safe="-._~")


FORMAL_ROBOT_ARTICULATION_ENTITY_IDS = {
    role: entity_id(prim_path)
    for role, prim_path in FORMAL_ROBOT_ARTICULATION_PRIMS.items()
}


def rigid_root_pose_path(prim_path: str, *, initial: bool = False) -> str:
    state = "initial_state" if initial else "states"
    return f"{DATA_ROOT}/{state}/rigid_object/{entity_id(prim_path)}/root_pose"


def rigid_root_velocity_path(prim_path: str, *, initial: bool = False) -> str:
    state = "initial_state" if initial else "states"
    return f"{DATA_ROOT}/{state}/rigid_object/{entity_id(prim_path)}/root_velocity"


def articulation_state_path(prim_path: str, field: str, *, initial: bool = False) -> str:
    if field not in _ARTICULATION_STATE_FIELDS:
        raise SchemaValidationError(f"unsupported articulation state field: {field!r}")
    state = "initial_state" if initial else "states"
    return f"{DATA_ROOT}/{state}/articulation/{entity_id(prim_path)}/{field}"


def articulation_feedback_path(prim_path: str, field: str) -> str:
    if field not in _ARTICULATION_FEEDBACK_FIELDS:
        raise SchemaValidationError(f"unsupported articulation feedback field: {field!r}")
    return f"{DATA_ROOT}/feedback/articulation/{entity_id(prim_path)}/{field}"


def contact_pair_id(source_prim: str, target_prim: str) -> str:
    source_id = entity_id(source_prim)
    target_id = entity_id(target_prim)
    delimiter = "__with__"
    if delimiter in source_id or delimiter in target_id:
        raise SchemaValidationError(
            "contact entity Prim paths must not contain the reserved "
            f"delimiter {delimiter!r}"
        )
    return f"{source_id}{delimiter}{target_id}"


def contact_field_path(source_prim: str, target_prim: str, field: str) -> str:
    if field not in _CONTACT_FIELDS:
        raise SchemaValidationError(f"unsupported contact field: {field!r}")
    return f"{DATA_ROOT}/sensors/contact/{contact_pair_id(source_prim, target_prim)}/{field}"


def task_reference_pose_path(prim_path: str) -> str:
    return f"{DATA_ROOT}/task_reference/{entity_id(prim_path)}/pose"


def gripper_closure_path(prim_path: str) -> str:
    return f"{DATA_ROOT}/states/gripper/{entity_id(prim_path)}/closure"


def _prim_path_text(prim: Any) -> str:
    path = prim.GetPath() if hasattr(prim, "GetPath") else getattr(prim, "path", prim)
    return str(path)


def _descendant_prims(stage: Any, root_path: str) -> Iterable[Any]:
    root = stage.GetPrimAtPath(root_path)
    if root is None or (hasattr(root, "IsValid") and not root.IsValid()):
        raise SchemaValidationError(f"USD Prim does not exist: {root_path}")
    if hasattr(stage, "Traverse"):
        prefix = root_path.rstrip("/") + "/"
        return (prim for prim in stage.Traverse() if _prim_path_text(prim) == root_path or _prim_path_text(prim).startswith(prefix))
    if hasattr(root, "GetDescendants"):
        return (root, *tuple(root.GetDescendants()))
    raise SchemaValidationError("stage does not expose USD traversal")


def _is_joint(prim: Any) -> bool:
    try:
        from pxr import UsdPhysics

        return bool(prim.IsA(UsdPhysics.Joint))
    except (ImportError, AttributeError, TypeError):
        type_name = str(prim.GetTypeName()) if hasattr(prim, "GetTypeName") else str(getattr(prim, "type_name", ""))
        return type_name.endswith("Joint")


def _is_rigid_body(prim: Any) -> bool:
    try:
        from pxr import UsdPhysics

        return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
    except (ImportError, AttributeError, TypeError):
        return bool(getattr(prim, "is_rigid_body", False))


def sorted_joint_paths(stage: Any, articulation_prim: str) -> tuple[str, ...]:
    root = _absolute_prim_path(articulation_prim)
    return tuple(sorted(_prim_path_text(prim) for prim in _descendant_prims(stage, root) if _is_joint(prim)))


def sorted_body_paths(stage: Any, articulation_prim: str) -> tuple[str, ...]:
    root = _absolute_prim_path(articulation_prim)
    return tuple(sorted(_prim_path_text(prim) for prim in _descendant_prims(stage, root) if _is_rigid_body(prim)))


def _ordered_index(paths: tuple[str, ...], selected: str, kind: str) -> int:
    selected = _absolute_prim_path(selected)
    try:
        return paths.index(selected)
    except ValueError as error:
        raise SchemaValidationError(f"{kind} Prim is not part of the standard ordering: {selected}") from error


def joint_index(stage: Any, articulation_prim: str, joint_prim: str) -> int:
    return _ordered_index(sorted_joint_paths(stage, articulation_prim), joint_prim, "joint")


def body_index(stage: Any, articulation_prim: str, body_prim: str) -> int:
    return _ordered_index(sorted_body_paths(stage, articulation_prim), body_prim, "body")


def contact_index(stage: Any, source_prim: str, body_prim: str) -> int:
    return _ordered_index(sorted_body_paths(stage, source_prim), body_prim, "contact body")


def _numeric_dataset(hdf5_file: h5py.File, path: str, trailing_shape: tuple[int, ...]) -> h5py.Dataset:
    node = hdf5_file.get(path)
    if not isinstance(node, h5py.Dataset):
        raise SchemaValidationError(f"required HDF5 dataset is missing: {path}")
    if node.dtype.kind not in "fiu":
        raise SchemaValidationError(f"HDF5 dataset must be numeric: {path}")
    if node.ndim != len(trailing_shape) + 1 or tuple(node.shape[1:]) != trailing_shape or node.shape[0] < 1:
        raise SchemaValidationError(f"HDF5 dataset has invalid shape: {path} {tuple(node.shape)}")
    return node


def validate_submission_structure(hdf5_file: h5py.File) -> None:
    """Validate the universal control-step portion of one V4 action HDF5 file."""

    demo = hdf5_file.get(DATA_ROOT)
    if not isinstance(demo, h5py.Group):
        raise SchemaValidationError(f"HDF5 is missing {DATA_ROOT}")
    position = _numeric_dataset(hdf5_file, TRACKED_POSITION, (3,))
    orientation = _numeric_dataset(hdf5_file, TRACKED_ORIENTATION, (4,))
    linear = _numeric_dataset(hdf5_file, TRACKED_LINEAR_VELOCITY, (3,))
    angular = _numeric_dataset(hdf5_file, TRACKED_ANGULAR_VELOCITY, (3,))
    lengths = {position.shape[0], orientation.shape[0], linear.shape[0], angular.shape[0]}
    if len(lengths) != 1:
        raise SchemaValidationError("tracked_pose datasets must share the same sample count")
    for dataset in (position, orientation, linear, angular):
        if dataset.dtype.kind == "f" and not np.isfinite(np.asarray(dataset[...])).all():
            raise SchemaValidationError(f"HDF5 dataset contains NaN or infinity: {dataset.name}")
    quaternions = np.asarray(orientation[...], dtype=np.float64)
    if np.any(np.linalg.norm(quaternions, axis=1) <= 1e-12):
        raise SchemaValidationError("tracked orientation contains a zero quaternion")


def _reject_runtime_namespace(hdf5_file: h5py.File) -> None:
    forbidden_encoded = entity_id(FORBIDDEN_RUNTIME_PRIM_PREFIX)
    offending: list[str] = []

    def visitor(name: str, _node: Any) -> None:
        if forbidden_encoded in name or FORBIDDEN_RUNTIME_PRIM_PREFIX in unquote(name):
            offending.append("/" + name)

    hdf5_file.visititems(visitor)
    if offending:
        raise SchemaValidationError(
            "HDF5 uses forbidden Runner runtime namespace /World/envs/env_0; "
            f"first occurrence: {offending[0]}"
        )


def detect_artifact_role(hdf5_file: h5py.File) -> str:
    """Identify exactly one registered robot from initial articulation identity."""

    root = hdf5_file.get(f"{DATA_ROOT}/initial_state/articulation")
    if not isinstance(root, h5py.Group):
        raise SchemaValidationError(
            f"HDF5 is missing {DATA_ROOT}/initial_state/articulation"
        )
    matches = [
        role
        for role, encoded_id in FORMAL_ROBOT_ARTICULATION_ENTITY_IDS.items()
        if encoded_id in root
    ]
    if len(matches) != 1:
        expected = ", ".join(
            f"{role}={encoded_id}"
            for role, encoded_id in FORMAL_ROBOT_ARTICULATION_ENTITY_IDS.items()
        )
        raise SchemaValidationError(
            "HDF5 must contain exactly one registered robot articulation identity under "
            f"{root.name}; found {matches or 'none'}, expected one of: {expected}"
        )
    return matches[0]


def _required_dataset(hdf5_file: h5py.File, path: str) -> h5py.Dataset:
    node = hdf5_file.get(path)
    if not isinstance(node, h5py.Dataset):
        raise SchemaValidationError(f"required HDF5 dataset is missing: {path}")
    return node


def _require_exact_dtype(dataset: h5py.Dataset, dtype: object) -> None:
    expected = np.dtype(dtype)
    if dataset.dtype != expected:
        raise SchemaValidationError(
            f"HDF5 dataset has invalid dtype: {dataset.name} "
            f"{dataset.dtype} (expected {expected})"
        )


def _require_finite(dataset: h5py.Dataset) -> None:
    if dataset.dtype.kind == "f" and not np.isfinite(np.asarray(dataset[...])).all():
        raise SchemaValidationError(
            f"HDF5 dataset contains NaN or infinity: {dataset.name}"
        )


def _metadata_text(attributes: h5py.AttributeManager, name: str) -> str:
    value = attributes[name]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SchemaValidationError(
                f"metadata.{name} must be valid UTF-8 text"
            ) from error
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"metadata.{name} must be non-empty text")
    return value.strip()


def validate_safety_submission_structure(
    hdf5_file: h5py.File,
    *,
    expected_task_id: str | None = None,
) -> None:
    """Validate the safety Runner's mandatory, task-independent HDF5 surface.

    Entity-specific evidence is validated separately against the generated
    HDF5 requirements manifest. Keeping that distinction lets a Runner
    validate its fixed output contract without reading a Rule Package.
    """

    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group):
        raise SchemaValidationError("HDF5 is missing /metadata")
    attributes = metadata.attrs
    required_attributes = {
        "schema_name",
        "schema_version",
        "task_id",
        "instruction",
        "complete",
    }
    missing = sorted(required_attributes - set(attributes))
    if missing:
        raise SchemaValidationError(
            f"HDF5 metadata is missing required attributes: {missing}"
        )
    if _metadata_text(attributes, "schema_name") != SCHEMA_NAME:
        raise SchemaValidationError(
            f"metadata.schema_name must be {SCHEMA_NAME!r}"
        )
    raw_schema_version = attributes["schema_version"]
    if (
        isinstance(raw_schema_version, (bool, np.bool_))
        or not isinstance(raw_schema_version, (int, np.integer))
    ):
        raise SchemaValidationError("metadata.schema_version must be integer 1")
    schema_version = int(raw_schema_version)
    if schema_version != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"metadata.schema_version must be {SCHEMA_VERSION}"
        )
    raw_task_id = attributes["task_id"]
    task_id = _metadata_text(attributes, "task_id")
    if isinstance(raw_task_id, bytes):
        raw_task_id = raw_task_id.decode("utf-8")
    if raw_task_id != task_id:
        raise SchemaValidationError(
            "metadata.task_id must not contain leading or trailing whitespace"
        )
    if expected_task_id is not None and (
        not isinstance(expected_task_id, str)
        or not expected_task_id
        or expected_task_id != expected_task_id.strip()
    ):
        raise SchemaValidationError("expected_task_id must be non-empty text")
    if expected_task_id is not None and task_id != expected_task_id:
        raise SchemaValidationError(
            f"metadata.task_id {task_id!r} does not match expected task "
            f"{expected_task_id!r}"
        )
    _metadata_text(attributes, "instruction")
    complete = attributes["complete"]
    if not isinstance(complete, (bool, np.bool_)) or not bool(complete):
        raise SchemaValidationError("metadata.complete must be true")

    validate_submission_structure(hdf5_file)
    tracked = (
        _required_dataset(hdf5_file, TRACKED_POSITION),
        _required_dataset(hdf5_file, TRACKED_ORIENTATION),
        _required_dataset(hdf5_file, TRACKED_LINEAR_VELOCITY),
        _required_dataset(hdf5_file, TRACKED_ANGULAR_VELOCITY),
    )
    for dataset in tracked:
        _require_exact_dtype(dataset, np.float32)
    frame_count = int(tracked[0].shape[0])

    actions = _required_dataset(hdf5_file, f"{DATA_ROOT}/actions")
    _require_exact_dtype(actions, np.float32)
    if actions.ndim != 2 or actions.shape[0] != frame_count or actions.shape[1] < 1:
        raise SchemaValidationError(
            f"HDF5 actions must have shape [T,A] with A>=1: {tuple(actions.shape)}"
        )
    _require_finite(actions)

    step_index = _required_dataset(hdf5_file, f"{DATA_ROOT}/time/step_index")
    action_time = _required_dataset(hdf5_file, f"{DATA_ROOT}/time/action_time_s")
    state_time = _required_dataset(hdf5_file, f"{DATA_ROOT}/time/state_time_s")
    _require_exact_dtype(step_index, np.uint64)
    _require_exact_dtype(action_time, np.float64)
    _require_exact_dtype(state_time, np.float64)
    for dataset in (step_index, action_time, state_time):
        if dataset.ndim != 1 or dataset.shape[0] != frame_count:
            raise SchemaValidationError(
                f"HDF5 time dataset must have shape [T]: {dataset.name} "
                f"{tuple(dataset.shape)}"
            )
        _require_finite(dataset)
    expected_steps = np.arange(frame_count, dtype=np.uint64)
    if not np.array_equal(np.asarray(step_index[...]), expected_steps):
        raise SchemaValidationError(
            "HDF5 time/step_index must start at 0 and be contiguous"
        )
    for dataset in (action_time, state_time):
        values = np.asarray(dataset[...], dtype=np.float64)
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise SchemaValidationError(
                f"HDF5 time dataset must be strictly increasing: {dataset.name}"
            )


def validate_formal_submission_structure(
    hdf5_file: h5py.File,
    *,
    expected_task_id: str | None = None,
    expected_role: str | None = None,
) -> str:
    """Validate the strict HDF5 v1 baseline and registered robot identity."""

    validate_safety_submission_structure(
        hdf5_file,
        expected_task_id=expected_task_id,
    )
    _reject_runtime_namespace(hdf5_file)
    role = detect_artifact_role(hdf5_file)
    if expected_role is not None:
        if expected_role not in FORMAL_ROBOT_ARTICULATION_ENTITY_IDS:
            raise SchemaValidationError(
                f"unsupported expected role: {expected_role!r}"
            )
        if role != expected_role:
            raise SchemaValidationError(
                "HDF5 artifact role does not match expectation: "
                f"{role!r} != {expected_role!r}"
            )
    return role
