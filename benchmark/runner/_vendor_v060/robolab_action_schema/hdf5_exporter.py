"""Atomic exporter for formal RoboLab competition HDF5 submissions.

The Runner owns sampling.  This module owns only deterministic normalization,
path generation, fixed-evidence closure checks, postflight validation, and
atomic publication.  It deliberately has no Isaac Sim dependency and never
reads a Rule Package.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .fixed_evidence_surface import (
    FixedEvidenceSurface,
    FixedEvidenceValidationResult,
    validate_fixed_evidence_surface,
)
from .v4_hdf5_schema import (
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
    gripper_closure_path,
    rigid_root_pose_path,
    rigid_root_velocity_path,
    task_reference_pose_path,
)

EXPORTER_VERSION = "0.4.0rc4"

_REQUIRED_FEEDBACK_FIELDS = frozenset(
    {
        "joint_position",
        "joint_velocity",
        "body_pose",
        "body_velocity",
        "step_index",
        "sim_time_s",
    }
)
_OPTIONAL_FEEDBACK_FIELDS = frozenset(
    {
        "joint_position_target",
        "joint_velocity_target",
        "joint_effort_target",
        "applied_torque",
        "computed_torque",
    }
)


class FormalHdf5ExportError(ValueError):
    """Raised before an invalid or incomplete artifact can be published."""


@dataclass(frozen=True)
class RigidObjectRecord:
    initial_pose: Any
    initial_velocity: Any
    pose: Any
    velocity: Any


@dataclass(frozen=True)
class ArticulationRecord:
    initial_root_pose: Any
    initial_root_velocity: Any
    initial_joint_position: Any
    initial_joint_velocity: Any
    root_pose: Any
    root_velocity: Any
    joint_position: Any
    joint_velocity: Any
    feedback: Mapping[str, Any]


@dataclass(frozen=True)
class ContactRecord:
    net_forces_w: Any
    step_index: Any
    sim_time_s: Any


@dataclass(frozen=True)
class FormalEpisodeRecord:
    """Normalized Runner buffers for one completed episode.

    Mapping keys are formal absolute Prim paths.  Contact keys are ordered
    ``(source_prim, target_prim)`` pairs.  Runtime namespaces, display names,
    and stream names must be resolved by the Runner adapter before constructing
    this record.
    """

    task_id: str
    instruction: str
    actions: Any
    step_index: Any
    action_time_s: Any
    state_time_s: Any
    tracked_position: Any
    tracked_orientation: Any
    tracked_linear_velocity: Any
    tracked_angular_velocity: Any
    rigid_objects: Mapping[str, RigidObjectRecord] = field(default_factory=dict)
    articulations: Mapping[str, ArticulationRecord] = field(default_factory=dict)
    static_poses: Mapping[str, Any] = field(default_factory=dict)
    gripper_closures: Mapping[str, Any] = field(default_factory=dict)
    contacts: Mapping[tuple[str, str], ContactRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class FormalHdf5ExportResult:
    destination: Path
    sha256: str
    size_bytes: int
    fixed_evidence_sha256: str
    validation: FixedEvidenceValidationResult


def _exact_keys(name: str, actual: set[Any], expected: set[Any]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise FormalHdf5ExportError(
            f"{name} does not exactly match the fixed evidence surface; "
            f"missing={missing}, extra={extra}"
        )


def _validate_record_closure(
    surface: FixedEvidenceSurface,
    record: FormalEpisodeRecord,
) -> None:
    if not isinstance(record.task_id, str) or not record.task_id.strip():
        raise FormalHdf5ExportError("record.task_id must be a non-empty string")
    if surface.task_id is None:
        raise FormalHdf5ExportError(
            "production export requires surface.task_id to be frozen"
        )
    if record.task_id != surface.task_id:
        raise FormalHdf5ExportError(
            f"record task_id does not match fixed surface: {record.task_id!r} != "
            f"{surface.task_id!r}"
        )
    if not isinstance(record.instruction, str) or not record.instruction.strip():
        raise FormalHdf5ExportError("record.instruction must be a non-empty string")
    if surface.formal_stage_digest is None:
        raise FormalHdf5ExportError(
            "production export requires surface.formal_stage_digest to be frozen"
        )

    _exact_keys(
        "rigid object records",
        set(record.rigid_objects),
        set(surface.rigid_objects),
    )
    _exact_keys(
        "articulation records",
        set(record.articulations),
        {item.prim_path for item in surface.articulations},
    )
    _exact_keys(
        "static pose records",
        set(record.static_poses),
        set(surface.task_references) | set(surface.static_contact_targets),
    )
    _exact_keys(
        "gripper closure records",
        set(record.gripper_closures),
        set(surface.gripper_closure_prims),
    )
    _exact_keys(
        "contact records",
        set(record.contacts),
        {(item.source_prim, item.target_prim) for item in surface.contact_pairs},
    )

    allowed_feedback = _REQUIRED_FEEDBACK_FIELDS | _OPTIONAL_FEEDBACK_FIELDS
    for prim_path, articulation in record.articulations.items():
        fields = set(articulation.feedback)
        missing = sorted(_REQUIRED_FEEDBACK_FIELDS - fields)
        unknown = sorted(fields - allowed_feedback)
        if missing or unknown:
            raise FormalHdf5ExportError(
                f"articulation feedback fields are invalid for {prim_path}; "
                f"missing={missing}, unknown={unknown}"
            )


def _put(
    hdf5_file: h5py.File,
    path: str,
    value: Any,
    dtype: Any,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        # A zero-DOF articulation has [0] initial joint arrays and [T,0]
        # control/feedback arrays.  No other empty formal dataset is valid.
        joint_dataset = path.endswith(("/joint_position", "/joint_velocity"))
        zero_dof_initial = array.shape == (0,)
        zero_dof_stream = (
            array.ndim >= 2 and array.shape[0] > 0 and 0 in array.shape[1:]
        )
        if not (joint_dataset and (zero_dof_initial or zero_dof_stream)):
            raise FormalHdf5ExportError(f"empty dataset: {path}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise FormalHdf5ExportError(f"non-finite dataset: {path}")
    keyword_args: dict[str, Any] = {}
    if compression is not None and array.ndim > 0 and array.size > 0:
        keyword_args["compression"] = compression
        if compression_opts is not None:
            keyword_args["compression_opts"] = compression_opts
        keyword_args["shuffle"] = True
    hdf5_file.create_dataset(path, data=array, **keyword_args)


def _write_payload(
    hdf5_file: h5py.File,
    surface: FixedEvidenceSurface,
    record: FormalEpisodeRecord,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    def put(path: str, value: Any, dtype: Any) -> None:
        _put(
            hdf5_file,
            path,
            value,
            dtype,
            compression=compression,
            compression_opts=compression_opts,
        )

    metadata = hdf5_file.create_group("metadata")
    metadata.attrs["schema_name"] = SCHEMA_NAME
    metadata.attrs["schema_version"] = SCHEMA_VERSION
    metadata.attrs["task_id"] = record.task_id
    metadata.attrs["instruction"] = record.instruction
    metadata.attrs["complete"] = False
    metadata.attrs["artifact_role"] = surface.robot_role
    metadata.attrs["formal_stage_digest"] = surface.formal_stage_digest
    metadata.attrs["fixed_evidence_surface_schema"] = surface.canonical_audit_dict()[
        "schema"
    ]
    metadata.attrs["fixed_evidence_surface_sha256"] = surface.audit_sha256()
    metadata.attrs["exporter_version"] = EXPORTER_VERSION

    put(f"{DATA_ROOT}/actions", record.actions, np.float32)
    put(f"{DATA_ROOT}/time/step_index", record.step_index, np.uint64)
    put(f"{DATA_ROOT}/time/action_time_s", record.action_time_s, np.float64)
    put(f"{DATA_ROOT}/time/state_time_s", record.state_time_s, np.float64)
    put(TRACKED_POSITION, record.tracked_position, np.float32)
    put(TRACKED_ORIENTATION, record.tracked_orientation, np.float32)
    put(TRACKED_LINEAR_VELOCITY, record.tracked_linear_velocity, np.float32)
    put(TRACKED_ANGULAR_VELOCITY, record.tracked_angular_velocity, np.float32)

    for prim_path in sorted(record.rigid_objects):
        state = record.rigid_objects[prim_path]
        put(rigid_root_pose_path(prim_path, initial=True), state.initial_pose, np.float32)
        put(
            rigid_root_velocity_path(prim_path, initial=True),
            state.initial_velocity,
            np.float32,
        )
        put(rigid_root_pose_path(prim_path), state.pose, np.float32)
        put(rigid_root_velocity_path(prim_path), state.velocity, np.float32)

    for prim_path in sorted(record.articulations):
        state = record.articulations[prim_path]
        initial = {
            "root_pose": state.initial_root_pose,
            "root_velocity": state.initial_root_velocity,
            "joint_position": state.initial_joint_position,
            "joint_velocity": state.initial_joint_velocity,
        }
        control = {
            "root_pose": state.root_pose,
            "root_velocity": state.root_velocity,
            "joint_position": state.joint_position,
            "joint_velocity": state.joint_velocity,
        }
        for field_name, value in initial.items():
            put(
                articulation_state_path(prim_path, field_name, initial=True),
                value,
                np.float32,
            )
        for field_name, value in control.items():
            put(articulation_state_path(prim_path, field_name), value, np.float32)
        for field_name in sorted(state.feedback):
            dtype = np.float64 if field_name == "sim_time_s" else np.uint64 if field_name == "step_index" else np.float32
            put(
                articulation_feedback_path(prim_path, field_name),
                state.feedback[field_name],
                dtype,
            )

    for prim_path in sorted(record.static_poses):
        put(task_reference_pose_path(prim_path), record.static_poses[prim_path], np.float32)
    for prim_path in sorted(record.gripper_closures):
        put(gripper_closure_path(prim_path), record.gripper_closures[prim_path], np.float32)
    for source_prim, target_prim in sorted(record.contacts):
        contact = record.contacts[(source_prim, target_prim)]
        put(
            contact_field_path(source_prim, target_prim, "net_forces_w"),
            contact.net_forces_w,
            np.float32,
        )
        put(
            contact_field_path(source_prim, target_prim, "step_index"),
            contact.step_index,
            np.uint64,
        )
        put(
            contact_field_path(source_prim, target_prim, "sim_time_s"),
            contact.sim_time_s,
            np.float64,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_submission_atomically(
    destination: str | os.PathLike[str],
    surface: FixedEvidenceSurface,
    record: FormalEpisodeRecord,
    *,
    overwrite: bool = False,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> FormalHdf5ExportResult:
    """Validate and atomically publish one formal ``submission.hdf5``.

    ``complete`` becomes true only inside the unpublished temporary file.  The
    complete fixed-evidence postflight must then pass before ``os.replace`` can
    expose the destination.
    """

    if not isinstance(surface, FixedEvidenceSurface):
        raise TypeError("surface must be a FixedEvidenceSurface")
    if not isinstance(record, FormalEpisodeRecord):
        raise TypeError("record must be a FormalEpisodeRecord")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    if compression not in (None, "gzip", "lzf"):
        raise FormalHdf5ExportError("compression must be None, 'gzip', or 'lzf'")
    if compression != "gzip" and compression_opts is not None:
        raise FormalHdf5ExportError(
            "compression_opts is supported only with gzip compression"
        )
    if compression == "gzip" and (
        not isinstance(compression_opts, int)
        or isinstance(compression_opts, bool)
        or not 0 <= compression_opts <= 9
    ):
        raise FormalHdf5ExportError("gzip compression_opts must be an integer in [0,9]")

    output = Path(destination).resolve()
    if output.name != "submission.hdf5":
        raise FormalHdf5ExportError(
            "formal Runner output filename must be submission.hdf5"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _validate_record_closure(surface, record)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    validation: FixedEvidenceValidationResult | None = None
    try:
        with h5py.File(temporary, "w") as hdf5_file:
            _write_payload(
                hdf5_file,
                surface,
                record,
                compression=compression,
                compression_opts=compression_opts,
            )
            hdf5_file["/metadata"].attrs.modify("complete", True)
            hdf5_file.flush()
            validation = validate_fixed_evidence_surface(hdf5_file, surface)
            hdf5_file.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    if validation is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("export completed without fixed-evidence validation")
    return FormalHdf5ExportResult(
        destination=output,
        sha256=_sha256(output),
        size_bytes=output.stat().st_size,
        fixed_evidence_sha256=surface.audit_sha256(),
        validation=validation,
    )
