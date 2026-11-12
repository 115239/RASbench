#!/usr/bin/env python3
"""Export one real Runner capture with the v0.7.2 safety handoff.

The Runner owns both input files: its ordinary control trace and the fixed
evidence sidecar frozen from the same runtime Stage.  This adapter performs no
Prim-path inference and does not read a Rule Package.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
VENDOR_ROOT = ROOT / "_vendor_v072"
if VENDOR_ROOT.is_dir() and str(VENDOR_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(VENDOR_ROOT))
os.sys.path.insert(0, str(ROOT / "m20_runtime"))
from rubric_evidence_io import export as export_rubric_evidence, validate as validate_rubric_evidence

from robolab_action_schema import (
    EXPORTER_VERSION,
    ArticulationEvidence,
    ArticulationRecord,
    ContactPairEvidence,
    ContactRecord,
    FixedEvidenceSurface,
    FormalEpisodeRecord,
    JointCoordinateEvidence,
    RigidObjectRecord,
    UsdBooleanAttributeRecord,
    UsdBooleanAttributeSpec,
    export_submission_atomically,
    validate_fixed_evidence_surface,
    validate_usd_boolean_attribute_evidence,
    write_discovered_usd_boolean_attribute_evidence,
)


RAW_ROOT = "/data/demo_0"
DEFAULT_SYSTEM_ID = os.environ.get("PAPER200_SYSTEM_ID", "m20-piper-groot-vln-vla-baseline")
DEFAULT_SYSTEM_VERSION = os.environ.get("PAPER200_SYSTEM_VERSION", "mock-v1")


def write_default_system_identity(hdf5_file: h5py.File) -> dict[str, str]:
    """Preserve the handoff fields and identify the actual collection system."""
    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group):
        raise ValueError("formal submission is missing /metadata")
    identity = {
        "system_id": DEFAULT_SYSTEM_ID,
        "system_version": DEFAULT_SYSTEM_VERSION,
    }
    for name, value in identity.items():
        metadata.attrs[name] = value
    return identity


def validate_default_system_identity(hdf5_file: h5py.File) -> dict[str, str]:
    """Fail closed if the exported HDF5 lost or changed the default identity."""
    metadata = hdf5_file.get("/metadata")
    if not isinstance(metadata, h5py.Group):
        raise ValueError("formal submission is missing /metadata")
    expected = {
        "system_id": DEFAULT_SYSTEM_ID,
        "system_version": DEFAULT_SYSTEM_VERSION,
    }
    actual: dict[str, str] = {}
    for name, value in expected.items():
        raw = metadata.attrs.get(name)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw != value:
            raise ValueError(
                f"metadata.{name}={raw!r}, expected default value {value!r}"
            )
        actual[name] = str(raw)
    return actual


def dataset(hdf5_file: h5py.File, path: str, dtype: Any) -> np.ndarray:
    node = hdf5_file.get(path)
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"required Runner evidence is missing: {path}")
    value = np.asarray(node[...], dtype=dtype)
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"Runner evidence contains NaN or infinity: {path}")
    return value


def load_surface(sidecar: h5py.File) -> FixedEvidenceSurface:
    node = sidecar.get("surface_json")
    if not isinstance(node, h5py.Dataset):
        raise ValueError("fixed evidence sidecar is missing surface_json")
    value = json.loads(bytes(np.asarray(node[...], dtype=np.uint8)).decode("utf-8"))
    if value.get("schema") != "robolab-embodied-fixed-evidence-surface/v0.3":
        raise ValueError("unsupported fixed evidence sidecar schema")
    return FixedEvidenceSurface(
        robot_role=str(value["artifact_role"]),
        robot_articulation_prim=str(value["robot_articulation_prim"]),
        task_id=value.get("task_id"),
        formal_stage_digest=value.get("formal_stage_digest"),
        rigid_objects=tuple(value["rigid_objects"]),
        articulations=tuple(
            ArticulationEvidence(
                str(item["prim_path"]),
                tuple(item["joint_paths"]),
                tuple(
                    JointCoordinateEvidence(
                        joint_path=str(joint["joint_path"]),
                        joint_type=str(joint["joint_type"]),
                        body0_path=(
                            str(joint["body0_path"])
                            if joint.get("body0_path") is not None
                            else None
                        ),
                        body1_path=(
                            str(joint["body1_path"])
                            if joint.get("body1_path") is not None
                            else None
                        ),
                        axis=str(joint["axis"]),
                        local_pos0_m=tuple(float(v) for v in joint["local_pos0_m"]),
                        local_rot0_wxyz=tuple(
                            float(v) for v in joint["local_rot0_wxyz"]
                        ),
                        local_pos1_m=tuple(float(v) for v in joint["local_pos1_m"]),
                        local_rot1_wxyz=tuple(
                            float(v) for v in joint["local_rot1_wxyz"]
                        ),
                        lower_limit=(
                            float(joint["lower_limit"])
                            if joint.get("lower_limit") is not None
                            else None
                        ),
                        upper_limit=(
                            float(joint["upper_limit"])
                            if joint.get("upper_limit") is not None
                            else None
                        ),
                        unit=str(joint["unit"]),
                    )
                    for joint in item["joints"]
                ),
                tuple(item["body_paths"]),
            )
            for item in value["articulations"]
        ),
        task_references=tuple(value["task_references"]),
        static_contact_targets=tuple(value["static_contact_targets"]),
        gripper_closure_prims=tuple(value["gripper_closure_prims"]),
        contact_pairs=tuple(
            ContactPairEvidence(
                str(item["source"]),
                str(item["target"]),
                tuple(item["contact_body_paths"]),
            )
            for item in value["contact_pairs"]
        ),
    )


def load_usd_boolean_records(
    sidecar: h5py.File,
) -> tuple[
    tuple[UsdBooleanAttributeSpec, ...],
    dict[tuple[str, str], UsdBooleanAttributeRecord],
]:
    """Load the exact Stage-discovered surface and aligned same-run samples."""

    surface_node = sidecar.get("usd_boolean_attribute/surface_json")
    initial_node = sidecar.get("usd_boolean_attribute/initial_value")
    values_node = sidecar.get("usd_boolean_attribute/values")
    if not all(
        isinstance(node, h5py.Dataset)
        for node in (surface_node, initial_node, values_node)
    ):
        raise ValueError("fixed evidence sidecar is missing USD boolean evidence")
    discovered_json = json.loads(
        bytes(np.asarray(surface_node[...], dtype=np.uint8)).decode("utf-8")
    )
    if not isinstance(discovered_json, list):
        raise ValueError("USD boolean discovery surface must be a JSON list")
    discovered = tuple(
        UsdBooleanAttributeSpec(
            prim_path=str(item["prim_path"]),
            attribute_name=str(item["attribute_name"]),
        )
        for item in discovered_json
    )
    if discovered != tuple(sorted(discovered)) or len(set(discovered)) != len(discovered):
        raise ValueError("USD boolean discovery surface is not unique and sorted")
    initial = np.asarray(initial_node[...])
    values = np.asarray(values_node[...])
    if initial.dtype != np.dtype(np.uint8) or initial.shape != (len(discovered),):
        raise ValueError("USD boolean initial evidence does not close the discovery surface")
    if (
        values.dtype != np.dtype(np.uint8)
        or values.ndim != 2
        or values.shape[1] != len(discovered)
    ):
        raise ValueError("USD boolean step evidence does not close the discovery surface")
    if np.any((initial != 0) & (initial != 1)) or np.any(
        (values != 0) & (values != 1)
    ):
        raise ValueError("USD boolean evidence must contain only 0 or 1")
    records = {
        (spec.prim_path, spec.attribute_name): UsdBooleanAttributeRecord(
            initial_value=bool(initial[index]),
            values=values[:, index],
        )
        for index, spec in enumerate(discovered)
    }
    return discovered, records


def _sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(
    raw: h5py.File,
    sidecar: h5py.File,
    surface: FixedEvidenceSurface,
) -> FormalEpisodeRecord:
    if not bool(sidecar.attrs.get("complete", False)):
        raise ValueError("fixed evidence sidecar is incomplete")

    actions = dataset(raw, f"{RAW_ROOT}/actions", np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError(f"Runner actions must be [T,A], got {actions.shape}")
    count = int(actions.shape[0])
    step_index = dataset(raw, f"{RAW_ROOT}/time/step_index", np.uint64)
    sidecar_steps = dataset(sidecar, "steps/step_index", np.uint64)
    sidecar_times = dataset(sidecar, "steps/sim_time_s", np.float64)
    if not np.array_equal(step_index, np.arange(count, dtype=np.uint64)):
        raise ValueError("Runner step_index is not exactly 0..T-1")
    if not np.array_equal(sidecar_steps, step_index):
        raise ValueError("Runner and fixed evidence step axes differ")
    if sidecar_times.shape != (count,) or (
        count > 1 and np.any(np.diff(sidecar_times) <= 0)
    ):
        raise ValueError("fixed evidence time axis is not strictly increasing")

    initial_roots = dataset(sidecar, "initial/articulation_root", np.float32)
    initial_joints = dataset(sidecar, "initial/articulation_joint", np.float32)
    roots = dataset(sidecar, "steps/articulation_root", np.float32)
    joints = dataset(sidecar, "steps/articulation_joint", np.float32)
    bodies = dataset(sidecar, "steps/articulation_body", np.float32)
    initial_rigids = dataset(sidecar, "initial/rigid", np.float32)
    rigids = dataset(sidecar, "steps/rigid", np.float32)
    contacts = dataset(sidecar, "steps/contact", np.float32)
    static_poses = dataset(sidecar, "initial/static_pose", np.float32)
    if any(array.shape[0] != count for array in (roots, joints, bodies, rigids, contacts)):
        raise ValueError("Runner and fixed evidence sample counts differ")

    articulation_records: dict[str, ArticulationRecord] = {}
    joint_cursor = 0
    body_cursor = 0
    for index, articulation in enumerate(surface.articulations):
        joint_stop = joint_cursor + len(articulation.joint_paths)
        body_stop = body_cursor + len(articulation.body_paths)
        joint_slice = slice(joint_cursor, joint_stop)
        body_slice = slice(body_cursor, body_stop)
        articulation_records[articulation.prim_path] = ArticulationRecord(
            initial_root_pose=initial_roots[index, :7],
            initial_root_velocity=initial_roots[index, 7:],
            initial_joint_position=initial_joints[joint_slice, 0],
            initial_joint_velocity=initial_joints[joint_slice, 1],
            root_pose=roots[:, index, :7],
            root_velocity=roots[:, index, 7:],
            joint_position=joints[:, joint_slice, 0],
            joint_velocity=joints[:, joint_slice, 1],
            feedback={
                "step_index": sidecar_steps,
                "sim_time_s": sidecar_times,
                "joint_position": joints[:, joint_slice, 0],
                "joint_velocity": joints[:, joint_slice, 1],
                "body_pose": bodies[:, body_slice, :7],
                "body_velocity": bodies[:, body_slice, 7:],
            },
        )
        joint_cursor = joint_stop
        body_cursor = body_stop
    if joint_cursor != initial_joints.shape[0] or body_cursor != bodies.shape[1]:
        raise ValueError("fixed evidence articulation axes do not close")

    rigid_records = {
        prim: RigidObjectRecord(
            initial_pose=initial_rigids[index, :7],
            initial_velocity=initial_rigids[index, 7:],
            pose=rigids[:, index, :7],
            velocity=rigids[:, index, 7:],
        )
        for index, prim in enumerate(surface.rigid_objects)
    }
    if len(rigid_records) != initial_rigids.shape[0] or len(rigid_records) != rigids.shape[1]:
        raise ValueError("fixed evidence rigid-object axis does not close")

    static_prims = surface.task_references + surface.static_contact_targets
    if len(static_prims) != static_poses.shape[0]:
        raise ValueError("fixed evidence static-pose axis does not close")
    static_records = {prim: static_poses[index] for index, prim in enumerate(static_prims)}

    gripper_records: dict[str, np.ndarray] = {}
    if surface.gripper_closure_prims:
        closures = dataset(sidecar, "steps/gripper_closure", np.float32)
        if closures.shape != (count, len(surface.gripper_closure_prims)):
            raise ValueError("fixed evidence gripper-closure axis does not close")
        gripper_records = {
            prim: closures[:, index]
            for index, prim in enumerate(surface.gripper_closure_prims)
        }

    contact_records: dict[tuple[str, str], ContactRecord] = {}
    contact_cursor = 0
    for pair in surface.contact_pairs:
        stop = contact_cursor + pair.contact_axis_size
        contact_records[(pair.source_prim, pair.target_prim)] = ContactRecord(
            net_forces_w=contacts[:, contact_cursor:stop],
            step_index=sidecar_steps,
            sim_time_s=sidecar_times,
        )
        contact_cursor = stop
    if contact_cursor != contacts.shape[1]:
        raise ValueError("fixed evidence contact axis does not close")

    metadata = raw.get("metadata")
    if not isinstance(metadata, h5py.Group):
        raise ValueError("Runner capture is missing metadata")
    instruction = metadata.attrs.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Runner capture has no instruction")

    return FormalEpisodeRecord(
        task_id=str(surface.task_id),
        instruction=instruction,
        actions=actions,
        step_index=step_index,
        action_time_s=dataset(raw, f"{RAW_ROOT}/time/action_time_s", np.float64),
        state_time_s=dataset(raw, f"{RAW_ROOT}/time/state_time_s", np.float64),
        tracked_position=dataset(raw, f"{RAW_ROOT}/tracked_pose/position", np.float32),
        tracked_orientation=dataset(raw, f"{RAW_ROOT}/tracked_pose/orientation", np.float32),
        tracked_linear_velocity=dataset(raw, f"{RAW_ROOT}/tracked_pose/linear_velocity", np.float32),
        tracked_angular_velocity=dataset(raw, f"{RAW_ROOT}/tracked_pose/angular_velocity", np.float32),
        rigid_objects=rigid_records,
        articulations=articulation_records,
        static_poses=static_records,
        gripper_closures=gripper_records,
        contacts=contact_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a real Runner trace and final-Stage evidence with v0.7.2"
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixed-evidence-sidecar", type=Path, required=True)
    parser.add_argument("--rubric-evidence-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with h5py.File(args.raw.resolve(), "r") as raw, h5py.File(
        args.fixed_evidence_sidecar.resolve(), "r"
    ) as sidecar:
        surface = load_surface(sidecar)
        record = build_record(raw, sidecar, surface)
        discovered, task_state_records = load_usd_boolean_records(sidecar)
    destination = args.output.expanduser().resolve()
    if destination.name != "submission.hdf5":
        raise ValueError("formal Runner output filename must be submission.hdf5")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rubric_report = None
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.v072.", dir=destination.parent
    ) as temporary_directory:
        unpublished = Path(temporary_directory) / "submission.hdf5"
        result = export_submission_atomically(
            unpublished,
            surface,
            record,
            overwrite=False,
        )
        with h5py.File(unpublished, "r+") as hdf5_file:
            metadata = hdf5_file["/metadata"]
            metadata.attrs.modify("complete", np.bool_(False))
            system_identity = write_default_system_identity(hdf5_file)
            if args.rubric_evidence_sidecar is not None:
                with h5py.File(args.rubric_evidence_sidecar, "r") as rubric_sidecar:
                    validate_rubric_evidence(rubric_sidecar, expected_task_id=surface.task_id,
                                             expected_steps=None)
                    rubric_times = rubric_sidecar["/extensions/bboard_rubric_v04/sim_time_s"][...]
                    if not all(np.any(np.isclose(rubric_times, t + .005, rtol=0, atol=1e-7)) for t in record.state_time_s):
                        raise ValueError("rubric observations and Runner state-time axes differ")
                    rubric_report = export_rubric_evidence(rubric_sidecar, hdf5_file)
            task_state_validation = write_discovered_usd_boolean_attribute_evidence(
                hdf5_file,
                discovered,
                task_state_records,
            )
            metadata.attrs.modify("complete", np.bool_(True))
            hdf5_file.flush()
            fixed_validation = validate_fixed_evidence_surface(hdf5_file, surface)
            if validate_default_system_identity(hdf5_file) != system_identity:
                raise RuntimeError("system identity validation changed after write")
            validated_task_state = validate_usd_boolean_attribute_evidence(hdf5_file)
            if validated_task_state != task_state_validation:
                raise RuntimeError("USD boolean evidence validation changed after write")
            hdf5_file.flush()
        os.replace(unpublished, destination)
    report = {
        "status": "SUCCEEDED",
        "handoff_version": "0.7.2",
        "exporter_version": EXPORTER_VERSION,
        "source": "real Runner capture plus same-run fixed evidence sidecar",
        "rubric_evidence": rubric_report,
        "task_id": surface.task_id,
        "artifact_role": surface.robot_role,
        "system_identity": system_identity,
        "control_steps": fixed_validation.control_steps,
        "articulations": fixed_validation.articulation_count,
        "rigid_objects": fixed_validation.rigid_object_count,
        "static_contact_targets": fixed_validation.static_contact_target_count,
        "contact_pairs": fixed_validation.contact_pair_count,
        "fixed_evidence_surface_sha256": result.fixed_evidence_sha256,
        "usd_boolean_attribute_evidence": {
            "discovery_source": "final_composed_stage",
            "attribute_count": task_state_validation.attribute_count,
            "frame_count": task_state_validation.frame_count,
            "extension_written": bool(task_state_validation.attribute_count),
        },
        "submission": {
            "file": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
        },
    }
    report_path = args.report or destination.with_name("protocol_v072_export_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
