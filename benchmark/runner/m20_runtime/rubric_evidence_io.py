"""Write/export/validate the v04 observation extension without physical aliases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np

ROOT = "/extensions/bboard_rubric_v04"
SCHEMA = "bboard-rubric-observations/v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def object_path(prim):
    return ROOT + "/objects/" + quote(prim, safe="")


def contact_path(source, target):
    return ROOT + "/contacts/" + quote(source, safe="") + "__with__" + quote(target, safe="")


def selectors(manifest):
    objects = {}
    for obj in manifest["objects"]:
        base = object_path(obj["prim"])
        fields = ["root_pose", "root_velocity"]
        if obj["colliders"]:
            fields += ["world_aabb", "collider_world_aabb"]
        objects[obj["prim"]] = {f: base + "/" + f for f in fields}
        objects[obj["prim"]]["initial_root_pose"] = base + "/initial/root_pose"
    contacts = {p["source"] + " -> " + p["target"]:
                {field: contact_path(p["source"], p["target"]) + "/" + field
                 for field in ["net_forces_w", "impulse_w", "active_fraction", "step_index", "sim_time_s"]}
                for p in manifest["contacts"]}
    return {"schema": SCHEMA, "task_id": manifest["task_id"], "rubric_id": manifest["rubric_id"],
            "rubric_sha256": manifest["rubric_sha256"], "objects": objects, "contacts": contacts,
            "step_index": ROOT + "/step_index", "sim_time_s": ROOT + "/sim_time_s"}


class RubricEvidenceWriter:
    def __init__(self, path, manifest):
        self.path = Path(path)
        self.file = h5py.File(self.path, "x")
        self.manifest = manifest
        self.count = 0
        self.initialized = False
        self.completed = False
        group = self.file.require_group(ROOT)
        group.attrs.update(schema=SCHEMA, complete=False, task_id=manifest["task_id"],
                           rubric_sha256=manifest["rubric_sha256"],
                           manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest())
        group.create_dataset("manifest_json", data=np.frombuffer(canonical(manifest), dtype=np.uint8))
        group.create_dataset("selectors_json", data=np.frombuffer(canonical(selectors(manifest)), dtype=np.uint8))
        for obj in manifest["objects"]:
            self.file.require_group(object_path(obj["prim"])).attrs.update(
                prim=obj["prim"], runtime_prim=obj["runtime_prim"], pose_source=obj["pose_source"])
        self.path.with_name("rubric_evidence_contract.json").write_text(
            json.dumps(selectors(manifest), ensure_ascii=False, indent=2) + "\n")

    def initial(self, rows):
        if self.initialized or self.completed:
            raise ValueError("rubric initial sample already recorded")
        for obj, row in zip(self.manifest["objects"], rows, strict=True):
            group = self.file.require_group(object_path(obj["prim"]) + "/initial")
            for field, value in row.items():
                group.create_dataset(field, data=value, dtype=np.float32)
        self.initialized = True

    def formal_initial(self, rows, *, physics_step, sim_time_s):
        if not self.initialized or self.completed:
            raise ValueError("formal reference requires an active initialized recorder")
        root = self.file[ROOT]
        if 'formal_reference_physics_step' in root.attrs:
            raise ValueError("formal reference already captured")
        for obj, row in zip(self.manifest["objects"], rows, strict=True):
            group = self.file.create_group(object_path(obj["prim"]) + "/formal_initial")
            for field, value in row.items():
                value = np.asarray(value, dtype=np.float32)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite formal reference")
                group.create_dataset(field, data=value)
        root.attrs.update(formal_reference_physics_step=int(physics_step),
                          formal_reference_sim_time_s=float(sim_time_s),
                          formal_reference_phase='TASK_RUNNING')
        self.file.flush()

    def _append(self, path, value, dtype):
        value = np.asarray(value, dtype=dtype)
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite rubric observation: {path}")
        if path not in self.file:
            parent, name = path.rsplit("/", 1)
            node = self.file.require_group(parent).create_dataset(
                name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                chunks=(1, *value.shape), dtype=dtype, compression="gzip", compression_opts=4)
        else:
            node = self.file[path]
        if node.shape != (self.count, *value.shape):
            raise ValueError(f"rubric observation shape/step changed: {path}")
        node.resize(self.count + 1, axis=0)
        node[self.count] = value

    def step(self, step_index, sim_time_s, rows, contacts):
        if not self.initialized or self.completed or step_index != self.count:
            raise ValueError("rubric observation steps must be contiguous")
        if not np.isfinite(sim_time_s) or sim_time_s < 0:
            raise ValueError("invalid rubric observation time")
        self._append(ROOT + "/step_index", step_index, np.uint64)
        self._append(ROOT + "/sim_time_s", sim_time_s, np.float64)
        self._append(ROOT + "/physics_steps", contacts["physics_steps"], np.uint64)
        for obj, row in zip(self.manifest["objects"], rows, strict=True):
            for field, value in row.items():
                self._append(object_path(obj["prim"]) + "/" + field, value, np.float32)
        for i, pair in enumerate(self.manifest["contacts"]):
            base = contact_path(pair["source"], pair["target"])
            for field in ["net_forces_w", "impulse_w", "active_fraction"]:
                value = contacts[field][i]
                if field != "active_fraction":
                    value = value[None, :]  # one explicitly selected semantic source
                self._append(base + "/" + field, value, np.float32)
        self.count += 1
        if self.count % 120 == 0:
            self.file.flush()

    def complete(self, expected_steps):
        if not self.initialized or self.count != expected_steps or not self.count:
            raise ValueError("incomplete rubric observations")
        for pair in self.manifest["contacts"]:
            base = self.file[contact_path(pair["source"], pair["target"])]
            for field in ["step_index", "sim_time_s"]:
                base[field] = self.file[ROOT + "/" + field]
        self.file[ROOT].attrs.modify("complete", True)
        validate(self.file, expected_task_id=self.manifest["task_id"])
        self.file.flush()
        self.completed = True

    def close(self):
        self.file.close()


def validate(h5, *, expected_task_id=None, expected_plan=None, expected_steps=None):
    if ROOT not in h5:
        raise ValueError("RUBRIC_EVIDENCE_MISSING")
    group = h5[ROOT]
    if group.attrs.get("schema") != SCHEMA or not bool(group.attrs.get("complete")):
        raise ValueError("RUBRIC_EVIDENCE_INCOMPLETE")
    manifest = json.loads(bytes(group["manifest_json"][...]).decode())
    if hashlib.sha256(canonical(manifest)).hexdigest() != group.attrs["manifest_sha256"]:
        raise ValueError("RUBRIC_MANIFEST_HASH_MISMATCH")
    if expected_task_id and manifest["task_id"] != expected_task_id:
        raise ValueError("RUBRIC_TASK_ID_MISMATCH")
    if expected_plan:
        for key in ["task_id", "rubric_id", "rubric_sha256", "rubric_prim_references", "contacts"]:
            if manifest[key] != expected_plan[key]:
                raise ValueError(f"RUBRIC_PLAN_MISMATCH: {key}")
        if len(manifest["objects"]) != len(expected_plan["objects"]) or any(
                any(observed.get(k) != v for k, v in required.items())
                for observed, required in zip(manifest["objects"], expected_plan["objects"])):
            raise ValueError("RUBRIC_PLAN_MISMATCH: objects")
    if json.loads(bytes(group["selectors_json"][...]).decode()) != selectors(manifest):
        raise ValueError("RUBRIC_SELECTOR_MANIFEST_MISMATCH")
    count = len(group["step_index"])
    if count < 1 or expected_steps is not None and count != expected_steps:
        raise ValueError("RUBRIC_SAMPLE_COUNT_MISMATCH")
    def check(path, shape, dtype):
        if path not in h5 or not isinstance(h5[path], h5py.Dataset):
            raise ValueError(f"RUBRIC_SELECTOR_MISSING: {path}")
        node = h5[path]
        if node.shape != shape or node.dtype != np.dtype(dtype):
            raise ValueError(f"RUBRIC_SELECTOR_SHAPE_OR_TYPE: {path}")
        value = node[...]
        if not np.isfinite(value).all():
            raise ValueError(f"RUBRIC_SELECTOR_NONFINITE: {path}")
        return value
    steps = check(ROOT + "/step_index", (count,), np.uint64)
    times = check(ROOT + "/sim_time_s", (count,), np.float64)
    if not np.array_equal(steps, np.arange(count)) or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("RUBRIC_TIME_AXIS_INVALID")
    physics_steps = check(ROOT + "/physics_steps", (count,), np.uint64)
    if np.any(physics_steps < 1):
        raise ValueError("RUBRIC_PHYSICS_OBSERVATION_GAP")
    if count > 1 and not np.allclose(np.diff(times), physics_steps[1:] * manifest["physics_dt"], rtol=0, atol=1e-7):
        raise ValueError("RUBRIC_PHYSICS_TIME_MISMATCH")
    ids = {o["prim"] for o in manifest["objects"]}
    if not set(manifest["rubric_prim_references"]) <= ids:
        raise ValueError("RUBRIC_PRIM_COVERAGE_FAILED")
    for obj in manifest["objects"]:
        base = object_path(obj["prim"])
        for initial in [False, True]:
            prefix = base + ("/initial" if initial else "")
            pose = check(prefix + "/root_pose", (7,) if initial else (count, 7), np.float32)
            check(prefix + "/root_velocity", (6,) if initial else (count, 6), np.float32)
            if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1, atol=1e-4):
                raise ValueError("RUBRIC_QUATERNION_INVALID")
            if obj["colliders"]:
                bound = check(prefix + "/world_aabb", (2, 3) if initial else (count, 2, 3), np.float32)
                parts = check(prefix + "/collider_world_aabb", (len(obj["colliders"]), 2, 3) if initial else (count, len(obj["colliders"]), 2, 3), np.float32)
                if np.any(bound[..., 0, :] > bound[..., 1, :]) or np.any(parts[..., 0, :] > parts[..., 1, :]):
                    raise ValueError("RUBRIC_AABB_INVALID")
                union = np.stack([parts[..., 0, :].min(axis=-2), parts[..., 1, :].max(axis=-2)], axis=-2)
                if not np.allclose(bound, union, rtol=0, atol=1e-5):
                    raise ValueError("RUBRIC_AABB_UNION_MISMATCH")
    for pair in manifest["contacts"]:
        base = contact_path(pair["source"], pair["target"])
        force = check(base + "/net_forces_w", (count, 1, 3), np.float32)
        impulse = check(base + "/impulse_w", (count, 1, 3), np.float32)
        active = check(base + "/active_fraction", (count,), np.float32)
        if np.any((active < 0) | (active > 1)):
            raise ValueError("RUBRIC_CONTACT_ACTIVITY_INVALID")
        if not np.allclose(force * physics_steps[:, None, None] * manifest["physics_dt"], impulse, atol=1e-5, rtol=1e-5):
            raise ValueError("RUBRIC_CONTACT_FORCE_INTERVAL_MISMATCH")
        if not np.array_equal(check(base + "/step_index", (count,), np.uint64), steps) or not np.array_equal(check(base + "/sim_time_s", (count,), np.float64), times):
            raise ValueError("RUBRIC_CONTACT_TIME_MISMATCH")
    return {"status": "VALID", "schema": SCHEMA, "rubric_id": manifest["rubric_id"],
            "rubric_sha256": manifest["rubric_sha256"], "control_steps": count,
            "objects": len(manifest["objects"]), "contacts": len(manifest["contacts"]),
            "rubric_prims_covered": len(manifest["rubric_prim_references"])}


def export(sidecar, destination):
    report = validate(sidecar)
    if ROOT in destination:
        raise ValueError("refusing to overwrite rubric observation extension")
    parent, name = ROOT.rsplit("/", 1)
    sidecar.copy(sidecar[ROOT], destination.require_group(parent), name=name)
    validate(destination)
    return report
