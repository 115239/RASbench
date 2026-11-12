"""Same-run fixed-evidence sidecar writer for the formal rc3 exporter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


_INITIAL_FIELDS = {
    "articulation_root_pose",
    "articulation_root_velocity",
    "articulation_joint_position",
    "articulation_joint_velocity",
    "rigid_root_pose",
    "rigid_root_velocity",
    "static_pose",
}
_CONTROL_FIELDS = {
    "articulation_root_pose",
    "articulation_root_velocity",
    "articulation_joint_position",
    "articulation_joint_velocity",
    "articulation_body_pose",
    "articulation_body_velocity",
    "rigid_root_pose",
    "rigid_root_velocity",
    "contact_net_forces_w",
    "gripper_closure",
}


def _row(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim < 1 or result.shape[0] != 1:
        raise ValueError(f"fixed evidence {name} must contain exactly one environment row")
    result = np.ascontiguousarray(result[0], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"fixed evidence {name} contains NaN or infinity")
    return result


def _exact_sample(value: Mapping[str, Any], expected: set[str], kind: str) -> dict[str, np.ndarray]:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"fixed evidence {kind} fields differ: missing={missing}, extra={extra}")
    return {name: _row(item, f"{kind}/{name}") for name, item in value.items()}


def _boolean_vector(value: Any, count: int, kind: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype not in {np.dtype(np.bool_), np.dtype(np.uint8)}:
        raise ValueError(f"USD boolean {kind} values must use bool or uint8")
    if result.shape != (count,) or np.any((result != 0) & (result != 1)):
        raise ValueError(
            f"USD boolean {kind} values must be a complete 0/1 vector of shape {(count,)}"
        )
    return np.ascontiguousarray(result, dtype=np.uint8)


class FixedEvidenceSidecarWriter:
    """Write only the Stage-derived fixed surface sampled in this episode."""

    def __init__(
        self,
        path: str | Path,
        manifest: Mapping[str, Any],
        *,
        usd_boolean_attributes: Iterable[Any] = (),
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.name != "fixed_evidence_sidecar.hdf5":
            raise ValueError("fixed evidence sidecar must be named fixed_evidence_sidecar.hdf5")
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite fixed evidence sidecar: {self.path}")
        surface = manifest.get("surface")
        if manifest.get("schema_name") != "robolab-fixed-evidence-runtime-plan" or int(
            manifest.get("schema_version", -1)
        ) != 3:
            raise ValueError("fixed evidence runtime manifest must be v3")
        if not isinstance(surface, Mapping) or surface.get("schema") != (
            "robolab-embodied-fixed-evidence-surface/v0.3"
        ):
            raise ValueError("fixed evidence surface must be v0.3")
        surface_json = json.dumps(
            dict(surface), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "x", libver="latest")
        self._file.attrs["schema_name"] = "robolab-fixed-evidence-sidecar"
        self._file.attrs["schema_version"] = np.int64(1)
        self._file.attrs["complete"] = np.bool_(False)
        self._file.attrs["surface_sha256"] = str(manifest["surface_sha256"])
        self._file.attrs["task_id"] = str(manifest["task_id"])
        self._file.attrs["artifact_role"] = str(manifest["artifact_role"])
        self._file.attrs["formal_stage_digest"] = str(surface["formal_stage_digest"])
        self._file.create_dataset("surface_json", data=np.frombuffer(surface_json, dtype=np.uint8))
        self._initial = self._file.create_group("initial")
        self._steps = self._file.create_group("steps")
        discovered = tuple(
            {
                "prim_path": str(item.prim_path),
                "attribute_name": str(item.attribute_name),
            }
            for item in usd_boolean_attributes
        )
        discovered_identities = tuple(
            (item["prim_path"], item["attribute_name"]) for item in discovered
        )
        if discovered_identities != tuple(sorted(discovered_identities)) or len(
            set(discovered_identities)
        ) != len(discovered_identities):
            raise ValueError(
                "USD boolean discovery must be unique and sorted by Prim path/attribute name"
            )
        task_state = self._file.create_group("usd_boolean_attribute")
        task_state.attrs["discovery_source"] = "final_composed_stage"
        task_state.attrs["discovery_rule"] = "state:* with exact USD bool type"
        task_state.attrs["attribute_count"] = np.uint64(len(discovered))
        surface_json = json.dumps(
            discovered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        task_state.create_dataset(
            "surface_json", data=np.frombuffer(surface_json, dtype=np.uint8)
        )
        self._usd_boolean_group = task_state
        self._usd_boolean_count = len(discovered)
        self._usd_boolean_steps: h5py.Dataset | None = None
        if self._usd_boolean_count:
            self._usd_boolean_steps = task_state.create_dataset(
                "values",
                shape=(0, self._usd_boolean_count),
                maxshape=(None, self._usd_boolean_count),
                chunks=(1, self._usd_boolean_count),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
            )
        self._step_count = 0
        self._zero_streams: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
        self._initial_recorded = False
        self._completed = False

    @staticmethod
    def _create_initial(group: h5py.Group, name: str, value: np.ndarray) -> None:
        group.create_dataset(name, data=value, dtype=np.float32)

    def _append(self, group: h5py.Group, name: str, value: np.ndarray, dtype: Any) -> None:
        array = np.asarray(value, dtype=dtype)
        if 0 in array.shape:
            expected_shape, expected_dtype, count = self._zero_streams.get(
                name,
                (array.shape, array.dtype, 0),
            )
            if expected_shape != array.shape or expected_dtype != array.dtype:
                raise ValueError(f"fixed evidence sidecar shape changed for {name}")
            self._zero_streams[name] = (expected_shape, expected_dtype, count + 1)
            return
        dataset = group.get(name)
        if dataset is None:
            dataset = group.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=(1, *array.shape),
                dtype=array.dtype,
                compression="gzip",
                compression_opts=4,
            )
        if not isinstance(dataset, h5py.Dataset) or dataset.shape[1:] != array.shape:
            raise ValueError(f"fixed evidence sidecar shape changed for {name}")
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def record_initial(
        self,
        sample: Mapping[str, Any],
        *,
        usd_boolean_values: Any,
    ) -> None:
        if self._initial_recorded or self._completed:
            raise ValueError("fixed evidence initial sample is not writable")
        value = _exact_sample(sample, _INITIAL_FIELDS, "initial")
        self._create_initial(
            self._initial,
            "articulation_root",
            np.concatenate(
                (value["articulation_root_pose"], value["articulation_root_velocity"]), axis=-1
            ),
        )
        self._create_initial(
            self._initial,
            "articulation_joint",
            np.stack(
                (value["articulation_joint_position"], value["articulation_joint_velocity"]),
                axis=-1,
            ),
        )
        self._create_initial(
            self._initial,
            "rigid",
            np.concatenate((value["rigid_root_pose"], value["rigid_root_velocity"]), axis=-1),
        )
        self._create_initial(self._initial, "static_pose", value["static_pose"])
        self._usd_boolean_group.create_dataset(
            "initial_value",
            data=_boolean_vector(
                usd_boolean_values,
                self._usd_boolean_count,
                "initial",
            ),
            dtype=np.uint8,
        )
        self._initial_recorded = True
        self._file.flush()

    def record_step(
        self,
        *,
        step_index: int,
        sim_time_s: float,
        sample: Mapping[str, Any],
        usd_boolean_values: Any,
    ) -> None:
        if not self._initial_recorded or self._completed or step_index != self._step_count:
            raise ValueError("fixed evidence control step is non-contiguous")
        timestamp = float(sim_time_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("fixed evidence sim_time_s must be finite and non-negative")
        value = _exact_sample(sample, _CONTROL_FIELDS, "steps")
        self._append(self._steps, "step_index", np.asarray(step_index, dtype=np.uint64), np.uint64)
        self._append(self._steps, "sim_time_s", np.asarray(timestamp, dtype=np.float64), np.float64)
        self._append(
            self._steps,
            "articulation_root",
            np.concatenate(
                (value["articulation_root_pose"], value["articulation_root_velocity"]), axis=-1
            ),
            np.float32,
        )
        self._append(
            self._steps,
            "articulation_joint",
            np.stack(
                (value["articulation_joint_position"], value["articulation_joint_velocity"]),
                axis=-1,
            ),
            np.float32,
        )
        self._append(
            self._steps,
            "articulation_body",
            np.concatenate(
                (value["articulation_body_pose"], value["articulation_body_velocity"]), axis=-1
            ),
            np.float32,
        )
        self._append(
            self._steps,
            "rigid",
            np.concatenate((value["rigid_root_pose"], value["rigid_root_velocity"]), axis=-1),
            np.float32,
        )
        self._append(self._steps, "contact", value["contact_net_forces_w"], np.float32)
        self._append(self._steps, "gripper_closure", value["gripper_closure"], np.float32)
        boolean_values = _boolean_vector(
            usd_boolean_values,
            self._usd_boolean_count,
            "control-step",
        )
        if self._usd_boolean_steps is not None:
            self._usd_boolean_steps.resize(self._step_count + 1, axis=0)
            self._usd_boolean_steps[self._step_count] = boolean_values
        self._step_count += 1
        if self._step_count % 120 == 0:
            self._file.flush()

    def complete(self, expected_steps: int) -> None:
        if not self._initial_recorded or self._step_count != int(expected_steps):
            raise ValueError("fixed evidence sidecar is incomplete")
        for name, (shape, dtype, count) in self._zero_streams.items():
            if count != self._step_count:
                raise ValueError(f"fixed evidence zero-width stream is incomplete: {name}")
            self._steps.create_dataset(
                name,
                data=np.empty((self._step_count, *shape), dtype=dtype),
            )
        if self._usd_boolean_steps is None:
            self._usd_boolean_group.create_dataset(
                "values",
                data=np.empty((self._step_count, 0), dtype=np.uint8),
            )
        elif self._usd_boolean_steps.shape != (
            self._step_count,
            self._usd_boolean_count,
        ):
            raise ValueError("USD boolean control-step evidence is incomplete")
        self._file.attrs.modify("complete", np.bool_(True))
        self._file.flush()
        self._completed = True

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            try:
                os.fsync(descriptor)
            except OSError:
                if os.name != "nt":
                    raise
        finally:
            os.close(descriptor)

    def abort(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


__all__ = ["FixedEvidenceSidecarWriter"]
