# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming official Runner recorder for one V4 submission episode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from robolab_runner.contract import V4TaskContract
from robolab_runner.errors import SubmissionError
from robolab_runner.schema import DATA_GROUP, DEMO_NAME, METADATA_GROUP, write_submission_metadata


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "detach"):
        return np.stack([item.detach().cpu().numpy() for item in value])
    return np.asarray(value)


def float32_array(value: Any, *, path: str) -> np.ndarray:
    result = as_numpy(value)
    if not np.issubdtype(result.dtype, np.number):
        raise SubmissionError(f"{path} is not numeric (dtype={result.dtype})")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise SubmissionError(f"{path} contains NaN or infinity")
    return result


def slice_env_tree(value: Any, env_id: int, *, prefix: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): slice_env_tree(item, env_id, prefix=f"{prefix}/{key}")
            for key, item in value.items()
        }
    array = float32_array(value, path=prefix)
    if array.ndim == 0 or array.shape[0] <= env_id:
        raise SubmissionError(
            f"{prefix} must have a leading environment dimension containing env {env_id}; got {array.shape}"
        )
    return array[env_id : env_id + 1]


class V4SubmissionRecorder:
    """Write one ``/data/demo_0`` to a new ``submission.hdf5.partial`` only."""

    def __init__(
        self,
        path: str | Path,
        contract: V4TaskContract,
        *,
        runner_version: str,
        endpoint_scope: str,
        policy_id: str,
        policy_version: str,
        env_id: int = 0,
        flush_interval: int = 50,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.contract = contract
        self.env_id = int(env_id)
        self.flush_interval = max(1, int(flush_interval))
        self._file: h5py.File | None = None
        self._demo: h5py.Group | None = None
        self._policy: h5py.Group | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._initial_recorded = False
        self._actions = 0
        self._post_steps = 0
        self._completed = False
        self._last_control_time: float | None = None
        self._queries: list[tuple[int, float, int]] = []
        if contract.uses_full_capture:
            raise SubmissionError("full-capture formats must use CaptureSession")
        if self.path.name != "submission.hdf5.partial":
            raise SubmissionError("V4 recorder output must be named submission.hdf5.partial")
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite partial submission: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = h5py.File(self.path, "x", libver="latest")
            self._file.create_group(DATA_GROUP)
            write_submission_metadata(
                self._file,
                contract,
                runner_version=runner_version,
                endpoint_scope=endpoint_scope,
                policy_id=policy_id,
                policy_version=policy_version,
            )
            self._demo = self._file[DATA_GROUP].create_group(DEMO_NAME)
            self._demo.attrs["seed"] = np.int64(contract.seed)
            self._policy = self._demo.create_group("policy")
            self._file.flush()
        except Exception:
            self.abort()
            raise

    @property
    def num_samples(self) -> int:
        return self._actions

    @property
    def query_count(self) -> int:
        return len(self._queries)

    def _compression(self) -> str | None:
        for candidate in ("gzip", "lzf", None):
            if candidate in self.contract.limits.allowed_compressions:
                return candidate
        raise SubmissionError("contract has no recorder-supported compression")

    def _create_fixed_tree(self, group: h5py.Group, value: Mapping[str, Any], *, prefix: str) -> None:
        for key, item in value.items():
            path = f"{prefix}/{key}"
            if isinstance(item, Mapping):
                self._create_fixed_tree(group.create_group(key), item, prefix=path)
                continue
            array = float32_array(item, path=path)
            if array.ndim < 2 or array.shape[0] != 1:
                raise SubmissionError(f"{path} must have leading shape (1, ...), got {array.shape}")
            group.create_dataset(key, data=array, chunks=True, compression=self._compression())

    def _append(self, group: h5py.Group, key: str, value: Any, *, prefix: str) -> None:
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            child = group[key] if key in group else group.create_group(key)
            if not isinstance(child, h5py.Group):
                raise SubmissionError(f"{path} changed from a dataset to a group")
            for sub_key, sub_value in value.items():
                self._append(child, str(sub_key), sub_value, prefix=path)
            return
        array = float32_array(value, path=path)
        if array.ndim < 2 or array.shape[0] != 1:
            raise SubmissionError(f"{path} must contain exactly one row, got {array.shape}")
        cache_key = f"{group.name}/{key}"
        dataset = self._datasets.get(cache_key)
        if dataset is None:
            self._datasets[cache_key] = group.create_dataset(
                key,
                data=array,
                maxshape=(None,) + array.shape[1:],
                chunks=True,
                compression=self._compression(),
            )
            return
        if dataset.shape[1:] != array.shape[1:]:
            raise SubmissionError(f"{path} shape changed from {dataset.shape[1:]} to {array.shape[1:]}")
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = array[0]

    def record_initial_state(self, env: Any) -> None:
        if self._initial_recorded:
            raise SubmissionError("initial_state was already recorded")
        if self._demo is None or self._file is None:
            raise SubmissionError("recorder is closed")
        state = slice_env_tree(env.scene.get_state(is_relative=True), self.env_id, prefix="initial_state")
        self._create_fixed_tree(self._demo.create_group("initial_state"), state, prefix="initial_state")
        self._initial_recorded = True
        self._file.flush()

    def record_policy_query(self, *, step: int, inference_duration_s: float, chunk_length: int) -> None:
        if self._policy is None or not self._initial_recorded:
            raise SubmissionError("initial_state must be recorded before a policy query")
        if step != self._actions:
            raise SubmissionError(f"policy query step {step} differs from next action index {self._actions}")
        if self._queries:
            prior_step, _prior_duration, prior_chunk = self._queries[-1]
            if step != prior_step + prior_chunk:
                raise SubmissionError("policy queried before the previous action chunk was consumed")
        duration = float(inference_duration_s)
        if not np.isfinite(duration) or duration < 0:
            raise SubmissionError("policy inference duration must be finite and non-negative")
        chunk = int(chunk_length)
        if not 1 <= chunk <= self.contract.policy_limits.max_action_horizon:
            raise SubmissionError("policy chunk length is outside contract limits")
        self._queries.append((step, duration, chunk))
        values = {
            "query_step": np.int64(step),
            "inference_duration_s": np.float64(duration),
            "chunk_length": np.int64(chunk),
        }
        for name, value in values.items():
            dataset = self._policy.get(name)
            array = np.asarray([value])
            if dataset is None:
                self._policy.create_dataset(name, data=array, maxshape=(None,), chunks=True)
            else:
                index = dataset.shape[0]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value

    def record_pre_step(self, action: Any, *, control_time_s: float) -> None:
        if not self._initial_recorded or self._demo is None:
            raise SubmissionError("initial_state must be recorded before the first action")
        if not self._queries or self._actions >= self._queries[-1][0] + self._queries[-1][2]:
            raise SubmissionError("every action must come from the current policy action chunk")
        if self._actions >= self.contract.limits.max_steps:
            raise SubmissionError(f"episode exceeds max_steps={self.contract.limits.max_steps}")
        array = float32_array(action, path="actions")
        if array.ndim == 1:
            array = array[None, :]
        if array.shape != (1, self.contract.control.action_dim):
            raise SubmissionError(
                f"actions must have shape (1, {self.contract.control.action_dim}), got {array.shape}"
            )
        timestamp = float(control_time_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise SubmissionError("control_time_s must be finite and non-negative")
        if self._last_control_time is not None and timestamp <= self._last_control_time:
            raise SubmissionError("control_time_s must be strictly increasing")
        self._append(self._demo, "actions", array, prefix="")
        time_dataset = self._demo.get("control_time")
        if time_dataset is None:
            self._demo.create_dataset(
                "control_time", data=np.asarray([timestamp], dtype=np.float64), maxshape=(None,), chunks=True
            )
        else:
            index = time_dataset.shape[0]
            time_dataset.resize(index + 1, axis=0)
            time_dataset[index] = timestamp
        self._last_control_time = timestamp
        self._actions += 1

    def _end_effector_state(self, env: Any) -> dict[str, np.ndarray]:
        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        target = self.contract.primary_tracking_body
        if target in body_names:
            body_index = body_names.index(target)
        else:
            matches = [index for index, name in enumerate(body_names) if target in name]
            if len(matches) != 1:
                raise SubmissionError(f"end-effector body {target!r} is not uniquely present in {body_names}")
            body_index = matches[0]
        env_slice = slice(self.env_id, self.env_id + 1)
        position = robot.data.body_pos_w[env_slice, body_index, :] - env.scene.env_origins[env_slice, 0:3]
        return {
            "position": float32_array(position, path="ee_pose/position"),
            "orientation": float32_array(
                robot.data.body_quat_w[env_slice, body_index, :], path="ee_pose/orientation"
            ),
            "linear_velocity": float32_array(
                robot.data.body_lin_vel_w[env_slice, body_index, :], path="ee_pose/linear_velocity"
            ),
            "angular_velocity": float32_array(
                robot.data.body_ang_vel_w[env_slice, body_index, :], path="ee_pose/angular_velocity"
            ),
        }

    def record_post_step(self, env: Any) -> None:
        if self._demo is None or self._file is None:
            raise SubmissionError("recorder is closed")
        if self._post_steps >= self._actions:
            raise SubmissionError("post-step state has no matching pre-step action")
        state = slice_env_tree(env.scene.get_state(is_relative=True), self.env_id, prefix="states")
        self._append(self._demo, "states", state, prefix="")
        self._append(self._demo, "ee_pose", self._end_effector_state(env), prefix="")
        self._post_steps += 1
        if self._post_steps % self.flush_interval == 0:
            self._file.flush()

    def complete(self, participant_result: Mapping[str, Any] | None = None) -> None:
        if self._demo is None or self._file is None:
            raise SubmissionError("recorder is closed")
        if not self._initial_recorded or self._actions < 1 or self._actions != self._post_steps:
            raise SubmissionError(
                f"incomplete alignment: initial={self._initial_recorded}, actions={self._actions}, "
                f"post_steps={self._post_steps}"
            )
        if not self._queries or self._queries[-1][0] + self._queries[-1][2] < self._actions:
            raise SubmissionError("policy query audit does not cover every recorded action")
        result_json = json.dumps(dict(participant_result or {}), ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(result_json.encode("utf-8")) > self.contract.limits.max_attribute_bytes:
            raise SubmissionError("participant_result_json exceeds max_attribute_bytes")
        self._demo.attrs["num_samples"] = np.int64(self._actions)
        self._demo.attrs["participant_result_json"] = result_json
        if "success" in (participant_result or {}):
            self._demo.attrs["success"] = np.bool_(participant_result["success"])
        self._file[METADATA_GROUP].attrs.modify("complete", np.bool_(True))
        self._file.flush()
        self._completed = True

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._demo = None
            self._policy = None
            try:
                descriptor = os.open(self.path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass

    def abort(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._demo = None
                self._policy = None

    def __enter__(self) -> "V4SubmissionRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and self._completed:
            self.close()
        else:
            self.abort()


__all__ = ["V4SubmissionRecorder", "as_numpy", "float32_array", "slice_env_tree"]
