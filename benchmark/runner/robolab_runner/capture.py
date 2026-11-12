# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming V4.1 participant capture with one HDF5 and one H.264 video."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from robolab_runner.contract import CaptureStream, V4TaskContract
from robolab_runner.errors import SubmissionError
from robolab_runner.recorder import as_numpy, float32_array, slice_env_tree
from robolab_runner.schema import DATA_GROUP, DEMO_NAME, METADATA_GROUP, write_submission_metadata

_NETWORK_MESSAGE_TYPES = frozenset({"handshake", "reset", "infer", "end_episode"})
_NETWORK_STATUSES = frozenset({"success", "timeout", "disconnected", "protocol_error"})


class FFmpegVideoWriter:
    """Stream RGB24 frames directly to an H.264/yuv420p partial MP4."""

    def __init__(self, path: str | Path, *, width: int, height: int, fps: float) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.name != "episode.mp4.partial":
            raise SubmissionError("V4.1 video output must be named episode.mp4.partial")
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite partial video: {self.path}")
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frames = 0
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            "-n",
            str(self.path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise SubmissionError(f"failed to start ffmpeg: {error}") from error

    def write(self, rgb: Any) -> None:
        array = np.asarray(rgb)
        expected = (self.height, self.width, 3)
        if array.dtype != np.dtype(np.uint8) or array.shape != expected:
            raise SubmissionError(f"video frame must be uint8 RGB {expected}, got {array.dtype} {array.shape}")
        if self._process.stdin is None:
            raise SubmissionError("ffmpeg stdin is unavailable")
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes(order="C"))
        except (BrokenPipeError, OSError) as error:
            detail = self._read_error()
            raise SubmissionError(f"ffmpeg stopped while encoding frame {self.frames}: {detail}") from error
        self.frames += 1

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "no encoder diagnostics"
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")[-2000:]
        except OSError:
            return "failed to read encoder diagnostics"

    def close(self) -> None:
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            self._process.stdin = None
        return_code = self._process.wait()
        detail = self._read_error()
        if return_code != 0:
            raise SubmissionError(f"ffmpeg exited with status {return_code}: {detail}")
        if self.frames < 1 or not self.path.is_file() or self.path.stat().st_size <= 0:
            raise SubmissionError("ffmpeg produced no V4.1 video frames")
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def abort(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
                self._process.stdin = None
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class CaptureSession:
    """Write all contract-declared V4.1 streams without duplicating producer samples."""

    def __init__(
        self,
        hdf5_path: str | Path,
        video_path: str | Path,
        contract: V4TaskContract,
        *,
        runner_version: str,
        endpoint_scope: str,
        policy_id: str,
        policy_version: str,
        env_id: int = 0,
    ) -> None:
        if not contract.uses_full_capture or contract.runner_capture is None:
            raise SubmissionError("CaptureSession requires a full-capture contract")
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.video_path = Path(video_path).expanduser().resolve()
        if self.hdf5_path.name != "submission.hdf5.partial":
            raise SubmissionError("full-capture HDF5 output must be named submission.hdf5.partial")
        if self.hdf5_path.parent != self.video_path.parent:
            raise SubmissionError("full-capture partial HDF5 and video must share a directory")
        for path in (self.hdf5_path, self.video_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite partial capture: {path}")
        self.contract = contract
        self.capture = contract.runner_capture
        self.env_id = int(env_id)
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: h5py.File | None = None
        self._demo: h5py.Group | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._stream_groups: dict[tuple[str, str], h5py.Group] = {}
        self._stream_last: dict[tuple[str, str], tuple[int, float]] = {}
        self._initial_recorded = False
        self._actions = 0
        self._post_steps = 0
        self._direct_steps = 0
        self._completed = False
        self._logical_bytes = 0
        self._objects = 1
        self._last_action_time: float | None = None
        self._last_state_time: float | None = None
        self._queries: list[tuple[int, float, int]] = []
        self._direct_lidar_profile: dict[str, Any] | None = None
        self._video = FFmpegVideoWriter(
            self.video_path,
            width=self.capture.video.width,
            height=self.capture.video.height,
            fps=self.capture.video.fps,
        )
        try:
            self._file = h5py.File(self.hdf5_path, "x", libver="latest")
            data = self._create_group(self._file, DATA_GROUP)
            write_submission_metadata(
                self._file,
                contract,
                runner_version=runner_version,
                endpoint_scope=endpoint_scope,
                policy_id=policy_id,
                policy_version=policy_version,
            )
            self._objects += 1
            self._demo = self._create_group(data, DEMO_NAME)
            self._demo.attrs["seed"] = np.int64(contract.seed)
            self._create_group(self._demo, "policy")
            self._create_group(self._demo["policy"], "network")
            self._create_group(self._demo, "time")
            if contract.is_v4_3:
                self._create_group(self._demo, "task")
                self._create_group(self._demo, "events")
            self._file.flush()
        except Exception:
            self.abort()
            raise

    @property
    def num_samples(self) -> int:
        return self._actions

    @property
    def video_frames(self) -> int:
        return self._video.frames

    @property
    def query_count(self) -> int:
        return len(self._queries)

    def _check_capacity(self, added_bytes: int = 0) -> None:
        self._logical_bytes += int(added_bytes)
        if self._logical_bytes > self.contract.limits.max_logical_bytes:
            raise SubmissionError("V4.1 logical capture capacity exceeded")
        if self._objects > self.contract.limits.max_hdf5_objects:
            raise SubmissionError("V4.1 HDF5 object limit exceeded")
        if self.hdf5_path.exists() and self.hdf5_path.stat().st_size > self.contract.limits.max_file_bytes:
            raise SubmissionError("V4.1 HDF5 file capacity exceeded")

    def _create_group(self, parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
        group = parent.create_group(name)
        self._objects += 1
        self._check_capacity()
        return group

    def _compression_kwargs(self) -> dict[str, Any]:
        return {"compression": self.capture.compression, "compression_opts": self.capture.compression_level}

    def _create_dataset(self, parent: h5py.Group, name: str, **kwargs: Any) -> h5py.Dataset:
        dataset = parent.create_dataset(name, **kwargs)
        self._objects += 1
        self._check_capacity(int(dataset.size) * int(dataset.dtype.itemsize))
        return dataset

    def _append_array(
        self,
        parent: h5py.Group,
        name: str,
        value: Any,
        *,
        dtype: str | np.dtype,
        shape: tuple[int, ...],
        compress: bool = True,
    ) -> None:
        array = np.asarray(value)
        expected_dtype = np.dtype(dtype)
        if array.dtype != expected_dtype or array.shape != shape:
            raise SubmissionError(
                f"{parent.name}/{name} must be {expected_dtype.name}{shape}, got {array.dtype}{array.shape}"
            )
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise SubmissionError(f"{parent.name}/{name} contains NaN or infinity")
        key = f"{parent.name}/{name}"
        dataset = self._datasets.get(key)
        if dataset is None:
            kwargs = self._compression_kwargs() if compress else {}
            dataset = self._create_dataset(
                parent,
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                chunks=(1,) + shape if shape else (1024,),
                dtype=expected_dtype,
                **kwargs,
            )
            self._datasets[key] = dataset
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = array
        self._check_capacity(array.nbytes)

    def _append_scalar(self, parent: h5py.Group, name: str, value: Any, dtype: str) -> None:
        self._append_array(parent, name, np.asarray(value, dtype=np.dtype(dtype)), dtype=dtype, shape=())

    def _create_fixed_tree(self, group: h5py.Group, value: Mapping[str, Any], *, prefix: str) -> None:
        for key, item in value.items():
            path = f"{prefix}/{key}"
            if isinstance(item, Mapping):
                self._create_fixed_tree(self._create_group(group, str(key)), item, prefix=path)
                continue
            array = float32_array(item, path=path)
            if array.ndim < 2 or array.shape[0] != 1:
                raise SubmissionError(f"{path} must have leading shape (1, ...), got {array.shape}")
            self._create_dataset(group, str(key), data=array, chunks=True, **self._compression_kwargs())

    def _append_tree(self, group: h5py.Group, value: Mapping[str, Any], *, prefix: str) -> None:
        for key, item in value.items():
            path = f"{prefix}/{key}"
            if isinstance(item, Mapping):
                child = group.get(str(key))
                if child is None:
                    child = self._create_group(group, str(key))
                if not isinstance(child, h5py.Group):
                    raise SubmissionError(f"{path} changed from dataset to group")
                self._append_tree(child, item, prefix=path)
                continue
            array = float32_array(item, path=path)
            if array.ndim < 2 or array.shape[0] != 1:
                raise SubmissionError(f"{path} must contain exactly one environment row")
            self._append_array(group, str(key), array[0], dtype="float32", shape=array.shape[1:])

    def record_initial_state(self, env: Any) -> None:
        if self._initial_recorded or self._demo is None or self._file is None:
            raise SubmissionError("initial_state cannot be recorded in the current session state")
        state = slice_env_tree(env.scene.get_state(is_relative=True), self.env_id, prefix="initial_state")
        self._create_fixed_tree(self._create_group(self._demo, "initial_state"), state, prefix="initial_state")
        self._initial_recorded = True
        self._file.flush()

    def record_policy_query(self, *, step: int, inference_duration_s: float, chunk_length: int) -> None:
        if step != self._actions:
            raise SubmissionError("policy query step differs from the next V4.1 action")
        duration = float(inference_duration_s)
        chunk = int(chunk_length)
        if not np.isfinite(duration) or duration < 0:
            raise SubmissionError("policy inference duration must be finite and non-negative")
        if not 1 <= chunk <= self.contract.policy_limits.max_action_horizon:
            raise SubmissionError("policy action chunk is outside the contract limit")
        if self._queries and step != self._queries[-1][0] + self._queries[-1][2]:
            raise SubmissionError("policy queried before its previous action chunk was consumed")
        self._queries.append((step, duration, chunk))

    def record_network_event(self, event: Mapping[str, Any]) -> None:
        if self._demo is None:
            raise SubmissionError("capture is closed")
        group = self._demo["policy/network"]
        message_type = str(event.get("message_type", ""))
        status = str(event.get("status", ""))
        if message_type not in _NETWORK_MESSAGE_TYPES or status not in _NETWORK_STATUSES:
            raise SubmissionError("network audit event has an unsupported message type or status")
        sequence = int(event.get("sequence", -1))
        control_step = int(event.get("control_step", -1))
        request_bytes = int(event.get("request_bytes", -1))
        response_bytes = int(event.get("response_bytes", -1))
        session_time_s = float(event.get("session_time_s", np.nan))
        round_trip_s = float(event.get("round_trip_s", np.nan))
        if sequence < 0 or control_step < -1 or request_bytes < 0 or response_bytes < 0:
            raise SubmissionError("network audit integer values are outside their allowed range")
        if not np.isfinite([session_time_s, round_trip_s]).all() or session_time_s < 0 or round_trip_s < 0:
            raise SubmissionError("network audit times must be finite and non-negative")
        values = {
            "sequence": (np.asarray(sequence, dtype=np.uint64), "uint64"),
            "message_type": (np.asarray(message_type.encode("ascii"), dtype="S16"), "S16"),
            "control_step": (np.asarray(control_step, dtype=np.int64), "int64"),
            "session_time_s": (np.asarray(session_time_s, dtype=np.float64), "float64"),
            "request_bytes": (np.asarray(request_bytes, dtype=np.uint64), "uint64"),
            "response_bytes": (np.asarray(response_bytes, dtype=np.uint64), "uint64"),
            "round_trip_s": (np.asarray(round_trip_s, dtype=np.float64), "float64"),
            "status": (np.asarray(status.encode("ascii"), dtype="S16"), "S16"),
        }
        for name, (value, dtype) in values.items():
            self._append_array(group, name, value, dtype=dtype, shape=())

    def record_pre_step(self, action: Any, *, step_index: int, action_time_s: float) -> None:
        if not self._initial_recorded or self._demo is None:
            raise SubmissionError("initial_state must precede actions")
        if step_index != self._actions or self._actions >= self.contract.limits.max_steps:
            raise SubmissionError("V4.1 action step is non-contiguous or exceeds max_steps")
        if not self._queries or self._actions >= self._queries[-1][0] + self._queries[-1][2]:
            raise SubmissionError("every action must come from the current policy action chunk")
        array = float32_array(action, path="actions")
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (self.contract.control.action_dim,):
            raise SubmissionError(f"V4.1 action must have shape ({self.contract.control.action_dim},)")
        timestamp = float(action_time_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise SubmissionError("action_time_s must be finite and non-negative")
        if self._last_action_time is not None and timestamp <= self._last_action_time:
            raise SubmissionError("action_time_s must be strictly increasing")
        self._append_array(self._demo, "actions", array, dtype="float32", shape=array.shape)
        time_group = self._demo["time"]
        self._append_scalar(time_group, "step_index", step_index, "uint64")
        self._append_scalar(time_group, "action_time_s", timestamp, "float64")
        self._last_action_time = timestamp
        self._actions += 1

    def _tracking_state(self, env: Any) -> dict[str, np.ndarray]:
        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        target = self.contract.primary_tracking_body
        matches = [index for index, name in enumerate(body_names) if name == target or target in name]
        if len(matches) != 1:
            raise SubmissionError(f"tracking body {target!r} is not unique in {body_names}")
        body_index = matches[0]
        env_slice = slice(self.env_id, self.env_id + 1)
        position = robot.data.body_pos_w[env_slice, body_index, :] - env.scene.env_origins[env_slice, 0:3]
        return {
            "position": float32_array(position, path="tracked_pose/position"),
            "orientation": float32_array(
                robot.data.body_quat_w[env_slice, body_index, :], path="tracked_pose/orientation"
            ),
            "linear_velocity": float32_array(
                robot.data.body_lin_vel_w[env_slice, body_index, :], path="tracked_pose/linear_velocity"
            ),
            "angular_velocity": float32_array(
                robot.data.body_ang_vel_w[env_slice, body_index, :], path="tracked_pose/angular_velocity"
            ),
        }

    def record_post_step(self, env: Any, *, step_index: int, state_time_s: float) -> None:
        if self._demo is None or self._file is None or step_index != self._post_steps:
            raise SubmissionError("V4.1 post-step state is non-contiguous")
        if self._post_steps >= self._actions:
            raise SubmissionError("post-step state has no matching action")
        timestamp = float(state_time_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise SubmissionError("state_time_s must be finite and non-negative")
        if self._last_state_time is not None and timestamp <= self._last_state_time:
            raise SubmissionError("state_time_s must be strictly increasing")
        state = slice_env_tree(env.scene.get_state(is_relative=True), self.env_id, prefix="states")
        states = self._demo.get("states")
        if states is None:
            states = self._create_group(self._demo, "states")
        self._append_tree(states, state, prefix="states")
        tracking = self._demo.get("tracked_pose")
        if tracking is None:
            tracking = self._create_group(self._demo, "tracked_pose")
        self._append_tree(tracking, self._tracking_state(env), prefix="tracked_pose")
        self._append_scalar(self._demo["time"], "state_time_s", timestamp, "float64")
        self._last_state_time = timestamp
        self._post_steps += 1
        if self._post_steps % self.capture.flush_interval_samples == 0:
            self._file.flush()
            self._check_capacity()

    def _stream_group(self, stream: CaptureStream) -> h5py.Group:
        identity = (stream.kind, stream.name)
        existing = self._stream_groups.get(identity)
        if existing is not None:
            return existing
        if self._demo is None:
            raise SubmissionError("capture is closed")
        if stream.kind == "articulation":
            path = ("feedback", "articulation", stream.name)
        else:
            path = ("sensors", stream.kind, stream.name)
        group: h5py.Group = self._demo
        for segment in path:
            child = group.get(segment)
            if child is None:
                child = self._create_group(group, segment)
            if not isinstance(child, h5py.Group):
                raise SubmissionError(f"capture path {path!r} is not a group")
            group = child
        self._stream_groups[identity] = group
        return group

    def _append_lidar_points(
        self,
        group: h5py.Group,
        stream: CaptureStream,
        values: Mapping[str, Any],
    ) -> None:
        fields = {field.name: field for field in stream.fields}
        points = np.asarray(values["points"])
        point_field = fields["points"]
        if points.dtype != np.dtype(point_field.dtype) or points.ndim != 2 or points.shape[1:] != point_field.shape:
            raise SubmissionError("LiDAR points differ from the frozen dtype/shape")
        if len(points) > self.capture.max_lidar_points_per_frame:
            raise SubmissionError("LiDAR frame exceeds max_lidar_points_per_frame; truncation is forbidden")
        if not np.isfinite(points).all():
            raise SubmissionError("LiDAR points contain NaN or infinity")
        dataset = group.get("points")
        if dataset is None:
            dataset = self._create_dataset(
                group,
                "points",
                shape=(0,) + point_field.shape,
                maxshape=(None,) + point_field.shape,
                chunks=(max(1, min(8192, self.capture.max_lidar_points_per_frame)),) + point_field.shape,
                dtype=np.dtype(point_field.dtype),
                **self._compression_kwargs(),
            )
            self._datasets[f"{group.name}/points"] = dataset
            self._create_dataset(
                group,
                "frame_offsets",
                data=np.asarray([0], dtype=np.uint64),
                maxshape=(None,),
                chunks=True,
                **self._compression_kwargs(),
            )
        start = int(dataset.shape[0])
        dataset.resize(start + len(points), axis=0)
        if len(points):
            dataset[start:] = points
        offsets = group["frame_offsets"]
        offsets.resize(offsets.shape[0] + 1, axis=0)
        offsets[-1] = start + len(points)
        self._check_capacity(points.nbytes + np.dtype(np.uint64).itemsize)
        if "ray_index" in fields:
            ray_index = np.asarray(values["ray_index"])
            field = fields["ray_index"]
            if ray_index.dtype != np.dtype(field.dtype) or ray_index.shape != (len(points),) + field.shape:
                raise SubmissionError("LiDAR ray_index must align one-to-one with points")
            ray_dataset = group.get("ray_index")
            if ray_dataset is None:
                ray_dataset = self._create_dataset(
                    group,
                    "ray_index",
                    shape=(0,) + field.shape,
                    maxshape=(None,) + field.shape,
                    chunks=(max(1, min(8192, self.capture.max_lidar_points_per_frame)),) + field.shape,
                    dtype=np.dtype(field.dtype),
                    **self._compression_kwargs(),
                )
            ray_dataset.resize(start + len(points), axis=0)
            if len(points):
                ray_dataset[start:] = ray_index
            self._check_capacity(ray_index.nbytes)

    def record_stream(
        self,
        kind: str,
        name: str,
        *,
        sample_id: int,
        step_index: int,
        sim_time_s: float,
        values: Mapping[str, Any],
        physics_step: int | None = None,
        direct_audit: Mapping[str, Any] | None = None,
        attack_profile: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append a producer update; return false only for the same previously seen sample."""
        stream = self.capture.stream(kind, name)
        identity = (kind, name)
        sample = int(sample_id)
        step = int(step_index)
        timestamp = float(sim_time_s)
        if sample < 0 or step < 0 or not np.isfinite(timestamp) or timestamp < 0:
            raise SubmissionError("stream sample axes must be finite and non-negative")
        prior = self._stream_last.get(identity)
        if prior is not None:
            if (sample, timestamp) == prior:
                return False
            if sample <= prior[0] or timestamp <= prior[1]:
                raise SubmissionError(f"producer timeline for {identity!r} is not strictly increasing")
        field_names = {field.name for field in stream.fields}
        if set(values) != field_names:
            raise SubmissionError(
                f"stream {identity!r} fields differ from contract: "
                f"missing={sorted(field_names - set(values))}, extra={sorted(set(values) - field_names)}"
            )
        group = self._stream_group(stream)
        if kind == "lidar":
            self._append_lidar_points(group, stream, values)
        for field in stream.fields:
            if kind == "lidar" and field.name in {"points", "ray_index"}:
                continue
            array = np.asarray(values[field.name])
            self._append_array(
                group,
                field.name,
                array,
                dtype=field.dtype,
                shape=field.shape,
            )
        self._append_scalar(group, "sample_id", sample, "uint64")
        self._append_scalar(group, "step_index", step, "uint64")
        self._append_scalar(group, "sim_time_s", timestamp, "float64")
        if self.contract.is_v4_3:
            physical = int(physics_step if physics_step is not None else -1)
            if physical < 1:
                raise SubmissionError("V4.3 producer samples require a positive physics_step")
            self._append_scalar(group, "physics_step", physical, "uint64")
            if kind == "lidar" and self.contract.direct_scoring and self.contract.direct_scoring.lidar_audit:
                self._append_direct_lidar_audit(group, values, direct_audit, attack_profile)
        self._stream_last[identity] = (sample, timestamp)
        if kind == "camera" and name == self.capture.video.camera:
            self._video.write(values["rgb"])
        return True

    def _append_direct_lidar_audit(
        self,
        group: h5py.Group,
        values: Mapping[str, Any],
        audit: Mapping[str, Any] | None,
        attack_profile: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(audit, Mapping) or not isinstance(attack_profile, Mapping):
            raise SubmissionError("V4.3 direct LiDAR sample lacks attack audit/profile")
        expected = {
            "attack_active": np.dtype(np.bool_),
            "attacked_point_count": np.dtype(np.uint64),
            "expected_phantom_count": np.dtype(np.uint64),
            "matched_phantom_count": np.dtype(np.uint64),
            "navigation_lidar_active": np.dtype(np.bool_),
            "raw_point_count": np.dtype(np.uint64),
            "raw_points": np.dtype(np.float32),
        }
        if set(audit) != set(expected):
            raise SubmissionError("V4.3 direct LiDAR audit fields differ from the frozen layout")
        raw_points = np.asarray(audit["raw_points"])
        if raw_points.dtype != np.float32 or raw_points.ndim != 2 or raw_points.shape[1:] != (3,):
            raise SubmissionError("V4.3 direct LiDAR raw_points must be float32[N,3]")
        if len(raw_points) > self.capture.max_lidar_points_per_frame or not np.isfinite(raw_points).all():
            raise SubmissionError("V4.3 direct LiDAR raw_points exceed limits or contain non-finite values")
        raw_dataset = group.get("raw_points")
        if raw_dataset is None:
            raw_dataset = self._create_dataset(
                group,
                "raw_points",
                shape=(0, 3),
                maxshape=(None, 3),
                chunks=(max(1, min(8192, self.capture.max_lidar_points_per_frame)), 3),
                dtype=np.float32,
                **self._compression_kwargs(),
            )
            self._create_dataset(
                group,
                "raw_frame_offsets",
                data=np.asarray([0], dtype=np.uint64),
                maxshape=(None,),
                chunks=True,
                **self._compression_kwargs(),
            )
        start = int(raw_dataset.shape[0])
        raw_dataset.resize(start + len(raw_points), axis=0)
        if len(raw_points):
            raw_dataset[start:] = raw_points
        offsets = group["raw_frame_offsets"]
        offsets.resize(offsets.shape[0] + 1, axis=0)
        offsets[-1] = start + len(raw_points)
        self._check_capacity(raw_points.nbytes + np.dtype(np.uint64).itemsize)
        for name, dtype in expected.items():
            if name == "raw_points":
                continue
            value = np.asarray(audit[name])
            if value.dtype != dtype or value.shape != ():
                raise SubmissionError(f"V4.3 direct LiDAR audit field {name!r} has invalid dtype/shape")
            self._append_array(group, name, value, dtype=dtype, shape=())
        if int(audit["raw_point_count"]) != len(raw_points):
            raise SubmissionError("V4.3 direct LiDAR raw_point_count differs from raw_points")
        if int(audit["attacked_point_count"]) != len(np.asarray(values["points"])):
            raise SubmissionError("V4.3 direct LiDAR attacked_point_count differs from points")
        normalized_profile = dict(attack_profile)
        if self._direct_lidar_profile is None:
            self._direct_lidar_profile = normalized_profile
        elif self._direct_lidar_profile != normalized_profile:
            raise SubmissionError("V4.3 direct LiDAR attack profile changed during capture")

    def record_direct_step(
        self,
        task_state: Mapping[str, Any],
        *,
        step_index: int,
        terminated: bool,
    ) -> None:
        """Append one V4.3 direct-scoring row after the matching environment step."""
        if not self.contract.is_v4_3 or self.contract.direct_scoring is None or self._demo is None:
            raise SubmissionError("direct scoring rows require a V4.3 capture session")
        if step_index != self._direct_steps or self._post_steps != self._direct_steps + 1:
            raise SubmissionError("V4.3 direct scoring row is not aligned with the post-step state")
        expected = {field.name for field in self.contract.direct_scoring.task_fields}
        if set(task_state) != expected:
            raise SubmissionError(
                f"V4.3 direct task fields differ from contract: missing={sorted(expected - set(task_state))}, "
                f"extra={sorted(set(task_state) - expected)}"
            )
        self._append_scalar(
            self._demo,
            "sim_time",
            (step_index + 1) * self.contract.control.control_dt,
            "float64",
        )
        self._append_scalar(self._demo, "terminated", bool(terminated), "bool")
        task_group = self._demo["task"]
        for field in self.contract.direct_scoring.task_fields:
            self._append_array(
                task_group,
                field.name,
                np.asarray(task_state[field.name], dtype=np.dtype(field.dtype)),
                dtype=field.dtype,
                shape=field.shape,
            )
        self._direct_steps += 1

    def complete(
        self,
        participant_result: Mapping[str, Any] | None = None,
        *,
        direct_final_state: Mapping[str, Any] | None = None,
        direct_events: list[Mapping[str, Any]] | None = None,
    ) -> None:
        if self._demo is None or self._file is None:
            raise SubmissionError("capture is closed")
        if not self._initial_recorded or self._actions < 1 or self._actions != self._post_steps:
            raise SubmissionError("V4.1 action/state alignment is incomplete")
        if not self._queries or self._queries[-1][0] + self._queries[-1][2] < self._actions:
            raise SubmissionError("policy query audit does not cover all V4.1 actions")
        missing_streams = {
            (stream.kind, stream.name) for stream in self.capture.streams
        } - set(self._stream_last)
        if missing_streams:
            raise SubmissionError(f"V4.1 capture is missing producer streams: {sorted(missing_streams)}")
        primary_identity = ("camera", self.capture.video.camera)
        primary_group = self._stream_groups[primary_identity]
        rgb_samples = int(primary_group["rgb"].shape[0])
        if self._video.frames != rgb_samples:
            raise SubmissionError("episode.mp4 frame count differs from primary camera RGB samples")
        if self.contract.is_v4_3:
            if self._direct_steps != self._actions or not isinstance(direct_final_state, Mapping):
                raise SubmissionError("V4.3 direct scoring rows/final state are incomplete")
            task_group = self._demo["task"]
            expected = {field.name for field in self.contract.direct_scoring.task_fields}
            if set(direct_final_state) != expected:
                raise SubmissionError("V4.3 final direct task fields differ from the contract")
            for field in self.contract.direct_scoring.task_fields:
                dataset = task_group[field.name]
                dataset[-1] = np.asarray(direct_final_state[field.name], dtype=np.dtype(field.dtype))
            encoded = [
                json.dumps(dict(event), ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
                for event in (direct_events or [])
            ]
            width = max([1, *(len(value) for value in encoded)])
            self._create_dataset(
                self._demo["events"],
                "jsonl",
                data=np.asarray(encoded, dtype=f"S{width}"),
                **self._compression_kwargs(),
            )
            metadata = self._file[METADATA_GROUP]
            metadata.attrs["direct_scoring_source"] = "participant_direct"
            metadata.attrs["direct_lidar_attack_profile_json"] = json.dumps(
                self._direct_lidar_profile or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        result_json = json.dumps(dict(participant_result or {}), ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(result_json.encode("utf-8")) > self.contract.limits.max_attribute_bytes:
            raise SubmissionError("participant_result_json exceeds max_attribute_bytes")
        self._demo.attrs["num_samples"] = np.int64(self._actions)
        self._demo.attrs["participant_result_json"] = result_json
        if "success" in (participant_result or {}):
            self._demo.attrs["success"] = np.bool_(participant_result["success"])
        self._file[METADATA_GROUP].attrs.modify("complete", np.bool_(True))
        self._file.flush()
        self._check_capacity()
        self._completed = True

    def close(self) -> None:
        video_error: Exception | None = None
        try:
            self._video.close()
        except Exception as error:
            video_error = error
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._demo = None
            descriptor = os.open(self.hdf5_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if video_error is not None:
            raise video_error

    def abort(self) -> None:
        self._video.abort()
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            finally:
                self._file = None
                self._demo = None

    def __enter__(self) -> "CaptureSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and self._completed:
            self.close()
        else:
            self.abort()


__all__ = ["CaptureSession", "FFmpegVideoWriter"]
