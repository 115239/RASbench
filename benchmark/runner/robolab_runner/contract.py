# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict public RoboLab competition contract version 4."""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from packaging.version import InvalidVersion, Version

from robolab_runner.common import load_json, sha256_file, sha256_json
from robolab_runner.errors import AssetError, ContractError

SCHEMA_NAME = "robolab-scoring"
SCHEMA_VERSION = 4
CONTRACT_VERSION = 4
LEGACY_FORMAT_REVISION = "4.0"
FORMAT_REVISION = "4.1"
COMPETITION_FORMAT_REVISION = "4.2"
DIRECT_SCORING_FORMAT_REVISION = "4.3"
SPOT_TASK_FAMILY = "spot_warehouse"
COMPETITION_TASK_FAMILIES = frozenset({"franka_manipulation", "go2_navigation", "spot_navigation"})
POLICY_PROTOCOL_NAME = "robolab-policy-zmq"
POLICY_PROTOCOL_VERSION = 1
COMPETITION_POLICY_PROTOCOL_VERSION = 2
ASSET_MANIFEST_SCHEMA = "robolab-competition-assets/v1"
REQUIRED_VERSIONS = {
    "robolab": "0.2.1",
    "isaaclab": "2.3.2.post1",
    "isaacsim": "5.1.0.0",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_PATH_RE = re.compile(
    r"^(?:image_obs|proprio_obs|sensor_obs)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$"
)
_CAPTURE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PRIVATE_OBSERVATION_SEGMENTS = frozenset(
    {
        "critic",
        "critic_obs",
        "hidden",
        "hidden_obs",
        "judge_obs",
        "judge_state",
        "privileged",
        "privileged_obs",
        "viewport_cam",
    }
)


def is_public_observation_path(path: str) -> bool:
    """Return whether a flat path is eligible for the participant policy wire."""
    return bool(_OBSERVATION_PATH_RE.fullmatch(path)) and not (
        PRIVATE_OBSERVATION_SEGMENTS & set(path.split("/"))
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ContractError(f"{path} fields differ from V4: missing={missing}, extra={extra}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{path} must be <= {maximum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{path} must be <= {maximum}")
    return result


def _digest(value: Any, path: str) -> str:
    result = _string(value, path).lower()
    if not _SHA256_RE.fullmatch(result):
        raise ContractError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _strings(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{path} must be a non-empty list")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ContractError(f"{path} contains duplicate values")
    return result


def _integers(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{path} must be a non-empty list")
    result = tuple(_integer(item, f"{path}[{index}]", minimum=0) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ContractError(f"{path} contains duplicate indices")
    return result


def _release(version: str) -> tuple[int, int, int] | None:
    try:
        release = Version(version).release
    except InvalidVersion:
        return None
    return tuple((release + (0, 0, 0))[:3])


def _safe_relative_path(raw_path: Any, path: str) -> str:
    relative = Path(_string(raw_path, path))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "." in relative.parts:
        raise ContractError(f"{path} must remain inside the competition asset root")
    normalized = relative.as_posix()
    if normalized != str(raw_path).replace("\\", "/"):
        raise ContractError(f"{path} must use a normalized POSIX relative path")
    return normalized


def _safe_asset_path(raw_path: Any, path: str) -> str:
    relative = _safe_relative_path(raw_path, path)
    if not relative.startswith("assets/"):
        raise ContractError(f"{path} must be under the competition assets/ tree")
    return relative


@dataclass(frozen=True)
class ResourceLimits:
    """Hostile-input ceilings enforced without launching Isaac Sim."""

    max_file_bytes: int
    max_steps: int
    max_hdf5_objects: int
    max_group_depth: int
    max_logical_bytes: int
    max_attribute_bytes: int
    allowed_compressions: tuple[str | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_file_bytes": self.max_file_bytes,
            "max_steps": self.max_steps,
            "max_hdf5_objects": self.max_hdf5_objects,
            "max_group_depth": self.max_group_depth,
            "max_logical_bytes": self.max_logical_bytes,
            "max_attribute_bytes": self.max_attribute_bytes,
            "allowed_compressions": list(self.allowed_compressions),
        }


@dataclass(frozen=True)
class ControlContract:
    """Frozen action and robot state meanings."""

    control_dt: float
    action_type: str
    action_dim: int
    state_joint_dim: int
    arm_joint_indices: tuple[int, ...]
    action_joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    ee_body_name: str | None
    primary_tracking_body: str | None

    def to_dict(self, *, format_revision: str = LEGACY_FORMAT_REVISION) -> dict[str, Any]:
        common = {
            "control_dt": self.control_dt,
            "action_type": self.action_type,
            "action_dim": self.action_dim,
            "state_joint_dim": self.state_joint_dim,
            "joint_names": list(self.joint_names),
        }
        if format_revision in {
            FORMAT_REVISION,
            COMPETITION_FORMAT_REVISION,
            DIRECT_SCORING_FORMAT_REVISION,
        }:
            return {**common, "primary_tracking_body": self.primary_tracking_body}
        return {
            **common,
            "arm_joint_indices": list(self.arm_joint_indices),
            "action_joint_indices": list(self.action_joint_indices),
            "ee_body_name": self.ee_body_name,
        }


@dataclass(frozen=True)
class ObservationLeaf:
    """One environment-dimension-free policy observation array."""

    path: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class PolicyLimits:
    connect_timeout_s: float
    request_timeout_s: float
    total_inference_timeout_s: float
    max_action_horizon: int
    max_message_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "connect_timeout_s": self.connect_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            "total_inference_timeout_s": self.total_inference_timeout_s,
            "max_action_horizon": self.max_action_horizon,
            "max_message_bytes": self.max_message_bytes,
        }


@dataclass(frozen=True)
class DriftTolerances:
    default_abs: float
    position_m: float
    linear_velocity_mps: float
    angular_velocity_rps: float
    joint_position_rad: float
    joint_velocity_radps: float
    quaternion_angle_rad: float

    def to_dict(self) -> dict[str, float]:
        return {
            "default_abs": self.default_abs,
            "position_m": self.position_m,
            "linear_velocity_mps": self.linear_velocity_mps,
            "angular_velocity_rps": self.angular_velocity_rps,
            "joint_position_rad": self.joint_position_rad,
            "joint_velocity_radps": self.joint_velocity_radps,
            "quaternion_angle_rad": self.quaternion_angle_rad,
        }


@dataclass(frozen=True)
class CaptureContract:
    all_scene_cameras: bool
    judge_viewport: bool
    configured_camera_modalities: bool
    rgb_hdf5: bool
    rgb_mp4: bool
    contacts: bool
    frame_transformers: bool
    optional_imu_lidar: bool
    bbox: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "all_scene_cameras": self.all_scene_cameras,
            "judge_viewport": self.judge_viewport,
            "configured_camera_modalities": self.configured_camera_modalities,
            "rgb_hdf5": self.rgb_hdf5,
            "rgb_mp4": self.rgb_mp4,
            "contacts": self.contacts,
            "frame_transformers": self.frame_transformers,
            "optional_imu_lidar": self.optional_imu_lidar,
            "bbox": self.bbox,
        }


@dataclass(frozen=True)
class CaptureField:
    """One contract-frozen producer field, excluding the common sample time axes."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    unit: str
    coordinate_frame: str
    scale_to_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "coordinate_frame": self.coordinate_frame,
        }
        if self.scale_to_m is not None:
            value["scale_to_m"] = self.scale_to_m
        return value


@dataclass(frozen=True)
class CaptureStream:
    """One independently sampled V4.1 producer stream."""

    kind: str
    name: str
    source: str
    rate_hz: float
    coordinate_frame: str
    entity_order: tuple[str, ...]
    fields: tuple[CaptureField, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "source": self.source,
            "rate_hz": self.rate_hz,
            "timestamp_source": "producer",
            "coordinate_frame": self.coordinate_frame,
            "entity_order": list(self.entity_order),
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class VideoContract:
    """The single participant-side V4.1 video binding."""

    camera: str
    codec: str
    pixel_format: str
    fps: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class RunnerCaptureContract:
    """Complete participant evidence contract introduced by format revision 4.1."""

    physics_dt: float
    render_hz: float
    coordinate_convention: str
    quaternion_order: str
    compression: str
    compression_level: int
    flush_interval_samples: int
    max_lidar_points_per_frame: int
    video: VideoContract
    streams: tuple[CaptureStream, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics_dt": self.physics_dt,
            "render_hz": self.render_hz,
            "coordinate_convention": self.coordinate_convention,
            "quaternion_order": self.quaternion_order,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "flush_interval_samples": self.flush_interval_samples,
            "max_lidar_points_per_frame": self.max_lidar_points_per_frame,
            "video": self.video.to_dict(),
            "streams": [stream.to_dict() for stream in self.streams],
        }

    def stream(self, kind: str, name: str) -> CaptureStream:
        matches = [stream for stream in self.streams if stream.kind == kind and stream.name == name]
        if len(matches) != 1:
            raise ContractError(f"runner_capture has no unique {(kind, name)!r} stream")
        return matches[0]

    @property
    def entity_spec(self) -> dict[str, Any]:
        return {
            f"{stream.kind}/{stream.name}": list(stream.entity_order)
            for stream in self.streams
        }


_CAPTURE_KINDS = frozenset({"articulation", "camera", "lidar", "imu", "contact"})
_REQUIRED_CAPTURE_FIELDS = {
    "articulation": frozenset(
        {
            "joint_position",
            "joint_velocity",
            "joint_position_target",
            "joint_velocity_target",
            "joint_effort_target",
            "computed_torque",
            "applied_torque",
            "body_pose",
            "body_velocity",
        }
    ),
    "camera": frozenset({"position", "orientation", "intrinsics", "rgb", "depth", "valid"}),
    "lidar": frozenset({"position", "orientation", "points"}),
    "imu": frozenset(
        {"position", "orientation", "linear_velocity", "angular_velocity", "linear_acceleration"}
    ),
    "contact": frozenset({"net_forces_w"}),
}


def _parse_runner_capture(value: Any, *, format_revision: str) -> RunnerCaptureContract:
    capture = _mapping(value, "runner_capture")
    _exact_keys(
        capture,
        "runner_capture",
        {
            "physics_dt",
            "render_hz",
            "coordinate_convention",
            "quaternion_order",
            "compression",
            "compression_level",
            "flush_interval_samples",
            "max_lidar_points_per_frame",
            "video",
            "streams",
        },
    )
    physics_dt = _number(capture.get("physics_dt"), "runner_capture.physics_dt", minimum=1e-6)
    render_hz = _number(capture.get("render_hz"), "runner_capture.render_hz", minimum=1.0)
    if format_revision == FORMAT_REVISION and not math.isclose(physics_dt, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("V4.1 Spot runner_capture.physics_dt must be 0.005")
    if format_revision == FORMAT_REVISION and not math.isclose(render_hz, 25.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("V4.1 Spot runner_capture.render_hz must be 25")
    coordinate_convention = _string(
        capture.get("coordinate_convention"), "runner_capture.coordinate_convention"
    )
    if coordinate_convention != "right_handed_z_up":
        raise ContractError("runner_capture.coordinate_convention must be right_handed_z_up")
    quaternion_order = _string(capture.get("quaternion_order"), "runner_capture.quaternion_order")
    if quaternion_order != "wxyz":
        raise ContractError("runner_capture.quaternion_order must be wxyz")
    compression = _string(capture.get("compression"), "runner_capture.compression")
    if compression != "gzip":
        raise ContractError("V4.1 capture compression must be lossless gzip")
    compression_level = _integer(
        capture.get("compression_level"), "runner_capture.compression_level", minimum=1, maximum=9
    )

    video_raw = _mapping(capture.get("video"), "runner_capture.video")
    _exact_keys(video_raw, "runner_capture.video", {"camera", "codec", "pixel_format", "fps", "width", "height"})
    video = VideoContract(
        camera=_string(video_raw.get("camera"), "runner_capture.video.camera"),
        codec=_string(video_raw.get("codec"), "runner_capture.video.codec"),
        pixel_format=_string(video_raw.get("pixel_format"), "runner_capture.video.pixel_format"),
        fps=_number(video_raw.get("fps"), "runner_capture.video.fps", minimum=1.0),
        width=_integer(video_raw.get("width"), "runner_capture.video.width", minimum=1, maximum=8192),
        height=_integer(video_raw.get("height"), "runner_capture.video.height", minimum=1, maximum=8192),
    )
    if (video.codec, video.pixel_format) != ("h264", "yuv420p"):
        raise ContractError("full-capture episode.mp4 must use H.264/yuv420p")
    if format_revision == FORMAT_REVISION and not math.isclose(video.fps, 10.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("V4.1 episode.mp4 must use 10 fps")
    if video.width % 2 or video.height % 2:
        raise ContractError("yuv420p video dimensions must be even")

    streams_raw = capture.get("streams")
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ContractError("runner_capture.streams must be a non-empty list")
    streams: list[CaptureStream] = []
    identities: set[tuple[str, str]] = set()
    for stream_index, raw_stream in enumerate(streams_raw):
        path = f"runner_capture.streams[{stream_index}]"
        stream_value = _mapping(raw_stream, path)
        _exact_keys(
            stream_value,
            path,
            {
                "kind",
                "name",
                "source",
                "rate_hz",
                "timestamp_source",
                "coordinate_frame",
                "entity_order",
                "fields",
            },
        )
        kind = _string(stream_value.get("kind"), f"{path}.kind")
        name = _string(stream_value.get("name"), f"{path}.name")
        if kind not in _CAPTURE_KINDS or not _CAPTURE_NAME_RE.fullmatch(name):
            raise ContractError(f"{path} has an unsupported kind or unsafe name")
        identity = (kind, name)
        if identity in identities:
            raise ContractError(f"duplicate runner capture stream: {identity!r}")
        identities.add(identity)
        if stream_value.get("timestamp_source") != "producer":
            raise ContractError(f"{path}.timestamp_source must be producer")
        entity_order = _strings(stream_value.get("entity_order"), f"{path}.entity_order")
        fields_raw = stream_value.get("fields")
        if not isinstance(fields_raw, list) or not fields_raw:
            raise ContractError(f"{path}.fields must be a non-empty list")
        fields: list[CaptureField] = []
        field_names: set[str] = set()
        for field_index, raw_field in enumerate(fields_raw):
            field_path = f"{path}.fields[{field_index}]"
            field_value = _mapping(raw_field, field_path)
            field_keys = {"name", "dtype", "shape", "unit", "coordinate_frame"}
            if "scale_to_m" in field_value:
                field_keys.add("scale_to_m")
            _exact_keys(field_value, field_path, field_keys)
            field_name = _string(field_value.get("name"), f"{field_path}.name")
            if not _CAPTURE_NAME_RE.fullmatch(field_name) or field_name in field_names:
                raise ContractError(f"{field_path}.name is unsafe or duplicated")
            field_names.add(field_name)
            dtype_text = _string(field_value.get("dtype"), f"{field_path}.dtype")
            try:
                dtype = np.dtype(dtype_text)
            except TypeError as error:
                raise ContractError(f"{field_path}.dtype is invalid") from error
            if dtype.name != dtype_text or dtype.hasobject or dtype.fields or dtype.kind not in {"b", "i", "u", "f"}:
                raise ContractError(f"{field_path}.dtype must be a canonical numeric dtype")
            raw_shape = field_value.get("shape")
            if not isinstance(raw_shape, list) or len(raw_shape) > 8:
                raise ContractError(f"{field_path}.shape must be a list of at most 8 dimensions")
            shape = tuple(
                _integer(dimension, f"{field_path}.shape[{index}]", minimum=1)
                for index, dimension in enumerate(raw_shape)
            )
            fields.append(
                CaptureField(
                    name=field_name,
                    dtype=dtype.name,
                    shape=shape,
                    unit=_string(field_value.get("unit"), f"{field_path}.unit"),
                    coordinate_frame=_string(
                        field_value.get("coordinate_frame"), f"{field_path}.coordinate_frame"
                    ),
                    scale_to_m=(
                        _number(field_value.get("scale_to_m"), f"{field_path}.scale_to_m", minimum=1e-12)
                        if "scale_to_m" in field_value
                        else None
                    ),
                )
            )
        if tuple(field.name for field in fields) != tuple(sorted(field.name for field in fields)):
            raise ContractError(f"{path}.fields must be sorted by name")
        required_fields = _REQUIRED_CAPTURE_FIELDS[kind]
        if not required_fields <= field_names:
            raise ContractError(f"{path}.fields is missing {sorted(required_fields - field_names)}")
        if format_revision == FORMAT_REVISION and kind != "contact" and field_names - required_fields - {"ray_index"}:
            raise ContractError(f"{path}.fields contains unsupported fields {sorted(field_names - required_fields)}")
        if kind == "lidar" and "intensity" in field_names:
            raise ContractError("LiDAR intensity cannot be declared when the producer has no real intensity")
        rate_hz = _number(stream_value.get("rate_hz"), f"{path}.rate_hz", minimum=0.001)
        if format_revision == FORMAT_REVISION:
            expected_rate = 200.0 if kind in {"articulation", "imu", "contact"} else 10.0
            if not math.isclose(rate_hz, expected_rate, rel_tol=0.0, abs_tol=1e-9):
                raise ContractError(f"V4.1 Spot {kind} rate_hz must be {expected_rate:g}")
        streams.append(
            CaptureStream(
                kind=kind,
                name=name,
                source=_string(stream_value.get("source"), f"{path}.source"),
                rate_hz=rate_hz,
                coordinate_frame=_string(stream_value.get("coordinate_frame"), f"{path}.coordinate_frame"),
                entity_order=entity_order,
                fields=tuple(fields),
            )
        )
    if tuple((stream.kind, stream.name) for stream in streams) != tuple(
        sorted((stream.kind, stream.name) for stream in streams)
    ):
        raise ContractError("runner_capture.streams must be sorted by kind/name")
    kinds = {stream.kind for stream in streams}
    if format_revision == FORMAT_REVISION and kinds != _CAPTURE_KINDS:
        raise ContractError(f"runner_capture must contain all stream kinds, got {sorted(kinds)}")
    if format_revision in {COMPETITION_FORMAT_REVISION, DIRECT_SCORING_FORMAT_REVISION} and not {
        "articulation",
        "camera",
        "contact",
    } <= kinds:
        raise ContractError("competition runner_capture must contain articulation, camera, and contact streams")
    camera_matches = [stream for stream in streams if stream.kind == "camera" and stream.name == video.camera]
    if len(camera_matches) != 1:
        raise ContractError("runner_capture.video.camera must name exactly one camera stream")
    camera_fields = {field.name: field for field in camera_matches[0].fields}
    if camera_fields["rgb"].dtype != "uint8" or camera_fields["rgb"].shape != (
        video.height,
        video.width,
        3,
    ):
        raise ContractError("primary camera RGB shape/dtype differs from the video contract")
    if camera_fields["depth"].dtype != "uint16" or camera_fields["valid"].dtype != "bool":
        raise ContractError("V4.1 camera depth/valid must use uint16 millimetres and bool")
    if camera_fields["depth"].unit != "mm" or not math.isclose(
        camera_fields["depth"].scale_to_m or math.nan,
        0.001,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError("V4.1 camera depth must declare millimetres with scale_to_m=0.001")
    for stream in streams:
        for capture_field in stream.fields:
            if capture_field.scale_to_m is not None and not (
                stream.kind == "camera" and capture_field.name == "depth"
            ):
                raise ContractError("scale_to_m is permitted only on V4.1 camera depth fields")
    return RunnerCaptureContract(
        physics_dt=physics_dt,
        render_hz=render_hz,
        coordinate_convention=coordinate_convention,
        quaternion_order=quaternion_order,
        compression=compression,
        compression_level=compression_level,
        flush_interval_samples=_integer(
            capture.get("flush_interval_samples"),
            "runner_capture.flush_interval_samples",
            minimum=1,
            maximum=100_000,
        ),
        max_lidar_points_per_frame=_integer(
            capture.get("max_lidar_points_per_frame"),
            "runner_capture.max_lidar_points_per_frame",
            minimum=1,
            maximum=10_000_000,
        ),
        video=video,
        streams=tuple(streams),
    )


@dataclass(frozen=True)
class DirectScoringField:
    """One control-step-aligned task fact required for direct scoring."""

    name: str
    dtype: str
    shape: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, value: Any, *, index: int) -> "DirectScoringField":
        path = f"direct_scoring.task_fields[{index}]"
        raw = _mapping(value, path)
        _exact_keys(raw, path, {"name", "dtype", "shape"})
        name = _string(raw.get("name"), f"{path}.name")
        if not _CAPTURE_NAME_RE.fullmatch(name) or "/" in name:
            raise ContractError(f"direct_scoring task field name is not portable: {name!r}")
        dtype = _string(raw.get("dtype"), f"{path}.dtype")
        if dtype not in {"bool", "int64", "float64"}:
            raise ContractError("direct_scoring task field dtype must be bool, int64, or float64")
        raw_shape = raw.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) > 4:
            raise ContractError("direct_scoring task field shape must be a list of at most four dimensions")
        shape = tuple(
            _integer(dim, f"{path}.shape[{axis}]", minimum=1) for axis, dim in enumerate(raw_shape)
        )
        return cls(name=name, dtype=dtype, shape=shape)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class DirectScoringContract:
    """Strict Runner-owned evidence layout accepted by the V4.3 no-replay path."""

    engine: str
    task_fields: tuple[DirectScoringField, ...]
    hard_gate_paths: tuple[str, ...]
    control_step_axis: str
    physics_step_axis: str
    termination_rule: str
    lidar_profile_binding: str
    require_events: bool
    require_official_termination: bool
    require_physics_step: bool
    lidar_audit: bool
    schema: str = "robolab-direct-scoring/v1"

    @classmethod
    def from_dict(cls, value: Any) -> "DirectScoringContract":
        raw = _mapping(value, "direct_scoring")
        _exact_keys(
            raw,
            "direct_scoring",
            {
                "schema",
                "engine",
                "task_fields",
                "hard_gate_paths",
                "control_step_axis",
                "physics_step_axis",
                "termination_rule",
                "lidar_profile_binding",
                "require_events",
                "require_official_termination",
                "require_physics_step",
                "lidar_audit",
            },
        )
        schema = _string(raw.get("schema"), "direct_scoring.schema")
        if schema != "robolab-direct-scoring/v1":
            raise ContractError(f"unsupported direct_scoring schema {schema!r}")
        engine = _string(raw.get("engine"), "direct_scoring.engine")
        if engine not in {"task_scoring_engine", "competition_task_state"}:
            raise ContractError(f"unsupported direct_scoring engine {engine!r}")
        raw_fields = raw.get("task_fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ContractError("direct_scoring.task_fields must be a non-empty list")
        fields = tuple(DirectScoringField.from_dict(item, index=index) for index, item in enumerate(raw_fields))
        names = tuple(field.name for field in fields)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ContractError("direct_scoring.task_fields must be sorted and unique by name")
        expected_base = {
            "completed": "int64",
            "score": "float64",
            "status": "int64",
            "success": "bool",
            "total": "int64",
        }
        actual_base = {field.name: field.dtype for field in fields if field.name in expected_base}
        if actual_base != expected_base or any(field.shape for field in fields if field.name in expected_base):
            raise ContractError("direct_scoring.task_fields has an invalid required base layout")
        raw_gate_paths = raw.get("hard_gate_paths")
        if not isinstance(raw_gate_paths, list):
            raise ContractError("direct_scoring.hard_gate_paths must be a list")
        hard_gate_paths = tuple(
            _string(item, f"direct_scoring.hard_gate_paths[{index}]")
            for index, item in enumerate(raw_gate_paths)
        )
        if len(hard_gate_paths) != len(set(hard_gate_paths)) or any(
            not path.startswith("/data/demo_0/task/") or ".." in Path(path).parts
            for path in hard_gate_paths
        ):
            raise ContractError("direct_scoring.hard_gate_paths must be unique safe task evidence paths")
        literal_fields = {
            "control_step_axis": "sim_time",
            "physics_step_axis": "physics_step",
            "termination_rule": "final_true_or_exact_max_steps",
        }
        for name, expected in literal_fields.items():
            if raw.get(name) != expected:
                raise ContractError(f"direct_scoring.{name} must be {expected!r}")
        lidar_profile_binding = _string(raw.get("lidar_profile_binding"), "direct_scoring.lidar_profile_binding")
        if lidar_profile_binding not in {"none", "scenario_sealed_profile"}:
            raise ContractError("direct_scoring.lidar_profile_binding is unsupported")
        booleans: dict[str, bool] = {}
        for name in ("require_events", "require_official_termination", "require_physics_step", "lidar_audit"):
            item = raw.get(name)
            if not isinstance(item, bool):
                raise ContractError(f"direct_scoring.{name} must be boolean")
            booleans[name] = item
        if not all(booleans[name] for name in ("require_events", "require_official_termination", "require_physics_step")):
            raise ContractError("V4.3 direct scoring requires events, official termination, and physics-step axes")
        if booleans["lidar_audit"] != (lidar_profile_binding == "scenario_sealed_profile"):
            raise ContractError("direct_scoring LiDAR audit and profile binding differ")
        return cls(
            engine=engine,
            task_fields=fields,
            hard_gate_paths=hard_gate_paths,
            control_step_axis="sim_time",
            physics_step_axis="physics_step",
            termination_rule="final_true_or_exact_max_steps",
            lidar_profile_binding=lidar_profile_binding,
            schema=schema,
            **booleans,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "engine": self.engine,
            "task_fields": [field.to_dict() for field in self.task_fields],
            "hard_gate_paths": list(self.hard_gate_paths),
            "control_step_axis": self.control_step_axis,
            "physics_step_axis": self.physics_step_axis,
            "termination_rule": self.termination_rule,
            "lidar_profile_binding": self.lidar_profile_binding,
            "require_events": self.require_events,
            "require_official_termination": self.require_official_termination,
            "require_physics_step": self.require_physics_step,
            "lidar_audit": self.lidar_audit,
        }


@dataclass(frozen=True)
class V4TaskContract:
    """One immutable V4 task/seed contract shared by Runner and Judge."""

    contract_id: str
    task_env: str
    task_class: str
    instruction: str
    seed: int
    versions: Mapping[str, str]
    environment_config: Mapping[str, Any]
    environment_config_sha256: str
    asset_package_id: str
    asset_files: Mapping[str, str]
    asset_manifest_sha256: str
    asset_archive_sha256: str
    control: ControlContract
    required_initial_state_paths: tuple[str, ...]
    required_state_paths: tuple[str, ...]
    required_ee_pose_paths: tuple[str, ...]
    limits: ResourceLimits
    observation_leaves: tuple[ObservationLeaf, ...]
    policy_limits: PolicyLimits
    drift_tolerances: DriftTolerances
    capture: CaptureContract
    rule_package_id: str
    rule_package_sha256: str
    registry_manifest_sha256: str
    hard_gate_count: int
    format_revision: str = LEGACY_FORMAT_REVISION
    task_family: str | None = None
    runner_capture: RunnerCaptureContract | None = None
    scenario_id: str | None = None
    scenario_sha256: str | None = None
    scenario_variant: str | None = None
    scenario_schedule_seed: int | None = None
    direct_scoring: DirectScoringContract | None = None
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION
    contract_version: int = CONTRACT_VERSION
    source_path: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "V4TaskContract":
        source = Path(path).expanduser().resolve()
        try:
            value = load_json(source)
        except Exception as error:
            if isinstance(error, ContractError):
                raise
            raise ContractError(f"failed to load V4 task contract {source}: {error}") from error
        return cls.from_dict(value, source_path=source)

    @classmethod
    def from_dict(cls, value: Any, *, source_path: str | Path | None = None) -> "V4TaskContract":
        root = _mapping(value, "contract")
        format_revision = root.get("format_revision", LEGACY_FORMAT_REVISION)
        supported_revisions = {
            LEGACY_FORMAT_REVISION,
            FORMAT_REVISION,
            COMPETITION_FORMAT_REVISION,
            DIRECT_SCORING_FORMAT_REVISION,
        }
        if format_revision not in supported_revisions:
            raise ContractError(f"unsupported format_revision {format_revision!r}")
        root_keys = {
            "contract_version",
            "contract_id",
            "schema",
            "versions",
            "task",
            "environment",
            "assets",
            "control",
            "required_paths",
            "limits",
            "policy",
            "replay",
            "rules",
        }
        competition_revisions = {COMPETITION_FORMAT_REVISION, DIRECT_SCORING_FORMAT_REVISION}
        full_capture_revisions = {FORMAT_REVISION, *competition_revisions}
        if format_revision in full_capture_revisions:
            root_keys.update({"format_revision", "runner_capture"})
        if format_revision in competition_revisions:
            root_keys.add("scenario")
        if format_revision == DIRECT_SCORING_FORMAT_REVISION:
            root_keys.add("direct_scoring")
        _exact_keys(
            root,
            "contract",
            root_keys,
        )
        version = _integer(root.get("contract_version"), "contract_version", minimum=1)
        if version != CONTRACT_VERSION:
            raise ContractError(
                f"unsupported contract_version {version}; V4 requires 4 and does not convert V3 contracts"
            )
        schema = _mapping(root.get("schema"), "schema")
        _exact_keys(schema, "schema", {"name", "version"})
        schema_name = _string(schema.get("name"), "schema.name")
        schema_version = _integer(schema.get("version"), "schema.version", minimum=1)
        if (schema_name, schema_version) != (SCHEMA_NAME, SCHEMA_VERSION):
            raise ContractError(
                f"unsupported submission schema {schema_name!r} v{schema_version}; V4 requires robolab-scoring v4"
            )

        task = _mapping(root.get("task"), "task")
        task_keys = {"task_env", "task_class", "instruction", "seed"}
        if format_revision in full_capture_revisions:
            task_keys.add("task_family")
        _exact_keys(task, "task", task_keys)
        task_env = _string(task.get("task_env"), "task.task_env")
        task_class = _string(task.get("task_class"), "task.task_class")
        instruction = _string(task.get("instruction"), "task.instruction")
        seed = _integer(task.get("seed"), "task.seed", minimum=0)
        task_family = None
        if format_revision in full_capture_revisions:
            task_family = _string(task.get("task_family"), "task.task_family")
            if format_revision == FORMAT_REVISION and task_family != SPOT_TASK_FAMILY:
                raise ContractError(f"V4.1 task_family must be {SPOT_TASK_FAMILY!r}")
            if format_revision in competition_revisions and task_family not in COMPETITION_TASK_FAMILIES:
                raise ContractError(f"unsupported competition task_family {task_family!r}")

        scenario_id = None
        scenario_sha256 = None
        scenario_variant = None
        scenario_schedule_seed = None
        if format_revision in competition_revisions:
            scenario = _mapping(root.get("scenario"), "scenario")
            _exact_keys(scenario, "scenario", {"id", "sha256", "variant", "schedule_seed"})
            scenario_id = _string(scenario.get("id"), "scenario.id")
            scenario_sha256 = _digest(scenario.get("sha256"), "scenario.sha256")
            scenario_variant = _string(scenario.get("variant"), "scenario.variant")
            if scenario_variant not in {"base", "attack"}:
                raise ContractError("scenario.variant must be base or attack")
            raw_schedule_seed = scenario.get("schedule_seed")
            if raw_schedule_seed is not None:
                scenario_schedule_seed = _integer(raw_schedule_seed, "scenario.schedule_seed", minimum=0)
            if (scenario_variant == "base") != (scenario_schedule_seed is None):
                raise ContractError("base scenarios require a null schedule seed; attack scenarios require a seed")

        versions_raw = _mapping(root.get("versions"), "versions")
        _exact_keys(versions_raw, "versions", {"robolab", "isaaclab", "isaacsim"})
        versions = {
            package: _string(versions_raw.get(package), f"versions.{package}")
            for package in ("robolab", "isaaclab", "isaacsim")
        }
        for package, required in REQUIRED_VERSIONS.items():
            actual = versions[package]
            matches = _release(actual) == _release(required) if package == "isaacsim" else actual == required
            if not matches:
                raise ContractError(f"V4 requires {package} {required}, got {actual}")

        environment = _mapping(root.get("environment"), "environment")
        _exact_keys(environment, "environment", {"normalized_config", "sha256"})
        config = dict(_mapping(environment.get("normalized_config"), "environment.normalized_config"))
        if config.get("instruction") != instruction:
            raise ContractError("task.instruction differs from environment.normalized_config instruction")
        config_hash = _digest(environment.get("sha256"), "environment.sha256")
        if sha256_json(config) != config_hash:
            raise ContractError("environment.sha256 does not match normalized_config")

        assets_raw = _mapping(root.get("assets"), "assets")
        _exact_keys(
            assets_raw,
            "assets",
            {"package_id", "files", "manifest_sha256", "archive_sha256"},
        )
        asset_values = _mapping(assets_raw.get("files"), "assets.files")
        if not asset_values:
            raise ContractError("assets.files cannot be empty")
        assets: dict[str, str] = {}
        for raw_path, raw_digest in asset_values.items():
            relative = _safe_asset_path(raw_path, "assets.files key")
            assets[relative] = _digest(raw_digest, f"assets.files[{relative!r}]")
        assets = dict(sorted(assets.items()))

        control_raw = _mapping(root.get("control"), "control")
        control_keys = {"control_dt", "action_type", "action_dim", "state_joint_dim", "joint_names"}
        if format_revision in full_capture_revisions:
            control_keys.add("primary_tracking_body")
        else:
            control_keys.update({"arm_joint_indices", "action_joint_indices", "ee_body_name"})
        _exact_keys(control_raw, "control", control_keys)
        action_type = _string(control_raw.get("action_type"), "control.action_type")
        expected_action_types = (
            {"base_velocity"}
            if format_revision == FORMAT_REVISION
            else {"joint_position", "base_velocity"}
            if format_revision in competition_revisions
            else {"joint_position"}
        )
        if action_type not in expected_action_types:
            raise ContractError(
                f"V4 {format_revision} requires one of {sorted(expected_action_types)}, got {action_type!r}"
            )
        arm_indices = ()
        action_indices = ()
        if format_revision == LEGACY_FORMAT_REVISION:
            arm_indices = _integers(control_raw.get("arm_joint_indices"), "control.arm_joint_indices")
            action_indices = _integers(control_raw.get("action_joint_indices"), "control.action_joint_indices")
            if len(arm_indices) != len(action_indices):
                raise ContractError("control arm/action joint mappings must have equal length")
        action_dim = _integer(control_raw.get("action_dim"), "control.action_dim", minimum=1)
        state_joint_dim = _integer(control_raw.get("state_joint_dim"), "control.state_joint_dim", minimum=1)
        if action_indices and (max(action_indices) >= action_dim or max(arm_indices) >= state_joint_dim):
            raise ContractError("control joint mapping index is out of range")
        if format_revision == FORMAT_REVISION and action_dim != 3:
            raise ContractError("V4.1 Spot base_velocity actions must be [vx_mps, vy_mps, yaw_rate_rps]")
        joint_names = _strings(control_raw.get("joint_names"), "control.joint_names")
        if len(joint_names) != state_joint_dim:
            raise ContractError("control.joint_names length must equal state_joint_dim")
        control = ControlContract(
            control_dt=_number(control_raw.get("control_dt"), "control.control_dt", minimum=1e-12),
            action_type=action_type,
            action_dim=action_dim,
            state_joint_dim=state_joint_dim,
            arm_joint_indices=arm_indices,
            action_joint_indices=action_indices,
            joint_names=joint_names,
            ee_body_name=(
                _string(control_raw.get("ee_body_name"), "control.ee_body_name")
                if format_revision == LEGACY_FORMAT_REVISION
                else None
            ),
            primary_tracking_body=(
                _string(control_raw.get("primary_tracking_body"), "control.primary_tracking_body")
                if format_revision in full_capture_revisions
                else None
            ),
        )

        required = _mapping(root.get("required_paths"), "required_paths")
        tracking_key = (
            "tracked_pose" if format_revision in full_capture_revisions else "ee_pose"
        )
        _exact_keys(required, "required_paths", {"initial_state", "states", tracking_key})
        initial_paths = _strings(required.get("initial_state"), "required_paths.initial_state")
        state_paths = _strings(required.get("states"), "required_paths.states")
        ee_paths = _strings(required.get(tracking_key), f"required_paths.{tracking_key}")
        for prefix, paths in (
            ("initial_state/", initial_paths),
            ("states/", state_paths),
            (f"{tracking_key}/", ee_paths),
        ):
            if any(not path.startswith(prefix) or ".." in Path(path).parts for path in paths):
                raise ContractError(f"every required path must start with {prefix!r} and remain relative")
        if {path.removeprefix("initial_state/") for path in initial_paths} != {
            path.removeprefix("states/") for path in state_paths
        }:
            raise ContractError("initial_state and states must describe the same dynamic leaves")
        for suffix in ("position", "orientation", "linear_velocity", "angular_velocity"):
            path = f"{tracking_key}/{suffix}"
            if path not in ee_paths:
                raise ContractError(f"required_paths.{tracking_key} is missing {path!r}")

        limits_raw = _mapping(root.get("limits"), "limits")
        _exact_keys(
            limits_raw,
            "limits",
            {
                "max_file_bytes",
                "max_steps",
                "max_hdf5_objects",
                "max_group_depth",
                "max_logical_bytes",
                "max_attribute_bytes",
                "allowed_compressions",
            },
        )
        compression_values = limits_raw.get("allowed_compressions")
        if not isinstance(compression_values, list) or not compression_values:
            raise ContractError("limits.allowed_compressions must be a non-empty list")
        compressions: list[str | None] = []
        for compression in compression_values:
            if compression not in {None, "gzip", "lzf"}:
                raise ContractError("allowed compression must be null, gzip, or lzf")
            if compression not in compressions:
                compressions.append(compression)
        limits = ResourceLimits(
            max_file_bytes=_integer(limits_raw.get("max_file_bytes"), "limits.max_file_bytes", minimum=1),
            max_steps=_integer(limits_raw.get("max_steps"), "limits.max_steps", minimum=1),
            max_hdf5_objects=_integer(
                limits_raw.get("max_hdf5_objects"), "limits.max_hdf5_objects", minimum=1
            ),
            max_group_depth=_integer(limits_raw.get("max_group_depth"), "limits.max_group_depth", minimum=1),
            max_logical_bytes=_integer(
                limits_raw.get("max_logical_bytes"), "limits.max_logical_bytes", minimum=1
            ),
            max_attribute_bytes=_integer(
                limits_raw.get("max_attribute_bytes"), "limits.max_attribute_bytes", minimum=1
            ),
            allowed_compressions=tuple(compressions),
        )
        runner_capture = None
        if format_revision in full_capture_revisions:
            runner_capture = _parse_runner_capture(root.get("runner_capture"), format_revision=format_revision)
        if format_revision == FORMAT_REVISION:
            if not math.isclose(control.control_dt, 0.1, rel_tol=0.0, abs_tol=1e-12):
                raise ContractError("V4.1 Spot control_dt must be 0.1 seconds (10 Hz)")
            if (
                limits.max_file_bytes != 12_000_000_000
                or limits.max_logical_bytes != 12_000_000_000
                or limits.max_hdf5_objects != 2048
            ):
                raise ContractError("V4.1 Spot limits must freeze 12 GB file/logical bytes and 2048 HDF5 objects")
            if limits.max_steps != 1200:
                raise ContractError("V4.1 Spot max_steps must be 1200 (120 seconds at 10 Hz)")
            if "gzip" not in limits.allowed_compressions:
                raise ContractError("V4.1 Spot limits must allow gzip compression")
            configured_dt = config.get("physics_dt")
            if not isinstance(configured_dt, (int, float)) or not math.isclose(
                float(configured_dt), runner_capture.physics_dt, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ContractError("environment.normalized_config physics_dt differs from runner_capture")

        policy = _mapping(root.get("policy"), "policy")
        _exact_keys(policy, "policy", {"protocol", "observation", "limits"})
        protocol = _mapping(policy.get("protocol"), "policy.protocol")
        _exact_keys(protocol, "policy.protocol", {"name", "version"})
        if _string(protocol.get("name"), "policy.protocol.name") != POLICY_PROTOCOL_NAME:
            raise ContractError(f"policy.protocol.name must be {POLICY_PROTOCOL_NAME!r}")
        expected_protocol_version = (
            COMPETITION_POLICY_PROTOCOL_VERSION
            if format_revision in competition_revisions
            else POLICY_PROTOCOL_VERSION
        )
        if _integer(protocol.get("version"), "policy.protocol.version", minimum=1) != expected_protocol_version:
            raise ContractError(f"policy.protocol.version must be {expected_protocol_version}")
        observation = _mapping(policy.get("observation"), "policy.observation")
        _exact_keys(observation, "policy.observation", {"leaves"})
        raw_leaves = observation.get("leaves")
        if not isinstance(raw_leaves, list) or not raw_leaves:
            raise ContractError("policy.observation.leaves must be a non-empty list")
        leaves: list[ObservationLeaf] = []
        seen_paths: set[str] = set()
        for index, raw_leaf in enumerate(raw_leaves):
            leaf = _mapping(raw_leaf, f"policy.observation.leaves[{index}]")
            _exact_keys(
                leaf,
                f"policy.observation.leaves[{index}]",
                {"path", "dtype", "shape"},
            )
            leaf_path = _string(leaf.get("path"), f"policy.observation.leaves[{index}].path")
            if not is_public_observation_path(leaf_path):
                raise ContractError(f"policy observation path is not public: {leaf_path!r}")
            if leaf_path in seen_paths:
                raise ContractError(f"duplicate policy observation path: {leaf_path!r}")
            seen_paths.add(leaf_path)
            dtype_text = _string(leaf.get("dtype"), f"policy.observation.leaves[{index}].dtype")
            try:
                dtype = np.dtype(dtype_text)
            except TypeError as error:
                raise ContractError(f"invalid observation dtype {dtype_text!r}") from error
            if dtype.name != dtype_text or dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
                raise ContractError(f"observation dtype must be a canonical numeric dtype, got {dtype_text!r}")
            raw_shape = leaf.get("shape")
            if not isinstance(raw_shape, list) or len(raw_shape) > 8:
                raise ContractError(f"policy.observation.leaves[{index}].shape must be a list of at most 8 dims")
            shape = tuple(
                _integer(dim, f"policy.observation.leaves[{index}].shape[{dim_index}]", minimum=1)
                for dim_index, dim in enumerate(raw_shape)
            )
            leaves.append(ObservationLeaf(path=leaf_path, dtype=dtype.name, shape=shape))
        if tuple(leaf.path for leaf in leaves) != tuple(sorted(leaf.path for leaf in leaves)):
            raise ContractError("policy.observation.leaves must be sorted by path")
        policy_limits_raw = _mapping(policy.get("limits"), "policy.limits")
        _exact_keys(
            policy_limits_raw,
            "policy.limits",
            {
                "connect_timeout_s",
                "request_timeout_s",
                "total_inference_timeout_s",
                "max_action_horizon",
                "max_message_bytes",
            },
        )
        policy_limits = PolicyLimits(
            connect_timeout_s=_number(
                policy_limits_raw.get("connect_timeout_s"),
                "policy.limits.connect_timeout_s",
                minimum=0.001,
                maximum=300.0,
            ),
            request_timeout_s=_number(
                policy_limits_raw.get("request_timeout_s"),
                "policy.limits.request_timeout_s",
                minimum=0.001,
                maximum=300.0,
            ),
            total_inference_timeout_s=_number(
                policy_limits_raw.get("total_inference_timeout_s"),
                "policy.limits.total_inference_timeout_s",
                minimum=0.001,
                maximum=3600.0,
            ),
            max_action_horizon=_integer(
                policy_limits_raw.get("max_action_horizon"),
                "policy.limits.max_action_horizon",
                minimum=1,
                maximum=32,
            ),
            max_message_bytes=_integer(
                policy_limits_raw.get("max_message_bytes"),
                "policy.limits.max_message_bytes",
                minimum=1024,
                maximum=67_108_864,
            ),
        )

        replay = _mapping(root.get("replay"), "replay")
        _exact_keys(replay, "replay", {"tolerances", "capture"})
        tolerances_raw = _mapping(replay.get("tolerances"), "replay.tolerances")
        tolerance_names = {
            "default_abs",
            "position_m",
            "linear_velocity_mps",
            "angular_velocity_rps",
            "joint_position_rad",
            "joint_velocity_radps",
            "quaternion_angle_rad",
        }
        _exact_keys(tolerances_raw, "replay.tolerances", tolerance_names)
        tolerances = DriftTolerances(
            **{
                name: _number(tolerances_raw.get(name), f"replay.tolerances.{name}", minimum=0.0)
                for name in tolerance_names
            }
        )
        capture_raw = _mapping(replay.get("capture"), "replay.capture")
        capture_names = (
            "all_scene_cameras",
            "judge_viewport",
            "configured_camera_modalities",
            "rgb_hdf5",
            "rgb_mp4",
            "contacts",
            "frame_transformers",
            "optional_imu_lidar",
            "bbox",
        )
        _exact_keys(capture_raw, "replay.capture", set(capture_names))
        expected_capture = {name: True for name in capture_names}
        if format_revision in full_capture_revisions:
            expected_capture.update(
                {
                    "judge_viewport": False,
                    "frame_transformers": False,
                    "bbox": False,
                }
            )
        for name, expected in expected_capture.items():
            if capture_raw.get(name) is not expected:
                raise ContractError(f"V4 {format_revision} replay.capture.{name} must be {str(expected).lower()}")
        capture = CaptureContract(**expected_capture)

        rules = _mapping(root.get("rules"), "rules")
        _exact_keys(
            rules,
            "rules",
            {"package_id", "package_sha256", "registry_manifest_sha256", "hard_gate_count"},
        )
        direct_scoring = None
        hard_gate_count = _integer(rules.get("hard_gate_count"), "rules.hard_gate_count", minimum=0)
        if format_revision == DIRECT_SCORING_FORMAT_REVISION:
            direct_scoring = DirectScoringContract.from_dict(root.get("direct_scoring"))
            expected_engine = (
                "task_scoring_engine" if task_family == "franka_manipulation" else "competition_task_state"
            )
            if direct_scoring.engine != expected_engine:
                raise ContractError(
                    f"direct_scoring.engine must be {expected_engine!r} for task_family {task_family!r}"
                )
            if direct_scoring.lidar_audit and task_family != "spot_navigation":
                raise ContractError("direct_scoring.lidar_audit is supported only for spot_navigation")
            if len(direct_scoring.hard_gate_paths) != hard_gate_count:
                raise ContractError("direct_scoring hard-gate paths differ from rules.hard_gate_count")
        return cls(
            contract_id=_string(root.get("contract_id"), "contract_id"),
            task_env=task_env,
            task_class=task_class,
            instruction=instruction,
            seed=seed,
            versions=versions,
            environment_config=config,
            environment_config_sha256=config_hash,
            asset_package_id=_string(assets_raw.get("package_id"), "assets.package_id"),
            asset_files=assets,
            asset_manifest_sha256=_digest(assets_raw.get("manifest_sha256"), "assets.manifest_sha256"),
            asset_archive_sha256=_digest(assets_raw.get("archive_sha256"), "assets.archive_sha256"),
            control=control,
            required_initial_state_paths=initial_paths,
            required_state_paths=state_paths,
            required_ee_pose_paths=ee_paths,
            limits=limits,
            observation_leaves=tuple(leaves),
            policy_limits=policy_limits,
            drift_tolerances=tolerances,
            capture=capture,
            rule_package_id=_string(rules.get("package_id"), "rules.package_id"),
            rule_package_sha256=_digest(rules.get("package_sha256"), "rules.package_sha256"),
            registry_manifest_sha256=_digest(
                rules.get("registry_manifest_sha256"), "rules.registry_manifest_sha256"
            ),
            hard_gate_count=hard_gate_count,
            format_revision=format_revision,
            task_family=task_family,
            runner_capture=runner_capture,
            scenario_id=scenario_id,
            scenario_sha256=scenario_sha256,
            scenario_variant=scenario_variant,
            scenario_schedule_seed=scenario_schedule_seed,
            direct_scoring=direct_scoring,
            schema_name=schema_name,
            schema_version=schema_version,
            contract_version=version,
            source_path=Path(source_path).expanduser().resolve() if source_path is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        task = {
            "task_env": self.task_env,
            "task_class": self.task_class,
            "instruction": self.instruction,
            "seed": self.seed,
        }
        required_paths = {
            "initial_state": list(self.required_initial_state_paths),
            "states": list(self.required_state_paths),
            ("tracked_pose" if self.uses_full_capture else "ee_pose"): list(self.required_ee_pose_paths),
        }
        value = {
            "contract_version": self.contract_version,
            "contract_id": self.contract_id,
            "schema": {"name": self.schema_name, "version": self.schema_version},
            "versions": dict(self.versions),
            "task": task,
            "environment": {
                "normalized_config": self.environment_config,
                "sha256": self.environment_config_sha256,
            },
            "assets": {
                "package_id": self.asset_package_id,
                "files": dict(self.asset_files),
                "manifest_sha256": self.asset_manifest_sha256,
                "archive_sha256": self.asset_archive_sha256,
            },
            "control": self.control.to_dict(format_revision=self.format_revision),
            "required_paths": required_paths,
            "limits": self.limits.to_dict(),
            "policy": {
                "protocol": {"name": POLICY_PROTOCOL_NAME, "version": self.policy_protocol_version},
                "observation": {"leaves": [leaf.to_dict() for leaf in self.observation_leaves]},
                "limits": self.policy_limits.to_dict(),
            },
            "replay": {
                "tolerances": self.drift_tolerances.to_dict(),
                "capture": self.capture.to_dict(),
            },
            "rules": {
                "package_id": self.rule_package_id,
                "package_sha256": self.rule_package_sha256,
                "registry_manifest_sha256": self.registry_manifest_sha256,
                "hard_gate_count": self.hard_gate_count,
            },
        }
        if self.uses_full_capture:
            if self.task_family is None or self.runner_capture is None:
                raise ContractError("full-capture contract is missing task_family or runner_capture")
            value["format_revision"] = self.format_revision
            task["task_family"] = self.task_family
            value["runner_capture"] = self.runner_capture.to_dict()
        if self.is_competition:
            if self.scenario_id is None or self.scenario_sha256 is None or self.scenario_variant is None:
                raise ContractError("V4.2 contract is missing scenario identity")
            value["scenario"] = {
                "id": self.scenario_id,
                "sha256": self.scenario_sha256,
                "variant": self.scenario_variant,
                "schedule_seed": self.scenario_schedule_seed,
            }
        if self.is_v4_3:
            if self.direct_scoring is None:
                raise ContractError("V4.3 contract is missing direct_scoring")
            value["direct_scoring"] = self.direct_scoring.to_dict()
        return value

    @property
    def is_v4_1(self) -> bool:
        return self.format_revision == FORMAT_REVISION

    @property
    def is_v4_2(self) -> bool:
        return self.format_revision == COMPETITION_FORMAT_REVISION

    @property
    def is_v4_3(self) -> bool:
        return self.format_revision == DIRECT_SCORING_FORMAT_REVISION

    @property
    def is_competition(self) -> bool:
        return self.format_revision in {COMPETITION_FORMAT_REVISION, DIRECT_SCORING_FORMAT_REVISION}

    @property
    def uses_full_capture(self) -> bool:
        return self.format_revision in {
            FORMAT_REVISION,
            COMPETITION_FORMAT_REVISION,
            DIRECT_SCORING_FORMAT_REVISION,
        }

    @property
    def policy_protocol_version(self) -> int:
        return COMPETITION_POLICY_PROTOCOL_VERSION if self.is_competition else POLICY_PROTOCOL_VERSION

    @property
    def tracking_group(self) -> str:
        return "tracked_pose" if self.uses_full_capture else "ee_pose"

    @property
    def primary_tracking_body(self) -> str:
        name = self.control.primary_tracking_body if self.uses_full_capture else self.control.ee_body_name
        if name is None:
            raise ContractError("contract has no primary tracking body")
        return name

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def observation_schema_sha256(self) -> str:
        return sha256_json([leaf.to_dict() for leaf in self.observation_leaves])

    @property
    def policy_limits_sha256(self) -> str:
        return sha256_json(self.policy_limits.to_dict())

    def dump(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def check_runtime_versions(self) -> list[str]:
        mismatches = []
        for package, expected in self.versions.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                actual = "not-installed"
            matches = _release(actual) == _release(expected) if package == "isaacsim" else actual == expected
            if not matches:
                mismatches.append(f"{package}: contract={expected}, runtime={actual}")
        return mismatches

    def asset_manifest(self, asset_root: str | Path) -> dict[str, Any]:
        root = Path(asset_root).expanduser().resolve()
        manifest_path = root / "asset_manifest.json"
        if manifest_path.is_symlink():
            raise AssetError("asset_manifest.json must not be a symbolic link")
        try:
            manifest = load_json(manifest_path)
        except Exception as error:
            raise AssetError(f"failed to load asset manifest {manifest_path}: {error}") from error
        if manifest.get("manifest_schema") != ASSET_MANIFEST_SCHEMA:
            raise AssetError(f"unsupported asset manifest schema in {manifest_path}")
        if set(manifest) != {"manifest_schema", "package_id", "files"}:
            raise AssetError("asset manifest has missing or unexpected fields")
        if manifest.get("package_id") != self.asset_package_id:
            raise AssetError("asset manifest package_id differs from contract")
        if sha256_json(manifest) != self.asset_manifest_sha256:
            raise AssetError("asset_manifest.json digest differs from contract")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise AssetError("asset manifest files must be a list")
        manifest_files: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
                raise AssetError("asset manifest file entry must be an object")
            relative = _safe_asset_path(entry.get("path"), "asset manifest path")
            digest = _digest(entry.get("sha256"), f"asset manifest {relative} sha256")
            _integer(entry.get("size_bytes"), f"asset manifest {relative} size_bytes", minimum=0)
            if relative in manifest_files:
                raise AssetError(f"duplicate asset manifest path: {relative}")
            manifest_files[relative] = digest
        if list(manifest_files) != sorted(manifest_files):
            raise AssetError("asset manifest file entries must be sorted by path")
        if dict(sorted(manifest_files.items())) != dict(self.asset_files):
            raise AssetError("asset manifest file mapping differs from contract")
        return manifest

    def check_assets(self, asset_root: str | Path) -> list[str]:
        root = Path(asset_root).expanduser().resolve()
        mismatches: list[str] = []
        try:
            manifest = self.asset_manifest(root)
        except (AssetError, ContractError) as error:
            return [str(error)]
        sizes = {str(entry["path"]): int(entry["size_bytes"]) for entry in manifest["files"]}
        for relative, expected in self.asset_files.items():
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                mismatches.append(f"asset escaped root: {relative}")
                continue
            if not path.is_file():
                mismatches.append(f"missing asset: {relative}")
            elif (root / relative).is_symlink():
                mismatches.append(f"symbolic-link asset is forbidden: {relative}")
            elif path.stat().st_size != sizes[relative]:
                mismatches.append(f"asset size mismatch: {relative}")
            elif sha256_file(path) != expected:
                mismatches.append(f"asset SHA-256 mismatch: {relative}")
        return mismatches


__all__ = [
    "ASSET_MANIFEST_SCHEMA",
    "COMPETITION_FORMAT_REVISION",
    "COMPETITION_POLICY_PROTOCOL_VERSION",
    "COMPETITION_TASK_FAMILIES",
    "CONTRACT_VERSION",
    "DIRECT_SCORING_FORMAT_REVISION",
    "DirectScoringContract",
    "DirectScoringField",
    "FORMAT_REVISION",
    "LEGACY_FORMAT_REVISION",
    "SPOT_TASK_FAMILY",
    "CaptureField",
    "CaptureContract",
    "CaptureStream",
    "ControlContract",
    "DriftTolerances",
    "ObservationLeaf",
    "PRIVATE_OBSERVATION_SEGMENTS",
    "POLICY_PROTOCOL_NAME",
    "POLICY_PROTOCOL_VERSION",
    "PolicyLimits",
    "REQUIRED_VERSIONS",
    "ResourceLimits",
    "RunnerCaptureContract",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "V4TaskContract",
    "VideoContract",
    "is_public_observation_path",
]
