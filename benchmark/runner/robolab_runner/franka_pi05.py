# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLab contributors.
# SPDX-License-Identifier: Apache-2.0

"""Strict PI0.5 policy bridge for the fixed Rubik's-cube-to-bowl challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from robolab_runner.errors import ProtocolError
from robolab_runner.policy.handler import PolicyHandler, serve_policy
from robolab_runner.spot_pi05 import JsonlTrace, OpenPiWebSocketClient, Pi05ModelClient

EXPECTED_PROTOCOL_VERSION = 2
EXPECTED_TASK_ENV = "RoboLabCompetition-FrankaFixedRubiksBowl-C01-S2026072902-Attack-v0"
ACTION_DIM = 8
ACTION_HORIZON = 15
EXPECTED_OBSERVATION_SCHEMA: dict[str, tuple[str, tuple[int, ...]]] = {
    "image_obs/over_shoulder_left_camera": ("uint8", (720, 1280, 3)),
    "image_obs/wrist_cam": ("uint8", (720, 1280, 3)),
    "proprio_obs/arm_joint_pos": ("float32", (7,)),
    "proprio_obs/ee_pos": ("float32", (3,)),
    "proprio_obs/ee_quat": ("float32", (4,)),
    "proprio_obs/eef_pos": ("float32", (3,)),
    "proprio_obs/eef_quat": ("float32", (4,)),
    "proprio_obs/gripper_pos": ("float32", (1,)),
}


def _validate_observations(observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(observations) != set(EXPECTED_OBSERVATION_SCHEMA):
        missing = sorted(set(EXPECTED_OBSERVATION_SCHEMA) - set(observations))
        extra = sorted(set(observations) - set(EXPECTED_OBSERVATION_SCHEMA))
        raise ProtocolError(f"Franka PI0.5 observations differ from the frozen schema: missing={missing}, extra={extra}")
    result: dict[str, np.ndarray] = {}
    for path, (dtype_text, shape) in EXPECTED_OBSERVATION_SCHEMA.items():
        value = observations[path]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype_text) or value.shape != shape:
            raise ProtocolError(f"Franka PI0.5 observation {path!r} must be {dtype_text}{shape}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ProtocolError(f"Franka PI0.5 observation {path!r} contains NaN or infinity")
        result[path] = np.ascontiguousarray(value)
    return result


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """Match OpenPI's PIL bilinear ``resize_with_pad`` preprocessing."""
    image = np.asarray(image)
    if image.dtype != np.dtype(np.uint8) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("PI0.5 images must be uint8[height,width,3]")
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new(resized.mode, (width, height), 0)
    padded.paste(resized, (max(0, (width - resized_width) // 2), max(0, (height - resized_height) // 2)))
    return np.ascontiguousarray(np.asarray(padded, dtype=np.uint8))


def build_openpi_request(
    observations: Mapping[str, np.ndarray],
    *,
    effective_instruction: str,
) -> dict[str, Any]:
    """Build the exact DROID joint-position request expected by pi05_droid_jointpos."""
    if not isinstance(effective_instruction, str) or not effective_instruction.strip():
        raise ValueError("effective_instruction must be a non-empty string")
    values = _validate_observations(observations)
    return {
        "observation/exterior_image_1_left": resize_with_pad(
            values["image_obs/over_shoulder_left_camera"]
        ),
        "observation/wrist_image_left": resize_with_pad(values["image_obs/wrist_cam"]),
        "observation/joint_position": values["proprio_obs/arm_joint_pos"],
        "observation/gripper_position": values["proprio_obs/gripper_pos"],
        "prompt": effective_instruction,
    }


def validate_action_chunk(actions: Any) -> np.ndarray:
    """Validate and post-process one 15-by-8 PI0.5 action chunk."""
    raw = np.asarray(actions)
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"PI0.5 actions must be a real numeric array, got {raw.dtype}")
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"PI0.5 actions must have shape {(ACTION_HORIZON, ACTION_DIM)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("PI0.5 actions contain NaN or infinity")
    chunk = np.ascontiguousarray(values.copy())
    chunk[:, -1] = (chunk[:, -1] > 0.5).astype(np.float32)
    return chunk


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "norm": float(np.linalg.norm(array)),
    }


def _input_summary(request: Mapping[str, Any]) -> dict[str, Any]:
    exterior = np.asarray(request["observation/exterior_image_1_left"])
    wrist = np.asarray(request["observation/wrist_image_left"])
    instruction = str(request["prompt"])
    return {
        "exterior": {
            "dtype": exterior.dtype.name,
            "shape": list(exterior.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(exterior).tobytes()).hexdigest(),
        },
        "wrist": {
            "dtype": wrist.dtype.name,
            "shape": list(wrist.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(wrist).tobytes()).hexdigest(),
        },
        "joint_position": np.asarray(request["observation/joint_position"], dtype=np.float32).tolist(),
        "gripper_position": np.asarray(request["observation/gripper_position"], dtype=np.float32).tolist(),
        "instruction_bytes": len(instruction.encode("utf-8")),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "ignored_observation_paths": [
            "proprio_obs/ee_pos",
            "proprio_obs/ee_quat",
            "proprio_obs/eef_pos",
            "proprio_obs/eef_quat",
        ],
    }


@dataclass
class _EpisodeState:
    last_infer_step: int = -1
    chunk: np.ndarray | None = None
    chunk_index: int = ACTION_HORIZON


class FrankaPi05Policy(PolicyHandler):
    """VLA-only Franka policy with the checkpoint's frozen 15-step open-loop horizon."""

    policy_id = "robolab-franka-pi05"
    policy_version = "1"

    def __init__(
        self,
        model_client: Pi05ModelClient,
        *,
        trace: JsonlTrace | None = None,
        task_envs: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.model_client = model_client
        self.trace = trace
        raw_task_envs = (EXPECTED_TASK_ENV,) if task_envs is None else tuple(task_envs)
        if not raw_task_envs or any(not isinstance(value, str) or not value.strip() for value in raw_task_envs):
            raise ValueError("task_envs must contain at least one non-empty environment ID")
        self.task_envs = frozenset(value.strip() for value in raw_task_envs)
        self._episode_states: dict[str, _EpisodeState] = {}

    def on_handshake(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        if payload.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
            raise ProtocolError("Franka PI0.5 accepts only robolab-policy-zmq/2")
        if payload.get("task_env") not in self.task_envs:
            allowed = ", ".join(sorted(self.task_envs))
            raise ProtocolError(f"Franka PI0.5 accepts only the configured task environments: {allowed}")
        if payload.get("action_dim") != ACTION_DIM or payload.get("max_action_horizon") != 1:
            raise ProtocolError("Franka PI0.5 requires action_dim=8 and max_action_horizon=1")
        raw_leaves = payload.get("observation_leaves")
        if not isinstance(raw_leaves, list):
            raise ProtocolError("Franka PI0.5 handshake has no observation schema")
        try:
            schema = {
                str(leaf["path"]): (str(leaf["dtype"]), tuple(int(value) for value in leaf["shape"]))
                for leaf in raw_leaves
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("Franka PI0.5 handshake observation schema is malformed") from error
        if schema != EXPECTED_OBSERVATION_SCHEMA:
            raise ProtocolError("Franka PI0.5 accepts only the frozen challenge observation schema")
        self._episode_states[episode_id] = _EpisodeState()

    def reset(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        if episode_id not in self._episode_states:
            raise ProtocolError("Franka PI0.5 reset requires a validated handshake")
        self._episode_states[episode_id] = _EpisodeState()

    def infer(self, observations: Mapping[str, np.ndarray], *, step: int, episode_id: str) -> np.ndarray:
        del observations, step, episode_id
        raise ProtocolError("Franka PI0.5 accepts only robolab-policy-zmq/2 infer requests")

    def infer_v2(
        self,
        observations: Mapping[str, np.ndarray],
        *,
        step: int,
        episode_id: str,
        effective_instruction: str,
    ) -> np.ndarray:
        state = self._episode_states.get(episode_id)
        if state is None:
            raise ProtocolError("Franka PI0.5 infer requires a validated, reset episode")
        if step != state.last_infer_step + 1:
            raise ProtocolError("Franka PI0.5 infer steps must be contiguous and start at zero")
        state.last_infer_step = step
        request = build_openpi_request(observations, effective_instruction=effective_instruction)
        input_summary = _input_summary(request)
        model_call = state.chunk is None or state.chunk_index >= ACTION_HORIZON
        started = time.monotonic()
        response_summary: dict[str, Any] | None = None
        try:
            if model_call:
                response = self.model_client.infer(request)
                if not isinstance(response, Mapping) or "actions" not in response:
                    raise ValueError("OpenPI response must contain an actions array")
                response_summary = _array_summary(np.asarray(response["actions"]))
                state.chunk = validate_action_chunk(response["actions"])
                state.chunk_index = 0
            if state.chunk is None:
                raise RuntimeError("PI0.5 action cache was not populated")
            action = np.ascontiguousarray(state.chunk[state.chunk_index].copy(), dtype=np.float32)
            state.chunk_index += 1
            latency_s = time.monotonic() - started
            if self.trace is not None:
                self.trace.write(
                    {
                        "trace_schema": "robolab-franka-pi05-trace/v1",
                        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "episode_id": episode_id,
                        "step": step,
                        "status": "success",
                        "model_call": model_call,
                        "chunk_index": state.chunk_index - 1,
                        "latency_s": latency_s,
                        "input": input_summary,
                        "response": response_summary,
                        "action": action.tolist(),
                        "action_norm": float(np.linalg.norm(action)),
                    }
                )
            return action.reshape(1, ACTION_DIM)
        except Exception as error:
            if self.trace is not None:
                self.trace.write(
                    {
                        "trace_schema": "robolab-franka-pi05-trace/v1",
                        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "episode_id": episode_id,
                        "step": step,
                        "status": "error",
                        "model_call": model_call,
                        "latency_s": time.monotonic() - started,
                        "input": input_summary,
                        "response": response_summary,
                        "error": {"type": type(error).__name__, "message": str(error)[:1000]},
                    }
                )
            raise

    def end_episode(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        self._episode_states.pop(episode_id, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-policy-franka-pi05")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--openpi-uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--task-env", action="append", dest="task_envs")
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--model-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-message-bytes", type=int, default=67_108_864)
    args = parser.parse_args(argv)
    if not math.isfinite(args.model_timeout_s) or args.model_timeout_s <= 0.0:
        parser.error("--model-timeout-s must be finite and positive")
    if args.max_message_bytes < 1:
        parser.error("--max-message-bytes must be positive")

    trace = JsonlTrace(args.trace_jsonl)
    model_client: OpenPiWebSocketClient | None = None
    try:
        model_client = OpenPiWebSocketClient(args.openpi_uri, timeout_s=args.model_timeout_s)
        print(
            json.dumps(
                {
                    "status": "OPENPI_CONNECTED",
                    "openpi_uri": args.openpi_uri,
                    "metadata": model_client.metadata_summary,
                    "action_horizon": ACTION_HORIZON,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        handler = FrankaPi05Policy(model_client, trace=trace, task_envs=args.task_envs)
        serve_policy(handler, args.endpoint, max_message_bytes=args.max_message_bytes)
    except KeyboardInterrupt:
        pass
    finally:
        if model_client is not None:
            model_client.close()
        trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "EXPECTED_OBSERVATION_SCHEMA",
    "EXPECTED_TASK_ENV",
    "FrankaPi05Policy",
    "build_openpi_request",
    "main",
    "resize_with_pad",
    "validate_action_chunk",
]
