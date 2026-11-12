# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLab contributors.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5-only policy bridge for the frozen Spot S-aisle attack scenario."""

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
from typing import Any, Mapping, Protocol

import msgpack
import numpy as np
import websockets.sync.client

from robolab_runner.errors import ProtocolError
from robolab_runner.policy.handler import PolicyHandler, serve_policy

EXPECTED_PROTOCOL_VERSION = 2
EXPECTED_TASK_ENV = "RoboLabCompetition-SpotSAisle-Attack-v0"
EXPECTED_OBSERVATION_SCHEMA: dict[str, tuple[str, tuple[int, ...]]] = {
    "image_obs/front_rgb": ("uint8", (480, 640, 3)),
    "proprio_obs/base": ("float32", (13,)),
    "proprio_obs/goal_relative": ("float32", (3,)),
    "proprio_obs/joints": ("float32", (24,)),
    "sensor_obs/lidar_scan": ("float32", (360,)),
}
SPOT_COMMAND_LIMITS = np.asarray(
    [[-0.10, 0.55], [-0.20, 0.20], [-0.70, 0.70]],
    dtype=np.float32,
)


def _pack_openpi_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in {"V", "O", "c"}:
        raise ValueError(f"OpenPI does not support NumPy dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": np.ascontiguousarray(value).tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_openpi_array(value: dict[Any, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_openpi_message(value: Mapping[str, Any]) -> bytes:
    """Encode the safe NumPy MessagePack dialect used by OpenPI."""
    return msgpack.packb(dict(value), default=_pack_openpi_array)


def unpack_openpi_message(value: bytes) -> dict[str, Any]:
    """Decode one OpenPI response without enabling object deserialization."""
    decoded = msgpack.unpackb(value, object_hook=_unpack_openpi_array)
    if not isinstance(decoded, dict):
        raise RuntimeError("OpenPI response must be a MessagePack map")
    return decoded


class Pi05ModelClient(Protocol):
    """Small injectable boundary used by the bridge and its offline tests."""

    metadata_summary: Mapping[str, Any]

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class OpenPiWebSocketClient:
    """One fail-closed, no-retry OpenPI WebSocket session."""

    def __init__(self, uri: str, *, timeout_s: float) -> None:
        if not isinstance(uri, str) or not uri.startswith(("ws://", "wss://")):
            raise ValueError("--openpi-uri must use ws:// or wss://")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("model timeout must be finite and positive")
        self.uri = uri
        self.timeout_s = float(timeout_s)
        self._failed = False
        self._connection = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=self.timeout_s,
            close_timeout=min(10.0, self.timeout_s),
            proxy=None,
        )
        try:
            metadata_wire = self._connection.recv(timeout=self.timeout_s)
            if isinstance(metadata_wire, str):
                raise RuntimeError(f"OpenPI metadata handshake returned text: {metadata_wire[:500]}")
            metadata = unpack_openpi_message(metadata_wire)
        except Exception:
            self._failed = True
            self._connection.close()
            raise
        self.metadata_summary = {
            "sha256": hashlib.sha256(metadata_wire).hexdigest(),
            "keys": sorted(str(key) for key in metadata),
            "value_types": {str(key): type(metadata[key]).__name__ for key in sorted(metadata, key=str)},
        }

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._failed:
            raise RuntimeError("OpenPI session is closed after a previous model failure")
        started = time.monotonic()
        try:
            self._connection.send(pack_openpi_message(request))
            remaining = self.timeout_s - (time.monotonic() - started)
            if remaining <= 0.0:
                raise TimeoutError(f"OpenPI inference exceeded {self.timeout_s:.3f}s")
            response_wire = self._connection.recv(timeout=remaining)
            if isinstance(response_wire, str):
                raise RuntimeError(f"OpenPI inference returned an error: {response_wire[:500]}")
            return unpack_openpi_message(response_wire)
        except Exception:
            self._failed = True
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()


def _validate_spot_observations(observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(observations) != set(EXPECTED_OBSERVATION_SCHEMA):
        raise ProtocolError("Spot PI0.5 observations differ from the frozen S-aisle schema")
    result: dict[str, np.ndarray] = {}
    for path, (dtype_text, shape) in EXPECTED_OBSERVATION_SCHEMA.items():
        value = observations[path]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype_text) or value.shape != shape:
            raise ProtocolError(f"Spot PI0.5 observation {path!r} must be {dtype_text}{shape}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ProtocolError(f"Spot PI0.5 observation {path!r} contains NaN or infinity")
        result[path] = np.ascontiguousarray(value)
    return result


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm):
        raise ValueError("base quaternion must have a finite non-zero norm")
    # Runner doctor deliberately exercises a policy with zero_observations().
    # Treat only its exact all-zero quaternion sentinel as identity; malformed
    # near-zero quaternions from a real episode still fail closed.
    if norm == 0.0 and not np.any(quaternion):
        return 0.0
    if norm <= 1e-8:
        raise ValueError("base quaternion must have a finite non-zero norm")
    w, x, y, z = (float(value) / norm for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_pi05_state(base: np.ndarray, goal_relative: np.ndarray) -> np.ndarray:
    """Convert frozen Spot pose/body-goal observations to the legacy normalized 7-D state."""
    base = np.asarray(base)
    goal_relative = np.asarray(goal_relative)
    if base.shape != (13,) or base.dtype != np.dtype(np.float32) or not np.isfinite(base).all():
        raise ValueError("base observation must be finite float32[13]")
    if (
        goal_relative.shape != (3,)
        or goal_relative.dtype != np.dtype(np.float32)
        or not np.isfinite(goal_relative).all()
    ):
        raise ValueError("goal_relative observation must be finite float32[3]")
    yaw = _yaw_from_wxyz(base[3:7])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    goal_world_x = cosine * float(goal_relative[0]) - sine * float(goal_relative[1])
    goal_world_y = sine * float(goal_relative[0]) + cosine * float(goal_relative[1])
    return np.asarray(
        [
            float(base[0]) / 12.0,
            float(base[1]) / 4.5,
            float(base[2]),
            yaw / math.pi,
            goal_world_x / 20.0,
            goal_world_y / 6.0,
            float(np.linalg.norm(goal_relative[:2])) / 20.0,
        ],
        dtype=np.float32,
    )


def build_openpi_request(
    observations: Mapping[str, np.ndarray],
    *,
    effective_instruction: str,
) -> dict[str, Any]:
    """Build exactly the dual-RGB PI0.5 request used by the legacy Spot integration."""
    if not isinstance(effective_instruction, str) or not effective_instruction.strip():
        raise ValueError("effective_instruction must be a non-empty string")
    values = _validate_spot_observations(observations)
    rgb = values["image_obs/front_rgb"]
    state = build_pi05_state(values["proprio_obs/base"], values["proprio_obs/goal_relative"])
    return {
        "observation/exterior_image_1_left": rgb,
        "observation/wrist_image_left": rgb,
        "observation/joint_position": state,
        "observation/gripper_position": np.asarray([0.0], dtype=np.float32),
        "prompt": effective_instruction,
    }


def pi05_action_features(actions: Any) -> np.ndarray:
    """Extract the legacy 33 statistics from a PI0.5 action chunk."""
    raw_values = np.asarray(actions)
    if raw_values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"PI0.5 actions must be a real numeric array, got {raw_values.dtype}")
    values = np.asarray(raw_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 8:
        raise ValueError(f"PI0.5 actions must have shape [T, >=8], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("PI0.5 actions contain NaN or infinity")
    chunk = values[:, :8]
    features = np.concatenate(
        (chunk.mean(axis=0), chunk.std(axis=0), chunk[0], chunk[-1], np.ones(1, dtype=np.float32))
    )
    if features.shape != (33,) or features.dtype != np.dtype(np.float32) or not np.isfinite(features).all():
        raise ValueError("PI0.5 feature extraction did not produce finite float32[33]")
    return np.ascontiguousarray(features)


def map_pi05_actions(actions: Any, weights: np.ndarray) -> np.ndarray:
    """Map PI0.5 action statistics to a clipped Spot ``[vx, vy, yaw_rate]`` command."""
    weights = np.asarray(weights)
    if weights.dtype != np.dtype(np.float32) or weights.shape != (33, 3) or not np.isfinite(weights).all():
        raise ValueError("PI0.5 adapter weights must be finite float32[33,3]")
    proposal = pi05_action_features(actions) @ weights
    if proposal.shape != (3,) or not np.isfinite(proposal).all():
        raise ValueError("PI0.5 adapter returned an invalid three-dimensional command")
    return np.ascontiguousarray(
        np.clip(proposal, SPOT_COMMAND_LIMITS[:, 0], SPOT_COMMAND_LIMITS[:, 1]),
        dtype=np.float32,
    )


def load_adapter_weights(path: Path) -> np.ndarray:
    """Load and validate only the legacy adapter's trained 33-by-3 weight matrix."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PI0.5 adapter is missing: {resolved}")
    with np.load(resolved, allow_pickle=False) as adapter:
        if "weights" not in adapter:
            raise ValueError("PI0.5 adapter does not contain weights")
        weights = np.asarray(adapter["weights"])
    if weights.dtype != np.dtype(np.float32) or weights.shape != (33, 3) or not np.isfinite(weights).all():
        raise ValueError("PI0.5 adapter weights must be finite float32[33,3]")
    return np.ascontiguousarray(weights.copy())


class JsonlTrace:
    """Exclusive, flush-on-write policy sidecar that never stores image bytes."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


@dataclass
class _EpisodeState:
    last_infer_step: int = -1
    last_decision_step: int = -1
    cached_action: np.ndarray | None = None


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
    rgb = np.asarray(request["observation/exterior_image_1_left"])
    instruction = str(request["prompt"])
    return {
        "rgb_dtype": rgb.dtype.name,
        "rgb_shape": list(rgb.shape),
        "rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "state": np.asarray(request["observation/joint_position"], dtype=np.float32).tolist(),
        "instruction_bytes": len(instruction.encode("utf-8")),
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "ignored_observation_paths": ["proprio_obs/joints", "sensor_obs/lidar_scan"],
    }


class SpotPi05Policy(PolicyHandler):
    """Strict VLA-only Spot policy: no planner, mixing, retry, or zero-action fallback."""

    policy_id = "robolab-spot-pi05"
    policy_version = "1"

    def __init__(
        self,
        model_client: Pi05ModelClient,
        weights: np.ndarray,
        *,
        decision_interval_steps: int = 3,
        trace: JsonlTrace | None = None,
        task_envs: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(decision_interval_steps, bool) or not isinstance(decision_interval_steps, int):
            raise ValueError("decision interval must be an integer")
        if decision_interval_steps < 1:
            raise ValueError("decision interval must be at least one step")
        self.model_client = model_client
        self.weights = np.asarray(weights)
        if self.weights.dtype != np.dtype(np.float32) or self.weights.shape != (33, 3):
            raise ValueError("PI0.5 adapter weights must be float32[33,3]")
        if not np.isfinite(self.weights).all():
            raise ValueError("PI0.5 adapter weights contain NaN or infinity")
        self.decision_interval_steps = decision_interval_steps
        self.trace = trace
        raw_task_envs = (EXPECTED_TASK_ENV,) if task_envs is None else tuple(task_envs)
        if not raw_task_envs or any(not isinstance(value, str) or not value.strip() for value in raw_task_envs):
            raise ValueError("task_envs must contain at least one non-empty environment ID")
        self.task_envs = frozenset(value.strip() for value in raw_task_envs)
        self._episode_states: dict[str, _EpisodeState] = {}

    def on_handshake(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        if payload.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
            raise ProtocolError("Spot PI0.5 accepts only robolab-policy-zmq/2")
        if payload.get("task_env") not in self.task_envs:
            allowed = ", ".join(sorted(self.task_envs))
            raise ProtocolError(f"Spot PI0.5 accepts only the configured task environments: {allowed}")
        if payload.get("action_dim") != 3:
            raise ProtocolError("Spot PI0.5 requires a three-dimensional base action")
        raw_leaves = payload.get("observation_leaves")
        schema = {
            str(leaf["path"]): (str(leaf["dtype"]), tuple(int(value) for value in leaf["shape"]))
            for leaf in raw_leaves
        }
        if schema != EXPECTED_OBSERVATION_SCHEMA:
            raise ProtocolError("Spot PI0.5 accepts only the frozen S-aisle observation schema")
        self._episode_states[episode_id] = _EpisodeState()

    def reset(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        if episode_id not in self._episode_states:
            raise ProtocolError("Spot PI0.5 reset requires a validated handshake")
        self._episode_states[episode_id] = _EpisodeState()

    def infer(self, observations: Mapping[str, np.ndarray], *, step: int, episode_id: str) -> np.ndarray:
        del observations, step, episode_id
        raise ProtocolError("Spot PI0.5 accepts only robolab-policy-zmq/2 infer requests")

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
            raise ProtocolError("Spot PI0.5 infer requires a validated, reset episode")
        if step <= state.last_infer_step:
            raise ProtocolError("Spot PI0.5 infer steps must be strictly increasing")
        state.last_infer_step = step
        request = build_openpi_request(observations, effective_instruction=effective_instruction)
        input_summary = _input_summary(request)
        reuse_cache = (
            state.cached_action is not None
            and step - state.last_decision_step < self.decision_interval_steps
        )
        started = time.monotonic()
        response_summary: dict[str, Any] | None = None
        try:
            if reuse_cache:
                action = state.cached_action.copy()
            else:
                response = self.model_client.infer(request)
                if not isinstance(response, Mapping) or "actions" not in response:
                    raise ValueError("OpenPI response must contain an actions array")
                raw_actions = response["actions"]
                response_summary = _array_summary(np.asarray(raw_actions))
                action = map_pi05_actions(raw_actions, self.weights)
                state.cached_action = action.copy()
                state.last_decision_step = step
            latency_s = time.monotonic() - started
            if self.trace is not None:
                self.trace.write(
                    {
                        "trace_schema": "robolab-spot-pi05-trace/v1",
                        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "episode_id": episode_id,
                        "step": step,
                        "status": "success",
                        "cached": reuse_cache,
                        "latency_s": latency_s,
                        "input": input_summary,
                        "response": response_summary,
                        "action": action.tolist(),
                        "action_norm": float(np.linalg.norm(action)),
                    }
                )
            return action.reshape(1, 3)
        except Exception as error:
            if self.trace is not None:
                self.trace.write(
                    {
                        "trace_schema": "robolab-spot-pi05-trace/v1",
                        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                        "episode_id": episode_id,
                        "step": step,
                        "status": "error",
                        "cached": reuse_cache,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-policy-spot-pi05")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--openpi-uri", default="ws://127.0.0.1:8010")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--task-env",
        action="append",
        dest="task_envs",
        help=(
            "Exact frozen task environment ID accepted by the bridge; repeat to allow more than one. "
            f"Defaults to {EXPECTED_TASK_ENV}."
        ),
    )
    parser.add_argument("--decision-interval-steps", type=int, default=3)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--model-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-message-bytes", type=int, default=67_108_864)
    args = parser.parse_args(argv)
    if args.decision_interval_steps < 1:
        parser.error("--decision-interval-steps must be at least 1")
    if not math.isfinite(args.model_timeout_s) or args.model_timeout_s <= 0.0:
        parser.error("--model-timeout-s must be finite and positive")
    if args.max_message_bytes < 1:
        parser.error("--max-message-bytes must be positive")

    adapter = args.adapter.expanduser().resolve()
    weights = load_adapter_weights(adapter)
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
                    "adapter": str(adapter),
                    "adapter_sha256": _sha256_file(adapter),
                    "decision_interval_steps": args.decision_interval_steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        handler = SpotPi05Policy(
            model_client,
            weights,
            decision_interval_steps=args.decision_interval_steps,
            trace=trace,
            task_envs=args.task_envs,
        )
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
    "EXPECTED_OBSERVATION_SCHEMA",
    "EXPECTED_TASK_ENV",
    "JsonlTrace",
    "OpenPiWebSocketClient",
    "SPOT_COMMAND_LIMITS",
    "SpotPi05Policy",
    "build_openpi_request",
    "build_pi05_state",
    "load_adapter_weights",
    "main",
    "map_pi05_actions",
    "pack_openpi_message",
    "pi05_action_features",
    "unpack_openpi_message",
]
