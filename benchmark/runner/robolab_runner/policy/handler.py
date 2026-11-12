# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python policy-service SDK for RoboLab policy protocols v1 and v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Mapping

import numpy as np
import zmq

from robolab_runner.common import sha256_json
from robolab_runner.contract import (
    COMPETITION_POLICY_PROTOCOL_VERSION,
    POLICY_PROTOCOL_NAME,
    POLICY_PROTOCOL_VERSION,
    is_public_observation_path,
)
from robolab_runner.errors import ProtocolError, RunnerError
from robolab_runner.policy.client import PROTOCOL_ID, protocol_id

_SUPPORTED_PROTOCOL_VERSIONS = frozenset({POLICY_PROTOCOL_VERSION, COMPETITION_POLICY_PROTOCOL_VERSION})
_SUPPORTED_PROTOCOL_IDS = frozenset(protocol_id(version) for version in _SUPPORTED_PROTOCOL_VERSIONS)
from robolab_runner.policy.codec import pack_message, unpack_message


@dataclass
class _Session:
    contract_sha256: str
    protocol_version: int
    handshake: dict[str, Any]
    observation_schema: dict[str, tuple[np.dtype, tuple[int, ...]]]
    reset_payload: dict[str, Any] | None = None
    request_ids: set[str] = field(default_factory=set)


class PolicyHandler(ABC):
    """Subclass this handler and implement :meth:`infer` in a separate process."""

    policy_id = "unnamed-policy"
    policy_version = "0"

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def on_handshake(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        """Optional hook called after protocol and schema fields are validated."""

    def reset(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        """Optional hook for clearing policy state before an episode."""

    @abstractmethod
    def infer(self, observations: Mapping[str, np.ndarray], *, step: int, episode_id: str) -> np.ndarray:
        """Return finite ``float32[H, action_dim]`` actions."""

    def infer_v2(
        self,
        observations: Mapping[str, np.ndarray],
        *,
        step: int,
        episode_id: str,
        effective_instruction: str,
    ) -> np.ndarray:
        """Handle protocol v2; override to consume the per-step effective instruction."""
        del effective_instruction
        return self.infer(observations, step=step, episode_id=episode_id)

    def end_episode(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        """Optional hook called after the Runner finishes the episode."""

    @staticmethod
    def _payload(request: Mapping[str, Any]) -> dict[str, Any]:
        payload = request.get("payload")
        if not isinstance(payload, dict):
            raise ProtocolError("request payload must be a map")
        return payload

    @staticmethod
    def _observation_schema(raw_leaves: Any) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
        if not isinstance(raw_leaves, list) or not raw_leaves:
            raise ProtocolError("handshake observation_leaves must be a non-empty list")
        result: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
        for raw_leaf in raw_leaves:
            if not isinstance(raw_leaf, dict) or set(raw_leaf) != {"path", "dtype", "shape"}:
                raise ProtocolError("handshake observation leaf is malformed")
            path = raw_leaf["path"]
            dtype_text = raw_leaf["dtype"]
            raw_shape = raw_leaf["shape"]
            if not isinstance(path, str) or not is_public_observation_path(path) or path in result:
                raise ProtocolError("handshake observation path is private or duplicated")
            if not isinstance(dtype_text, str):
                raise ProtocolError("handshake observation dtype must be a string")
            try:
                dtype = np.dtype(dtype_text)
            except TypeError as error:
                raise ProtocolError(f"handshake observation dtype is invalid: {dtype_text!r}") from error
            if dtype.name != dtype_text or dtype.hasobject or dtype.fields or dtype.kind not in {"b", "i", "u", "f"}:
                raise ProtocolError(f"handshake observation dtype is unsafe: {dtype_text!r}")
            if not isinstance(raw_shape, list) or len(raw_shape) > 8 or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in raw_shape
            ):
                raise ProtocolError("handshake observation shape is invalid")
            shape = tuple(raw_shape)
            result[path] = (dtype, shape)
        if list(result) != sorted(result):
            raise ProtocolError("handshake observation leaves must be sorted")
        return result

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_protocol = request.get("protocol")
        if request_protocol not in _SUPPORTED_PROTOCOL_IDS:
            raise ProtocolError(f"unsupported protocol {request.get('protocol')!r}")
        protocol_version = int(str(request_protocol).rsplit("/", 1)[1])
        message_type = request.get("type")
        if message_type not in {"handshake", "reset", "infer", "end_episode"}:
            raise ProtocolError(f"unsupported message type {message_type!r}")
        request_id = request.get("request_id")
        episode_id = request.get("episode_id")
        contract_sha256 = request.get("contract_sha256")
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("request_id must be a non-empty string")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProtocolError("episode_id must be a non-empty string")
        if (
            not isinstance(contract_sha256, str)
            or len(contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in contract_sha256)
        ):
            raise ProtocolError("contract_sha256 must be a SHA-256 digest")
        payload = self._payload(request)

        session = self._sessions.get(episode_id)
        if message_type == "handshake":
            if session is not None:
                raise ProtocolError("episode_id already completed a handshake")
            if (
                payload.get("protocol_name") != POLICY_PROTOCOL_NAME
                or payload.get("protocol_version") != protocol_version
            ):
                raise ProtocolError(f"handshake must request {request_protocol}")
            action_dim = payload.get("action_dim")
            max_horizon = payload.get("max_action_horizon")
            leaves = payload.get("observation_leaves")
            if (
                isinstance(action_dim, bool)
                or not isinstance(action_dim, int)
                or action_dim < 1
                or isinstance(max_horizon, bool)
                or not isinstance(max_horizon, int)
                or not 1 <= max_horizon <= 32
            ):
                raise ProtocolError("handshake action/schema fields are invalid")
            observation_schema = self._observation_schema(leaves)
            canonical_leaves = [
                {"path": path, "dtype": dtype.name, "shape": list(shape)}
                for path, (dtype, shape) in observation_schema.items()
            ]
            if payload.get("observation_schema_sha256") != sha256_json(canonical_leaves):
                raise ProtocolError("handshake observation schema digest is invalid")
            session = _Session(
                contract_sha256=contract_sha256,
                protocol_version=protocol_version,
                handshake=dict(payload),
                observation_schema=observation_schema,
            )
            session.request_ids.add(request_id)
            self._sessions[episode_id] = session
            self.on_handshake(episode_id=episode_id, payload=payload)
            return {
                "protocol_name": POLICY_PROTOCOL_NAME,
                "protocol_version": protocol_version,
                "policy_id": str(self.policy_id),
                "policy_version": str(self.policy_version),
            }

        if session is None:
            raise ProtocolError("handshake must precede episode messages")
        if session.contract_sha256 != contract_sha256:
            raise ProtocolError("contract_sha256 changed within an episode")
        if request_id in session.request_ids:
            raise ProtocolError("request_id was already used in this episode")
        session.request_ids.add(request_id)

        if message_type == "reset":
            if session.reset_payload is not None:
                raise ProtocolError("episode was already reset")
            if payload.get("contract_id") != session.handshake.get("contract_id") or payload.get(
                "task_env"
            ) != session.handshake.get("task_env"):
                raise ProtocolError("reset contract/task identity differs from the handshake")
            session.reset_payload = dict(payload)
            self.reset(episode_id=episode_id, payload=payload)
            return {"accepted": True}
        if session.reset_payload is None:
            raise ProtocolError("reset must precede infer/end_episode")
        if message_type == "infer":
            step = payload.get("step")
            observations = payload.get("observations")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ProtocolError("infer step must be a non-negative integer")
            if not isinstance(observations, dict) or not all(
                isinstance(path, str) and isinstance(value, np.ndarray)
                for path, value in observations.items()
            ):
                raise ProtocolError("infer observations must be a map of encoded ndarrays")
            if set(observations) != set(session.observation_schema):
                raise ProtocolError("infer observation paths differ from the negotiated schema")
            for path, (expected_dtype, expected_shape) in session.observation_schema.items():
                array = observations[path]
                if array.dtype != expected_dtype or array.shape != expected_shape:
                    raise ProtocolError(f"infer observation {path!r} differs from the negotiated schema")
                if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                    raise ProtocolError(f"infer observation {path!r} contains NaN or infinity")
            if session.protocol_version == COMPETITION_POLICY_PROTOCOL_VERSION:
                effective_instruction = payload.get("effective_instruction")
                if not isinstance(effective_instruction, str) or not effective_instruction.strip():
                    raise ProtocolError("protocol v2 infer requires a non-empty effective_instruction")
                actions = self.infer_v2(
                    observations,
                    step=step,
                    episode_id=episode_id,
                    effective_instruction=effective_instruction,
                )
            else:
                if "effective_instruction" in payload:
                    raise ProtocolError("protocol v1 infer cannot contain effective_instruction")
                actions = self.infer(observations, step=step, episode_id=episode_id)
            if not isinstance(actions, np.ndarray):
                raise ProtocolError("PolicyHandler.infer must return a NumPy ndarray")
            return {"actions": actions}
        self.end_episode(episode_id=episode_id, payload=payload)
        del self._sessions[episode_id]
        return {"accepted": True}


def _response_envelope(request: Mapping[str, Any], *, ok: bool, payload: Mapping[str, Any]) -> dict[str, Any]:
    message_type = request.get("type")
    return {
        "protocol": request.get("protocol", PROTOCOL_ID),
        "type": f"{message_type}_response",
        "request_id": request.get("request_id", "invalid"),
        "episode_id": request.get("episode_id", "invalid"),
        "contract_sha256": request.get("contract_sha256", "invalid"),
        "ok": ok,
        **dict(payload),
    }


def serve_policy(
    handler: PolicyHandler,
    endpoint: str,
    *,
    max_message_bytes: int = 67_108_864,
    stop_event: Event | None = None,
) -> None:
    """Bind a single-threaded REP service. The Runner never starts this process."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, int(max_message_bytes))
    socket.bind(endpoint)
    try:
        while stop_event is None or not stop_event.is_set():
            if not socket.poll(100, zmq.POLLIN):
                continue
            request: dict[str, Any] = {}
            try:
                raw = socket.recv()
                request = unpack_message(raw, max_bytes=max_message_bytes)
                response_payload = handler.dispatch(request)
                response = _response_envelope(request, ok=True, payload={"payload": response_payload})
            except Exception as error:
                code = error.code if isinstance(error, RunnerError) else "POLICY_HANDLER_ERROR"
                response = _response_envelope(
                    request,
                    ok=False,
                    payload={"error": {"code": code, "message": str(error)[:1000]}},
                )
            socket.send(pack_message(response, max_bytes=max_message_bytes))
    finally:
        socket.close(linger=0)
        context.term()


__all__ = ["PolicyHandler", "serve_policy"]
