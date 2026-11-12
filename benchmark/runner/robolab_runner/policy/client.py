# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed ZMQ REQ client for the versioned RoboLab policy protocol."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import zmq

from robolab_runner.contract import POLICY_PROTOCOL_NAME, POLICY_PROTOCOL_VERSION, V4TaskContract
from robolab_runner.errors import (
    InvalidAction,
    PolicyConnectTimeout,
    PolicyDisconnected,
    PolicyRemoteError,
    PolicyRequestTimeout,
    ProtocolError,
)
from robolab_runner.policy.codec import pack_message, unpack_message
from robolab_runner.policy.endpoint import EndpointInfo, inspect_endpoint
from robolab_runner.policy.observations import validate_observations

PROTOCOL_ID = f"{POLICY_PROTOCOL_NAME}/{POLICY_PROTOCOL_VERSION}"
_POLICY_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def protocol_id(version: int) -> str:
    """Return the canonical wire identifier for one negotiated protocol version."""
    return f"{POLICY_PROTOCOL_NAME}/{int(version)}"


@dataclass(frozen=True)
class PolicyIdentity:
    policy_id: str
    policy_version: str


@dataclass(frozen=True)
class PolicyNetworkEvent:
    """Size/timing/status audit for one Runner-to-policy REQ/REP exchange."""

    sequence: int
    message_type: str
    control_step: int
    session_time_s: float
    request_bytes: int
    response_bytes: int
    round_trip_s: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "message_type": self.message_type,
            "control_step": self.control_step,
            "session_time_s": self.session_time_s,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "round_trip_s": self.round_trip_s,
            "status": self.status,
        }


class PolicyClient:
    """One no-retry policy session bound to one contract and episode."""

    def __init__(
        self,
        contract: V4TaskContract,
        endpoint: str,
        *,
        episode_id: str | None = None,
        allow_remote: bool = False,
        context: zmq.Context | None = None,
    ) -> None:
        self.contract = contract
        self.protocol_id = protocol_id(contract.policy_protocol_version)
        self.endpoint_info: EndpointInfo = inspect_endpoint(endpoint, allow_remote=allow_remote)
        self.episode_id = episode_id or uuid.uuid4().hex
        self._context = context or zmq.Context.instance()
        self._owns_context = False
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.setsockopt(zmq.SNDTIMEO, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.MAXMSGSIZE, contract.policy_limits.max_message_bytes)
        self._socket.connect(self.endpoint_info.endpoint)
        self._request_sequence = 0
        self._closed = False
        self._handshaken = False
        self._reset = False
        self._identity: PolicyIdentity | None = None
        self._inference_wait_s = 0.0
        self._query_count = 0
        self._last_request_wait_s = 0.0
        self._session_started = time.monotonic()
        self._network_events: list[PolicyNetworkEvent] = []

    @property
    def endpoint_scope(self) -> str:
        return self.endpoint_info.scope

    @property
    def identity(self) -> PolicyIdentity | None:
        return self._identity

    @property
    def inference_wait_s(self) -> float:
        return self._inference_wait_s

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def remaining_inference_timeout_s(self) -> float:
        return max(0.0, self.contract.policy_limits.total_inference_timeout_s - self._inference_wait_s)

    def drain_network_events(self) -> list[dict[str, Any]]:
        events = [event.to_dict() for event in self._network_events]
        self._network_events.clear()
        return events

    def _next_request_id(self) -> str:
        self._request_sequence += 1
        return f"{self._request_sequence:016x}"

    def _close_failed_socket(self) -> None:
        if not self._closed:
            self._socket.close(linger=0)
            self._closed = True

    def _request(
        self,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float,
        connect_phase: bool = False,
        control_step: int = -1,
    ) -> dict[str, Any]:
        if self._closed:
            raise PolicyDisconnected("policy session is closed after a previous failure")
        self._last_request_wait_s = 0.0
        request_id = self._next_request_id()
        envelope = {
            "protocol": self.protocol_id,
            "type": message_type,
            "request_id": request_id,
            "episode_id": self.episode_id,
            "contract_sha256": self.contract.digest,
            "payload": dict(payload),
        }
        packed = b""
        response_bytes = b""
        status = "protocol_error"
        wait_started = time.monotonic()
        try:
            packed = pack_message(envelope, max_bytes=self.contract.policy_limits.max_message_bytes)
            deadline = time.monotonic() + max(0.0, timeout_s)
            transport_started = time.monotonic()
            try:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if not self._socket.poll(remaining_ms, zmq.POLLOUT):
                    self._close_failed_socket()
                    error_type = PolicyConnectTimeout if connect_phase else PolicyRequestTimeout
                    raise error_type(f"policy {message_type} send timed out after {timeout_s:.3f}s")
                self._socket.send(packed, flags=zmq.NOBLOCK)
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if not self._socket.poll(remaining_ms, zmq.POLLIN):
                    self._close_failed_socket()
                    error_type = PolicyConnectTimeout if connect_phase else PolicyRequestTimeout
                    raise error_type(f"policy {message_type} response timed out after {timeout_s:.3f}s")
                response_bytes = self._socket.recv(flags=zmq.NOBLOCK)
            except (PolicyConnectTimeout, PolicyRequestTimeout):
                status = "timeout"
                raise
            except zmq.Again as error:
                status = "disconnected"
                self._close_failed_socket()
                raise PolicyDisconnected(f"policy disconnected during {message_type}") from error
            except zmq.ZMQError as error:
                status = "disconnected"
                self._close_failed_socket()
                raise PolicyDisconnected(f"ZMQ failure during {message_type}: {error}") from error
            finally:
                # The total inference budget covers REQ/REP transport waiting,
                # not observation packing or response validation.
                self._last_request_wait_s = time.monotonic() - transport_started

            response = unpack_message(response_bytes, max_bytes=self.contract.policy_limits.max_message_bytes)
            expected = {
                "protocol": self.protocol_id,
                "type": f"{message_type}_response",
                "request_id": request_id,
                "episode_id": self.episode_id,
                "contract_sha256": self.contract.digest,
            }
            for field, expected_value in expected.items():
                if response.get(field) != expected_value:
                    self._close_failed_socket()
                    raise ProtocolError(
                        f"policy response {field}={response.get(field)!r}, expected {expected_value!r}"
                    )
            if response.get("ok") is not True:
                error = response.get("error")
                if not isinstance(error, Mapping):
                    raise ProtocolError("policy error response lacks an error object")
                code = str(error.get("code", "POLICY_ERROR"))
                message = str(error.get("message", "policy service returned an error"))
                raise PolicyRemoteError(f"{code}: {message}")
            response_payload = response.get("payload")
            if not isinstance(response_payload, dict):
                raise ProtocolError("policy response payload must be a map")
            status = "success"
            return response_payload
        except (ProtocolError, PolicyRemoteError):
            status = "protocol_error"
            raise
        finally:
            self._network_events.append(
                PolicyNetworkEvent(
                    sequence=self._request_sequence,
                    message_type=message_type,
                    control_step=int(control_step),
                    session_time_s=max(0.0, time.monotonic() - self._session_started),
                    request_bytes=len(packed),
                    response_bytes=len(response_bytes),
                    round_trip_s=max(0.0, self._last_request_wait_s),
                    status=status,
                )
            )

    def handshake(self) -> PolicyIdentity:
        if self._handshaken:
            if self._identity is None:
                raise ProtocolError("policy session lost its handshake identity")
            return self._identity
        payload = self._request(
            "handshake",
            {
                "protocol_name": POLICY_PROTOCOL_NAME,
                "protocol_version": self.contract.policy_protocol_version,
                "contract_id": self.contract.contract_id,
                "task_env": self.contract.task_env,
                "action_dim": self.contract.control.action_dim,
                "max_action_horizon": self.contract.policy_limits.max_action_horizon,
                "observation_schema_sha256": self.contract.observation_schema_sha256,
                "observation_leaves": [leaf.to_dict() for leaf in self.contract.observation_leaves],
            },
            timeout_s=self.contract.policy_limits.connect_timeout_s,
            connect_phase=True,
        )
        policy_id = payload.get("policy_id")
        policy_version = payload.get("policy_version")
        if not isinstance(policy_id, str) or not _POLICY_IDENTITY_RE.fullmatch(policy_id):
            raise ProtocolError("handshake response policy_id must be a portable identifier")
        if not isinstance(policy_version, str) or not _POLICY_IDENTITY_RE.fullmatch(policy_version):
            raise ProtocolError("handshake response policy_version must be a portable identifier")
        if (
            payload.get("protocol_name") != POLICY_PROTOCOL_NAME
            or payload.get("protocol_version") != self.contract.policy_protocol_version
        ):
            raise ProtocolError(f"policy service did not negotiate {self.protocol_id}")
        self._identity = PolicyIdentity(policy_id=policy_id, policy_version=policy_version)
        self._handshaken = True
        return self._identity

    def reset(self) -> None:
        if not self._handshaken:
            raise ProtocolError("handshake must complete before reset")
        self._request(
            "reset",
            {
                "contract_id": self.contract.contract_id,
                "task_env": self.contract.task_env,
                "instruction": self.contract.instruction,
                "seed": self.contract.seed,
            },
            timeout_s=self.contract.policy_limits.request_timeout_s,
        )
        self._reset = True

    def infer(
        self,
        observations: Mapping[str, Any],
        *,
        step: int,
        effective_instruction: str | None = None,
    ) -> tuple[np.ndarray, float]:
        if not self._reset:
            raise ProtocolError("reset must complete before infer")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ProtocolError("infer step must be a non-negative integer")
        public_observations = validate_observations(observations, self.contract)
        infer_payload: dict[str, Any] = {"step": step, "observations": public_observations}
        if self.contract.is_competition:
            if not isinstance(effective_instruction, str) or not effective_instruction.strip():
                raise ProtocolError("policy protocol v2 requires a non-empty effective_instruction")
            infer_payload["effective_instruction"] = effective_instruction
        elif effective_instruction is not None:
            raise ProtocolError("effective_instruction is available only in policy protocol v2")
        remaining = self.remaining_inference_timeout_s
        if remaining <= 0:
            raise PolicyRequestTimeout("total inference timeout budget is exhausted")
        timeout = min(self.contract.policy_limits.request_timeout_s, remaining)
        duration = 0.0
        try:
            payload = self._request(
                "infer",
                infer_payload,
                timeout_s=timeout,
                control_step=step,
            )
        finally:
            duration = self._last_request_wait_s
            self._inference_wait_s += duration
            self._query_count += 1
        actions = payload.get("actions")
        if not isinstance(actions, np.ndarray):
            raise InvalidAction("infer response actions must be an encoded ndarray")
        if actions.dtype != np.dtype(np.float32):
            raise InvalidAction(f"infer response actions must be float32, got {actions.dtype}")
        expected_width = self.contract.control.action_dim
        maximum_horizon = self.contract.policy_limits.max_action_horizon
        if actions.ndim != 2 or actions.shape[1:] != (expected_width,):
            raise InvalidAction(
                f"infer response actions must have shape [H, {expected_width}], got {actions.shape}"
            )
        if not 1 <= actions.shape[0] <= maximum_horizon:
            raise InvalidAction(f"infer action horizon must be in [1, {maximum_horizon}], got {actions.shape[0]}")
        if not np.isfinite(actions).all():
            raise InvalidAction("infer response actions contain NaN or infinity")
        return np.ascontiguousarray(actions), duration

    def end_episode(self, *, steps: int, status: str) -> None:
        if not self._reset or self._closed:
            return
        self._request(
            "end_episode",
            {"steps": int(steps), "status": str(status)},
            timeout_s=self.contract.policy_limits.request_timeout_s,
        )
        self._reset = False

    def close(self) -> None:
        if not self._closed:
            self._socket.close(linger=0)
            self._closed = True
        if self._owns_context:
            self._context.term()

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["PROTOCOL_ID", "PolicyClient", "PolicyIdentity", "PolicyNetworkEvent", "protocol_id"]
