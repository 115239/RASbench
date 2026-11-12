# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable public error types and failure codes."""

from __future__ import annotations


class RunnerError(RuntimeError):
    """Base class for an expected Runner failure."""

    code = "RUNNER_ERROR"


class ContractError(RunnerError, ValueError):
    code = "INVALID_CONTRACT"


class AssetError(RunnerError):
    code = "ASSET_MISMATCH"


class ProtocolError(RunnerError):
    code = "POLICY_PROTOCOL_ERROR"


class PolicyConnectTimeout(RunnerError, TimeoutError):
    code = "POLICY_CONNECT_TIMEOUT"


class PolicyRequestTimeout(RunnerError, TimeoutError):
    code = "POLICY_REQUEST_TIMEOUT"


class PolicyDisconnected(RunnerError, ConnectionError):
    code = "POLICY_DISCONNECTED"


class PolicyRemoteError(RunnerError):
    code = "POLICY_REMOTE_ERROR"


class InvalidAction(RunnerError, ValueError):
    code = "INVALID_ACTION"


class SubmissionError(RunnerError):
    code = "INVALID_SUBMISSION"


__all__ = [
    "AssetError",
    "ContractError",
    "InvalidAction",
    "PolicyConnectTimeout",
    "PolicyDisconnected",
    "PolicyRemoteError",
    "PolicyRequestTimeout",
    "ProtocolError",
    "RunnerError",
    "SubmissionError",
]

