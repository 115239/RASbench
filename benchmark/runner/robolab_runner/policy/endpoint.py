# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy endpoint validation and scope classification."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit

from robolab_runner.errors import ProtocolError


@dataclass(frozen=True)
class EndpointInfo:
    endpoint: str
    scope: str
    host: str
    port: int


def inspect_endpoint(endpoint: str, *, allow_remote: bool = False) -> EndpointInfo:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ProtocolError("policy endpoint must be a non-empty URI")
    if endpoint != endpoint.strip():
        raise ProtocolError("policy endpoint URI must not contain surrounding whitespace")
    if "@" in endpoint:
        raise ProtocolError("policy endpoint URI must not contain credentials")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp":
        raise ProtocolError("V4 policy endpoint must use tcp://")
    if parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment:
        raise ProtocolError("policy endpoint URI must not contain credentials, path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as error:
        raise ProtocolError(f"invalid policy endpoint port: {error}") from error
    host = parsed.hostname
    if host is None or port is None or not (1 <= port <= 65535):
        raise ProtocolError("policy endpoint must include a host and port")
    is_loopback = host.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    scope = "loopback" if is_loopback else "remote"
    if scope == "remote" and not allow_remote:
        raise ProtocolError("remote policy endpoint is disabled; use --allow-remote-policy explicitly")
    return EndpointInfo(endpoint=endpoint, scope=scope, host=host, port=port)


__all__ = ["EndpointInfo", "inspect_endpoint"]
