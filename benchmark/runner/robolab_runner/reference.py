# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference zero-action policy service used for acceptance testing."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import numpy as np

from robolab_runner.policy.handler import PolicyHandler, serve_policy


class ZeroActionPolicy(PolicyHandler):
    policy_id = "robolab-reference-zero"
    policy_version = "1"

    def __init__(self, *, horizon: int = 1) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self._action_dims: dict[str, int] = {}
        self._max_horizons: dict[str, int] = {}

    def on_handshake(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        self._action_dims[episode_id] = int(payload["action_dim"])
        self._max_horizons[episode_id] = int(payload["max_action_horizon"])

    def infer(self, observations: Mapping[str, np.ndarray], *, step: int, episode_id: str) -> np.ndarray:
        del observations, step
        horizon = min(max(1, self.horizon), self._max_horizons[episode_id])
        return np.zeros((horizon, self._action_dims[episode_id]), dtype=np.float32)

    def end_episode(self, *, episode_id: str, payload: Mapping[str, Any]) -> None:
        del payload
        self._action_dims.pop(episode_id, None)
        self._max_horizons.pop(episode_id, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-policy-zero")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-message-bytes", type=int, default=67_108_864)
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= 32:
        parser.error("--horizon must be in [1, 32]")
    serve_policy(
        ZeroActionPolicy(horizon=args.horizon),
        args.endpoint,
        max_message_bytes=args.max_message_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ZeroActionPolicy", "main"]

