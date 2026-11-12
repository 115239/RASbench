# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone policy protocol conformance probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robolab_runner.contract import V4TaskContract
from robolab_runner.errors import RunnerError
from robolab_runner.policy.client import PolicyClient
from robolab_runner.policy.observations import zero_observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolab-policy-conformance")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--allow-remote-policy", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract = V4TaskContract.load(args.contract)
        with PolicyClient(
            contract,
            args.policy_endpoint,
            episode_id="conformance",
            allow_remote=args.allow_remote_policy,
        ) as client:
            identity = client.handshake()
            client.reset()
            actions, duration = client.infer(zero_observations(contract), step=0)
            client.end_episode(steps=0, status="conformance")
        print(
            json.dumps(
                {
                    "status": "CONFORMANT",
                    "policy_id": identity.policy_id,
                    "policy_version": identity.policy_version,
                    "action_shape": list(actions.shape),
                    "inference_duration_s": duration,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (RunnerError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"robolab-policy-conformance: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
