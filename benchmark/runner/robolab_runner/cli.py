# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public ``robolab-runner`` command line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from robolab_runner.errors import RunnerError
from robolab_runner.runtime import doctor, record, validate_directory

_RECORD_WORKER_ENV = "ROBOLAB_RUNNER_INTERNAL_RECORD_WORKER"


def _policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument(
        "--allow-remote-policy",
        action="store_true",
        help="Explicitly allow a non-loopback unencrypted ZMQ endpoint (qualification only).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robolab-runner",
        description="Public RoboLab V4 competition Runner. It connects to, but never starts, a policy service.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="Verify contract/assets and exercise the policy protocol.")
    _policy_arguments(doctor_parser)
    record_parser = commands.add_parser("record", help="Record one fixed V4 submission directory.")
    _policy_arguments(record_parser)
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--device", default="cuda:0")
    validate_parser = commands.add_parser("validate", help="Validate a movable submission directory.")
    validate_parser.add_argument("--contract", type=Path, required=True)
    validate_parser.add_argument("--submission-dir", type=Path, required=True)
    return parser


def _record_in_subprocess(args: argparse.Namespace) -> int:
    """Keep a parent alive to verify publication after Isaac shuts its worker down."""
    command = [
        sys.executable,
        "-m",
        "robolab_runner.cli",
        "record",
        "--contract",
        str(args.contract),
        "--asset-root",
        str(args.asset_root),
        "--policy-endpoint",
        str(args.policy_endpoint),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
    ]
    if args.allow_remote_policy:
        command.append("--allow-remote-policy")
    environment = dict(os.environ)
    environment[_RECORD_WORKER_ENV] = "1"
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        return 2
    result = validate_directory(args.contract, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result.get("status") == "VALID_SUBMISSION" else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record" and os.environ.get(_RECORD_WORKER_ENV) != "1":
        try:
            return _record_in_subprocess(args)
        except (RunnerError, FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"robolab-runner: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
    try:
        if args.command == "doctor":
            result = doctor(
                args.contract,
                args.asset_root,
                args.policy_endpoint,
                allow_remote_policy=args.allow_remote_policy,
            )
        elif args.command == "record":
            record(
                args.contract,
                args.asset_root,
                args.policy_endpoint,
                args.output_dir,
                device=args.device,
                allow_remote_policy=args.allow_remote_policy,
            )
            return 0
        else:
            result = validate_directory(args.contract, args.submission_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result.get("status") in {"READY", "VALID_SUBMISSION"} else 2
    except (RunnerError, FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"robolab-runner: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
