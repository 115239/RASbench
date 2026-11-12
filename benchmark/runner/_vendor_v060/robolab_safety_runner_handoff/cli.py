"""Command-line entry point for safety Runner delivery validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from robolab_action_schema import FORMAL_ROBOT_PROFILES

from .validation import DeliveryValidationError, validate_runner_delivery


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robolab-safety-runner-validate",
        description=(
            "Validate a finalized safety Runner submission.hdf5 against the "
            "formal robot and fixed-evidence baseline."
        ),
    )
    parser.add_argument("submission", help="path to submission.hdf5")
    parser.add_argument(
        "--robot-role",
        choices=tuple(FORMAL_ROBOT_PROFILES),
        default="m20_piper",
        help="expected formal robot role (default: m20_piper)",
    )
    parser.add_argument(
        "--task-id",
        help="expected task ID when no requirements file supplies it",
    )
    parser.add_argument(
        "--output",
        help="optional JSON report path; stdout is always populated",
    )
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _validate_output_target(args: argparse.Namespace) -> None:
    if args.output is None:
        return
    destination = Path(args.output).expanduser().resolve()
    protected = {Path(args.submission).expanduser().resolve()}
    if destination in protected:
        raise DeliveryValidationError(
            "OUTPUT_PATH_CONFLICT",
            "validation report must not overwrite the submission",
            details={"output": str(destination)},
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_output_target(args)
        report = validate_runner_delivery(
            args.submission,
            expected_task_id=args.task_id,
            expected_role=args.robot_role,
        )
    except DeliveryValidationError as error:
        _emit(
            error.as_report(),
            None if error.code == "OUTPUT_PATH_CONFLICT" else args.output,
        )
        return 2
    _emit(report, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
