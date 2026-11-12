# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge a whitelist-owned environment producer API into :class:`CaptureSession`."""

from __future__ import annotations

from typing import Any

from robolab_runner.capture import CaptureSession
from robolab_runner.errors import SubmissionError


def capture_physics_substep(
    session: CaptureSession,
    env: Any,
    control_step: int,
    physics_step: int,
    sim_time_s: float,
) -> None:
    direct = bool(session.contract.is_v4_3 and session.contract.direct_scoring and session.contract.direct_scoring.lidar_audit)
    producer_name = "competition_direct_scoring_capture_samples" if direct else "competition_capture_samples"
    producer = getattr(env, producer_name, None)
    if not callable(producer):
        raise SubmissionError(f"competition environment does not expose {producer_name}")
    samples = producer(
        control_step=int(control_step),
        physics_step=int(physics_step),
        sim_time_s=float(sim_time_s),
    )
    if not isinstance(samples, list):
        raise SubmissionError("competition_capture_samples must return a list")
    for sample in samples:
        if not isinstance(sample, dict):
            raise SubmissionError("competition capture sample must be a mapping")
        required = {"kind", "name", "sample_id", "step_index", "sim_time_s", "values"}
        allowed = required | {"judge_physics_step", "judge_audit", "attack_profile"}
        if not required <= set(sample) or not set(sample) <= allowed:
            raise SubmissionError("competition capture sample has missing or unexpected keys")
        session.record_stream(
            str(sample["kind"]),
            str(sample["name"]),
            sample_id=int(sample["sample_id"]),
            step_index=int(sample["step_index"]),
            sim_time_s=float(sample["sim_time_s"]),
            values=sample["values"],
            physics_step=int(sample.get("judge_physics_step", physics_step)),
            direct_audit=sample.get("judge_audit"),
            attack_profile=sample.get("attack_profile"),
        )


__all__ = ["capture_physics_substep"]
