"""Temporal reduction for atomic predicate traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .trace import PredicateTrace, PredicateValidationError, TrueRun, crop_after_step, finite_float, true_runs


@dataclass(frozen=True, slots=True)
class TemporalResult:
    passed: bool
    matched_step_start: int | None
    matched_step_end: int | None
    matched_duration_s: float
    mode: str


def _result(passed: bool, mode: str, run: TrueRun | None = None) -> TemporalResult:
    return TemporalResult(
        passed=passed,
        matched_step_start=run.start_step if run else None,
        matched_step_end=run.end_step if run else None,
        matched_duration_s=run.duration_s if run else 0.0,
        mode=mode,
    )


def reduce_trace(
    trace: PredicateTrace,
    temporal: Mapping[str, Any],
    after_step: int | None = None,
) -> TemporalResult:
    if not isinstance(temporal, Mapping):
        raise PredicateValidationError("temporal must be a mapping")
    mode = temporal.get("mode")
    if mode not in {"eventually", "final", "always", "held_for"}:
        raise PredicateValidationError("temporal.mode must be eventually, final, always, or held_for")
    allowed = {"mode", "min_duration_s"} if mode == "held_for" else {"mode"}
    if set(temporal) != allowed:
        raise PredicateValidationError(f"temporal keys for {mode} must be exactly {sorted(allowed)}")
    current = crop_after_step(trace, after_step)
    if not len(current.control_steps):
        return _result(False, mode)
    runs = true_runs(current)
    if mode == "eventually":
        if not runs:
            return _result(False, mode)
        first = runs[0]
        # ``eventually`` is confirmed by the first matching control step.  Do
        # not extend its dependency completion point to the end of a sustained
        # true run, otherwise a grasp that remains true until release would
        # incorrectly block the intervening lift action.
        decisive = TrueRun(
            start_index=first.start_index,
            end_index=first.start_index,
            start_step=first.start_step,
            end_step=first.start_step,
            duration_s=0.0,
        )
        return _result(True, mode, decisive)
    if mode == "final":
        if not current.matched[-1]:
            return _result(False, mode)
        return _result(True, mode, runs[-1])
    if mode == "always":
        if not np.all(current.matched):
            return _result(False, mode)
        return _result(True, mode, runs[0])
    duration = finite_float(temporal["min_duration_s"], "min_duration_s", nonnegative=True)
    run = next((item for item in runs if item.duration_s + 1e-12 >= duration), None)
    if run is None:
        return _result(False, mode)
    # A held condition is confirmed at the earliest sample whose real elapsed
    # time reaches the requested duration, not at the eventual end of a longer
    # true run.  This value is the dependency completion boundary.
    indices = np.arange(run.start_index, run.end_index + 1)
    elapsed = current.time_s[indices] - current.time_s[run.start_index]
    decisive_end = int(indices[np.flatnonzero(elapsed + 1e-12 >= duration)[0]])
    decisive = TrueRun(
        start_index=run.start_index,
        end_index=decisive_end,
        start_step=run.start_step,
        end_step=int(current.control_steps[decisive_end]),
        duration_s=float(current.time_s[decisive_end] - current.time_s[run.start_index]),
    )
    return _result(True, mode, decisive)
