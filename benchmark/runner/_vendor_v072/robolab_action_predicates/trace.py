"""Validated predicate traces and dependency-time helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class PredicateValidationError(ValueError):
    """The declared evidence cannot be evaluated without guessing."""


def finite_float(value: object, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PredicateValidationError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise PredicateValidationError(f"{name} must be a finite number")
    if nonnegative and result < 0:
        raise PredicateValidationError(f"{name} must be nonnegative")
    if positive and result <= 0:
        raise PredicateValidationError(f"{name} must be positive")
    return result


def finite_array(value: object, name: str, *, ndim: int | None = None, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PredicateValidationError(f"{name} must be a finite numeric array") from error
    if ndim is not None and array.ndim != ndim:
        raise PredicateValidationError(f"{name} must have {ndim} dimensions")
    if shape_tail and (array.ndim < len(shape_tail) or array.shape[-len(shape_tail) :] != shape_tail):
        raise PredicateValidationError(f"{name} must end with shape {shape_tail}")
    if not np.all(np.isfinite(array)):
        raise PredicateValidationError(f"{name} must contain only finite values")
    return array


def vector_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name, ndim=2, shape_tail=(3,))
    if array.shape[0] != length:
        raise PredicateValidationError(f"{name} must have {length} samples")
    return array


def scalar_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name, ndim=1)
    if array.shape[0] != length:
        raise PredicateValidationError(f"{name} must have {length} samples")
    return array


def pose_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name, ndim=2, shape_tail=(7,))
    if array.shape[0] != length:
        raise PredicateValidationError(f"{name} must have {length} samples")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    if np.any(norms <= 1e-12):
        raise PredicateValidationError(f"{name} contains a zero quaternion")
    result = array.copy()
    result[:, 3:7] /= norms[:, None]
    return result


def constant_or_pose_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name)
    if array.shape == (7,):
        array = np.broadcast_to(array, (length, 7))
    elif array.shape != (length, 7):
        raise PredicateValidationError(f"{name} must have shape [7] or [{length}, 7]")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    if np.any(norms <= 1e-12):
        raise PredicateValidationError(f"{name} contains a zero quaternion")
    result = np.array(array, dtype=np.float64, copy=True)
    result[:, 3:7] /= norms[:, None]
    return result


def constant_or_vector_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name)
    if array.shape == (3,):
        return np.broadcast_to(array, (length, 3))
    if array.shape == (length, 3):
        return array
    raise PredicateValidationError(f"{name} must have shape [3] or [{length}, 3]")


def contact_force_series(value: object, name: str, length: int) -> np.ndarray:
    array = finite_array(value, name)
    if array.ndim not in {2, 3} or array.shape[0] != length or array.shape[-1] != 3:
        raise PredicateValidationError(f"{name} must have shape [T, 3] or [T, C, 3]")
    return array


def force_norm(force: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(force, axis=-1)
    return np.max(norms, axis=1) if norms.ndim == 2 else norms


def _validate_metric(value: Any, name: str = "metrics") -> None:
    if isinstance(value, np.ndarray):
        if value.size > 1024:
            raise PredicateValidationError(f"{name} arrays are limited to 1024 values")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise PredicateValidationError(f"{name} contains non-finite values")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PredicateValidationError(f"{name} keys must be strings")
            _validate_metric(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        if len(value) > 1024:
            raise PredicateValidationError(f"{name} lists are limited to 1024 values")
        for item in value:
            _validate_metric(item, name)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise PredicateValidationError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class PredicateTrace:
    control_steps: np.ndarray
    time_s: np.ndarray
    matched: np.ndarray
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        steps = np.asarray(self.control_steps)
        times = np.asarray(self.time_s, dtype=np.float64)
        matched = np.asarray(self.matched)
        if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
            raise PredicateValidationError("control_steps must be a one-dimensional integer array")
        if np.any(steps < 0) or (len(steps) > 1 and np.any(np.diff(steps.astype(np.int64)) <= 0)):
            raise PredicateValidationError("control_steps must be unique, nonnegative, and strictly increasing")
        if times.shape != steps.shape or not np.all(np.isfinite(times)):
            raise PredicateValidationError("time_s must be a finite one-dimensional array matching control_steps")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise PredicateValidationError("time_s must be strictly increasing")
        if matched.shape != steps.shape or matched.dtype != np.bool_:
            raise PredicateValidationError("matched must be a boolean array matching control_steps")
        if not isinstance(self.metrics, dict):
            raise PredicateValidationError("metrics must be a dictionary")
        _validate_metric(self.metrics)
        steps = steps.astype(np.uint64, copy=True)
        times = times.astype(np.float64, copy=True)
        matched = matched.astype(np.bool_, copy=True)
        steps.setflags(write=False)
        times.setflags(write=False)
        matched.setflags(write=False)
        object.__setattr__(self, "control_steps", steps)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "matched", matched)


@dataclass(frozen=True, slots=True)
class TrueRun:
    start_index: int
    end_index: int
    start_step: int
    end_step: int
    duration_s: float


def true_runs(trace: PredicateTrace) -> tuple[TrueRun, ...]:
    indices = np.flatnonzero(trace.matched)
    if not indices.size:
        return ()
    boundaries = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(indices) - 1]
    result: list[TrueRun] = []
    for start_at, end_at in zip(starts, ends, strict=True):
        start = int(indices[start_at])
        end = int(indices[end_at])
        result.append(
            TrueRun(
                start_index=start,
                end_index=end,
                start_step=int(trace.control_steps[start]),
                end_step=int(trace.control_steps[end]),
                duration_s=float(trace.time_s[end] - trace.time_s[start]),
            )
        )
    return tuple(result)


def crop_after_step(trace: PredicateTrace, after_step: int | None) -> PredicateTrace:
    if after_step is None:
        return trace
    if isinstance(after_step, bool) or not isinstance(after_step, (int, np.integer)) or int(after_step) < 0:
        raise PredicateValidationError("after_step must be a nonnegative integer")
    keep = trace.control_steps > int(after_step)
    return PredicateTrace(
        control_steps=trace.control_steps[keep],
        time_s=trace.time_s[keep],
        matched=trace.matched[keep],
        metrics=dict(trace.metrics),
    )
