# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner-side V4.3 task evidence using the same state engines as Judge replay."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from robolab_runner.contract import V4TaskContract
from robolab_runner.errors import RunnerError


def create_direct_engine(contract: V4TaskContract, env: Any) -> Any | None:
    """Create the contract-selected scoring engine after Isaac has started."""
    if not contract.is_v4_3:
        return None
    if contract.direct_scoring is None:
        raise RunnerError("V4.3 contract has no direct_scoring specification")
    if contract.direct_scoring.engine == "competition_task_state":
        getter = getattr(env, "competition_task_state", None)
        if not callable(getter):
            raise RunnerError("V4.3 environment does not expose competition_task_state")
        return None
    from robolab.core.scoring.task_engine import TaskScoringEngine

    engine = TaskScoringEngine(env)
    if engine.subtask_state_machine is None:
        raise RunnerError("V4.3 task does not expose TaskScoringEngine subtasks")
    return engine


def reset_direct_engine(contract: V4TaskContract, engine: Any | None) -> None:
    if contract.is_v4_3 and engine is not None:
        engine.reset()


def _raw_state(contract: V4TaskContract, env: Any, engine: Any | None, *, advance: bool) -> dict[str, Any]:
    if contract.direct_scoring is None:
        raise RunnerError("V4.3 contract has no direct_scoring specification")
    if contract.direct_scoring.engine == "task_scoring_engine":
        if engine is None:
            raise RunnerError("V4.3 TaskScoringEngine is unavailable")
        if advance:
            engine.step()
        machine = engine.subtask_state_machine
        success = bool(machine is not None and machine.is_complete())
        getter = getattr(env, "competition_rule_state", None)
        rule_state = dict(getter()) if callable(getter) else {}
        return {**dict(engine.info), **rule_state, "success": success}
    getter = getattr(env, "competition_task_state", None)
    if not callable(getter):
        raise RunnerError("V4.3 environment does not expose competition_task_state")
    value = getter()
    if not isinstance(value, Mapping):
        raise RunnerError("competition_task_state must return a mapping")
    result = dict(value)
    success = bool(result.get("success", False))
    result.update(
        {
            "status": int(result.get("status", 1 if success else 0)),
            "completed": int(result.get("completed", 1 if success else 0)),
            "total": int(result.get("total", 1)),
            "score": float(result.get("score", result.get("task_score", 0.0))),
            "success": success,
        }
    )
    return result


def normalize_direct_state(contract: V4TaskContract, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Select and type only contract-frozen task fields."""
    spec = contract.direct_scoring
    if spec is None:
        raise RunnerError("V4.3 contract has no direct_scoring specification")
    normalized: dict[str, np.ndarray] = {}
    for field in spec.task_fields:
        if field.name not in raw:
            raise RunnerError(f"V4.3 task state is missing contract field {field.name!r}")
        array = np.asarray(raw[field.name], dtype=np.dtype(field.dtype))
        if array.shape == (1,) and field.shape == ():
            array = array.reshape(())
        if array.shape != field.shape:
            raise RunnerError(
                f"V4.3 task field {field.name!r} must have shape {field.shape}, got {array.shape}"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise RunnerError(f"V4.3 task field {field.name!r} contains NaN or infinity")
        normalized[field.name] = array
    return normalized


def capture_direct_step(contract: V4TaskContract, env: Any, engine: Any | None) -> dict[str, np.ndarray]:
    return normalize_direct_state(contract, _raw_state(contract, env, engine, advance=True))


def finalize_direct_episode(
    contract: V4TaskContract,
    env: Any,
    engine: Any | None,
    *,
    steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Finalize task progress and return the sealed final row/result/events."""
    if contract.direct_scoring is None:
        raise RunnerError("V4.3 contract has no direct_scoring specification")
    if contract.direct_scoring.engine == "task_scoring_engine":
        if engine is None:
            raise RunnerError("V4.3 TaskScoringEngine is unavailable")
        machine = engine.subtask_state_machine
        task_complete = bool(machine is not None and machine.is_complete())
        if not task_complete:
            engine.finalize()
        raw = _raw_state(contract, env, engine, advance=False)
        raw["success"] = task_complete
        events = [dict(event) for event in engine.get_events(0)]
    else:
        raw = _raw_state(contract, env, engine, advance=False)
        task_complete = bool(raw.get("success", False))
        events = []
    final_row = normalize_direct_state(contract, raw)
    final_task_state = {
        name: value.item() if value.shape == () else value.tolist() for name, value in final_row.items()
    }
    final_result = {
        "authority": "runner_direct_capture",
        "scoring_source": "participant_direct",
        "success": task_complete,
        "task_complete": task_complete,
        "task_score": float(final_row["score"]),
        "status": int(final_row["status"]),
        "completed": int(final_row["completed"]),
        "total": int(final_row["total"]),
        "reason": str(raw.get("reason", raw.get("info", ""))),
        "steps_executed": int(steps),
        "submitted_steps": int(steps),
        "task": final_task_state,
    }
    return final_row, final_result, events


__all__ = [
    "capture_direct_step",
    "create_direct_engine",
    "finalize_direct_episode",
    "normalize_direct_state",
    "reset_direct_engine",
]
