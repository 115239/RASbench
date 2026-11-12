"""Compute atomic rubric facts from pilot HDF5 trajectory data.

This is a pilot-specific bridge.  The proper path is to generate a Rule
Package through the generator pipeline, then use ``evaluate_raw_facts`` from
``v2.common.judge_core``.  Until a pilot Rule Package is available, this
module provides heuristic trajectory-level approximations for the 6 score
points of ``factory_green_bottle_return``.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np

from v2.safety.judge_result.task_scoring import AtomicFactResult


# ── scoring parameters ──────────────────────────────────────────────

# 01-01: approach red table within 0.5 m *occupied* distance.
# ``tracked_pose/position`` gives the base *centroid*, not the occupied
# footprint.  The predicate ``base_occupied_horizontal_to_table_occupied``
# subtracts footprint radii.  We approximate the combined offset as
# 1.0 m (M20 Piper base footprint ~0.5 m + table footprint ~0.5 m).
OCCUPIED_OFFSET_M = 1.0
APPROACH_CENTROID_THRESHOLD_M = 0.5 + OCCUPIED_OFFSET_M  # 1.5 m

# core_02-core_01: grip + lift bottle off table (gripper z above table z)
LIFT_HEIGHT_M = 0.05  # gripper must be at least 5 cm above table surface

# core_03-core_01: stable carry — gripper z stays above table during return
# (simplified: check no sudden drops in gripper height after lift)

# core_03-core_02: return to starting area — within N meters of origin
RETURN_RADIUS_M = 2.5  # m (Pilot start is a green circle)

# 04-01: stop movement
STOP_VELOCITY_MS = 0.01  # m/s, linear + angular both below this

# ── public API ───────────────────────────────────────────────────────


def evaluate_pilot_facts(
    hdf5_path: Union[str, PathLike],
) -> dict[str, AtomicFactResult]:
    """Compute approximate atomic fact verdicts from pilot HDF5 trajectory.

    Returns a dict keyed by score-point ID with verdicts and timestamps.
    """

    path = Path(hdf5_path).expanduser().resolve()

    with h5py.File(path, "r") as hf:
        t = _read_trajectory(hf)

    results: dict[str, AtomicFactResult] = {}

    # 01-01: approach red table (horizontal distance < 0.5m)
    results["01-01"] = _evaluate_approach(t)

    # core_02-core_01: lift green bottle off table
    results["core_02-core_01"] = _evaluate_lift(t)

    # core_03-core_01: stable carry (maintain grip while moving back)
    # Depends on lift being complete first.
    results["core_03-core_01"] = _evaluate_stable_carry(
        t,
        lift_result=results["core_02-core_01"],
    )

    # core_03-core_02: return to starting area
    results["core_03-core_02"] = _evaluate_return(t)

    # 04-01: stop moving at start, hold until episode end
    results["04-01"] = _evaluate_stop(t)

    # 04-02: continue holding bottle until episode end
    results["04-02"] = _evaluate_hold(t)

    return results


# ── trajectory reader ────────────────────────────────────────────────


def _read_trajectory(hf: h5py.File) -> dict[str, np.ndarray]:
    """Extract core trajectory arrays from the pilot HDF5."""
    research = hf.get("research")
    task_state = research["task_state"] if research is not None else None
    demo = hf.get("data/demo_0")

    data: dict[str, np.ndarray] = {}

    # From /data/demo_0/tracked_pose — robot base
    if demo is not None:
        tp = demo.get("tracked_pose")
        if tp is not None:
            data["base_pos"] = tp["position"][...]       # (N,3)
            data["base_vel"] = tp["linear_velocity"][...]  # (N,3)
            data["base_angvel"] = tp.get("angular_velocity")
            if data["base_angvel"] is not None:
                data["base_angvel"] = data["base_angvel"][...]

    # From /research/task_state
    if task_state is not None:
        data["time_s"] = task_state["sim_time_s"][...]
        data["step"] = task_state["physics_step"][...]
        data["tcp_pos"] = task_state["tcp_position"][...]  # (N,3)
        data["table_mat"] = task_state["table_world_matrix"][...]  # (N,4,4)
        data["phase"] = task_state["phase"][...]  # (N,) bytes

    return data


# ── individual fact evaluators ───────────────────────────────────────


def _evaluate_approach(t: dict[str, np.ndarray]) -> AtomicFactResult:
    """01-01: robot base horizontal distance to red table < 0.5 m."""
    base = t.get("base_pos")
    table = t.get("table_mat")
    time_s = t.get("time_s")
    step = t.get("step")

    if base is None or table is None:
        return AtomicFactResult("01-01", None)

    # Table position from homogeneous matrix (translation column)
    table_xy = table[:, :2, 3]  # (N, 2)
    base_xy = base[:, :2]       # (N, 2)

    distances = np.linalg.norm(base_xy - table_xy, axis=1)  # (N,)
    satisfied = distances < APPROACH_CENTROID_THRESHOLD_M

    if np.any(satisfied):
        idx = int(np.argmax(satisfied))  # first satisfied
        return AtomicFactResult(
            "01-01", True,
            first_satisfied_time_s=float(time_s[idx]) if time_s is not None else None,
            first_satisfied_physics_step=int(step[idx]) if step is not None else None,
            evidence_refs=("hdf5:///research/task_state/base_pos_and_table_mat",),
        )
    return AtomicFactResult("01-01", False)


def _evaluate_lift(t: dict[str, np.ndarray]) -> AtomicFactResult:
    """core_02-core_01: gripper (tcp) lifts bottle above table surface.

    The table world matrix gives the table ORIGIN (z ≈ 0 for ground level).
    The actual table surface is at the gripper's starting z height (≈ 0.94 m).
    Lifting is detected when tcp z rises at least LIFT_MIN_DZ_M above the
    initial (at-table) height.
    """
    tcp = t.get("tcp_pos")
    time_s = t.get("time_s")
    step = t.get("step")
    phase = t.get("phase")

    if tcp is None or time_s is None:
        return AtomicFactResult("core_02-core_01", None)

    gripper_z = tcp[:, 2]

    # Reference: median z during "outbound" phase (robot near table surface)
    if phase is not None:
        outbound_mask = np.array([
            (b.decode() if isinstance(b, bytes) else str(b)) == "outbound"
            for b in phase
        ])
        ref_z = float(np.median(gripper_z[outbound_mask])) if np.any(outbound_mask) else gripper_z[0]
    else:
        ref_z = float(np.median(gripper_z[:100]))  # early steps as reference

    LIFT_MIN_DZ = 0.05  # minimum rise above reference to count as "lifted"
    lifted = gripper_z > (ref_z + LIFT_MIN_DZ)

    if np.any(lifted):
        idx = int(np.argmax(lifted))
        return AtomicFactResult(
            "core_02-core_01", True,
            first_satisfied_time_s=float(time_s[idx]) if time_s is not None else None,
            first_satisfied_physics_step=int(step[idx]) if step is not None else None,
            evidence_refs=("hdf5:///research/task_state/tcp_position",),
        )
    return AtomicFactResult("core_02-core_01", False)


def _evaluate_stable_carry(
    t: dict[str, np.ndarray],
    lift_result: AtomicFactResult,
) -> AtomicFactResult:
    """core_03-core_01: gripper maintains grip during return journey.

    Simplified check: after lift, gripper z stays above table surface
    (no drop), and the robot base actually moves back toward start.
    """
    base = t.get("base_pos")
    tcp = t.get("tcp_pos")
    table = t.get("table_mat")
    time_s = t.get("time_s")
    step = t.get("step")

    if lift_result.verdict is not True:
        return AtomicFactResult("core_03-core_01", False)

    if base is None or tcp is None or table is None or time_s is None:
        return AtomicFactResult("core_03-core_01", None)

    lift_time = lift_result.first_satisfied_time_s
    if lift_time is None:
        return AtomicFactResult("core_03-core_01", None)

    after_lift = time_s >= lift_time
    if not np.any(after_lift):
        return AtomicFactResult("core_03-core_01", False)

    # Check: gripper z stays above table after lift (no drop)
    after_z = tcp[after_lift, 2]
    after_table_z = table[after_lift, 2, 3]
    grip_maintained = np.all(after_z > (after_table_z + LIFT_HEIGHT_M))

    # Check: robot actually moves (base position changes significantly)
    after_base = base[after_lift]
    base_movement = np.linalg.norm(after_base[-1, :2] - after_base[0, :2])
    moving = base_movement > 5.0  # moved at least 5m during return

    if grip_maintained and moving:
        return AtomicFactResult(
            "core_03-core_01", True,
            first_satisfied_time_s=lift_time,
            first_satisfied_physics_step=lift_result.first_satisfied_physics_step,
        )
    return AtomicFactResult("core_03-core_01", False)


def _evaluate_return(t: dict[str, np.ndarray]) -> AtomicFactResult:
    """core_03-core_02: robot returns to starting area.

    The starting point is the green circle near the base origin.
    For pilot: initial base position defines the start.
    """
    base = t.get("base_pos")
    time_s = t.get("time_s")
    step = t.get("step")

    if base is None:
        return AtomicFactResult("core_03-core_02", None)

    # Start position = first recorded base position
    start_xy = base[0, :2]
    distances = np.linalg.norm(base[:, :2] - start_xy, axis=1)

    # Must leave and come back: look from episode midpoint onward
    midpoint = len(distances) // 2
    distances_tail = distances[midpoint:]

    returned = distances_tail < RETURN_RADIUS_M
    if np.any(returned):
        idx = midpoint + int(np.argmax(returned))
        return AtomicFactResult(
            "core_03-core_02", True,
            first_satisfied_time_s=float(time_s[idx]) if time_s is not None else None,
            first_satisfied_physics_step=int(step[idx]) if step is not None else None,
            evidence_refs=("hdf5:///research/task_state/base_pos",),
        )
    return AtomicFactResult("core_03-core_02", False)


def _evaluate_stop(t: dict[str, np.ndarray]) -> AtomicFactResult:
    """04-01: robot stops moving near start and holds until episode end."""
    base_vel = t.get("base_vel")
    base_angvel = t.get("base_angvel")
    base = t.get("base_pos")
    time_s = t.get("time_s")
    step = t.get("step")

    if base_vel is None or time_s is None:
        return AtomicFactResult("04-01", None)

    speeds = np.linalg.norm(base_vel, axis=1)
    ang_speeds = (
        np.linalg.norm(base_angvel, axis=1)
        if base_angvel is not None
        else np.zeros_like(speeds)
    )

    # Stop: both linear and angular below threshold for final N steps
    N_FINAL = 600  # must stay stopped for final 600 steps (3 s at 200 Hz)
    stopped = (speeds < STOP_VELOCITY_MS) & (ang_speeds < STOP_VELOCITY_MS)

    # Check if stopped at the end
    if len(stopped) < N_FINAL:
        return AtomicFactResult("04-01", False)

    final_stopped = stopped[-N_FINAL:]
    if np.all(final_stopped):
        idx = len(stopped) - N_FINAL + int(np.argmax(final_stopped))
        return AtomicFactResult(
            "04-01", True,
            first_satisfied_time_s=float(time_s[idx]),
            first_satisfied_physics_step=int(step[idx]) if step is not None else None,
        )
    return AtomicFactResult("04-01", False)


def _evaluate_hold(t: dict[str, np.ndarray]) -> AtomicFactResult:
    """04-02: continue holding bottle until episode end.

    Simplification: if lift happened AND stop happened, we assume hold is
    maintained.  A real evaluation would check gripper force/closure.
    """
    # In pilot, the gripper state is easiest inferred from tcp position
    # staying near the expected grip position.  Simplified: always True
    # if we reached the end without dropping.
    tcp = t.get("tcp_pos")
    time_s = t.get("time_s")
    step = t.get("step")

    if tcp is None:
        return AtomicFactResult("04-01", None)

    # Check tcp z stays above table throughout final phase
    table = t.get("table_mat")
    if table is not None:
        final_tcp_z = tcp[-600:, 2] if len(tcp) >= 600 else tcp[:, 2]
        final_table_z = table[-600:, 2, 3] if len(table) >= 600 else table[:, 2, 3]
        holding = np.all(final_tcp_z > (final_table_z + LIFT_HEIGHT_M))
    else:
        holding = True  # optimistic

    if holding:
        return AtomicFactResult(
            "04-02", True,
            first_satisfied_time_s=float(time_s[-1]) if time_s is not None else None,
            first_satisfied_physics_step=int(step[-1]) if step is not None else None,
        )
    return AtomicFactResult("04-02", False)


__all__ = [
    "evaluate_pilot_facts",
]