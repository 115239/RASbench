"""Stateful trajectory predicates for stable object manipulation.

These evaluators deliberately retain the history of one manipulation attempt.
They are therefore different from pointwise contact or pose predicates, whose
temporal continuity is normally supplied by a rule package's temporal reducer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .contact import contact_present
from .trace import PredicateValidationError, finite_array, finite_float, pose_series


def _initial_pose(value: object) -> np.ndarray:
    """Validate and normalize the immutable initial world pose."""

    initial = finite_array(
        value,
        "initial_object_pose",
        ndim=1,
        shape_tail=(7,),
    ).copy()
    quaternion_norm = float(np.linalg.norm(initial[3:7]))
    if quaternion_norm <= 1e-12:
        raise PredicateValidationError("initial_object_pose contains a zero quaternion")
    initial[3:7] /= quaternion_norm
    return initial


def _quaternion_delta_deg(quaternions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return physical quaternion distance in degrees, invariant to q/-q."""

    cosine = np.clip(np.abs(quaternions @ reference), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(cosine))


def _maximum_or_zero(values: np.ndarray) -> float:
    """Keep empty traces metrically valid without distorting nonempty data."""

    return float(np.max(values)) if values.size else 0.0


def evaluate_object_stably_lifted(
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    time_s: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate one uninterrupted, orientation-stable grasp-and-lift attempt.

    ``gripper_contact_force`` establishes the attempt's start at its first
    effective-contact sample.  From that sample onward, either loss of contact
    or an orientation excursion is terminal: a later re-grasp cannot make the
    same attempt pass.  At a passing sample the object's world ``+Z`` position
    must also be at least ``min_height_delta_m`` above its immutable initial
    pose.

    Inputs use the standard RoboLab conventions: poses are ``[T, 7]`` and
    quaternions are ``wxyz``.  The supplied initial pose is ``[7]``.
    """

    length = len(time_s)
    pose = pose_series(inputs["object_pose"], "object_pose", length)
    initial = _initial_pose(inputs["initial_object_pose"])
    min_contact_force_n = finite_float(
        params["min_contact_force_n"],
        "min_contact_force_n",
        nonnegative=True,
    )
    min_height_delta_m = finite_float(
        params["min_height_delta_m"],
        "min_height_delta_m",
        positive=True,
    )
    max_orientation_delta_deg = finite_float(
        params["max_orientation_delta_deg"],
        "max_orientation_delta_deg",
        nonnegative=True,
    )
    if max_orientation_delta_deg > 180.0:
        raise PredicateValidationError("max_orientation_delta_deg must be <= 180")

    in_contact, contact_norms = contact_present(
        inputs["gripper_contact_force"],
        length,
        min_contact_force_n,
        "gripper_contact_force",
    )
    height_delta_m = pose[:, 2] - initial[2]
    orientation_delta_deg = _quaternion_delta_deg(pose[:, 3:7], initial[3:7])

    first_contact_indices = np.flatnonzero(in_contact)
    matched = np.zeros(length, dtype=bool)
    first_contact_index: int | None = None
    first_contact_time_s: float | None = None
    first_interruption_index: int | None = None
    first_interruption_time_s: float | None = None

    if first_contact_indices.size:
        first_contact_index = int(first_contact_indices[0])
        # time_s is validated by PredicateTrace upstream.  Keeping it in the
        # metrics makes the grasp interval independently auditable.
        first_contact_time_s = float(time_s[first_contact_index])
        post_start_contact = in_contact[first_contact_index:]
        post_start_orientation = (
            orientation_delta_deg[first_contact_index:]
            <= max_orientation_delta_deg
        )
        uninterrupted_contact = np.logical_and.accumulate(post_start_contact)
        stable_orientation = np.logical_and.accumulate(post_start_orientation)
        valid_history = uninterrupted_contact & stable_orientation
        matched[first_contact_index:] = valid_history & (
            height_delta_m[first_contact_index:] >= min_height_delta_m
        )

        interruption = ~(post_start_contact & post_start_orientation)
        interruption_indices = np.flatnonzero(interruption)
        if interruption_indices.size:
            first_interruption_index = int(
                first_contact_index + interruption_indices[0]
            )
            first_interruption_time_s = float(time_s[first_interruption_index])

    return matched, {
        "first_grasp_contact_index": first_contact_index,
        "first_grasp_contact_time_s": first_contact_time_s,
        "first_stability_interruption_index": first_interruption_index,
        "first_stability_interruption_time_s": first_interruption_time_s,
        "maximum_height_delta_m": _maximum_or_zero(height_delta_m),
        "maximum_orientation_delta_deg": _maximum_or_zero(orientation_delta_deg),
        "maximum_gripper_contact_force_n": _maximum_or_zero(contact_norms),
        "min_contact_force_n": min_contact_force_n,
        "min_height_delta_m": min_height_delta_m,
        "max_orientation_delta_deg": max_orientation_delta_deg,
    }
