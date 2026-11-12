"""Offline contact, grasp, and release predicates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .trace import PredicateValidationError, contact_force_series, finite_float, force_norm


def contact_present(force: object, length: int, threshold_n: float, name: str = "contact_force") -> tuple[np.ndarray, np.ndarray]:
    values = contact_force_series(force, name, length)
    norms = force_norm(values)
    return norms >= threshold_n, norms


def evaluate_object_in_contact(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
    matched, norms = contact_present(inputs["contact_force"], len(time_s), threshold)
    return matched, {"maximum_contact_force_n": float(np.max(norms, initial=0.0)), "min_contact_force_n": threshold}


def evaluate_object_grabbed(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
    matched, norms = contact_present(inputs["gripper_contact_force"], len(time_s), threshold, "gripper_contact_force")
    return matched, {"maximum_gripper_contact_force_n": float(np.max(norms, initial=0.0)), "min_contact_force_n": threshold}


def evaluate_object_released(inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray):
    threshold = finite_float(params["min_contact_force_n"], "min_contact_force_n", nonnegative=True)
    contact, norms = contact_present(inputs["gripper_contact_force"], len(time_s), threshold, "gripper_contact_force")
    previously_grabbed = np.r_[False, np.maximum.accumulate(contact.astype(np.uint8))[:-1].astype(bool)]
    matched = previously_grabbed & ~contact
    if "gripper_open" in inputs:
        opened = np.asarray(inputs["gripper_open"])
        if opened.shape != (len(time_s),) or opened.dtype != np.bool_:
            raise PredicateValidationError("gripper_open must be a boolean array with shape [T]")
        matched &= opened
    return matched, {
        "maximum_gripper_contact_force_n": float(np.max(norms, initial=0.0)),
        "was_ever_grabbed": bool(np.any(contact)),
        "min_contact_force_n": threshold,
    }


def evaluate_force_magnitude_below(
    inputs: Mapping[str, Any], params: Mapping[str, Any], time_s: np.ndarray
):
    """Return a trace whose samples stay at or below a declared force limit."""

    maximum = finite_float(params["max_force_n"], "max_force_n", nonnegative=True)
    values = contact_force_series(inputs["force"], "force", len(time_s))
    norms = force_norm(values)
    matched = norms <= maximum
    return matched, {
        "maximum_observed_force_n": float(np.max(norms, initial=0.0)),
        "max_force_n": maximum,
    }
