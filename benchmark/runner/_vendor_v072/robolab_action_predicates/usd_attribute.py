"""Predicates over Runner-recorded, allowlisted USD attribute evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .trace import PredicateValidationError


def evaluate_usd_attribute_equals(
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    time_s: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compare a control-aligned USD boolean attribute with one expected value."""

    value = np.asarray(inputs["value"])
    if value.ndim != 1:
        raise PredicateValidationError("value must have one dimension")
    if value.shape != (len(time_s),):
        raise PredicateValidationError("value must have shape [T]")
    if value.dtype != np.dtype(np.uint8) and value.dtype != np.dtype(np.bool_):
        raise PredicateValidationError("value must use uint8 or boolean dtype")
    if np.any((value != 0) & (value != 1)):
        raise PredicateValidationError("value must contain only boolean 0 or 1")
    expected = params.get("expected_value")
    if not isinstance(expected, (bool, np.bool_)):
        raise PredicateValidationError("expected_value must be boolean")
    matched = value.astype(np.bool_, copy=False) == bool(expected)
    transitions = np.flatnonzero(np.diff(value.astype(np.int8)) != 0) + 1
    return matched, {
        "expected_value": bool(expected),
        "initial_observed_value": bool(value[0]),
        "final_observed_value": bool(value[-1]),
        "transition_count": int(transitions.size),
        "first_transition_index": (
            int(transitions[0]) if transitions.size else None
        ),
        "first_transition_time_s": (
            float(time_s[int(transitions[0])]) if transitions.size else None
        ),
    }
