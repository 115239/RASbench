# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract-exact extraction of public observations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from robolab_runner.contract import V4TaskContract
from robolab_runner.errors import ProtocolError


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_observations(observations: Mapping[str, Any], contract: V4TaskContract) -> dict[str, np.ndarray]:
    expected_paths = {leaf.path for leaf in contract.observation_leaves}
    if set(observations) != expected_paths:
        missing = sorted(expected_paths - set(observations))
        extra = sorted(set(observations) - expected_paths)
        raise ProtocolError(f"policy observation paths differ from contract: missing={missing}, extra={extra}")
    result: dict[str, np.ndarray] = {}
    for leaf in contract.observation_leaves:
        array = _numpy(observations[leaf.path])
        if array.dtype != np.dtype(leaf.dtype):
            raise ProtocolError(
                f"observation {leaf.path} dtype {array.dtype} differs from contract {leaf.dtype}"
            )
        if array.shape != leaf.shape:
            raise ProtocolError(
                f"observation {leaf.path} shape {array.shape} differs from contract {leaf.shape}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ProtocolError(f"observation {leaf.path} contains NaN or infinity")
        result[leaf.path] = np.ascontiguousarray(array)
    return result


def extract_observations(
    raw_obs: Mapping[str, Any],
    contract: V4TaskContract,
    *,
    env_id: int = 0,
) -> dict[str, np.ndarray]:
    """Extract only frozen image/proprio leaves and remove the environment dimension."""
    extracted: dict[str, np.ndarray] = {}
    for leaf in contract.observation_leaves:
        value: Any = raw_obs
        for segment in leaf.path.split("/"):
            if not isinstance(value, Mapping) or segment not in value:
                raise ProtocolError(f"live observation is missing contract path {leaf.path!r}")
            value = value[segment]
        array = _numpy(value)
        if array.ndim < 1 or array.shape[0] <= env_id:
            raise ProtocolError(
                f"live observation {leaf.path} lacks environment row {env_id}; shape={array.shape}"
            )
        extracted[leaf.path] = np.ascontiguousarray(array[env_id])
    return validate_observations(extracted, contract)


def zero_observations(contract: V4TaskContract) -> dict[str, np.ndarray]:
    return {leaf.path: np.zeros(leaf.shape, dtype=np.dtype(leaf.dtype)) for leaf in contract.observation_leaves}


__all__ = ["extract_observations", "validate_observations", "zero_observations"]
