# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HDF5 metadata encoding for participant schema V4."""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from robolab_runner.contract import POLICY_PROTOCOL_NAME, V4TaskContract

METADATA_GROUP = "metadata"
DATA_GROUP = "data"
DEMO_NAME = "demo_0"
V4_0_METADATA_ATTRIBUTES = frozenset(
    {
        "schema_name",
        "schema_version",
        "producer_type",
        "complete",
        "contract_id",
        "contract_sha256",
        "task_env",
        "task_class",
        "instruction",
        "seed",
        "robolab_version",
        "isaaclab_version",
        "isaacsim_version",
        "runner_version",
        "policy_protocol_name",
        "policy_protocol_version",
        "policy_observation_schema_sha256",
        "policy_limits_sha256",
        "policy_endpoint_scope",
        "policy_id",
        "policy_version",
        "environment_config_sha256",
        "asset_manifest_sha256",
        "rule_package_id",
        "rule_package_sha256",
        "action_type",
        "action_dim",
        "control_dt",
        "ee_body_name",
        "joint_names_json",
        "policy_observation_leaves_json",
        "policy_limits_json",
    }
)
V4_1_METADATA_ATTRIBUTES = frozenset(
    (V4_0_METADATA_ATTRIBUTES - {"ee_body_name"})
    | {
        "format_revision",
        "physics_dt",
        "coordinate_convention",
        "quaternion_order",
        "env_cfg_json",
        "entity_spec_json",
        "capture_spec_json",
        "primary_tracking_body",
    }
)
V4_3_METADATA_ATTRIBUTES = frozenset(
    V4_1_METADATA_ATTRIBUTES
    | {
        "direct_scoring_schema",
        "direct_scoring_source",
        "direct_lidar_attack_profile_json",
    }
)
SUBMISSION_METADATA_ATTRIBUTES = V4_0_METADATA_ATTRIBUTES


def _submission_metadata_values(
    contract: V4TaskContract,
    *,
    runner_version: str,
    endpoint_scope: str,
    policy_id: str,
    policy_version: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema_name": contract.schema_name,
        "schema_version": np.int64(contract.schema_version),
        "producer_type": "robolab_competition_runner",
        "complete": np.bool_(False),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest,
        "task_env": contract.task_env,
        "task_class": contract.task_class,
        "instruction": contract.instruction,
        "seed": np.int64(contract.seed),
        "robolab_version": contract.versions["robolab"],
        "isaaclab_version": contract.versions["isaaclab"],
        "isaacsim_version": contract.versions["isaacsim"],
        "runner_version": runner_version,
        "policy_protocol_name": POLICY_PROTOCOL_NAME,
        "policy_protocol_version": np.int64(contract.policy_protocol_version),
        "policy_observation_schema_sha256": contract.observation_schema_sha256,
        "policy_limits_sha256": contract.policy_limits_sha256,
        "policy_endpoint_scope": endpoint_scope,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "environment_config_sha256": contract.environment_config_sha256,
        "asset_manifest_sha256": contract.asset_manifest_sha256,
        "rule_package_id": contract.rule_package_id,
        "rule_package_sha256": contract.rule_package_sha256,
        "action_type": contract.control.action_type,
        "action_dim": np.int64(contract.control.action_dim),
        "control_dt": np.float64(contract.control.control_dt),
        "joint_names_json": json.dumps(list(contract.control.joint_names), ensure_ascii=False, separators=(",", ":")),
        "policy_observation_leaves_json": json.dumps(
            [leaf.to_dict() for leaf in contract.observation_leaves],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "policy_limits_json": json.dumps(
            contract.policy_limits.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }
    if contract.uses_full_capture:
        if contract.runner_capture is None:
            raise ValueError("V4.1 contract lacks runner_capture")
        values.update(
            {
                "format_revision": contract.format_revision,
                "physics_dt": np.float64(contract.runner_capture.physics_dt),
                "coordinate_convention": contract.runner_capture.coordinate_convention,
                "quaternion_order": contract.runner_capture.quaternion_order,
                "env_cfg_json": json.dumps(
                    contract.environment_config,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "entity_spec_json": json.dumps(
                    contract.runner_capture.entity_spec,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "capture_spec_json": json.dumps(
                    contract.runner_capture.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "primary_tracking_body": contract.primary_tracking_body,
            }
        )
        if contract.is_v4_3:
            if contract.direct_scoring is None:
                raise ValueError("V4.3 contract lacks direct_scoring")
            values.update(
                {
                    "direct_scoring_schema": contract.direct_scoring.schema,
                    "direct_scoring_source": "participant_direct",
                    "direct_lidar_attack_profile_json": "{}",
                }
            )
    else:
        values["ee_body_name"] = contract.primary_tracking_body
    return values


def _attribute_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, np.bytes_)):
        return len(bytes(value))
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U"}:
        return sum(_attribute_size(item) for item in array.reshape(-1))
    return int(array.nbytes)


def submission_metadata_attribute_bytes(
    contract: V4TaskContract,
    *,
    runner_version: str,
    endpoint_scope: str,
    policy_id: str,
    policy_version: str,
) -> int:
    """Return the validator-accounted bytes for one submission metadata group."""
    values = _submission_metadata_values(
        contract,
        runner_version=runner_version,
        endpoint_scope=endpoint_scope,
        policy_id=policy_id,
        policy_version=policy_version,
    )
    return sum(len(name.encode("utf-8")) + _attribute_size(value) for name, value in values.items())


def write_submission_metadata(
    hdf5_file: h5py.File,
    contract: V4TaskContract,
    *,
    runner_version: str,
    endpoint_scope: str,
    policy_id: str,
    policy_version: str,
) -> None:
    group = hdf5_file.create_group(METADATA_GROUP)
    values = _submission_metadata_values(
        contract,
        runner_version=runner_version,
        endpoint_scope=endpoint_scope,
        policy_id=policy_id,
        policy_version=policy_version,
    )
    for name, value in values.items():
        group.attrs[name] = value
    if not contract.uses_full_capture:
        group.create_dataset("arm_joint_indices", data=np.asarray(contract.control.arm_joint_indices, dtype=np.int64))
        group.create_dataset("action_joint_indices", data=np.asarray(contract.control.action_joint_indices, dtype=np.int64))


__all__ = [
    "DATA_GROUP",
    "DEMO_NAME",
    "METADATA_GROUP",
    "SUBMISSION_METADATA_ATTRIBUTES",
    "V4_0_METADATA_ATTRIBUTES",
    "V4_1_METADATA_ATTRIBUTES",
    "V4_3_METADATA_ATTRIBUTES",
    "submission_metadata_attribute_bytes",
    "write_submission_metadata",
]
