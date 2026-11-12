"""Validate one finalized safety Runner HDF5 before it is delivered."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py

from robolab_action_schema import (
    DATA_ROOT,
    SCHEMA_IMPLEMENTATION_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SAFETY_USD_ATTRIBUTE_EVIDENCE_SCHEMA,
    SchemaValidationError,
    get_formal_robot_profile,
    validate_fixed_evidence_metadata,
    validate_usd_boolean_attribute_evidence,
)

DELIVERY_VALIDATION_REPORT_SCHEMA = (
    "robolab-safety-runner-delivery-validation/v2"
)
REQUIRED_SCHEMA_IMPLEMENTATION_VERSION = "0.7.2"
DEFAULT_SAFETY_ROBOT_ROLE = "m20_piper"


class DeliveryValidationError(ValueError):
    """A stable, CLI-serializable Runner delivery validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def as_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "report_schema": DELIVERY_VALIDATION_REPORT_SCHEMA,
            "status": "INVALID",
            "error": {"code": self.code, "message": str(self)},
        }
        if self.details:
            report["error"]["details"] = self.details
        return report


def validate_runner_delivery(
    submission_hdf5: str | Path,
    legacy_requirements: str | Path | Mapping[str, Any] | None = None,
    *,
    expected_task_id: str | None = None,
    expected_role: str = DEFAULT_SAFETY_ROBOT_ROLE,
) -> dict[str, Any]:
    """Validate the formal HDF5 v1 and fixed-evidence delivery baseline.

    ``legacy_requirements`` remains only to fail closed for callers using the
    retired v1 API. The formal safety Runner contract has no per-task
    ``hdf5_requirements.json`` input.
    """

    if legacy_requirements is not None:
        raise DeliveryValidationError(
            "LEGACY_REQUIREMENTS_UNSUPPORTED",
            "hdf5_requirements.json is retired from the formal safety Runner contract",
        )
    if SCHEMA_IMPLEMENTATION_VERSION != REQUIRED_SCHEMA_IMPLEMENTATION_VERSION:
        raise DeliveryValidationError(
            "SCHEMA_IMPLEMENTATION_VERSION_MISMATCH",
            "Runner handoff loaded an incompatible shared Schema implementation",
            details={
                "expected": REQUIRED_SCHEMA_IMPLEMENTATION_VERSION,
                "actual": SCHEMA_IMPLEMENTATION_VERSION,
            },
        )
    try:
        profile = get_formal_robot_profile(expected_role)
    except ValueError as error:
        raise DeliveryValidationError(
            "ROBOT_ROLE_UNSUPPORTED",
            str(error),
            details={"expected_role": expected_role},
        ) from error

    submission_path = Path(submission_hdf5).expanduser().resolve()
    if not submission_path.is_file():
        raise DeliveryValidationError(
            "SUBMISSION_NOT_FOUND",
            f"submission HDF5 does not exist: {submission_path}",
            details={"submission": str(submission_path)},
        )
    if submission_path.name != "submission.hdf5":
        raise DeliveryValidationError(
            "SUBMISSION_FILENAME_INVALID",
            "final Runner evidence file must be named 'submission.hdf5'",
            details={"submission": str(submission_path)},
        )

    try:
        hdf5_file = h5py.File(submission_path, "r")
    except OSError as error:
        raise DeliveryValidationError(
            "SUBMISSION_UNREADABLE",
            f"cannot read submission HDF5: {error}",
            details={"submission": str(submission_path)},
        ) from error

    with hdf5_file:
        try:
            fixed_metadata = validate_fixed_evidence_metadata(
                hdf5_file,
                expected_task_id=expected_task_id,
                expected_role=expected_role,
            )
            detected_role = fixed_metadata["artifact_role"]
            metadata = hdf5_file["/metadata"]
            surface_schema = fixed_metadata["fixed_evidence_surface_schema"]
            fixed_evidence_sha256 = fixed_metadata[
                "fixed_evidence_surface_sha256"
            ]
            formal_stage_digest = fixed_metadata["formal_stage_digest"]
            exporter_version = fixed_metadata["exporter_version"]
            task_state_validation = validate_usd_boolean_attribute_evidence(
                hdf5_file
            )
            raw_task_id = metadata.attrs["task_id"]
            task_id = (
                raw_task_id.decode("utf-8")
                if isinstance(raw_task_id, bytes)
                else str(raw_task_id)
            )
            frame_count = int(hdf5_file[f"{DATA_ROOT}/actions"].shape[0])
        except SchemaValidationError as error:
            raise DeliveryValidationError(
                "HDF5_STRUCTURE_INVALID",
                str(error),
                details={"submission": str(submission_path)},
            ) from error
        except (KeyError, UnicodeError) as error:
            raise DeliveryValidationError(
                "SUBMISSION_UNREADABLE",
                f"cannot read submission HDF5: {error}",
                details={"submission": str(submission_path)},
            ) from error

    return {
        "report_schema": DELIVERY_VALIDATION_REPORT_SCHEMA,
        "status": "VALID",
        "submission": str(submission_path),
        "task_id": task_id,
        "frame_count": frame_count,
        "robot": {
            "role": detected_role,
            "formal_mount_prim": profile.formal_mount_prim,
            "articulation_prim": profile.articulation_prim,
            "joint_count": profile.joint_count,
            "body_count": profile.body_count,
            "asset_sha256": profile.asset_sha256,
        },
        "fixed_evidence": {
            "schema": surface_schema,
            "sha256": fixed_evidence_sha256,
            "formal_stage_digest": formal_stage_digest,
            "exporter_version": exporter_version,
        },
        "task_state": {
            "present": bool(task_state_validation.attribute_count),
            "schema": (
                SAFETY_USD_ATTRIBUTE_EVIDENCE_SCHEMA
                if task_state_validation.attribute_count
                else None
            ),
            "attribute_count": task_state_validation.attribute_count,
            "frame_count": task_state_validation.frame_count,
            "discovery_source": "final_composed_stage",
            "discovery_rule": "state:* with exact USD bool type",
        },
        "hdf5_schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "implementation_version": SCHEMA_IMPLEMENTATION_VERSION,
        },
    }
