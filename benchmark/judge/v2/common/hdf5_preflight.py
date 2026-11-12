"""Task-independent Action HDF5 identity and structure preflight."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

import h5py

from robolab_action_schema import (
    SchemaValidationError,
    validate_formal_submission_structure,
)


class SubmissionPreflightError(ValueError):
    """Raised when a module cannot trust the supplied Action HDF5."""


def validate_submission_identity(
    hdf5_path: Union[str, PathLike],
    *,
    expected_task_id: str,
    expected_robot_role: str,
) -> str:
    """Fail closed on malformed, incomplete, or misrouted Action HDF5."""

    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        raise SubmissionPreflightError(
            "submission HDF5 does not exist: {}".format(path)
        )
    try:
        with h5py.File(path, "r") as hdf5_file:
            return validate_formal_submission_structure(
                hdf5_file,
                expected_task_id=expected_task_id,
                expected_role=expected_robot_role,
            )
    except (OSError, SchemaValidationError, ValueError) as error:
        raise SubmissionPreflightError(
            "submission HDF5 identity/structure preflight failed: {}".format(
                error
            )
        ) from error


__all__ = ["SubmissionPreflightError", "validate_submission_identity"]
