"""Typed failures for the public, score-free Judge Core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


class JudgeCoreError(RuntimeError):
    """Base class for failures that must not produce partial Raw Facts."""


class JudgeCoreInputError(JudgeCoreError):
    """The trusted package, identity context, or HDF5 input is invalid."""


class JudgeCoreUnsupportedError(JudgeCoreError):
    """The package requests a feature not supported by the public core."""


class JudgeCorePreflightError(JudgeCoreInputError):
    """The HDF5 cannot satisfy the trusted Rule Package."""

    def __init__(self, message: str, report: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


class JudgeCoreEvaluationError(JudgeCoreError):
    """Complete evidence was present but predicate evaluation failed."""


class JudgeAuditCallbackError(JudgeCoreError):
    """The optional, non-scoring audit callback failed."""


__all__ = [
    "JudgeAuditCallbackError",
    "JudgeCoreError",
    "JudgeCoreEvaluationError",
    "JudgeCoreInputError",
    "JudgeCorePreflightError",
    "JudgeCoreUnsupportedError",
]
