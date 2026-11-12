"""Shared score-free Judge Core public interface."""

from .core import (
    SUPPORTED_PACKAGE_SCHEMAS,
    AuditCallback,
    RawFacts,
    RulePackageSource,
    evaluate_raw_facts,
)
from .errors import (
    JudgeAuditCallbackError,
    JudgeCoreError,
    JudgeCoreEvaluationError,
    JudgeCoreInputError,
    JudgeCorePreflightError,
    JudgeCoreUnsupportedError,
)

__all__ = [
    "AuditCallback",
    "JudgeAuditCallbackError",
    "JudgeCoreError",
    "JudgeCoreEvaluationError",
    "JudgeCoreInputError",
    "JudgeCorePreflightError",
    "JudgeCoreUnsupportedError",
    "RawFacts",
    "RulePackageSource",
    "SUPPORTED_PACKAGE_SCHEMAS",
    "evaluate_raw_facts",
]
