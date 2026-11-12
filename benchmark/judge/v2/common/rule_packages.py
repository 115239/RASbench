"""Trusted Rule Package loading shared by v2 competition modules."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Union

from robolab_action_judge.errors import ActionJudgeError
from robolab_action_judge.package import load_rule_package, validate_rule_package


RulePackageSource = Union[str, Path, Mapping[str, Any]]


class TrustedRulePackageError(ValueError):
    """Raised when an organizer-owned Rule Package cannot be materialized."""


def materialize_rule_package(source: RulePackageSource) -> Dict[str, Any]:
    """Load and validate a private copy of an organizer-owned Rule Package."""

    try:
        if isinstance(source, Mapping):
            package = copy.deepcopy(dict(source))
            validate_rule_package(package)
            return package
        return load_rule_package(source)
    except (ActionJudgeError, OSError, TypeError, ValueError) as error:
        raise TrustedRulePackageError(
            "invalid trusted Rule Package: {}".format(error)
        ) from error


__all__ = [
    "RulePackageSource",
    "TrustedRulePackageError",
    "materialize_rule_package",
]
