"""Safe JSON and mapping loaders for Rule Package v1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .errors import RulePackageV1Error
from .models import RulePackageV1, parse_rule_package_v1
from .validation import validate_rule_package_v1

MAX_PACKAGE_BYTES = 16 * 1024 * 1024
RulePackageV1Source = Union[str, Path, Mapping[str, Any]]


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RulePackageV1Error("duplicate JSON object key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RulePackageV1Error("non-finite JSON number is not allowed: {}".format(value))


def load_rule_package_v1(source: RulePackageV1Source) -> Dict[str, Any]:
    """Load a path or mapping and return a validated private dictionary."""

    if isinstance(source, Mapping):
        return validate_rule_package_v1(source)

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise RulePackageV1Error("Rule Package does not exist: {}".format(path))
    try:
        size = path.stat().st_size
        if size > MAX_PACKAGE_BYTES:
            raise RulePackageV1Error(
                "Rule Package exceeds {} bytes".format(MAX_PACKAGE_BYTES)
            )
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except RulePackageV1Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RulePackageV1Error("cannot load Rule Package v1: {}".format(error)) from error
    if not isinstance(value, Mapping):
        raise RulePackageV1Error("Rule Package root must be an object")
    return validate_rule_package_v1(value)


def load_rule_package_v1_model(source: RulePackageV1Source) -> RulePackageV1:
    """Load a path or mapping into the typed v1 data model."""

    return parse_rule_package_v1(load_rule_package_v1(source))


def materialize_rule_package_v1(source: RulePackageV1Source) -> Dict[str, Any]:
    """Return a validated private canonical dictionary for Judge/Generator use."""

    return parse_rule_package_v1(load_rule_package_v1(source)).to_dict()


__all__ = [
    "MAX_PACKAGE_BYTES",
    "RulePackageV1Source",
    "load_rule_package_v1",
    "load_rule_package_v1_model",
    "materialize_rule_package_v1",
]
