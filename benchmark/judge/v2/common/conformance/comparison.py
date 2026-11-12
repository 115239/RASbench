"""Reusable migration comparisons for public Judge and scoring outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from math import isclose
from types import MappingProxyType
from typing import Any, Tuple

from v2.common.tianchi import ModuleResult


@dataclass(frozen=True)
class ConformanceDifference:
    """One stable, machine-readable migration mismatch."""

    path: str
    expected: Any
    actual: Any


@dataclass(frozen=True)
class ConformanceReport:
    """Complete comparison result; mismatches are never silently dropped."""

    differences: Tuple[ConformanceDifference, ...]

    @property
    def matches(self) -> bool:
        return not self.differences


class ConformanceMismatchError(AssertionError):
    """Raised when a migration gate requires an exact match."""

    def __init__(self, report: ConformanceReport) -> None:
        self.report = report
        message = "; ".join(
            "{}: {!r} != {!r}".format(
                difference.path,
                difference.expected,
                difference.actual,
            )
            for difference in report.differences
        )
        super().__init__(message or "conformance comparison unexpectedly matched")


def compare_ordered_raw_facts(
    expected: Mapping[str, bool],
    actual: Mapping[str, bool],
) -> ConformanceReport:
    """Compare the exact public Raw Facts contract: IDs, order, and booleans."""

    differences = []
    if not isinstance(expected, Mapping):
        differences.append(
            ConformanceDifference("expected", "ordered mapping", type(expected).__name__)
        )
        return ConformanceReport(tuple(differences))
    if not isinstance(actual, Mapping):
        differences.append(
            ConformanceDifference("actual", "ordered mapping", type(actual).__name__)
        )
        return ConformanceReport(tuple(differences))

    expected_ids = tuple(expected.keys())
    actual_ids = tuple(actual.keys())
    if expected_ids != actual_ids:
        differences.append(
            ConformanceDifference("raw_facts.keys", expected_ids, actual_ids)
        )
    for item_id in expected_ids:
        expected_value = expected[item_id]
        if type(expected_value) is not bool:
            differences.append(
                ConformanceDifference(
                    "raw_facts.{}.expected_type".format(item_id),
                    "bool",
                    type(expected_value).__name__,
                )
            )
            continue
        if item_id not in actual:
            continue
        actual_value = actual[item_id]
        if type(actual_value) is not bool:
            differences.append(
                ConformanceDifference(
                    "raw_facts.{}.actual_type".format(item_id),
                    "bool",
                    type(actual_value).__name__,
                )
            )
        elif expected_value != actual_value:
            differences.append(
                ConformanceDifference(
                    "raw_facts.{}".format(item_id),
                    expected_value,
                    actual_value,
                )
            )
    return ConformanceReport(tuple(differences))


def compare_module_results(
    expected: object,
    actual: object,
    *,
    include_report: bool = False,
    score_tolerance: float = 0.0,
) -> ConformanceReport:
    """Compare public ModuleResult fields and optionally the private report."""

    expected_value = module_result_snapshot(expected, include_report=include_report)
    actual_value = module_result_snapshot(actual, include_report=include_report)
    differences = []
    for field_name in expected_value:
        left = expected_value[field_name]
        right = actual_value.get(field_name)
        if field_name == "score_ratio" and _scores_match(
            left, right, score_tolerance
        ):
            continue
        if left != right:
            differences.append(
                ConformanceDifference(
                    "module_result.{}".format(field_name), left, right
                )
            )
    extra = tuple(key for key in actual_value if key not in expected_value)
    if extra:
        differences.append(
            ConformanceDifference("module_result.extra_fields", (), extra)
        )
    return ConformanceReport(tuple(differences))


def module_result_snapshot(
    value: object,
    *,
    include_report: bool = False,
) -> Mapping[str, Any]:
    """Project a v2 or legacy result onto the common public result fields."""

    fields = {
        "module_id": _field(value, "module_id"),
        "module_version": _field(value, "module_version"),
        "status": _normalize(_field(value, "status")),
        "score_ratio": _field(value, "score_ratio"),
    }
    if include_report:
        fields["report"] = _normalize(_field(value, "report"))
    return MappingProxyType(fields)


def require_conformant(report: ConformanceReport) -> None:
    """Fail one release gate with the complete mismatch report."""

    if not report.matches:
        raise ConformanceMismatchError(report)


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError("ModuleResult mapping is missing {}".format(name))
        return value[name]
    if not isinstance(value, ModuleResult) and not hasattr(value, name):
        raise ValueError("object is not a ModuleResult-like value")
    return getattr(value, name)


def _scores_match(left: object, right: object, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(child)
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(_normalize(child) for child in value)
    return value


__all__ = [
    "ConformanceDifference",
    "ConformanceMismatchError",
    "ConformanceReport",
    "compare_module_results",
    "compare_ordered_raw_facts",
    "module_result_snapshot",
    "require_conformant",
]
