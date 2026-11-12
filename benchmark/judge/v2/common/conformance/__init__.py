"""Public migration and protocol conformance helpers."""

from .comparison import (
    ConformanceDifference,
    ConformanceMismatchError,
    ConformanceReport,
    compare_module_results,
    compare_ordered_raw_facts,
    module_result_snapshot,
    require_conformant,
)

__all__ = [
    "ConformanceDifference",
    "ConformanceMismatchError",
    "ConformanceReport",
    "compare_module_results",
    "compare_ordered_raw_facts",
    "module_result_snapshot",
    "require_conformant",
]
