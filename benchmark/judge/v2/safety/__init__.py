"""Safety-competition modules built on :mod:`v2.common`.

The package root is intentionally lazy: Generator tooling must remain importable
without installing the Safety Tianchi runtime and its deployment-only extras.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SAFETY_AGGREGATION_PROFILE",
    "SafetyAttackGatedScoringPipeline",
    "SafetyEntryResult",
    "SafetyScoringPipeline",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import scoring_modules

        return getattr(scoring_modules, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
"""Safety-private V2 adapters and scoring services."""
