"""Public data contracts for RoboLab v2 scoring orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


WEIGHT_TOTAL_BASIS_POINTS = 10_000


class TianchiFrameworkError(ValueError):
    """Raised when a scoring contract or trusted profile is invalid."""


class ModuleStatus(str, Enum):
    """Stable terminal states returned by a scoring module."""

    EVALUATED = "EVALUATED"
    UNDER_REVIEW = "UNDER_REVIEW"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


def _require_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TianchiFrameworkError(
            f"{field_name} must be a non-empty string"
        )


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TianchiFrameworkError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ModuleResult:
    """The only formal result shape consumed by a public aggregator."""

    module_id: str
    module_version: str
    status: ModuleStatus
    score_ratio: Optional[float]
    report: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.module_id, "module_id")
        _require_identifier(self.module_version, "module_version")
        if not isinstance(self.status, ModuleStatus):
            raise TianchiFrameworkError("status must be a ModuleStatus")
        object.__setattr__(
            self, "report", _freeze_mapping(self.report, "report")
        )

        if self.status is ModuleStatus.EVALUATED:
            score = self.score_ratio
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise TianchiFrameworkError(
                    "EVALUATED score_ratio must be finite and within [0, 1]"
                )
            object.__setattr__(self, "score_ratio", float(score))
        elif self.score_ratio is not None:
            raise TianchiFrameworkError(
                "non-EVALUATED ModuleResult must not contain score_ratio"
            )


@dataclass(frozen=True)
class AggregationProfile:
    """Trusted ordered module selection and aggregation policy."""

    modules: Tuple[str, ...]
    weights_basis_points: Optional[Mapping[str, Any]] = None
    aggregator: str = "weighted_sum"
    invalid_weight_policy: str = "equal"

    def __post_init__(self) -> None:
        if isinstance(self.modules, (str, bytes, bytearray)) or not isinstance(
            self.modules, Sequence
        ):
            raise TianchiFrameworkError("modules must be an ordered sequence")
        modules = tuple(self.modules)
        if not modules:
            raise TianchiFrameworkError("modules must not be empty")
        for module_id in modules:
            _require_identifier(module_id, "module_id")
        if len(set(modules)) != len(modules):
            raise TianchiFrameworkError("modules must not contain duplicates")
        object.__setattr__(self, "modules", modules)

        if isinstance(self.weights_basis_points, Mapping):
            object.__setattr__(
                self,
                "weights_basis_points",
                _freeze_mapping(
                    self.weights_basis_points, "weights_basis_points"
                ),
            )
        _require_identifier(self.aggregator, "aggregator")
        if self.invalid_weight_policy != "equal":
            raise TianchiFrameworkError(
                "invalid_weight_policy must be 'equal'"
            )


@dataclass(frozen=True)
class WeightResolution:
    """Auditable effective weights used for one successful aggregation."""

    mode: str
    effective_weights_basis_points: Mapping[str, int]
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in {"configured", "fallback_equal"}:
            raise TianchiFrameworkError("invalid weight resolution mode")
        object.__setattr__(
            self,
            "effective_weights_basis_points",
            _freeze_mapping(
                self.effective_weights_basis_points,
                "effective_weights_basis_points",
            ),
        )


@dataclass(frozen=True)
class AggregationResult:
    """Successful competition-level aggregation result."""

    score_ratio: float
    module_results: Tuple[ModuleResult, ...]
    weight_resolution: WeightResolution
    status: ModuleStatus = ModuleStatus.EVALUATED

    def __post_init__(self) -> None:
        if self.status is not ModuleStatus.EVALUATED:
            raise TianchiFrameworkError(
                "a formal AggregationResult must be EVALUATED"
            )
        score = self.score_ratio
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise TianchiFrameworkError(
                "aggregation score_ratio must be finite and within [0, 1]"
            )
        object.__setattr__(self, "score_ratio", float(score))
        object.__setattr__(self, "module_results", tuple(self.module_results))


__all__ = [
    "WEIGHT_TOTAL_BASIS_POINTS",
    "AggregationProfile",
    "AggregationResult",
    "ModuleResult",
    "ModuleStatus",
    "TianchiFrameworkError",
    "WeightResolution",
]
