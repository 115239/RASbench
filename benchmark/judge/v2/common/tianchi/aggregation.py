"""Deterministic public score aggregation for RoboLab v2."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple

from .models import (
    WEIGHT_TOTAL_BASIS_POINTS,
    AggregationProfile,
    AggregationResult,
    ModuleResult,
    ModuleStatus,
    TianchiFrameworkError,
    WeightResolution,
    _require_identifier,
)


class ScoreAggregator(Protocol):
    """Contract for a trusted competition score aggregation policy."""

    def aggregate(
        self,
        profile: AggregationProfile,
        module_results: Sequence[ModuleResult],
    ) -> AggregationResult:
        """Aggregate every selected result without dropping a module."""


AggregatorConstructor = Callable[[AggregationProfile], ScoreAggregator]


class AggregatorRegistry:
    """Deployment-owned registry of public aggregation policies."""

    def __init__(self) -> None:
        self._constructors: dict[str, AggregatorConstructor] = {}

    def register(
        self, aggregator: str, constructor: AggregatorConstructor
    ) -> None:
        _require_identifier(aggregator, "aggregator")
        if not callable(constructor):
            raise TianchiFrameworkError(
                "aggregator constructor must be callable"
            )
        if aggregator in self._constructors:
            raise TianchiFrameworkError(
                f"aggregator {aggregator!r} is already registered"
            )
        self._constructors[aggregator] = constructor

    def constructor_for(self, aggregator: str) -> AggregatorConstructor:
        _require_identifier(aggregator, "aggregator")
        try:
            return self._constructors[aggregator]
        except KeyError as error:
            raise TianchiFrameworkError(
                f"unknown aggregator {aggregator!r}"
            ) from error


class AggregatorFactory:
    """Construct an aggregator selected by a trusted profile."""

    def __init__(self, registry: AggregatorRegistry) -> None:
        if not isinstance(registry, AggregatorRegistry):
            raise TianchiFrameworkError(
                "registry must be an AggregatorRegistry"
            )
        self._registry = registry

    def create(self, profile: AggregationProfile) -> ScoreAggregator:
        aggregator = self._registry.constructor_for(profile.aggregator)(profile)
        if not callable(getattr(aggregator, "aggregate", None)):
            raise TianchiFrameworkError(
                "aggregator constructor returned a non-conforming object"
            )
        return aggregator


class WeightedSumAggregator:
    """Weighted sum with deterministic equal-weight fallback."""

    def aggregate(
        self,
        profile: AggregationProfile,
        module_results: Sequence[ModuleResult],
    ) -> AggregationResult:
        results = validate_ordered_results(profile, module_results)
        for result in results:
            if result.status is not ModuleStatus.EVALUATED:
                raise TianchiFrameworkError(
                    "every selected module must be EVALUATED before aggregation"
                )

        weights = resolve_weights(profile)
        score = sum(
            result.score_ratio
            * weights.effective_weights_basis_points[result.module_id]
            for result in results
        ) / WEIGHT_TOTAL_BASIS_POINTS
        return AggregationResult(
            score_ratio=score,
            module_results=results,
            weight_resolution=weights,
        )


def resolve_weights(profile: AggregationProfile) -> WeightResolution:
    """Resolve configured weights or deterministically fall back to equal."""

    configured = profile.weights_basis_points
    reason = _invalid_weight_reason(profile.modules, configured)
    if reason is None:
        return WeightResolution(
            mode="configured",
            effective_weights_basis_points={
                module_id: configured[module_id]
                for module_id in profile.modules
            },
        )
    return WeightResolution(
        mode="fallback_equal",
        effective_weights_basis_points=_equal_weights(profile.modules),
        fallback_reason=reason,
    )


def validate_ordered_results(
    profile: AggregationProfile,
    module_results: Sequence[ModuleResult],
) -> Tuple[ModuleResult, ...]:
    """Require an exact one-for-one result sequence in profile order."""

    if isinstance(module_results, (str, bytes, bytearray)) or not isinstance(
        module_results, Sequence
    ):
        raise TianchiFrameworkError(
            "module_results must be an ordered sequence"
        )
    results = tuple(module_results)
    if not all(isinstance(result, ModuleResult) for result in results):
        raise TianchiFrameworkError(
            "module_results must contain only ModuleResult values"
        )
    result_ids = tuple(result.module_id for result in results)
    if result_ids != profile.modules:
        raise TianchiFrameworkError(
            "module result IDs must exactly match profile order and selection"
        )
    return results


def aggregate_results(
    profile: AggregationProfile,
    module_results: Sequence[ModuleResult],
    factory: Optional[AggregatorFactory] = None,
) -> AggregationResult:
    """Aggregate with the built-in or caller-supplied trusted registry."""

    if factory is None:
        factory = AggregatorFactory(create_default_aggregator_registry())
    result = factory.create(profile).aggregate(profile, module_results)
    if not isinstance(result, AggregationResult):
        raise TianchiFrameworkError(
            "aggregator returned a non-conforming AggregationResult"
        )
    supplied = tuple(module_results)
    if len(result.module_results) != len(supplied) or any(
        actual is not expected
        for actual, expected in zip(result.module_results, supplied)
    ):
        raise TianchiFrameworkError(
            "aggregator must preserve every supplied ModuleResult instance"
        )
    expected_weights = resolve_weights(profile)
    if result.weight_resolution != expected_weights:
        raise TianchiFrameworkError(
            "aggregator must use framework-resolved weights"
        )
    return result


def create_default_aggregator_registry() -> AggregatorRegistry:
    registry = AggregatorRegistry()
    registry.register("weighted_sum", lambda profile: WeightedSumAggregator())
    return registry


def _invalid_weight_reason(
    module_ids: Tuple[str, ...], configured: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not isinstance(configured, Mapping):
        return "weights_basis_points is missing or not a mapping"
    if set(configured) != set(module_ids):
        return "weight keys must exactly match selected module IDs"
    total = 0
    for module_id in module_ids:
        value = configured[module_id]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "each weight must be a positive integer basis-point value"
        total += value
    if total != WEIGHT_TOTAL_BASIS_POINTS:
        return "weight basis points must sum exactly to 10000"
    return None


def _equal_weights(module_ids: Tuple[str, ...]) -> Mapping[str, int]:
    base, remainder = divmod(WEIGHT_TOTAL_BASIS_POINTS, len(module_ids))
    return {
        module_id: base + (1 if index < remainder else 0)
        for index, module_id in enumerate(module_ids)
    }


__all__ = [
    "AggregatorConstructor",
    "AggregatorFactory",
    "AggregatorRegistry",
    "ScoreAggregator",
    "WeightedSumAggregator",
    "aggregate_results",
    "create_default_aggregator_registry",
    "resolve_weights",
    "validate_ordered_results",
]
