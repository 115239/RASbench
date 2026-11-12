"""RoboLab v2 public Tianchi scoring framework."""

from .aggregation import (
    AggregatorFactory,
    AggregatorRegistry,
    ScoreAggregator,
    WeightedSumAggregator,
    aggregate_results,
    create_default_aggregator_registry,
    resolve_weights,
    validate_ordered_results,
)
from .models import (
    WEIGHT_TOTAL_BASIS_POINTS,
    AggregationProfile,
    AggregationResult,
    ModuleResult,
    ModuleStatus,
    TianchiFrameworkError,
    WeightResolution,
)
from .modules import (
    ScoringModule,
    ScoringModuleFactory,
    ScoringModuleRegistry,
    evaluate_modules,
)


__all__ = [
    "WEIGHT_TOTAL_BASIS_POINTS",
    "AggregationProfile",
    "AggregationResult",
    "AggregatorFactory",
    "AggregatorRegistry",
    "ModuleResult",
    "ModuleStatus",
    "ScoreAggregator",
    "ScoringModule",
    "ScoringModuleFactory",
    "ScoringModuleRegistry",
    "TianchiFrameworkError",
    "WeightResolution",
    "WeightedSumAggregator",
    "aggregate_results",
    "create_default_aggregator_registry",
    "evaluate_modules",
    "resolve_weights",
    "validate_ordered_results",
]
