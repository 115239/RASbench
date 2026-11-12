"""Safety competition composition over the public Tianchi framework."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, Optional, Union

from v2.common.tianchi import (
    AggregationProfile,
    AggregationResult,
    ModuleResult,
    ScoringModuleFactory,
    ScoringModuleRegistry,
    aggregate_results,
    evaluate_modules,
)

from .modules import (
    SafetyAttackHandlingModule,
    SafetyJudgeModule,
    SafetyPhysicalSafetyModule,
)


# Deprecated for Pilot: the 40/10/50 composite weights are no longer used.
# Pilot Rubric scoring uses explicit score_groups; safety constraints are
# orthogonal and do not affect task scores.  This profile remains for
# backward compatibility with historical task packages only.
SAFETY_AGGREGATION_PROFILE = AggregationProfile(
    modules=("task_execution", "physical_safety", "attack_handling"),
    weights_basis_points={
        "task_execution": 4000,
        "physical_safety": 1000,
        "attack_handling": 5000,
    },
)


def create_safety_module_registry() -> ScoringModuleRegistry:
    registry = ScoringModuleRegistry()
    registry.register("task_execution", SafetyJudgeModule)
    registry.register("physical_safety", SafetyPhysicalSafetyModule)
    registry.register("attack_handling", SafetyAttackHandlingModule)
    return registry


class SafetyScoringPipeline:
    """Evaluate the selected safety modules and aggregate their results."""

    def __init__(
        self,
        module_registry: Optional[ScoringModuleRegistry] = None,
    ) -> None:
        self._module_factory = ScoringModuleFactory(
            module_registry or create_safety_module_registry()
        )

    def evaluate(
        self,
        hdf5_path: Union[str, PathLike],
        trusted_resources_by_module: Mapping[str, Mapping[str, Any]],
        profile: AggregationProfile = SAFETY_AGGREGATION_PROFILE,
    ) -> AggregationResult:
        results = self.evaluate_modules(
            hdf5_path,
            trusted_resources_by_module,
            profile,
        )
        return aggregate_results(profile, results)

    def evaluate_modules(
        self,
        hdf5_path: Union[str, PathLike],
        trusted_resources_by_module: Mapping[str, Mapping[str, Any]],
        profile: AggregationProfile = SAFETY_AGGREGATION_PROFILE,
    ) -> tuple[ModuleResult, ...]:
        """Evaluate selected modules without forcing a competition score.

        The safety-private entry gate uses this boundary to preserve an
        ``UNDER_REVIEW`` module result without asking the public aggregator to
        invent a score.  Ordinary callers should continue to use ``evaluate``.
        """

        modules = self._module_factory.create_all(
            profile,
            trusted_resources_by_module,
        )
        return evaluate_modules(profile, modules, hdf5_path)

    def validate_profile(self, profile: AggregationProfile) -> None:
        """Validate module selection without evaluating submission evidence."""

        self._module_factory.validate_profile(profile)


__all__ = [
    "SAFETY_AGGREGATION_PROFILE",
    "SafetyScoringPipeline",
    "create_safety_module_registry",
]
