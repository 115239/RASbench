"""Module registry, construction, and HDF5-only execution contracts."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .models import (
    AggregationProfile,
    ModuleResult,
    TianchiFrameworkError,
    _require_identifier,
)


class ScoringModule(Protocol):
    """A competition-owned scorer bound to trusted resources at creation."""

    module_id: str
    module_version: str

    def evaluate(self, hdf5_path: Path) -> ModuleResult:
        """Evaluate one submission HDF5 and return one formal module result."""


ModuleConstructor = Callable[[Mapping[str, Any]], ScoringModule]


class ScoringModuleRegistry:
    """Deployment-owned mapping from stable module IDs to constructors."""

    def __init__(self) -> None:
        self._constructors: dict[str, ModuleConstructor] = {}

    def register(
        self, module_id: str, constructor: ModuleConstructor
    ) -> None:
        _require_identifier(module_id, "module_id")
        if not callable(constructor):
            raise TianchiFrameworkError("module constructor must be callable")
        if module_id in self._constructors:
            raise TianchiFrameworkError(
                f"module_id {module_id!r} is already registered"
            )
        self._constructors[module_id] = constructor

    def constructor_for(self, module_id: str) -> ModuleConstructor:
        _require_identifier(module_id, "module_id")
        try:
            return self._constructors[module_id]
        except KeyError as error:
            raise TianchiFrameworkError(
                f"unknown scoring module_id {module_id!r}"
            ) from error


class ScoringModuleFactory:
    """Create modules while binding organizer-owned trusted resources."""

    def __init__(self, registry: ScoringModuleRegistry) -> None:
        if not isinstance(registry, ScoringModuleRegistry):
            raise TianchiFrameworkError(
                "registry must be a ScoringModuleRegistry"
            )
        self._registry = registry

    def create(
        self,
        module_id: str,
        trusted_resources: Optional[Mapping[str, Any]] = None,
    ) -> ScoringModule:
        resources = _freeze_trusted_resources(trusted_resources)
        module = self._registry.constructor_for(module_id)(resources)
        _validate_module_instance(module, module_id)
        return module

    def validate_profile(self, profile: AggregationProfile) -> None:
        """Fail before execution if any selected module is unregistered."""

        if not isinstance(profile, AggregationProfile):
            raise TianchiFrameworkError(
                "profile must be an AggregationProfile"
            )
        for module_id in profile.modules:
            self._registry.constructor_for(module_id)

    def create_all(
        self,
        profile: AggregationProfile,
        trusted_resources_by_module: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
    ) -> Tuple[ScoringModule, ...]:
        resources_by_module = _validate_resource_bindings(
            profile, trusted_resources_by_module
        )
        return tuple(
            self.create(module_id, resources_by_module.get(module_id))
            for module_id in profile.modules
        )


def evaluate_modules(
    profile: AggregationProfile,
    modules: Sequence[ScoringModule],
    hdf5_path: Union[str, PathLike],
) -> Tuple[ModuleResult, ...]:
    """Evaluate modules in profile order against the same HDF5 path."""

    ordered_modules = _validate_ordered_modules(profile, modules)
    submission_path = _coerce_hdf5_path(hdf5_path)
    results = []
    for module_id, module in zip(profile.modules, ordered_modules):
        result = module.evaluate(submission_path)
        _validate_module_result(result, module_id, module.module_version)
        results.append(result)
    return tuple(results)


def _freeze_trusted_resources(
    resources: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if resources is None:
        resources = {}
    if not isinstance(resources, Mapping):
        raise TianchiFrameworkError("trusted_resources must be a mapping")
    return MappingProxyType(dict(resources))


def _validate_resource_bindings(
    profile: AggregationProfile,
    bindings: Optional[Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Mapping[str, Any]]:
    if bindings is None:
        return {}
    if not isinstance(bindings, Mapping):
        raise TianchiFrameworkError(
            "trusted_resources_by_module must be a mapping"
        )
    unknown_ids = set(bindings) - set(profile.modules)
    if unknown_ids:
        raise TianchiFrameworkError(
            "trusted resources contain unselected module IDs: "
            + ", ".join(sorted(str(item) for item in unknown_ids))
        )
    for module_id, resources in bindings.items():
        _require_identifier(module_id, "module_id")
        if not isinstance(resources, Mapping):
            raise TianchiFrameworkError(
                f"trusted resources for {module_id!r} must be a mapping"
            )
    return bindings


def _validate_module_instance(module: Any, expected_module_id: str) -> None:
    if getattr(module, "module_id", None) != expected_module_id:
        raise TianchiFrameworkError(
            "module constructor returned an object with the wrong module_id"
        )
    _require_identifier(getattr(module, "module_version", None), "module_version")
    if not callable(getattr(module, "evaluate", None)):
        raise TianchiFrameworkError(
            "module constructor returned a non-conforming object"
        )


def _validate_ordered_modules(
    profile: AggregationProfile,
    modules: Sequence[ScoringModule],
) -> Tuple[ScoringModule, ...]:
    if isinstance(modules, (str, bytes, bytearray)) or not isinstance(
        modules, Sequence
    ):
        raise TianchiFrameworkError("modules must be an ordered sequence")
    ordered = tuple(modules)
    actual_ids = tuple(getattr(module, "module_id", None) for module in ordered)
    if actual_ids != profile.modules:
        raise TianchiFrameworkError(
            "module IDs must exactly match profile order and selection"
        )
    for module_id, module in zip(profile.modules, ordered):
        _validate_module_instance(module, module_id)
    return ordered


def _validate_module_result(
    result: Any, expected_module_id: str, expected_module_version: str
) -> None:
    if not isinstance(result, ModuleResult):
        raise TianchiFrameworkError(
            "scoring module returned a non-conforming result"
        )
    if result.module_id != expected_module_id:
        raise TianchiFrameworkError("module result has the wrong module_id")
    if result.module_version != expected_module_version:
        raise TianchiFrameworkError(
            "module result has the wrong module_version"
        )


def _coerce_hdf5_path(value: Union[str, PathLike]) -> Path:
    if isinstance(value, str) and not value.strip():
        raise TianchiFrameworkError("hdf5_path must not be empty")
    if not isinstance(value, (str, PathLike)):
        raise TianchiFrameworkError("hdf5_path must be a filesystem path")
    return Path(value)


__all__ = [
    "ModuleConstructor",
    "ScoringModule",
    "ScoringModuleFactory",
    "ScoringModuleRegistry",
    "evaluate_modules",
]
