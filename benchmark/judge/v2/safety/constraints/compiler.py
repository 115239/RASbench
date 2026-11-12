"""Compile trusted task configuration into a deterministic constraint set."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class ConstraintConfigError(ValueError):
    """Raised when trusted safety configuration is malformed."""


@dataclass(frozen=True)
class CompiledConstraintConfig:
    safety_version: str
    constraint_set_id: str
    safety_types: Mapping[str, Mapping[str, Any]]
    scopes: Mapping[str, tuple[str, ...]]
    is_frozen: bool
    reason_codes: tuple[str, ...]
    digest: str


def compile_constraint_config(
    task_info: Mapping[str, Any],
) -> CompiledConstraintConfig:
    """Validate and normalize the trusted safety part of one task package."""

    if not isinstance(task_info, Mapping):
        raise ConstraintConfigError("task_info must be an object")
    config = _required_mapping(task_info.get("config"), "task_info.config")
    safety_version = _required_text(
        config.get("safety_version"), "config.safety_version"
    )
    constraint_set_id = _required_text(
        config.get("constraint_set_id"), "config.constraint_set_id"
    )

    raw_types = _required_mapping(
        config.get("safety_types"), "config.safety_types"
    )
    enabled_types = {}
    for type_id in sorted(raw_types):
        normalized_id = _required_text(type_id, "safety type id")
        spec = _required_mapping(
            raw_types[type_id], f"config.safety_types.{normalized_id}"
        )
        enabled = spec.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConstraintConfigError(
                f"config.safety_types.{normalized_id}.enabled must be boolean"
            )
        if not enabled:
            continue
        _required_text(
            spec.get("description"),
            f"config.safety_types.{normalized_id}.description",
        )
        rule = _required_mapping(
            spec.get("rule"),
            f"config.safety_types.{normalized_id}.rule",
        )
        _required_text(
            rule.get("kind"),
            f"config.safety_types.{normalized_id}.rule.kind",
        )
        enabled_types[normalized_id] = copy.deepcopy(dict(spec))

    raw_scopes = _required_mapping(config.get("scopes"), "config.scopes")
    scopes = {}
    for scope_id in sorted(raw_scopes):
        normalized_scope = _required_text(scope_id, "scope id")
        scope = _required_mapping(
            raw_scopes[scope_id], f"config.scopes.{normalized_scope}"
        )
        applicable = scope.get("applicable_safety_types")
        if (
            not isinstance(applicable, Sequence)
            or isinstance(applicable, (str, bytes))
        ):
            raise ConstraintConfigError(
                f"config.scopes.{normalized_scope}.applicable_safety_types must be an array"
            )
        normalized_types = tuple(
            _required_text(value, "applicable safety type") for value in applicable
        )
        if len(set(normalized_types)) != len(normalized_types):
            raise ConstraintConfigError(
                f"config.scopes.{normalized_scope} contains duplicate safety types"
            )
        missing = sorted(set(normalized_types).difference(enabled_types))
        if missing:
            raise ConstraintConfigError(
                "config.scopes.{} references unavailable safety types: {}".format(
                    normalized_scope, ", ".join(missing)
                )
            )
        scopes[normalized_scope] = tuple(sorted(normalized_types))

    draft = task_info.get("_draft")
    draft_status = draft.get("status") if isinstance(draft, Mapping) else None
    is_frozen = draft_status == "frozen"
    reason_codes = () if is_frozen else ("CONFIG_NOT_FROZEN",)

    digest_payload = {
        "safety_version": safety_version,
        "constraint_set_id": constraint_set_id,
        "safety_types": enabled_types,
        "scopes": scopes,
    }
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(canonical).hexdigest()

    return CompiledConstraintConfig(
        safety_version=safety_version,
        constraint_set_id=constraint_set_id,
        safety_types=MappingProxyType(enabled_types),
        scopes=MappingProxyType(scopes),
        is_frozen=is_frozen,
        reason_codes=reason_codes,
        digest=digest,
    )


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintConfigError(f"{field} must be an object")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConstraintConfigError(
            f"{field} must be normalized non-empty text"
        )
    return value


__all__ = [
    "CompiledConstraintConfig",
    "ConstraintConfigError",
    "compile_constraint_config",
]
