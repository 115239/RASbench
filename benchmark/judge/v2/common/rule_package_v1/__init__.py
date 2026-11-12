"""Unified, competition-neutral Rule Package v1 contract."""

from .errors import RulePackageV1Error
from .loading import (
    MAX_PACKAGE_BYTES,
    RulePackageV1Source,
    load_rule_package_v1,
    load_rule_package_v1_model,
    materialize_rule_package_v1,
)
from .judge_adapter import project_rule_package_v1_for_boolean_engine
from .models import RulePackageItemV1, RulePackageV1, parse_rule_package_v1
from .validation import (
    CANDIDATE_QUANTIFIER_V1_SCHEMA,
    RULE_PACKAGE_V1_SCHEMA,
    TRACE_PLAN_V1_SCHEMA,
    V1_PACKAGE_SCHEMA,
    validate_rule_package_v1,
)

__all__ = [
    "CANDIDATE_QUANTIFIER_V1_SCHEMA",
    "MAX_PACKAGE_BYTES",
    "RULE_PACKAGE_V1_SCHEMA",
    "RulePackageItemV1",
    "RulePackageV1",
    "RulePackageV1Error",
    "RulePackageV1Source",
    "TRACE_PLAN_V1_SCHEMA",
    "V1_PACKAGE_SCHEMA",
    "load_rule_package_v1",
    "load_rule_package_v1_model",
    "materialize_rule_package_v1",
    "parse_rule_package_v1",
    "project_rule_package_v1_for_boolean_engine",
    "validate_rule_package_v1",
]
