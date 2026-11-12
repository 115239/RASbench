"""Errors raised by the unified Rule Package v1 contract."""


class RulePackageV1Error(ValueError):
    """The supplied v1 package is not safe to publish or evaluate."""


__all__ = ["RulePackageV1Error"]
