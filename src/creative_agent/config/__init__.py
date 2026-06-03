"""Configuration loaders: Platform_Spec, Forbidden_Term dictionaries, env settings."""

from creative_agent.config.ad_group_quota import (
    AD_GROUP_QUOTA,
    DEFAULT_TARGET_COUNT,
    MIN_COMPLIANT_FLOOR,
    target_count_for,
)
from creative_agent.config.forbidden_loader import (
    ForbiddenTermEntry,
    find_term_matches,
    load_forbidden_dictionary,
)
from creative_agent.config.platform_loader import load_platform_spec

__all__ = [
    "ForbiddenTermEntry",
    "find_term_matches",
    "load_forbidden_dictionary",
    "load_platform_spec",
    "AD_GROUP_QUOTA",
    "DEFAULT_TARGET_COUNT",
    "MIN_COMPLIANT_FLOOR",
    "target_count_for",
]
