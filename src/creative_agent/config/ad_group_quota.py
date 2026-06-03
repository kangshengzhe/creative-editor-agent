"""Ad-group quota — per-creative-type target candidate counts.

Single source of truth for Requirement 5 (Ad-Group Quota). A single creative
request must return enough compliant candidates to fill a Google Ads ad group:
15 headlines and 10 descriptions. CTA and long copy keep the smaller default.

Both the Orchestrator and the Creative_Generator read :func:`target_count_for`
so the per-type target can never drift between components (Requirement 5.3).

Note on the description count (business decision, verified 2026-06):
    Google's Responsive Search Ads (RSA) policy caps a single ad at 15
    headlines but only **4** descriptions, and RSA has no standalone "CTA"
    asset type. The business deliberately targets **10** descriptions (and a
    CTA quota) as a cross-platform / optimise-then-select asset pool — copy is
    reused across Google, Facebook, and TikTok and human-allocated per platform
    — rather than as the Google RSA per-ad ceiling. So 10 (> Google's 4) is
    intentional, not a bug: do not "correct" it to 4 without a business sign-off.
"""

from __future__ import annotations

from creative_agent.models.enums import Creative_Type

__all__ = ["AD_GROUP_QUOTA", "DEFAULT_TARGET_COUNT", "MIN_COMPLIANT_FLOOR", "target_count_for"]

#: Hard minimum-viability floor (Requirement 5.6). If a run cannot surface at
#: least this many compliant candidates after all refill rounds, it degrades
#: to a failure exactly as before. Kept separate from the quota so lowering a
#: quota can never weaken the viability guarantee.
MIN_COMPLIANT_FLOOR: int = 3

#: Fallback target for any creative type not explicitly listed below.
DEFAULT_TARGET_COUNT: int = 5

#: Per-creative-type Ad_Group_Quota (Requirement 5.1). HEADLINE and
#: DESCRIPTION mirror the Google Ads ad-group capacity (15 + 10); CTA and
#: LONG_COPY keep the historical default of 5.
AD_GROUP_QUOTA: dict[Creative_Type, int] = {
    Creative_Type.HEADLINE: 15,
    Creative_Type.DESCRIPTION: 10,
    Creative_Type.CTA: 5,
    Creative_Type.LONG_COPY: 5,
}


def target_count_for(creative_type: Creative_Type) -> int:
    """Return the Target_Count for ``creative_type`` (Requirement 5.1).

    Falls back to :data:`DEFAULT_TARGET_COUNT` for any type not present in
    :data:`AD_GROUP_QUOTA`, so a newly added creative type can never produce a
    zero or undefined target.
    """
    return AD_GROUP_QUOTA.get(creative_type, DEFAULT_TARGET_COUNT)
