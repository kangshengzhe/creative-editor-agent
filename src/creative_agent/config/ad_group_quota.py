"""Ad-group quota — per-creative-type target candidate counts.

Single source of truth for Requirement 5 (Ad-Group Quota). A single creative
request returns this many compliant candidates per type to give operators
room to pick: 20 headlines, 15 descriptions, 10 CTAs. Long copy keeps the
smaller default of 5.

Both the Orchestrator and the Creative_Generator read :func:`target_count_for`
so the per-type target can never drift between components (Requirement 5.3).

Note on the counts (business decision, verified 2026-06):
    These targets are deliberately ABOVE any single platform's per-ad ceiling
    (e.g. Google RSA caps 15 headlines / 4 descriptions and has no standalone
    CTA asset). They form a cross-platform "optimise-then-select" asset pool —
    copy is reused across Google, Facebook, and TikTok and human-allocated per
    platform — with +5 over the prior 15/10/5 so operators have extra choices.
    Do not "correct" these to a single platform's limits without a business
    sign-off.
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

#: Per-creative-type Ad_Group_Quota (Requirement 5.1). Sized above any single
#: platform's per-ad limit to give operators a selection pool: HEADLINE 20,
#: DESCRIPTION 15, CTA 10; LONG_COPY keeps the default of 5.
AD_GROUP_QUOTA: dict[Creative_Type, int] = {
    Creative_Type.HEADLINE: 20,
    Creative_Type.DESCRIPTION: 15,
    Creative_Type.CTA: 10,
    Creative_Type.LONG_COPY: 5,
}


def target_count_for(creative_type: Creative_Type) -> int:
    """Return the Target_Count for ``creative_type`` (Requirement 5.1).

    Falls back to :data:`DEFAULT_TARGET_COUNT` for any type not present in
    :data:`AD_GROUP_QUOTA`, so a newly added creative type can never produce a
    zero or undefined target.
    """
    return AD_GROUP_QUOTA.get(creative_type, DEFAULT_TARGET_COUNT)
