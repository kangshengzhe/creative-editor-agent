"""Tests for the lightweight (stdlib, no-PyTorch) semantic embedding.

The default SemanticDiversityChecker now uses a pure-stdlib lexical embedding
(hashed bag-of-words + char trigrams + CJK per-char features) instead of
sentence-transformers. These tests lock in:
  * lightweight is the default and needs no ML dependency,
  * near-duplicate copy scores above the lightweight threshold,
  * genuinely distinct copy scores near zero,
  * CJK near-dups are caught (per-char features),
  * embeddings are deterministic across instances (stable hashing).
"""

from __future__ import annotations

from creative_agent.integration.semantic_diversity import SemanticDiversityChecker
from creative_agent.models import SemanticDiversityConfig


def _sim(c: SemanticDiversityChecker, a: str, b: str) -> float:
    return c.cosine_similarity(c.compute_embedding(a), c.compute_embedding(b))


class TestLightweightIsDefault:
    def test_default_uses_lightweight_embed_fn(self) -> None:
        c = SemanticDiversityChecker()
        assert c._embed_fn.__name__ == "_lightweight_embed_fn"

    def test_lightweight_threshold_nudged_to_0_50(self) -> None:
        # Default neural threshold (0.60) is lowered to 0.50 in lightweight mode.
        assert SemanticDiversityChecker()._threshold == 0.50

    def test_explicit_threshold_respected(self) -> None:
        cfg = SemanticDiversityConfig(similarity_threshold=0.42)
        assert SemanticDiversityChecker(cfg)._threshold == 0.42

    def test_neural_mode_opt_out(self) -> None:
        cfg = SemanticDiversityConfig(lightweight=False)
        c = SemanticDiversityChecker(cfg)
        assert c._embed_fn.__name__ == "_default_embed_fn"


class TestSeparation:
    def test_english_near_duplicates_score_high(self) -> None:
        c = SemanticDiversityChecker()
        assert _sim(c, "Quick topup bonus access", "Easy topup bonus access") > 0.50

    def test_distinct_copy_scores_near_zero(self) -> None:
        c = SemanticDiversityChecker()
        s = _sim(c, "Get 20% Bonus on Topup", "24/7 Support for Every Player")
        assert s < 0.20

    def test_cjk_near_duplicates_caught(self) -> None:
        c = SemanticDiversityChecker()
        assert _sim(c, "立即充值领取奖励", "立即充值获得奖励") > 0.50

    def test_cjk_distinct_low(self) -> None:
        c = SemanticDiversityChecker()
        assert _sim(c, "立即充值领取奖励", "全天候客服为您服务") < 0.20

    def test_identical_copy_is_one(self) -> None:
        c = SemanticDiversityChecker()
        assert _sim(c, "topup bonus now", "topup bonus now") == 1.0


class TestDeterminism:
    def test_embeddings_stable_across_instances(self) -> None:
        # Stable hashing → two separate instances produce the same vector.
        a = SemanticDiversityChecker().compute_embedding("topup bonus")
        b = SemanticDiversityChecker().compute_embedding("topup bonus")
        assert a == b


class TestCheckCandidate:
    async def test_rejects_near_duplicate(self) -> None:
        c = SemanticDiversityChecker()
        result = await c.check_candidate(
            "Easy topup bonus access", ["Quick topup bonus access"]
        )
        assert result.accepted is False

    async def test_accepts_distinct(self) -> None:
        c = SemanticDiversityChecker()
        result = await c.check_candidate(
            "24/7 support for every player", ["Get 20% bonus on topup"]
        )
        assert result.accepted is True

    async def test_empty_pool_accepts(self) -> None:
        c = SemanticDiversityChecker()
        result = await c.check_candidate("anything", [])
        assert result.accepted is True
