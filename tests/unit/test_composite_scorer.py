"""Tests for Composite Scorer — scoring formula and ranking logic."""
import pytest
from creative_agent.orchestrator.composite_scorer import compute_composite_score, rank_candidates
from creative_agent.models import (
    Creative_Candidate, Compliance_Report, AB_Ranking,
    Compliance_Severity, Target_Platform, Creative_Type,
)


def _make_candidate(
    compliance_score=1.0, keyword_coverage=1.0, cta_strength=0.5,
    generation_index=0, has_block=False,
):
    """Helper to build a minimal Creative_Candidate for testing."""
    violations = []
    if has_block:
        from creative_agent.models import Violation, ViolationCategory
        violations = [Violation(
            start=0, end=1, category=ViolationCategory.GAMBLING,
            severity=Compliance_Severity.BLOCK, suggestion="test",
        )]
        compliance_score = 0.0

    report = Compliance_Report(
        compliance_score=compliance_score,
        violations=violations,
        checked_at="2026-01-01T00:00:00Z",
        checker_version="test",
    )
    return Creative_Candidate(
        candidate_id=f"test_c{generation_index}",
        generation_index=generation_index,
        source_copy=f"Test copy {generation_index}",
        source_language="en",
        compliance_report=report,
        keyword_coverage=keyword_coverage,
        hit_keywords=["topup"] if keyword_coverage > 0 else [],
        skipped_keywords=[],
        cta_strength_score=cta_strength,
        cta_variants=None,
        localized_versions={},
        failed_languages=[],
        composite_score=0.0,
        target_platform=Target_Platform.GOOGLE_ADS,
        creative_type=Creative_Type.HEADLINE,
        warnings=[],
    )


class TestCompositeScore:
    def test_formula_basic(self):
        c = _make_candidate(compliance_score=1.0, keyword_coverage=1.0, cta_strength=1.0)
        score = compute_composite_score(c)
        assert abs(score - 1.0) < 1e-9

    def test_formula_weights(self):
        c = _make_candidate(compliance_score=0.8, keyword_coverage=0.6, cta_strength=0.4)
        expected = 0.5 * 0.8 + 0.25 * 0.6 + 0.25 * 0.4  # = 0.65
        score = compute_composite_score(c)
        assert abs(score - expected) < 1e-9

    def test_formula_zero(self):
        c = _make_candidate(compliance_score=0.0, keyword_coverage=0.0, cta_strength=0.0)
        assert compute_composite_score(c) == 0.0

    def test_score_in_range(self):
        c = _make_candidate(compliance_score=0.5, keyword_coverage=0.5, cta_strength=0.5)
        score = compute_composite_score(c)
        assert 0.0 <= score <= 1.0


class TestRankCandidates:
    def test_sorts_by_composite_desc(self):
        c1 = _make_candidate(compliance_score=1.0, keyword_coverage=1.0, cta_strength=0.8, generation_index=0)
        c2 = _make_candidate(compliance_score=1.0, keyword_coverage=0.5, cta_strength=0.5, generation_index=1)
        ranking = rank_candidates([c2, c1], request_id="test", total_generated=2)
        assert ranking.ranked_candidates[0].candidate_id == "test_c0"

    def test_filters_block(self):
        c1 = _make_candidate(generation_index=0)
        c2 = _make_candidate(generation_index=1, has_block=True)
        ranking = rank_candidates([c1, c2], request_id="test", total_generated=2)
        assert len(ranking.ranked_candidates) == 1
        assert ranking.total_candidates_filtered_out == 1

    def test_tiebreak_compliance(self):
        # Same composite but different compliance
        c1 = _make_candidate(compliance_score=0.8, keyword_coverage=0.8, cta_strength=0.8, generation_index=0)
        c2 = _make_candidate(compliance_score=1.0, keyword_coverage=0.6, cta_strength=0.6, generation_index=1)
        ranking = rank_candidates([c1, c2], request_id="test", total_generated=2)
        # Both have same composite (0.8), but c2 has higher compliance
        first = ranking.ranked_candidates[0]
        assert first.compliance_report.compliance_score >= ranking.ranked_candidates[1].compliance_report.compliance_score

    def test_empty_input(self):
        ranking = rank_candidates([], request_id="test", total_generated=0)
        assert ranking.ranked_candidates == []
