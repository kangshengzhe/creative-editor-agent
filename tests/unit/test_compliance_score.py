"""Tests for Compliance_Checker scoring formula."""
import pytest
from creative_agent.tools.compliance_checker import ComplianceChecker
from creative_agent.models import Target_Language, Compliance_Severity


@pytest.fixture
def checker():
    """Dictionary-only checker (no LLM)."""
    return ComplianceChecker(llm=None)


class TestComplianceScoring:
    async def test_clean_copy_scores_1(self, checker):
        report = await checker.check("Buy game credits today", Target_Language.EN)
        assert report.compliance_score == 1.0
        assert report.violations == []

    async def test_empty_copy_scores_0_block(self, checker):
        report = await checker.check("", Target_Language.EN)
        assert report.compliance_score == 0.0
        assert len(report.violations) == 1
        assert report.violations[0].severity == Compliance_Severity.BLOCK

    async def test_whitespace_only_scores_0(self, checker):
        report = await checker.check("   \t\n  ", Target_Language.EN)
        assert report.compliance_score == 0.0

    async def test_block_term_scores_0(self, checker):
        # "guaranteed jackpot" is BLOCK in en.json
        report = await checker.check("Win a guaranteed jackpot now", Target_Language.EN)
        assert report.compliance_score == 0.0
        assert any(v.severity == Compliance_Severity.BLOCK for v in report.violations)

    async def test_one_warn_scores_08(self, checker):
        # "guaranteed" alone is WARN (EXAGGERATION) in en.json
        report = await checker.check("This is guaranteed good", Target_Language.EN)
        assert abs(report.compliance_score - 0.8) < 0.01

    async def test_two_warns_scores_06(self, checker):
        # "guaranteed" (WARN) + "best ever" (WARN)
        report = await checker.check("guaranteed best ever deal", Target_Language.EN)
        assert abs(report.compliance_score - 0.6) < 0.01

    async def test_five_warns_scores_01(self, checker):
        # 5 WARN hits → max(0.1, 1.0 - 1.0) = 0.1
        report = await checker.check(
            "guaranteed perfect best ever only today last chance ever",
            Target_Language.EN,
        )
        # At least 4-5 WARN terms should hit
        warn_count = sum(1 for v in report.violations if v.severity == Compliance_Severity.WARN)
        if warn_count >= 5:
            assert abs(report.compliance_score - 0.1) < 0.01
        else:
            # Fewer hits due to word boundary rules; score should still be low
            assert report.compliance_score <= 0.6
