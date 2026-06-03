"""Example-based unit tests for Semantic_Diversity_Checker fallback & logging.

Feature: creative-localization-diversity (task 5.4).

These tests cover the failure/degradation semantics described in design.md
(§ Error Handling / Semantic Diversity Failures) and Requirements 2.6 & 2.8:

* **Requirement 2.8** — When the sentence-embedding model is unavailable, or
  embedding computation exceeds the configured ``timeout_seconds``, the
  :class:`~creative_agent.integration.semantic_diversity.SemanticDiversityChecker`
  degrades to *text-dedup only*: it returns a
  :class:`~creative_agent.integration.semantic_diversity.DiversityResult` with
  ``fallback=True`` and ``accepted=True``, logs a ``semantic_diversity.fallback``
  warning, and raises **nothing**. Because no exception escapes the component,
  the global circuit-breaker failure counter is left untouched — the tests
  assert on that signal by confirming ``check_candidate`` never raises.

* **Requirement 2.6** — Every rejection is logged via the
  ``semantic_diversity.rejected`` event with the candidate pair, the
  Cosine_Similarity score, and the applied Similarity_Threshold. The tests
  assert both on the structured log record (captured with
  ``structlog.testing.capture_logs``) and on the returned ``DiversityResult``
  fields (``rejected_pair``, ``similarity_score``) plus the configured
  threshold.

The embedding function is injected (``embed_fn``) so these tests are fully
deterministic and require no ML dependency. The project runs with
``asyncio_mode = "auto"`` (pytest-asyncio), so plain ``async def test_...``
coroutines are awaited automatically; ``embed_fn`` runs via
``asyncio.to_thread`` so a blocking ``time.sleep`` correctly triggers the
``asyncio.wait_for`` timeout.
"""

from __future__ import annotations

import time
from typing import Sequence

import pytest
from structlog.testing import capture_logs

from creative_agent.integration.semantic_diversity import (
    DiversityResult,
    EmbeddingUnavailableError,
    SemanticDiversityChecker,
)
from creative_agent.models import SemanticDiversityConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_event(records: list[dict], event: str) -> dict:
    """Return the single captured structlog record with the given event name."""
    matches = [r for r in records if r.get("event") == event]
    assert matches, f"expected a {event!r} log record; captured: {records!r}"
    assert len(matches) == 1, f"expected exactly one {event!r} record; got {matches!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# Requirement 2.8 — embedding model unavailable -> text-dedup-only fallback
# ---------------------------------------------------------------------------

class TestEmbeddingUnavailableFallback:
    """When ``embed_fn`` raises ``EmbeddingUnavailableError`` the checker
    degrades to text-dedup only without raising or tripping the breaker."""

    @staticmethod
    def _unavailable_embed_fn(text: str) -> Sequence[float]:
        raise EmbeddingUnavailableError("model not installed")

    async def test_returns_fallback_result_when_embedding_unavailable(self) -> None:
        checker = SemanticDiversityChecker(embed_fn=self._unavailable_embed_fn)

        result = await checker.check_candidate(
            "A brand new candidate headline",
            ["A previously accepted headline"],
        )

        assert isinstance(result, DiversityResult)
        # Degraded path: not rejected on semantic grounds, flagged as fallback.
        assert result.fallback is True
        assert result.accepted is True
        assert result.rejected_pair is None
        # No comparison was performed, so no similarity score is reported.
        assert result.similarity_score is None

    async def test_does_not_raise_when_embedding_unavailable(self) -> None:
        """No exception escapes -> the circuit-breaker counter is untouched."""
        checker = SemanticDiversityChecker(embed_fn=self._unavailable_embed_fn)

        # The call completing normally (returning a result) is the signal the
        # Orchestrator relies on to NOT increment the breaker (Requirement 2.8).
        result = await checker.check_candidate("new", ["accepted"])
        assert result.fallback is True

    async def test_logs_fallback_warning_when_unavailable(self) -> None:
        checker = SemanticDiversityChecker(embed_fn=self._unavailable_embed_fn)

        with capture_logs() as records:
            await checker.check_candidate("new candidate", ["accepted one"])

        record = _find_event(records, "semantic_diversity.fallback")
        assert record["log_level"] == "warning"
        assert record["reason"] == "model_unavailable"


# ---------------------------------------------------------------------------
# Requirement 2.8 — embedding timeout (> timeout_seconds) -> fallback
# ---------------------------------------------------------------------------

class TestEmbeddingTimeoutFallback:
    """When embedding computation exceeds ``timeout_seconds`` the checker
    degrades to text-dedup only (same contract as model-unavailable)."""

    @staticmethod
    def _slow_embed_fn(text: str) -> Sequence[float]:
        # embed_fn runs in a worker thread via asyncio.to_thread, so a blocking
        # sleep here is bounded by the surrounding asyncio.wait_for timeout.
        time.sleep(0.5)
        return [1.0, 0.0, 0.0]

    async def test_returns_fallback_result_on_timeout(self) -> None:
        config = SemanticDiversityConfig(timeout_seconds=0.05)
        checker = SemanticDiversityChecker(config, embed_fn=self._slow_embed_fn)

        start = time.perf_counter()
        result = await checker.check_candidate("candidate", ["accepted"])
        elapsed = time.perf_counter() - start

        assert result.fallback is True
        assert result.accepted is True
        assert result.rejected_pair is None
        assert result.similarity_score is None
        # The fallback fires on the timeout, well before the 0.5s embed sleep.
        assert elapsed < 0.5

    async def test_does_not_raise_on_timeout(self) -> None:
        config = SemanticDiversityConfig(timeout_seconds=0.05)
        checker = SemanticDiversityChecker(config, embed_fn=self._slow_embed_fn)

        result = await checker.check_candidate("candidate", ["accepted"])
        assert result.fallback is True

    async def test_logs_fallback_warning_on_timeout(self) -> None:
        config = SemanticDiversityConfig(timeout_seconds=0.05)
        checker = SemanticDiversityChecker(config, embed_fn=self._slow_embed_fn)

        with capture_logs() as records:
            await checker.check_candidate("candidate", ["accepted"])

        record = _find_event(records, "semantic_diversity.fallback")
        assert record["log_level"] == "warning"
        assert record["reason"] == "timeout"


# ---------------------------------------------------------------------------
# Requirement 2.6 — rejection logging contains pair, score, threshold
# ---------------------------------------------------------------------------

class TestRejectionLogging:
    """A candidate whose embedding is ~identical to a pool member is rejected,
    and the rejection is logged with the pair, similarity score, and threshold."""

    # Deterministic embeddings: "candidate" is identical to the accepted
    # "twin" (cosine 1.0) and orthogonal to "different" (cosine 0.0).
    _VECTORS: dict[str, list[float]] = {
        "candidate": [1.0, 0.0, 0.0],
        "twin": [1.0, 0.0, 0.0],
        "different": [0.0, 1.0, 0.0],
    }

    @classmethod
    def _deterministic_embed_fn(cls, text: str) -> Sequence[float]:
        return cls._VECTORS[text]

    async def test_rejects_near_identical_candidate(self) -> None:
        checker = SemanticDiversityChecker(embed_fn=self._deterministic_embed_fn)

        result = await checker.check_candidate("candidate", ["different", "twin"])

        assert result.accepted is False
        assert result.fallback is False
        # The rejected pair points at the most-similar accepted candidate.
        assert result.rejected_pair == ("candidate", "twin")
        # Highest cosine similarity observed against the pool (1.0 vs "twin").
        assert result.similarity_score == pytest.approx(1.0)

    async def test_rejection_log_contains_pair_score_threshold(self) -> None:
        # Use a non-default (but valid) threshold to prove the *configured*
        # threshold is what gets logged.
        config = SemanticDiversityConfig(similarity_threshold=0.7)
        checker = SemanticDiversityChecker(config, embed_fn=self._deterministic_embed_fn)

        with capture_logs() as records:
            result = await checker.check_candidate("candidate", ["different", "twin"])

        assert result.accepted is False

        record = _find_event(records, "semantic_diversity.rejected")
        assert record["log_level"] == "info"
        # Pair: new candidate + the most-similar accepted candidate.
        assert record["candidate"] == "candidate"
        assert record["most_similar_accepted"] == "twin"
        # Score is present and matches the observed cosine similarity (1.0).
        assert record["similarity_score"] == pytest.approx(1.0)
        # The applied threshold is the configured one.
        assert record["threshold"] == 0.7

    async def test_accepts_diverse_candidate(self) -> None:
        """Sanity check: an orthogonal candidate is accepted (no rejection)."""
        checker = SemanticDiversityChecker(embed_fn=self._deterministic_embed_fn)

        result = await checker.check_candidate("candidate", ["different"])

        assert result.accepted is True
        assert result.fallback is False
        assert result.rejected_pair is None
        # Best (and only) similarity vs the pool is 0.0 (orthogonal).
        assert result.similarity_score == pytest.approx(0.0)
