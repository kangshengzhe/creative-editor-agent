"""Tests for SemanticDiversityConfig (task 1.1, requirements 2.4, 2.8)."""

import pytest
from pydantic import ValidationError

from creative_agent.models import SemanticDiversityConfig


class TestSemanticDiversityConfigDefaults:
    def test_defaults(self):
        cfg = SemanticDiversityConfig()
        # Default calibrated to 0.60 for the multilingual MiniLM model on short
        # ad copy (see diversity_config module docstring); 0.85 let paraphrases
        # through.
        assert cfg.similarity_threshold == 0.60
        assert cfg.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
        assert cfg.timeout_seconds == 3.0
        assert cfg.enabled is True

    def test_overrides(self):
        cfg = SemanticDiversityConfig(
            similarity_threshold=0.5,
            embedding_model="custom-model",
            timeout_seconds=1.5,
            enabled=False,
        )
        assert cfg.similarity_threshold == 0.5
        assert cfg.embedding_model == "custom-model"
        assert cfg.timeout_seconds == 1.5
        assert cfg.enabled is False


class TestSimilarityThresholdBounds:
    def test_lower_bound_inclusive(self):
        assert SemanticDiversityConfig(similarity_threshold=0.30).similarity_threshold == 0.30

    def test_upper_bound_inclusive(self):
        assert SemanticDiversityConfig(similarity_threshold=0.99).similarity_threshold == 0.99

    def test_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            SemanticDiversityConfig(similarity_threshold=0.29)

    def test_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            SemanticDiversityConfig(similarity_threshold=1.0)


class TestTimeoutBounds:
    def test_positive_timeout_allowed(self):
        assert SemanticDiversityConfig(timeout_seconds=0.1).timeout_seconds == 0.1

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValidationError):
            SemanticDiversityConfig(timeout_seconds=0.0)

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValidationError):
            SemanticDiversityConfig(timeout_seconds=-1.0)


class TestExtraForbidden:
    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            SemanticDiversityConfig(unknown_field=123)
