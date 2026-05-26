"""Tests for API Gateway input validation."""
import pytest
from creative_agent.api.gateway import parse_and_validate, reset_request_counter
from creative_agent.errors import ValidationError


@pytest.fixture(autouse=True)
def _reset_counter():
    reset_request_counter()
    yield
    reset_request_counter()


def _valid_brief():
    return {
        "campaign_topic": "Test campaign",
        "target_platform": "GOOGLE_ADS",
        "target_market": "EN_GLOBAL",
        "creative_type": "HEADLINE",
    }


class TestParseAndValidate:
    async def test_valid_brief_returns_triple(self):
        request_id, brief, warnings = await parse_and_validate(_valid_brief())
        assert request_id.startswith("req_")
        assert brief.campaign_topic == "Test campaign"
        assert warnings == []

    async def test_missing_field_raises(self):
        data = _valid_brief()
        del data["campaign_topic"]
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate(data)
        assert exc_info.value.code == "MISSING_FIELD"

    async def test_empty_topic_raises(self):
        data = _valid_brief()
        data["campaign_topic"] = "   "
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate(data)
        assert exc_info.value.code == "MISSING_FIELD"

    async def test_invalid_platform_raises(self):
        data = _valid_brief()
        data["target_platform"] = "INVALID"
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate(data)
        assert exc_info.value.code == "INVALID_ENUM"

    async def test_invalid_market_raises(self):
        data = _valid_brief()
        data["target_market"] = "MARS"
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate(data)
        assert exc_info.value.code == "INVALID_ENUM"

    async def test_topic_too_long_raises(self):
        data = _valid_brief()
        data["campaign_topic"] = "x" * 201
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate(data)
        assert exc_info.value.code == "INVALID_LENGTH"

    async def test_keywords_truncated(self):
        data = _valid_brief()
        data["keywords"] = [f"kw{i}" for i in range(30)]
        _, brief, warnings = await parse_and_validate(data)
        assert len(brief.keywords) == 20
        assert len(warnings) == 1
        assert "truncated" in warnings[0]

    async def test_malformed_json_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            await parse_and_validate("{invalid json")
        assert exc_info.value.code == "MALFORMED_JSON"

    async def test_request_id_increments(self):
        id1, _, _ = await parse_and_validate(_valid_brief())
        id2, _, _ = await parse_and_validate(_valid_brief())
        # Extract sequence numbers
        seq1 = int(id1.split("_")[-1])
        seq2 = int(id2.split("_")[-1])
        assert seq2 > seq1
