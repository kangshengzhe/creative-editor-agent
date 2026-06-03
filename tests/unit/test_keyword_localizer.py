"""Unit tests for KeywordLocalizer (PPC keyword localization).

Verifies generic keywords get localized into the target language while brand /
proper nouns are kept verbatim, English targets are a no-op, and any LLM
failure degrades to the identity mapping (keywords are never lost).
"""

from __future__ import annotations

from creative_agent.errors.codes import ToolFailureError
from creative_agent.integration.keyword_localizer import KeywordLocalizer
from creative_agent.llm.mock_client import MockLLMClient


class TestEnglishAndEmptyAreNoOp:
    async def test_english_target_is_identity_without_llm_call(self) -> None:
        llm = MockLLMClient()
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup", "bonus"], "en")
        assert mapping == {"topup": "topup", "bonus": "bonus"}
        # No LLM call made for the English path.
        assert llm.calls == []

    async def test_empty_keywords_is_identity_without_llm_call(self) -> None:
        llm = MockLLMClient()
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize([], "es")
        assert mapping == {}
        assert llm.calls == []


class TestLocalization:
    async def test_generic_translated_brand_kept(self) -> None:
        llm = MockLLMClient()
        # Spanish: "topup" -> "recarga" (generic), "Coco" kept (brand).
        llm.set_default_response(
            {
                "keywords": [
                    {"original": "topup", "localized": "recarga", "kept_verbatim": False},
                    {"original": "Coco", "localized": "Coco", "kept_verbatim": True},
                ]
            }
        )
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup", "Coco"], "es")
        assert mapping == {"topup": "recarga", "Coco": "Coco"}

    async def test_missing_entry_defaults_to_identity(self) -> None:
        llm = MockLLMClient()
        # LLM only returns one of two keywords; the other must default to itself.
        llm.set_default_response(
            {"keywords": [{"original": "topup", "localized": "recarga"}]}
        )
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup", "bonus"], "es")
        assert mapping == {"topup": "recarga", "bonus": "bonus"}

    async def test_blank_localized_falls_back_to_original(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response(
            {"keywords": [{"original": "topup", "localized": "   "}]}
        )
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup"], "es")
        assert mapping == {"topup": "topup"}


class TestGracefulDegradation:
    async def test_llm_failure_degrades_to_identity(self) -> None:
        llm = MockLLMClient()
        llm.set_failure("topup", ToolFailureError(tool_name="LLMClient", message="boom"))
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup", "bonus"], "es")
        # Never drop keywords on failure.
        assert mapping == {"topup": "topup", "bonus": "bonus"}

    async def test_non_dict_payload_degrades_to_identity(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response("not a json object")
        loc = KeywordLocalizer(llm)
        mapping = await loc.localize(["topup"], "ar")
        assert mapping == {"topup": "topup"}
