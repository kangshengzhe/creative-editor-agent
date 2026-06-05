"""Unit tests for ReviewTranslator (HK operator-review translations).

Covers the batched-translation happy path, English-source language dropping,
graceful degradation on LLM failure, and — crucially — the patch-up retry that
fixes the real-world bug where the LLM omits the *last* entry of a long
numbered batch list, leaving the final headline / description untranslated.
"""

from __future__ import annotations

from creative_agent.errors.codes import ToolFailureError
from creative_agent.integration.review_translator import ReviewTranslator
from creative_agent.llm.mock_client import MockLLMClient


def _entry(i: int, *, en: bool = True) -> dict:
    d = {"index": i, "zh-Hans": f"简{i}", "zh-Hant": f"繁{i}"}
    if en:
        d["en"] = f"en{i}"
    return d


class TestEmptyAndBasics:
    async def test_empty_copies_returns_empty_without_llm_call(self) -> None:
        llm = MockLLMClient()
        rt = ReviewTranslator(llm)
        assert await rt.translate([], copy_is_english=False) == []
        assert llm.calls == []

    async def test_english_source_drops_en_language(self) -> None:
        llm = MockLLMClient()
        llm.set_default_response({"items": [{"index": 0, "zh-Hans": "简", "zh-Hant": "繁"}]})
        rt = ReviewTranslator(llm)
        out = await rt.translate(["Top up now"], copy_is_english=True)
        assert out[0] == {"zh-Hans": "简", "zh-Hant": "繁"}
        # The prompt must only request zh-Hans / zh-Hant for English source.
        assert "en" not in llm.calls[-1]["prompt"].split("requested: ")[1].split("\n")[0]


class TestPatchUpForDroppedLastItem:
    """The regression guard for the 'last item untranslated' bug."""

    async def test_dropped_last_item_is_patched_up(self) -> None:
        llm = MockLLMClient()
        # First (batch) call covers indices 0..2 but the model omits the LAST
        # one (index 2) — the exact failure observed against the real LLM.
        llm.set_response(
            "indexed 0 to 2",
            {"items": [_entry(0), _entry(1)]},
        )
        # Patch-up call re-requests ONLY the missing copy (a 1-item batch,
        # "indexed 0 to 0"); the model returns it this time.
        llm.set_response(
            "indexed 0 to 0",
            {"items": [_entry(0)]},
        )
        rt = ReviewTranslator(llm)
        out = await rt.translate(
            ["copy A", "copy B", "copy C"], copy_is_english=False
        )

        # Every item — including the last — now has a translation.
        assert all(out[i] for i in range(3)), out
        assert out[2] == {"zh-Hans": "简0", "zh-Hant": "繁0", "en": "en0"}
        # Exactly two LLM calls: the batch + one patch-up pass.
        assert len(llm.calls) == 2

    async def test_no_patchup_when_batch_complete(self) -> None:
        llm = MockLLMClient()
        llm.set_response(
            "indexed 0 to 2",
            {"items": [_entry(0), _entry(1), _entry(2)]},
        )
        rt = ReviewTranslator(llm)
        out = await rt.translate(
            ["copy A", "copy B", "copy C"], copy_is_english=False
        )
        assert all(out)
        # No missing items → no patch-up call.
        assert len(llm.calls) == 1

    async def test_patchup_failure_leaves_only_missing_item_empty(self) -> None:
        llm = MockLLMClient()
        llm.set_response("indexed 0 to 2", {"items": [_entry(0), _entry(1)]})
        # Patch-up call fails — the missing item stays empty, the rest survive.
        llm.set_failure("indexed 0 to 0", ToolFailureError(
            tool_name="LLMClient", message="simulated patch-up failure"
        ))
        rt = ReviewTranslator(llm)
        out = await rt.translate(
            ["copy A", "copy B", "copy C"], copy_is_english=False
        )
        assert out[0] and out[1]
        assert out[2] == {}


class TestGracefulDegradation:
    async def test_batch_failure_returns_all_empty(self) -> None:
        llm = MockLLMClient()
        llm.set_failure("indexed 0 to 1", ToolFailureError(
            tool_name="LLMClient", message="simulated batch failure"
        ))
        rt = ReviewTranslator(llm)
        out = await rt.translate(["a", "b"], copy_is_english=False)
        assert out == [{}, {}]
