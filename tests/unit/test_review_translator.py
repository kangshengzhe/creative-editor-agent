"""Unit tests for ReviewTranslator (HK operator-review translations).

The translator issues ONE LLM call per copy, concurrently. This guarantees the
result for copy i is the translation of copy i (no batch index to mislabel),
fixing two real bugs:

* "dropped last item": batched numbered-list calls reliably omitted the final
  entry, leaving the last headline/description untranslated.
* "wrong-candidate translation": the LLM occasionally mislabeled per-item
  indices in a batch, attaching a translation to the wrong candidate.

Covers: empty input, English-source language dropping, per-copy alignment under
partial failure, and graceful degradation.
"""

from __future__ import annotations

from typing import Optional

from creative_agent.errors.codes import ToolFailureError
from creative_agent.integration.review_translator import ReviewTranslator
from creative_agent.llm.client import LLMClient


class _ScriptedLLM(LLMClient):
    """LLM double that maps a copy (matched as a prompt substring) to a JSON
    response or an exception, and records call order.

    One entry per distinct copy lets tests prove that copy i's result comes
    from copy i's own call (per-copy translation, no batch index).
    """

    def __init__(self, by_copy: dict[str, object]) -> None:
        self._by_copy = by_copy
        self.json_calls: list[str] = []

    async def complete(self, *a, **k) -> str:  # pragma: no cover
        raise NotImplementedError

    async def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> dict:
        self.json_calls.append(prompt)
        for copy, resp in self._by_copy.items():
            if copy in prompt:
                if isinstance(resp, Exception):
                    raise resp
                return resp  # type: ignore[return-value]
        return {}


class TestEmptyAndBasics:
    async def test_empty_copies_returns_empty_without_llm_call(self) -> None:
        llm = _ScriptedLLM({})
        rt = ReviewTranslator(llm)
        assert await rt.translate([], copy_is_english=False) == []
        assert llm.json_calls == []

    async def test_english_source_drops_en_language(self) -> None:
        llm = _ScriptedLLM(
            {"Top up now": {"zh-Hans": "立即充值", "zh-Hant": "立即儲值", "en": "Top up now"}}
        )
        rt = ReviewTranslator(llm)
        out = await rt.translate(["Top up now"], copy_is_english=True)
        # English is dropped when the source copy is already English.
        assert out[0] == {"zh-Hans": "立即充值", "zh-Hant": "立即儲值"}
        # The prompt must not request the English review language.
        requested = llm.json_calls[-1].split("requested: ")[1].split("\n")[0]
        assert "en" not in requested


class TestPerCopyAlignment:
    """The regression guard for 'translation attached to wrong candidate'."""

    async def test_each_copy_gets_its_own_translation(self) -> None:
        llm = _ScriptedLLM(
            {
                "Get 20% more game credits": {
                    "zh-Hans": "获得20%更多游戏积分",
                    "zh-Hant": "獲得20%更多遊戲積分",
                    "en": "Get 20% more game credits",
                },
                "Always-on support for your top": {
                    "zh-Hans": "全天候支持您的顶级需求",
                    "zh-Hant": "全天候支援您的頂級需求",
                    "en": "Always-on support for your top",
                },
            }
        )
        rt = ReviewTranslator(llm)
        copies = ["Get 20% more game credits", "Always-on support for your top"]
        out = await rt.translate(copies, copy_is_english=True)

        # Copy 0's translation must reflect copy 0 (game credits), NOT topup.
        assert "游戏积分" in out[0]["zh-Hans"]
        # Copy 1's translation must reflect copy 1 (support), NOT copy 0.
        assert "全天候" in out[1]["zh-Hans"]
        # One call per copy.
        assert len(llm.json_calls) == 2

    async def test_partial_failure_isolated_to_its_copy(self) -> None:
        llm = _ScriptedLLM(
            {
                "good copy": {"zh-Hans": "好文案", "zh-Hant": "好文案", "en": "good copy"},
                "bad copy": ToolFailureError(
                    tool_name="LLMClient", message="simulated failure"
                ),
            }
        )
        rt = ReviewTranslator(llm)
        out = await rt.translate(["good copy", "bad copy"], copy_is_english=True)
        assert out[0]  # good copy translated
        assert out[1] == {}  # bad copy's failure does not corrupt others

    async def test_blank_copy_skipped_without_llm_call(self) -> None:
        llm = _ScriptedLLM({"real": {"zh-Hans": "真"}})
        rt = ReviewTranslator(llm)
        out = await rt.translate(["   ", "real"], copy_is_english=True)
        assert out[0] == {}
        assert out[1] == {"zh-Hans": "真"}
        # Only the non-blank copy triggered a call.
        assert len(llm.json_calls) == 1


class TestGracefulDegradation:
    async def test_all_failures_return_all_empty(self) -> None:
        llm = _ScriptedLLM(
            {
                "a": ToolFailureError(tool_name="LLMClient", message="x"),
                "b": ToolFailureError(tool_name="LLMClient", message="y"),
            }
        )
        rt = ReviewTranslator(llm)
        out = await rt.translate(["a", "b"], copy_is_english=False)
        assert out == [{}, {}]
