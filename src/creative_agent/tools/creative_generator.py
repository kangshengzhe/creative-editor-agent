"""Creative_Generator — generates ad copy candidates from a Creative_Brief.

Implements design.md § Components and Interfaces / 3. Creative_Generator and
Requirements 2.1 – 2.7.

Behaviour summary
-----------------

* Builds a Chinese system prompt + a structured user prompt from the
  ``Creative_Brief`` (campaign topic, selling points, audience, target
  platform/market, creative type, character limit, optional keywords and
  exclusion list) and asks the LLM for ``min_count + 2`` candidates so
  dedup losses still leave us with the required minimum (Requirement 2.1).
* Each candidate that exceeds ``platform_spec.char_limit(creative_type)`` is
  hard-truncated to that limit (Requirements 2.2 – 2.4, Property 8).
* Deduplication uses ``strip().lower()`` with punctuation stripped to
  guarantee mutually distinct content (Requirement 2.5) and to honour the
  refill loop's ``exclude_copies`` set (orchestrator support for
  Requirement 7.6).
* Up to 3 attempts (initial + 2 retries) — failed attempts sleep 100 ms then
  200 ms before the next try (Requirement 2.7). Every individual failure
  bumps ``tool_failure_counter[0]`` for the orchestrator's global breaker.
* The whole call is bounded by ``asyncio.wait_for(timeout_ms / 1000)`` so we
  honour the 5000 ms budget (Requirement 2.6); a timeout counts as one
  failed attempt and feeds back into the retry loop.
* Each candidate's downstream fields (compliance, keyword, CTA, localization)
  are populated with neutral placeholders that downstream tools overwrite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from creative_agent.errors.codes import (
    GenerationFailureError,
    ToolFailureError,
)
from creative_agent.llm.client import LLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.candidate import Creative_Candidate
from creative_agent.models.compliance import Compliance_Report
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.logging import get_logger

__all__ = ["CreativeGenerator", "GeneratorOutput"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Initial attempt + up to 2 retries (Requirement 2.7).
_MAX_ATTEMPTS: int = 3

#: Backoff between attempts, in seconds. Index 0 is unused (no sleep before
#: the first attempt). Position N is the sleep before attempt N+1.
_RETRY_BACKOFF_S: tuple[float, ...] = (0.0, 0.100, 0.200)

#: How many extra candidates to ask the LLM for over ``min_count`` so dedup
#: losses still leave us at or above the minimum.
_OVER_REQUEST: int = 2

#: Tokens budget for the JSON candidate response.
_LLM_MAX_TOKENS: int = 4096

#: Sampling temperature — high enough to encourage candidate diversity
#: (Requirement 2.5) without going off-prompt.
_LLM_TEMPERATURE: float = 0.8

#: Punctuation characters stripped during dedup normalisation. Limited to a
#: small ASCII + CJK set; any unknown punctuation just survives normalisation,
#: which is conservative — at worst we keep two near-duplicates.
_PUNCTUATION_CHARS: str = ' \t\r\n.,;:!?"\'`~()[]{}<>/\\|-_+=*&^%$#@！？，。；：、（）【】《》""''…—'

_SYSTEM_PROMPT: str = (
    "你是 Coco AI 平台的资深广告文案策划，专门为多市场（菲律宾、泰国、俄罗斯、英语全球）"
    "的游戏充值业务撰写合规且高转化的广告创意。\n\n"
    "你的任务：基于用户提供的活动 Brief，生成多条互不重复的广告文案候选。\n\n"
    "输出要求：\n"
    "1. 严格返回 JSON 对象，结构为：{\"candidates\": [{\"copy\": \"...\"}, ...]}。\n"
    "   不得输出任何 JSON 之外的文字、注释或 Markdown 代码块。\n"
    "2. 候选数量必须达到或超过用户在 user 消息中指定的数量。\n"
    "3. 每条 copy 的字符长度必须严格不超过用户指定的字符上限。\n"
    "4. 候选之间必须在角度、语气、卖点切入或结构上有明显差异化，避免雷同。\n"
    "5. 禁止包含违禁词或承诺类表述：博彩保证（如 100% win、guaranteed jackpot）、"
    "医疗承诺（如 cure、heal）、虚假紧迫感（如 last chance ever）、"
    "歧视性内容、敏感事件影射等。\n"
    "6. 使用用户指定的源语言进行输出（默认 English）。"
)


# ---------------------------------------------------------------------------
# Output envelope
# ---------------------------------------------------------------------------


@dataclass
class GeneratorOutput:
    """Output envelope for :meth:`CreativeGenerator.generate`.

    Mirrors design.md § 3. Creative_Generator's ``GeneratorOutput`` shape.
    """

    candidates: list[Creative_Candidate]
    generation_time_ms: int


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class CreativeGenerator:
    """LLM-backed creative copy generator.

    Args:
        llm_client: Concrete :class:`LLMClient`; lifecycle owned by the
            caller (typically the orchestrator).
        timeout_ms: Total wall-clock budget for one ``generate()`` call,
            in milliseconds. Default 5000 ms per Requirement 2.6.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        timeout_ms: int = 180000,
    ) -> None:
        self._llm = llm_client
        self._timeout_ms = timeout_ms
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        *,
        exclude_copies: Optional[list[str]] = None,
        min_count: int = 5,
        request_id: Optional[str] = None,
        tool_failure_counter: Optional[list[int]] = None,
    ) -> GeneratorOutput:
        """Generate candidate ad copies for the given brief.

        Args:
            brief: Validated :class:`Creative_Brief`.
            platform_spec: Spec for ``brief.target_platform``; supplies the
                per-creative-type character limit.
            exclude_copies: Previously seen copies that must not be re-emitted
                (used by the orchestrator's refill loop, Requirement 7.6).
            min_count: Desired minimum number of unique candidates. Default 5
                per Requirement 2.1.
            request_id: Optional request id used in candidate id construction
                and structured logs; falls back to ``"noreq"`` when absent.
            tool_failure_counter: Optional single-element list used as a
                shared counter; each retry-eligible failure increments
                ``tool_failure_counter[0]`` so the orchestrator can trigger
                the global circuit breaker (Requirement 9.6).

        Returns:
            A :class:`GeneratorOutput` whose ``candidates`` list is of length
            ``≥ min_count`` and whose ``generation_time_ms`` records the
            wall-clock time of the successful call.

        Raises:
            GenerationFailureError: When all 3 attempts (initial + 2 retries)
                fail to produce ``≥ min_count`` unique candidates, or when
                the overall ``timeout_ms`` budget is exceeded.
        """
        start = time.monotonic()
        excludes = list(exclude_copies) if exclude_copies else []
        request_count = max(min_count, 5) + _OVER_REQUEST
        char_limit = platform_spec.char_limit(brief.creative_type)
        timeout_s = self._timeout_ms / 1000.0

        self._log.info(
            "creative_generator.invoked",
            request_id=request_id,
            min_count=min_count,
            request_count=request_count,
            exclude_count=len(excludes),
            target_platform=brief.target_platform.value,
            target_market=brief.target_market.value,
            creative_type=brief.creative_type.value,
            char_limit=char_limit,
            timeout_ms=self._timeout_ms,
        )

        try:
            candidates = await asyncio.wait_for(
                self._generate_with_retry(
                    brief=brief,
                    platform_spec=platform_spec,
                    char_limit=char_limit,
                    excludes=excludes,
                    request_count=request_count,
                    min_count=min_count,
                    request_id=request_id,
                    tool_failure_counter=tool_failure_counter,
                ),
                timeout=timeout_s,
            )
        except GenerationFailureError:
            raise
        except asyncio.TimeoutError as exc:
            # Overall budget blown — every in-flight attempt counts as a
            # failure for the global breaker.
            _bump_failure_counter(tool_failure_counter)
            self._log.error(
                "creative_generator.timeout",
                request_id=request_id,
                timeout_ms=self._timeout_ms,
            )
            raise GenerationFailureError(
                message=(
                    "Creative_Generator exceeded "
                    f"{self._timeout_ms}ms total budget"
                ),
                request_id=request_id,
                details={
                    "timeout_ms": self._timeout_ms,
                    "original_exception": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "creative_generator.completed",
            request_id=request_id,
            count=len(candidates),
            ms=elapsed_ms,
        )
        return GeneratorOutput(
            candidates=candidates,
            generation_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------------

    async def _generate_with_retry(
        self,
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        char_limit: int,
        excludes: list[str],
        request_count: int,
        min_count: int,
        request_id: Optional[str],
        tool_failure_counter: Optional[list[int]],
    ) -> list[Creative_Candidate]:
        last_error: Optional[str] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Backoff before retries (no sleep before the first attempt).
            if attempt > 1:
                backoff = _RETRY_BACKOFF_S[attempt - 1]
                if backoff > 0:
                    await asyncio.sleep(backoff)

            try:
                raw_copies = await self._call_llm(
                    brief=brief,
                    platform_spec=platform_spec,
                    char_limit=char_limit,
                    request_count=request_count,
                    excludes=excludes,
                )
            except ToolFailureError as exc:
                last_error = exc.message
                _bump_failure_counter(tool_failure_counter)
                self._log.warning(
                    "creative_generator.retry",
                    request_id=request_id,
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=exc.message,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — defensive belt
                last_error = f"{type(exc).__name__}: {exc}"
                _bump_failure_counter(tool_failure_counter)
                self._log.warning(
                    "creative_generator.retry",
                    request_id=request_id,
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=last_error,
                )
                continue

            candidates = self._post_process(
                raw_copies=raw_copies,
                brief=brief,
                char_limit=char_limit,
                excludes=excludes,
                request_id=request_id,
            )

            if len(candidates) >= min_count:
                if len(candidates) < request_count:
                    # Shrinkage warning — still good enough to return.
                    self._log.warning(
                        "creative_generator.dedup_shrinkage",
                        request_id=request_id,
                        attempt=attempt,
                        produced=len(candidates),
                        requested=request_count,
                    )
                return candidates

            # Soft failure: insufficient unique candidates.
            last_error = (
                f"only {len(candidates)} unique candidate(s) after dedup, "
                f"need {min_count}"
            )
            _bump_failure_counter(tool_failure_counter)
            self._log.warning(
                "creative_generator.retry",
                request_id=request_id,
                attempt=attempt,
                max_attempts=_MAX_ATTEMPTS,
                error=last_error,
                raw_count=len(raw_copies),
            )

        self._log.error(
            "creative_generator.failed",
            request_id=request_id,
            attempts=_MAX_ATTEMPTS,
            last_error=last_error,
        )
        raise GenerationFailureError(
            message=(
                f"Creative_Generator failed after {_MAX_ATTEMPTS} attempt(s); "
                f"last error: {last_error or 'unknown'}"
            ),
            request_id=request_id,
            details={
                "attempts": _MAX_ATTEMPTS,
                "last_error": last_error,
                "min_count": min_count,
            },
        )

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        char_limit: int,
        request_count: int,
        excludes: list[str],
    ) -> list[str]:
        """Single ``complete_json`` call returning a list of raw copy strings."""
        prompt = self._build_prompt(
            brief=brief,
            platform_spec=platform_spec,
            char_limit=char_limit,
            request_count=request_count,
            excludes=excludes,
        )

        try:
            payload = await self._llm.complete_json(
                prompt,
                system=_SYSTEM_PROMPT,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                timeout_ms=self._timeout_ms,
            )
        except ToolFailureError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap any transport / parse error
            raise ToolFailureError(
                tool_name="Creative_Generator",
                message=f"LLM call failed: {exc}",
                original_exception=exc,
            ) from exc

        return self._extract_copies(payload)

    @staticmethod
    def _extract_copies(payload: Any) -> list[str]:
        """Extract candidate copies from the LLM JSON response.

        Strict path: ``{"candidates": [{"copy": "..."}, ...]}`` per the
        spec. We also tolerate a few common shapes (bare list, list of
        plain strings, ``"text"`` instead of ``"copy"``) so a slightly
        misbehaving model doesn't blow up the whole pipeline.
        """
        if isinstance(payload, list):
            raw_list: list[Any] = payload
        elif isinstance(payload, dict):
            cand = payload.get("candidates")
            if isinstance(cand, list):
                raw_list = cand
            elif isinstance(payload.get("copies"), list):  # tolerated alias
                raw_list = payload["copies"]
            else:
                raise ToolFailureError(
                    tool_name="Creative_Generator",
                    message=(
                        "LLM response missing 'candidates' list; "
                        f"got keys={list(payload.keys())!r}"
                    ),
                )
        else:
            raise ToolFailureError(
                tool_name="Creative_Generator",
                message=(
                    "LLM response must be a JSON object or array; "
                    f"got {type(payload).__name__}"
                ),
            )

        cleaned: list[str] = []
        for entry in raw_list:
            if isinstance(entry, str):
                if entry.strip():
                    cleaned.append(entry)
            elif isinstance(entry, dict):
                copy = entry.get("copy")
                if isinstance(copy, str) and copy.strip():
                    cleaned.append(copy)
                else:
                    text = entry.get("text")
                    if isinstance(text, str) and text.strip():
                        cleaned.append(text)
            # else: silently drop — non-string, non-dict entries
        return cleaned

    # ------------------------------------------------------------------
    # Post-processing: truncate, dedupe, build candidates
    # ------------------------------------------------------------------

    def _post_process(
        self,
        *,
        raw_copies: list[str],
        brief: Creative_Brief,
        char_limit: int,
        excludes: list[str],
        request_id: Optional[str],
    ) -> list[Creative_Candidate]:
        """Truncate, dedupe (incl. ``exclude_copies``), and build candidates."""
        seen_norm: set[str] = {self._normalise(c) for c in excludes if c}
        seen_norm.discard("")  # an empty exclude must not block real copies
        accepted_texts: list[str] = []

        for raw in raw_copies:
            truncated = self._truncate(raw, char_limit)
            if not truncated:
                continue
            norm = self._normalise(truncated)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            accepted_texts.append(truncated)

        return [
            self._build_candidate(
                brief=brief,
                source_copy=text,
                generation_index=i,
                request_id=request_id,
            )
            for i, text in enumerate(accepted_texts)
        ]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        char_limit: int,
        request_count: int,
        excludes: list[str],
    ) -> str:
        """Build the user-message prompt described in design § 3."""
        selling_points = (
            "; ".join(brief.selling_points)
            if brief.selling_points
            else "未指定"
        )
        target_audience = brief.target_audience or "未指定"
        brand_name = brief.brand_name or "未指定"
        forbidden_symbols = (
            list(platform_spec.forbidden_symbols)
            if platform_spec.forbidden_symbols
            else []
        )
        keywords = list(brief.keywords or [])

        body = (
            "请基于以下创意 Brief 生成多个差异化广告文案候选。\n\n"
            "活动信息：\n"
            f"- 活动主题：{brief.campaign_topic}\n"
            f"- 卖点：{selling_points}\n"
            f"- 目标受众：{target_audience}\n"
            f"- 品牌名称：{brand_name}\n"
            f"- 输出语言：{brief.source_language}\n\n"
            "投放设定：\n"
            f"- 目标平台：{brief.target_platform.value}\n"
            f"- 目标市场：{brief.target_market.value}\n"
            f"- 文案类型：{brief.creative_type.value}\n"
            f"- 字符上限：{char_limit}\n\n"
            f"请生成至少 {request_count} 条互不重复的广告文案候选。\n\n"
            "返回严格的 JSON 对象，结构示例：\n"
            '{"candidates": [{"copy": "..."}, {"copy": "..."}, ...]}\n\n'
            "硬性要求：\n"
            f"1. 每条 copy 的字符长度必须 ≤ {char_limit}。\n"
            f"2. 文案必须符合 {brief.target_platform.value} 平台的风格规范。\n"
            "3. 候选之间在角度 / 语气 / 卖点切入上必须有明显差异。\n"
            f"4. 使用 {brief.source_language} 作为输出语言。\n"
            "5. 禁止出现违禁词与承诺类表述（赌博保证、医疗承诺、虚假紧迫感等）。\n"
        )
        if forbidden_symbols:
            body += (
                f"6. 不得出现以下符号或符号组合：{forbidden_symbols}\n"
            )

        if keywords:
            body += (
                "\nSEO 关键词参考（按优先级，可融入但不强制）：\n- "
                + "\n- ".join(keywords)
                + "\n"
            )

        if excludes:
            preview = excludes[:30]
            body += (
                "\n请务必避免与以下已生成文案重复或近义改写：\n- "
                + "\n- ".join(preview)
                + "\n"
            )

        return body

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise a candidate for dedup: strip + lower + drop punctuation.

        The punctuation strip is conservative (a small ASCII + CJK set) and
        only collapses whitespace runs after substitution; characters not in
        the strip set survive unchanged so genuinely different copies cannot
        collide.
        """
        lowered = text.strip().lower()
        if not lowered:
            return ""
        # Replace each punctuation char with a single space, then collapse.
        translated = lowered.translate(
            {ord(ch): " " for ch in _PUNCTUATION_CHARS}
        )
        return " ".join(translated.split())

    @staticmethod
    def _truncate(text: str, char_limit: int) -> str:
        """Hard-truncate ``text`` to ``char_limit`` characters."""
        if char_limit <= 0:
            return ""
        stripped = text.strip()
        if len(stripped) <= char_limit:
            return stripped
        return stripped[:char_limit].rstrip()

    @staticmethod
    def _build_candidate(
        *,
        brief: Creative_Brief,
        source_copy: str,
        generation_index: int,
        request_id: Optional[str],
    ) -> Creative_Candidate:
        """Construct a Creative_Candidate with placeholder downstream fields."""
        rid = request_id or "noreq"
        candidate_id = f"{rid}_c{generation_index}_{uuid4().hex[:6]}"

        # Neutral placeholder — overwritten by Compliance_Checker downstream.
        # ``checked_at`` is left as an empty string per task spec; the
        # checker fills in a real ISO-8601 timestamp on first check.
        placeholder_report = Compliance_Report(
            compliance_score=0.0,
            violations=[],
            checked_at="",
            checker_version="pending",
        )

        return Creative_Candidate(
            candidate_id=candidate_id,
            generation_index=generation_index,
            source_copy=source_copy,
            source_language=brief.source_language,
            compliance_report=placeholder_report,
            keyword_coverage=0.0,
            hit_keywords=[],
            skipped_keywords=[],
            cta_strength_score=0.0,
            cta_variants=None,
            localized_versions={},
            failed_languages=[],
            composite_score=0.0,
            target_platform=brief.target_platform,
            creative_type=brief.creative_type,
            warnings=[],
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _bump_failure_counter(counter: Optional[list[int]]) -> None:
    """Increment the orchestrator's shared failure counter, if provided.

    The counter is modelled as a single-element list so callers (the
    orchestrator) get a mutable reference without forcing us to introduce a
    dedicated class for the global circuit breaker (Requirement 9.6).
    """
    if counter is not None and len(counter) > 0:
        counter[0] += 1
