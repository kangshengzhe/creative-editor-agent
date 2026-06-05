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
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from creative_agent.errors.codes import (
    GenerationFailureError,
    ToolFailureError,
)
from creative_agent.integration.angle_splitter import Angle, AngleSplitter
from creative_agent.integration.display_width import DisplayWidthCalculator
from creative_agent.integration.language_prompts import LanguagePromptSelector
from creative_agent.llm.client import LLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.candidate import Creative_Candidate
from creative_agent.models.compliance import Compliance_Report
from creative_agent.models.enums import Creative_Type, Target_Language, Target_Market
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.observability.logging import get_logger
from creative_agent.tools.localization_tool import LocalizationTool

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

#: Exactly how many candidates each angle-targeted LLM call requests
#: (Requirement 3.4). One call per angle, 3 candidates per call.
_CANDIDATES_PER_ANGLE_CALL: int = 3

#: Phase-2 wave overshoot. Each refill wave fires roughly this multiple of the
#: remaining gap (in calls) concurrently. Kept at 1.0 (cover the gap, no extra)
#: because overshooting generates surplus candidates that still cost full
#: downstream pipeline work (compliance/keyword/CTA) and inflate the reserve
#: pool, which measured *slower* overall, not faster. The win comes purely from
#: running the gap-covering calls of each wave in parallel rather than serially.
_WAVE_OVERSHOOT: float = 1.0

#: Hard cap on how many angle calls a single Phase-2 wave fires concurrently,
#: so a large ``min_count`` can't open an unbounded number of sockets at once.
_MAX_WAVE_CALLS: int = 12

#: Hard cap on the number of angle generation calls a single
#: ``generate_with_angles`` invocation will issue, so a pathological
#: ``min_count`` cannot spin the round-robin loop unbounded. Sized generously:
#: enough for ``max_angles`` (8) plus several refill cycles.
_MAX_ANGLE_CALLS: int = 64

#: Tokens budget for the JSON candidate response.
_LLM_MAX_TOKENS: int = 4096

#: Sampling temperature for creative generation — high enough to encourage
#: candidate diversity (Requirement 2.5) without going off-prompt. Configurable
#: via the ``GENERATION_TEMPERATURE`` environment variable (.env) so operators
#: can tune creativity without code changes; defaults to 0.8 and falls back to
#: 0.8 if the value is missing or unparseable. Only the *creative generation*
#: call uses this — precise tasks (translation, scoring, keyword matching) keep
#: their own low temperatures so raising this never destabilises them.
def _load_generation_temperature(default: float = 0.8) -> float:
    # Ensure .env is loaded even if this module is imported before the LLM
    # client (which also calls load_dotenv). load_dotenv is idempotent and does
    # not override variables already set in the real environment.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001 — dotenv is optional; env may be preset
        pass
    raw = os.getenv("GENERATION_TEMPERATURE")
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    # Clamp to the valid sampling range so a typo can't break API calls.
    return max(0.0, min(2.0, value))


_LLM_TEMPERATURE: float = _load_generation_temperature()

#: Language code for English (Req 1.7). English-primary markets keep the
#: standard English generation flow; the native-generation fallback also
#: generates in English before translating (Req 1.6).
_ENGLISH_CODE: str = Target_Language.EN.value

#: CJK markets whose copy is measured in Display_Units rather than raw
#: character counts (Req 4.8). For these markets — or for any copy that
#: contains a wide (width-2) character — char-limit checks and truncation use
#: the :class:`DisplayWidthCalculator` instead of ``len()`` / slicing.
_CJK_MARKETS: frozenset[Target_Market] = frozenset(
    {Target_Market.JP, Target_Market.KR, Target_Market.HK, Target_Market.TW}
)

#: Shared, stateless Display_Width_Calculator. Reused across all calls because
#: it is a pure function object with no per-request state (Req 4.10 — single
#: source of truth for Display_Unit conversion).
_DISPLAY_WIDTH = DisplayWidthCalculator()


def _should_use_display_width(
    brief: Creative_Brief,
    platform_spec: Platform_Spec,
    text: str = "",
) -> bool:
    """Return True when Display_Unit semantics apply for char-limit handling.

    Display-width semantics apply (Req 4.8 / 4.10) when any of the following
    hold:

    * the Platform_Spec declares its limits in Display_Units
      (``use_display_width``), OR
    * the brief targets a CJK market (JP, KR, HK, TW), OR
    * ``text`` contains at least one wide (width-2) character.

    The check is deliberately conservative: non-CJK English flows satisfy none
    of these conditions, so they keep ``len()``-based char-limit checks and
    slicing byte-for-byte unchanged.
    """
    if platform_spec.use_display_width:
        return True
    if brief.target_market in _CJK_MARKETS:
        return True
    return any(_DISPLAY_WIDTH.char_width(ch) == 2 for ch in text)

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
    "6. 使用用户指定的源语言进行输出（默认 English）。\n"
    "7. 【最重要】如果用户指定了 SEO 关键词，每条文案必须自然包含所有关键词。"
    "不得遗漏任何一个关键词。关键词匹配不区分大小写。"
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


@dataclass(frozen=True)
class _LanguagePlan:
    """Resolved generation-language plan for a single request (Req 1.1 / 1.7).

    Computed once per :meth:`CreativeGenerator.generate` call from the brief's
    target market via the :class:`LanguagePromptSelector`:

    * English-primary markets (SG, US, GB, EN_GLOBAL, …) → ``native=False``;
      the standard English generation flow runs unchanged (Req 1.7).
    * Non-English markets → ``native=True``; generation uses the
      target-language system prompt so the LLM composes as a native speaker
      (Req 1.1 / 1.2). On repeated native failure the generator falls back to
      English generation + translation (Req 1.6).

    Attributes:
        native: Whether native-language generation is active for this request.
        generation_language: Language code stamped onto every produced
            candidate's ``generation_language`` field — the market's primary
            language for native generation, otherwise ``"en"``.
        target_language: The primary :class:`Target_Language` for native
            generation / fallback translation; ``None`` for the English flow.
        system_prompt: System prompt handed to the LLM for generation — the
            native-language template for native generation, otherwise the
            standard English/Chinese system prompt.
    """

    native: bool
    generation_language: str
    target_language: Optional[Target_Language]
    system_prompt: str


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
        prompt_selector: Optional[LanguagePromptSelector] = None,
        localization_tool: Optional[LocalizationTool] = None,
        angle_splitter: Optional[AngleSplitter] = None,
    ) -> None:
        self._llm = llm_client
        self._timeout_ms = timeout_ms
        self._log = get_logger(__name__)
        # Routes non-English markets to native-language system prompts
        # (Requirements 1.1, 1.2, 1.7). Pure/stateless — safe to share.
        self._prompt_selector = prompt_selector or LanguagePromptSelector()
        # Used only for the native-generation fallback (Requirement 1.6):
        # English generation + Localization_Tool translate. Built from the
        # same LLM client when the caller does not inject one.
        self._localization_tool = localization_tool or LocalizationTool(llm_client)
        # Optional Angle_Splitter enabling angle-based round-robin generation
        # (Requirements 3.4 – 3.6, 3.8). When ``None`` the generator behaves
        # exactly as before: callers use :meth:`generate` (single-prompt). When
        # injected, callers can use :meth:`generate_with_angles`, which
        # decomposes the brief and drives per-angle round-robin generation,
        # falling back to single-prompt generation if decomposition fails.
        self._angle_splitter = angle_splitter

    @property
    def supports_angle_generation(self) -> bool:
        """Whether angle-based generation is available (an Angle_Splitter is wired).

        The orchestrator checks this to decide whether to call
        :meth:`generate_with_angles` (angle round-robin) or the existing
        :meth:`generate` (single-prompt) path. Keeps construction sites that do
        not inject an Angle_Splitter working unchanged (Requirement 3.8 default).
        """
        return self._angle_splitter is not None

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

        # Resolve the language plan for this market (Req 1.1 / 1.7). Non-English
        # markets generate natively with a target-language system prompt;
        # English-primary markets keep the standard English flow unchanged.
        plan = self._resolve_language_plan(brief)

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
            native_generation=plan.native,
            generation_language=plan.generation_language,
        )

        try:
            candidates = await asyncio.wait_for(
                self._run_generation(
                    brief=brief,
                    platform_spec=platform_spec,
                    char_limit=char_limit,
                    excludes=excludes,
                    request_count=request_count,
                    min_count=min_count,
                    request_id=request_id,
                    tool_failure_counter=tool_failure_counter,
                    plan=plan,
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
    # Public API — angle-based round-robin generation (Req 3.4 – 3.6, 3.8)
    # ------------------------------------------------------------------

    async def generate_with_angles(
        self,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        *,
        exclude_copies: Optional[list[str]] = None,
        min_count: int = 5,
        request_id: Optional[str] = None,
        tool_failure_counter: Optional[list[int]] = None,
        warnings: Optional[list[str]] = None,
        accepted_angle_counts: Optional[dict[str, int]] = None,
    ) -> GeneratorOutput:
        """Generate candidates via angle-based round-robin generation.

        Decomposes the brief into 4–8 distinct angles via the injected
        :class:`AngleSplitter`, then issues one LLM call per angle requesting
        exactly :data:`_CANDIDATES_PER_ANGLE_CALL` (3) candidates per call. All
        angles receive their first call in round-robin order before any angle
        receives a second call (Requirement 3.4). When more candidates are
        required, the loop cycles back through angles starting with the angle
        holding the fewest accepted candidates (Requirement 3.5). Every produced
        candidate carries its ``angle_label`` (Requirement 3.6).

        On decomposition failure (the Angle_Splitter raises
        :class:`ToolFailureError`, or no Angle_Splitter is wired) the call falls
        back to the existing single-prompt :meth:`generate` path and appends an
        angle-decomposition-failure warning to ``warnings`` (Requirement 3.8).

        Args:
            brief: Validated :class:`Creative_Brief`.
            platform_spec: Spec for ``brief.target_platform``.
            exclude_copies: Previously seen copies that must not be re-emitted
                (the orchestrator's refill loop, Requirement 7.6).
            min_count: Desired minimum number of unique candidates.
            request_id: Optional request id for candidate ids / structured logs.
            tool_failure_counter: Shared single-element failure counter for the
                orchestrator's global breaker.
            warnings: Optional request-level warning list. The
                angle-decomposition-failure warning (Requirement 3.8) is
                appended here when the fallback path is taken.
            accepted_angle_counts: Optional per-angle accepted-candidate counts
                accumulated by the orchestrator across earlier refill rounds.
                Seeds the lowest-count cycling so the first refill round
                continues to balance angles rather than restarting from zero
                (Requirement 3.5).

        Returns:
            A :class:`GeneratorOutput` whose candidates each carry an
            ``angle_label``.

        Raises:
            GenerationFailureError: When the underlying single-prompt fallback
                fails (decomposition failure path), or when angle generation
                cannot produce ``≥ min_count`` unique candidates after
                exhausting its call budget.
        """
        # No Angle_Splitter wired → behave exactly like single-prompt
        # generation (Requirement 3.8 default). No warning: angle generation
        # was simply never requested for this deployment.
        if self._angle_splitter is None:
            return await self.generate(
                brief,
                platform_spec,
                exclude_copies=exclude_copies,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
            )

        start = time.monotonic()

        # --- Decompose the brief into angles (Req 3.1 – 3.3) --------------
        try:
            angles = await self._angle_splitter.decompose(
                brief.selling_points,
                brief.campaign_topic,
                brief.target_audience,
            )
        except ToolFailureError as exc:
            # Decomposition failed after its own retries → single-prompt
            # fallback + request-level warning (Requirement 3.8).
            self._log.warning(
                "creative_generator.angle_decomposition_fallback",
                request_id=request_id,
                failure_reason=exc.message,
            )
            if warnings is not None:
                warnings.append(
                    "angle decomposition failed; fell back to single-prompt "
                    f"generation ({exc.message})"
                )
            return await self.generate(
                brief,
                platform_spec,
                exclude_copies=exclude_copies,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
            )

        if not angles:
            # Defensive: a well-behaved Angle_Splitter raises rather than
            # returning an empty list, but guard anyway (Requirement 3.8).
            self._log.warning(
                "creative_generator.angle_decomposition_empty",
                request_id=request_id,
            )
            if warnings is not None:
                warnings.append(
                    "angle decomposition returned no angles; fell back to "
                    "single-prompt generation"
                )
            return await self.generate(
                brief,
                platform_spec,
                exclude_copies=exclude_copies,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
            )

        # Resolve the language plan once (Req 1.1 / 1.7) — shared by every
        # angle call so native-language generation still applies per-angle.
        plan = self._resolve_language_plan(brief)
        char_limit = platform_spec.char_limit(brief.creative_type)
        timeout_s = self._timeout_ms / 1000.0

        self._log.info(
            "creative_generator.angle_generation.invoked",
            request_id=request_id,
            min_count=min_count,
            angle_count=len(angles),
            angle_labels=[a.label for a in angles],
            target_market=brief.target_market.value,
            creative_type=brief.creative_type.value,
            char_limit=char_limit,
            native_generation=plan.native,
            generation_language=plan.generation_language,
        )

        try:
            candidates = await asyncio.wait_for(
                self._run_angle_generation(
                    brief=brief,
                    platform_spec=platform_spec,
                    char_limit=char_limit,
                    angles=angles,
                    excludes=list(exclude_copies) if exclude_copies else [],
                    min_count=min_count,
                    request_id=request_id,
                    tool_failure_counter=tool_failure_counter,
                    plan=plan,
                    accepted_angle_counts=accepted_angle_counts,
                ),
                timeout=timeout_s,
            )
        except GenerationFailureError:
            raise
        except asyncio.TimeoutError as exc:
            _bump_failure_counter(tool_failure_counter)
            self._log.error(
                "creative_generator.angle_generation.timeout",
                request_id=request_id,
                timeout_ms=self._timeout_ms,
            )
            raise GenerationFailureError(
                message=(
                    "Creative_Generator (angle) exceeded "
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
            "creative_generator.angle_generation.completed",
            request_id=request_id,
            count=len(candidates),
            ms=elapsed_ms,
        )
        return GeneratorOutput(
            candidates=candidates,
            generation_time_ms=elapsed_ms,
        )

    async def _run_angle_generation(
        self,
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        char_limit: int,
        angles: list[Angle],
        excludes: list[str],
        min_count: int,
        request_id: Optional[str],
        tool_failure_counter: Optional[list[int]],
        plan: _LanguagePlan,
        accepted_angle_counts: Optional[dict[str, int]],
    ) -> list[Creative_Candidate]:
        """Drive the round-robin + lowest-count angle generation loop.

        Phase 1 (round-robin, Req 3.4): issue exactly one call per angle, in the
        order the angles were returned, each requesting 3 candidates — so every
        angle gets its first call before any angle gets a second.

        Phase 2 (cycling, Req 3.5): while more unique candidates are needed,
        issue a *wave* of lowest-count angle calls concurrently (each angle
        picked by fewest accepted candidates, ties broken by original order),
        then merge + dedup and repeat only if still short. This keeps the
        lowest-count balancing while collapsing the previously-serial refill
        round-trips into a few parallel waves.

        ``accepted_angle_counts`` seeds the per-angle accepted tally so cycling
        stays balanced across the orchestrator's refill rounds.
        """
        # Per-angle accepted-candidate tally, seeded from earlier refill rounds.
        accepted_counts: dict[str, int] = {a.label: 0 for a in angles}
        if accepted_angle_counts:
            for label, count in accepted_angle_counts.items():
                if label in accepted_counts:
                    accepted_counts[label] = count

        # Stable ordering index for deterministic tie-breaks in phase 2.
        order_index: dict[str, int] = {a.label: i for i, a in enumerate(angles)}
        angle_by_label: dict[str, Angle] = {a.label: a for a in angles}

        # Running set of accepted copies (seed with caller excludes) so each new
        # angle call avoids re-emitting an already accepted/seen copy.
        seen_copies: list[str] = list(excludes)
        accepted: list[Creative_Candidate] = []

        # Phase 1 — fire one call per angle CONCURRENTLY (Req 3.4: every angle
        # gets its first call before any second call; issuing them in parallel
        # trivially satisfies that and collapses N sequential LLM round-trips
        # into one wall-clock round-trip). Each call still requests 3 candidates
        # and is tagged with its angle. We pass the *same* base excludes to all
        # of them (they run simultaneously, so they can't see each other's
        # output); cross-angle duplicates are removed deterministically below.
        phase1_angles = angles[:_MAX_ANGLE_CALLS]
        call_budget = _MAX_ANGLE_CALLS - len(phase1_angles)

        phase1_results = await asyncio.gather(
            *[
                self._generate_one_angle_call(
                    brief=brief,
                    platform_spec=platform_spec,
                    char_limit=char_limit,
                    angle=angle,
                    excludes=seen_copies,
                    request_id=request_id,
                    tool_failure_counter=tool_failure_counter,
                    plan=plan,
                    start_index=0,  # reindexed below after merge
                )
                for angle in phase1_angles
            ]
        )

        # Merge concurrent results in angle order, deduping across angles by the
        # same normalised-text rule the generator uses within a call.
        seen_norm: set[str] = {self._normalise(c) for c in seen_copies if c}
        seen_norm.discard("")
        for angle, produced in zip(phase1_angles, phase1_results):
            for cand in produced:
                norm = self._normalise(cand.source_copy)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                cand.generation_index = len(accepted)
                accepted.append(cand)
                seen_copies.append(cand.source_copy)
                accepted_counts[angle.label] += 1

        # Phase 2 — cycle from the lowest-count angle until we hit min_count
        # (Req 3.5). Instead of one sequential call at a time (which serialised
        # 5–10 LLM round-trips for a 15-headline target), we issue a *wave* of
        # lowest-count angle calls CONCURRENTLY, then merge + dedup, and repeat
        # only if still short. This preserves the lowest-count balancing (each
        # wave picks the currently-neediest angles, breaking ties by angle
        # order) while collapsing the serial round-trips into a few parallel
        # waves. Bounded by the shared call budget so a chronically
        # under-producing LLM cannot loop forever.
        while len(accepted) < min_count and call_budget > 0:
            # How many more candidates do we need, and therefore how many
            # 3-candidate calls to fire this wave. We overshoot by _WAVE_OVERSHOOT
            # because the Orchestrator's semantic dedup discards much of each
            # batch downstream — sizing to the bare gap would mean many tiny
            # serial waves. Capped by _MAX_WAVE_CALLS and the remaining budget.
            needed = min_count - len(accepted)
            wave_calls = int(
                needed * _WAVE_OVERSHOOT // _CANDIDATES_PER_ANGLE_CALL
            ) + 1
            wave_calls = min(wave_calls, _MAX_WAVE_CALLS, call_budget)

            # Pick this wave's angles by repeatedly choosing the lowest-count
            # angle, charging a *provisional* count of one call's worth of
            # candidates per pick. Charging _CANDIDATES_PER_ANGLE_CALL (not 1)
            # makes a concurrent wave select the SAME angle sequence the old
            # serial loop would have, since each serial call added its ~3
            # produced candidates to the tally before the next pick. Provisional
            # counts are discarded after selection; real counts are updated from
            # actual accepted candidates below.
            provisional = dict(accepted_counts)
            wave_labels: list[str] = []
            for _ in range(wave_calls):
                label = self._select_lowest_count_angle(provisional, order_index)
                wave_labels.append(label)
                provisional[label] += _CANDIDATES_PER_ANGLE_CALL

            call_budget -= len(wave_labels)

            wave_results = await asyncio.gather(
                *[
                    self._generate_one_angle_call(
                        brief=brief,
                        platform_spec=platform_spec,
                        char_limit=char_limit,
                        angle=angle_by_label[label],
                        excludes=seen_copies,
                        request_id=request_id,
                        tool_failure_counter=tool_failure_counter,
                        plan=plan,
                        start_index=len(accepted),  # reindexed on merge
                    )
                    for label in wave_labels
                ]
            )

            progressed = False
            for label, produced in zip(wave_labels, wave_results):
                if not produced:
                    # This angle yielded nothing new. Bump its real tally so the
                    # next wave's selector moves on rather than re-picking a
                    # barren angle forever.
                    accepted_counts[label] += 1
                    continue
                for cand in produced:
                    norm = self._normalise(cand.source_copy)
                    if not norm or norm in seen_norm:
                        continue
                    seen_norm.add(norm)
                    cand.generation_index = len(accepted)
                    accepted.append(cand)
                    seen_copies.append(cand.source_copy)
                    accepted_counts[label] += 1
                    progressed = True

            if not progressed:
                # An entire wave produced nothing usable (all empty or all
                # duplicates). The angle tallies were bumped above so selection
                # advances; if the budget is exhausted the loop exits and the
                # insufficiency check below handles it.
                continue

        if len(accepted) < min_count:
            self._log.error(
                "creative_generator.angle_generation.insufficient",
                request_id=request_id,
                produced=len(accepted),
                min_count=min_count,
            )
            raise GenerationFailureError(
                message=(
                    "Creative_Generator (angle) produced "
                    f"{len(accepted)} unique candidate(s); need {min_count}"
                ),
                request_id=request_id,
                details={
                    "produced": len(accepted),
                    "min_count": min_count,
                    "angle_count": len(angles),
                },
            )

        return accepted

    @staticmethod
    def _select_lowest_count_angle(
        accepted_counts: dict[str, int],
        order_index: dict[str, int],
    ) -> str:
        """Return the angle label with the fewest accepted candidates (Req 3.5).

        Ties are broken by the angle's original decomposition order so the
        selection is deterministic.
        """
        return min(
            accepted_counts,
            key=lambda label: (accepted_counts[label], order_index[label]),
        )

    async def _generate_one_angle_call(
        self,
        *,
        brief: Creative_Brief,
        platform_spec: Platform_Spec,
        char_limit: int,
        angle: Angle,
        excludes: list[str],
        request_id: Optional[str],
        tool_failure_counter: Optional[list[int]],
        plan: _LanguagePlan,
        start_index: int,
    ) -> list[Creative_Candidate]:
        """Issue a single angle-focused LLM call requesting exactly 3 candidates.

        Returns the post-processed (truncated + deduped) candidates produced by
        this call, each stamped with ``angle.label`` (Requirement 3.6). A failed
        LLM call is non-fatal here — it bumps the shared failure counter, logs a
        warning, and returns an empty list so the round-robin / cycling loop can
        keep balancing the remaining angles. Insufficiency is decided by the
        caller against ``min_count``.
        """
        try:
            raw_copies = await self._call_llm(
                brief=brief,
                platform_spec=platform_spec,
                char_limit=char_limit,
                request_count=_CANDIDATES_PER_ANGLE_CALL,
                excludes=excludes,
                system_prompt=plan.system_prompt,
                output_language=plan.generation_language
                if plan.native
                else brief.source_language,
                angle=angle,
            )
        except ToolFailureError as exc:
            _bump_failure_counter(tool_failure_counter)
            self._log.warning(
                "creative_generator.angle_call_failed",
                request_id=request_id,
                angle_label=angle.label,
                error=exc.message,
            )
            return []
        except Exception as exc:  # noqa: BLE001 — defensive belt
            _bump_failure_counter(tool_failure_counter)
            self._log.warning(
                "creative_generator.angle_call_failed",
                request_id=request_id,
                angle_label=angle.label,
                error=f"{type(exc).__name__}: {exc}",
            )
            return []

        return self._post_process(
            raw_copies=raw_copies,
            brief=brief,
            char_limit=char_limit,
            excludes=excludes,
            request_id=request_id,
            generation_language=plan.generation_language,
            platform_spec=platform_spec,
            angle_label=angle.label,
            start_index=start_index,
        )

    # ------------------------------------------------------------------
    # Language plan resolution (Req 1.1 / 1.7)
    # ------------------------------------------------------------------

    def _resolve_language_plan(self, brief: Creative_Brief) -> _LanguagePlan:
        """Resolve the generation-language plan for ``brief`` (Req 1.1 / 1.7).

        English-primary markets keep the standard English generation flow with
        the default system prompt (``native=False``). Non-English markets
        switch to the target-language system prompt so the LLM composes as a
        native speaker (``native=True``). Unknown markets default to the
        English flow defensively rather than raising.
        """
        market = brief.target_market
        try:
            native_required = self._prompt_selector.is_native_generation_required(
                market
            )
        except KeyError:
            # Unknown market — fall back to the standard English flow.
            self._log.warning(
                "creative_generator.unknown_market",
                target_market=getattr(market, "value", str(market)),
            )
            native_required = False

        if not native_required:
            return _LanguagePlan(
                native=False,
                generation_language=_ENGLISH_CODE,
                target_language=None,
                system_prompt=_SYSTEM_PROMPT,
            )

        primary = self._prompt_selector.get_primary_language(market)
        system_prompt = self._prompt_selector.get_system_prompt(primary)
        try:
            target_language = Target_Language(primary)
        except ValueError:  # pragma: no cover — primary always a valid code
            target_language = None
        return _LanguagePlan(
            native=True,
            generation_language=primary,
            target_language=target_language,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Generation entry — native flow with English+translate fallback
    # ------------------------------------------------------------------

    async def _run_generation(
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
        plan: _LanguagePlan,
    ) -> list[Creative_Candidate]:
        """Run generation under ``plan``; handle the native-generation fallback.

        * Non-native (English) markets run the standard retry loop unchanged
          (Req 1.7).
        * Native markets run the retry loop with the target-language system
          prompt (Req 1.1 / 1.2). When all 3 attempts fail
          (:class:`GenerationFailureError`), fall back to English generation
          followed by Localization_Tool translation and log the fallback event
          with the target language, failure reason, and attempt count
          (Req 1.6).
        """
        if not plan.native:
            return await self._generate_with_retry(
                brief=brief,
                platform_spec=platform_spec,
                char_limit=char_limit,
                excludes=excludes,
                request_count=request_count,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
                system_prompt=plan.system_prompt,
                output_language=brief.source_language,
                generation_language=plan.generation_language,
            )

        # Native generation (Req 1.1 / 1.2).
        try:
            return await self._generate_with_retry(
                brief=brief,
                platform_spec=platform_spec,
                char_limit=char_limit,
                excludes=excludes,
                request_count=request_count,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
                system_prompt=plan.system_prompt,
                output_language=plan.generation_language,
                generation_language=plan.generation_language,
            )
        except GenerationFailureError as exc:
            # Native generation exhausted all attempts → English + translate
            # fallback (Req 1.6). Log the fallback with the required fields.
            self._log.warning(
                "creative_generator.native_generation_fallback",
                request_id=request_id,
                target_language=plan.generation_language,
                failure_reason=exc.message,
                attempt_count=_MAX_ATTEMPTS,
            )
            return await self._fallback_english_then_translate(
                brief=brief,
                platform_spec=platform_spec,
                char_limit=char_limit,
                excludes=excludes,
                request_count=request_count,
                min_count=min_count,
                request_id=request_id,
                tool_failure_counter=tool_failure_counter,
                plan=plan,
            )

    async def _fallback_english_then_translate(
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
        plan: _LanguagePlan,
    ) -> list[Creative_Candidate]:
        """English generation + Localization_Tool translate fallback (Req 1.6).

        Generates candidates in English with the standard flow, then translates
        each candidate's copy into the market's primary language via the
        existing Localization_Tool. A candidate whose translation succeeds is
        stamped with the target ``generation_language``; one whose translation
        fails keeps its English copy, records the per-language failure, and is
        flagged with a warning. If the English generation step itself fails,
        the :class:`GenerationFailureError` propagates to the orchestrator.
        """
        english_candidates = await self._generate_with_retry(
            brief=brief,
            platform_spec=platform_spec,
            char_limit=char_limit,
            excludes=excludes,
            request_count=request_count,
            min_count=min_count,
            request_id=request_id,
            tool_failure_counter=tool_failure_counter,
            system_prompt=_SYSTEM_PROMPT,
            output_language=_ENGLISH_CODE,
            generation_language=_ENGLISH_CODE,
        )

        target_language = plan.target_language
        if target_language is None:
            # No valid target language to translate into — return the English
            # candidates as-is (still better than failing the request).
            return english_candidates

        async def _translate(candidate: Creative_Candidate) -> Creative_Candidate:
            try:
                result = await self._localization_tool.translate(
                    candidate.source_copy,
                    source_language=Target_Language.EN,
                    target_languages=[target_language],
                    target_market=brief.target_market,
                )
            except Exception as exc:  # noqa: BLE001 — translation is best-effort
                _bump_failure_counter(tool_failure_counter)
                candidate.warnings = [
                    *candidate.warnings,
                    f"native_generation_fallback_translate_failed: {exc}",
                ]
                self._log.warning(
                    "creative_generator.fallback_translate_error",
                    request_id=request_id,
                    target_language=plan.generation_language,
                    error=str(exc),
                )
                return candidate

            translated = result.localized_versions.get(target_language)
            if translated:
                # Preserve the char-limit constraint on the translated copy.
                # The translated copy is in the market's primary language, which
                # for CJK markets is measured in Display_Units (Req 4.8 / 4.9),
                # so honour display-width semantics and refresh the candidate's
                # ``display_width`` to match the post-translation copy.
                use_display_width = _should_use_display_width(
                    brief, platform_spec, translated
                )
                candidate.source_copy = self._truncate(
                    translated, char_limit, use_display_width=use_display_width
                )
                candidate.generation_language = plan.generation_language
                candidate.display_width = _DISPLAY_WIDTH.text_width(
                    candidate.source_copy
                )
            else:
                # Translation failed for this language — keep English copy and
                # record the per-language failure (Req 9.3 semantics).
                candidate.failed_languages = list(result.failed_languages)
                candidate.warnings = [
                    *candidate.warnings,
                    "native_generation_fallback_translation_unavailable",
                ]
            return candidate

        return list(
            await asyncio.gather(*[_translate(c) for c in english_candidates])
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
        system_prompt: str,
        output_language: str,
        generation_language: str,
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
                    system_prompt=system_prompt,
                    output_language=output_language,
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
                generation_language=generation_language,
                platform_spec=platform_spec,
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
        system_prompt: str,
        output_language: str,
        angle: Optional[Angle] = None,
    ) -> list[str]:
        """Single ``complete_json`` call returning a list of raw copy strings.

        When ``angle`` is provided the user prompt is focused on that single
        creative angle (Requirement 3.4) so the call produces angle-specific
        copy; otherwise the standard multi-angle single-prompt is built.
        """
        prompt = self._build_prompt(
            brief=brief,
            platform_spec=platform_spec,
            char_limit=char_limit,
            request_count=request_count,
            excludes=excludes,
            output_language=output_language,
            angle=angle,
        )

        try:
            payload = await self._llm.complete_json(
                prompt,
                system=system_prompt,
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
        generation_language: str,
        platform_spec: Platform_Spec,
        angle_label: Optional[str] = None,
        start_index: int = 0,
    ) -> list[Creative_Candidate]:
        """Dedupe, build candidates, and prefer complete in-limit copies.

        Fit-first policy (方案 4): a copy the LLM wrote WITHIN ``char_limit`` is
        a complete sentence and is always kept; a copy that OVERFLOWS is only
        kept (word-boundary truncated) when this call produced no fitting copy
        at all, so the ad group is filled with complete sentences in the normal
        case and truncation is a rare last resort.

        ``generation_language`` is stamped onto every built candidate's
        ``generation_language`` field (Req 1.1) — the market's primary language
        for native generation, otherwise ``"en"``.

        ``angle_label`` is stamped onto every built candidate's ``angle_label``
        field when angle-based generation is active (Req 3.6); ``None`` for the
        single-prompt path. ``start_index`` offsets ``generation_index`` so
        candidates produced across successive angle calls keep monotonically
        increasing, unique indices within the request.

        Char-limit enforcement honours Display_Units for CJK markets (Req 4.8 /
        4.9): a market-level flag (CJK market or display-width Platform_Spec) is
        OR'd per copy with a wide-character check, so even a non-CJK market gets
        Display_Unit handling for copy that contains CJK characters, while
        pure-ASCII English flows keep ``len()`` semantics unchanged.
        """
        seen_norm: set[str] = {self._normalise(c) for c in excludes if c}
        seen_norm.discard("")  # an empty exclude must not block real copies

        # Hard must-avoid filter (requirement 6): operators can ban words this
        # run (e.g. a term an ad platform mis-flagged). The prompt asks the LLM
        # to avoid them, but LLMs don't always comply, so we ALSO drop any
        # candidate that still contains a banned phrase (case-insensitive
        # substring) — guaranteeing banned words never reach the output.
        must_avoid = [
            s.strip().lower()
            for s in (brief.must_avoid or [])
            if s and s.strip()
        ]

        # Market-level display-width decision computed once (CJK market or a
        # display-width Platform_Spec). Individual copies that contain wide
        # characters opt in even when the market itself does not (Req 4.8).
        market_display_width = _should_use_display_width(brief, platform_spec)

        def _banned(text: str) -> bool:
            if not must_avoid:
                return False
            lowered = text.lower()
            return any(b in lowered for b in must_avoid)

        # Fit-first strategy (no hard mid-sentence cuts). A copy the LLM already
        # wrote WITHIN the limit is a complete, well-formed sentence — always
        # preferred. A copy that OVERFLOWS would have to be truncated, which
        # leaves a clipped/awkward fragment; we keep those only as a fallback so
        # a call never returns empty. The orchestrator overshoots (~1.7x) and
        # refills, so dropping over-limit copies still fills the ad group with
        # complete sentences in the normal case. Truncation (word-boundary,
        # never mid-word) thus only surfaces when an entire call produced
        # nothing that fits — a rare last resort.
        accepted_texts: list[str] = []      # complete, within-limit copies
        fallback_texts: list[str] = []      # word-boundary-truncated overflow

        for raw in raw_copies:
            use_display_width = market_display_width or _should_use_display_width(
                brief, platform_spec, raw
            )
            stripped = raw.strip()
            if not stripped or _banned(stripped):
                continue

            if self._fits_within(stripped, char_limit, use_display_width):
                norm = self._normalise(stripped)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                accepted_texts.append(stripped)
                continue

            # Overflow → prepare a word-boundary-truncated fallback (used only
            # if this call yields no fitting copy at all).
            truncated = self._truncate(
                stripped, char_limit, use_display_width=use_display_width
            )
            if not truncated or _banned(truncated):
                continue
            norm = self._normalise(truncated)
            if not norm or norm in seen_norm:
                continue
            # Don't register the fallback's norm in ``seen_norm`` yet — a later
            # fitting copy that normalises the same should still win.
            fallback_texts.append(truncated)

        # Prefer complete sentences; only fall back to truncated copy when this
        # call produced none that fit (so the call isn't empty and generation
        # can still progress). Dedup the fallbacks against accepted norms.
        final_texts = list(accepted_texts)
        if not final_texts and fallback_texts:
            for text in fallback_texts:
                norm = self._normalise(text)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                final_texts.append(text)

        return [
            self._build_candidate(
                brief=brief,
                source_copy=text,
                generation_index=start_index + i,
                request_id=request_id,
                generation_language=generation_language,
                angle_label=angle_label,
            )
            for i, text in enumerate(final_texts)
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
        output_language: str,
        angle: Optional[Angle] = None,
    ) -> str:
        """Build the user-message prompt described in design § 3.

        ``output_language`` is the language the LLM must compose in — the
        market's primary language for native generation, ``brief.source_language``
        for the standard English flow, or ``"en"`` for the fallback English
        generation step (Req 1.1 / 1.2 / 1.6).

        ``angle`` focuses the call on a single creative angle (Req 3.4): when
        provided, the prompt instructs the model to lead every candidate with
        that angle's value proposition, and the requested count is the per-angle
        quota (3) rather than the whole batch.
        """
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
            f"- 输出语言：{output_language}\n\n"
            "投放设定：\n"
            f"- 目标平台：{brief.target_platform.value}\n"
            f"- 目标市场：{brief.target_market.value}\n"
            f"- 文案类型：{brief.creative_type.value}\n"
            f"- 字符上限：{char_limit}\n\n"
            f"请生成至少 {request_count} 条互不重复的广告文案候选。\n\n"
            "返回严格的 JSON 对象，结构示例：\n"
            '{"candidates": [{"copy": "..."}, {"copy": "..."}, ...]}\n\n'
            "硬性要求：\n"
            f"1. 每条 copy 的字符长度必须 ≤ {char_limit}，且必须是**语义完整的句子**。"
            f"宁可写短一些（例如 {max(char_limit - 8, 1)} 字符左右）也不要写到接近上限后被截断；"
            "绝对不要输出会被截断的半句话或残缺结尾。\n"
            f"2. 文案必须符合 {brief.target_platform.value} 平台的风格规范。\n"
            "3. 候选之间在角度 / 语气 / 卖点切入上必须有明显差异。\n"
            f"4. 使用 {output_language} 作为输出语言。\n"
            "5. 禁止出现违禁词与承诺类表述（赌博保证、医疗承诺、虚假紧迫感等）。\n"
        )
        if angle is not None:
            angle_desc = angle.description.strip() if angle.description else ""
            body += (
                f"\n【创意角度聚焦】本次只围绕单一创意角度「{angle.label}」展开，"
                f"每一条文案都必须以该角度为核心卖点切入。\n"
            )
            if angle_desc:
                body += f"角度说明：{angle_desc}\n"
            body += (
                f"请生成恰好 {request_count} 条都聚焦「{angle.label}」角度的差异化文案"
                "（彼此之间在措辞、结构、语气上仍需不同）。\n"
            )
        if forbidden_symbols:
            body += (
                f"6. 不得出现以下符号或符号组合：{forbidden_symbols}\n"
            )

        if keywords:
            if brief.creative_type == Creative_Type.CTA:
                body += (
                    "\nSEO 关键词参考（CTA 类型不强制包含，专注号召力即可）：\n- "
                    + "\n- ".join(keywords)
                    + "\n"
                )
            else:
                body += (
                    "\n【强制要求】以下 SEO 关键词必须出现在每条文案中（大小写不敏感，必须是完全一致的拼写，不要加空格、连字符或变复数）：\n- "
                    + "\n- ".join(keywords)
                    + "\n"
                    + "例如：关键词是 'topup' 就写 'topup'，不要写成 'top up' 或 'top-up' 或 'topups'。\n"
                )

        if excludes:
            preview = excludes[:30]
            body += (
                "\n请务必避免与以下已生成文案重复或近义改写：\n- "
                + "\n- ".join(preview)
                + "\n"
            )

        # --- Operator custom constraints (requirement 6) -------------------
        must_include = [s for s in (brief.must_include or []) if s and s.strip()]
        if must_include:
            body += (
                "\n【必须包含】以下短语必须自然出现在每条文案中：\n- "
                + "\n- ".join(must_include)
                + "\n"
            )

        must_avoid = [s for s in (brief.must_avoid or []) if s and s.strip()]
        if must_avoid:
            body += (
                "\n【必须规避】以下词语/表述绝对不能出现在任何文案中"
                "（广告审核可能误判，请彻底避开，包括其明显变体）：\n- "
                + "\n- ".join(must_avoid)
                + "\n"
            )

        extra = (brief.extra_instructions or "").strip()
        if extra:
            body += (
                "\n【额外创作要求】请同时遵循以下运营备注（自由指令）：\n"
                + extra
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
    def _fits_within(
        text: str, char_limit: int, use_display_width: bool = False
    ) -> bool:
        """Return True iff ``text`` is already within ``char_limit``.

        Uses Display_Unit width for CJK / wide-character copy (Req 4.8) and a
        plain ``len()`` count otherwise — mirroring :meth:`_truncate` so the
        "fits" check and the truncation use the same yardstick.
        """
        stripped = text.strip()
        if char_limit <= 0:
            return False
        if use_display_width:
            return _DISPLAY_WIDTH.fits_within_limit(stripped, char_limit)
        return len(stripped) <= char_limit

    @staticmethod
    def _truncate(
        text: str, char_limit: int, *, use_display_width: bool = False
    ) -> str:
        """Hard-truncate ``text`` to ``char_limit`` without splitting words.

        When ``use_display_width`` is True (CJK markets or copy containing wide
        characters, Req 4.8 / 4.9) the limit is a Display_Unit count enforced
        via the :class:`DisplayWidthCalculator`, which removes whole characters
        from the end until the total width fits. CJK / no-space scripts have no
        inter-word spaces, so character-level truncation never produces a
        partial "word" — this path is unchanged.

        When ``use_display_width`` is False (space-separated scripts such as
        English / Spanish / Arabic) the limit is a raw character count. A naive
        ``text[:limit]`` slice would cut mid-word (e.g. "20% more value, s" or
        "Your Topup Bo"), which no real ad would ship. We therefore retreat to
        the last whole-word boundary that fits and strip any dangling
        punctuation, so the result is always composed of complete words — at
        the cost of landing a few characters under the limit when the final
        word doesn't fit. If the limit is so small that not even the first word
        fits, we fall back to a hard slice (better a clipped single word than
        an empty string).
        """
        if char_limit <= 0:
            return ""
        stripped = text.strip()
        if use_display_width:
            if _DISPLAY_WIDTH.fits_within_limit(stripped, char_limit):
                return stripped
            return _DISPLAY_WIDTH.truncate_to_width(stripped, char_limit).rstrip()
        if len(stripped) <= char_limit:
            return stripped
        return CreativeGenerator._truncate_on_word_boundary(stripped, char_limit)

    @staticmethod
    def _truncate_on_word_boundary(text: str, char_limit: int) -> str:
        """Truncate ``text`` to ``char_limit`` chars without splitting a word.

        Assumes a space-separated script. Retreats to the last full word that
        fits and trims trailing separator punctuation (commas, dashes, etc.).
        Falls back to a hard slice when the first word alone exceeds the limit.
        """
        window = text[:char_limit]
        # If the character right after the cut is part of the same word (both
        # sides are word characters), the slice landed mid-word — drop the
        # partial trailing word by cutting at the last whitespace in ``window``.
        cut_is_mid_word = (
            len(text) > char_limit
            and not text[char_limit - 1].isspace()
            and not text[char_limit].isspace()
        )
        if cut_is_mid_word:
            last_space = window.rfind(" ")
            if last_space > 0:
                window = window[:last_space]
            # else: the whole window is one long word → keep the hard slice.
        # Trim trailing whitespace and dangling separator punctuation so the
        # copy doesn't end on a stray "," / "-" / "—" / etc.
        return window.rstrip().rstrip(",;:-—–·/&+ ").rstrip()

    @staticmethod
    def _build_candidate(
        *,
        brief: Creative_Brief,
        source_copy: str,
        generation_index: int,
        request_id: Optional[str],
        generation_language: str,
        angle_label: Optional[str] = None,
    ) -> Creative_Candidate:
        """Construct a Creative_Candidate with placeholder downstream fields.

        ``generation_language`` records the language the copy was generated in
        (Req 1.1) — the market's primary language for native generation,
        otherwise ``"en"``.

        ``angle_label`` records the creative angle this candidate was generated
        for (Req 3.6); ``None`` for the single-prompt path.

        The candidate's ``display_width`` is populated from the final source
        copy via the shared :class:`DisplayWidthCalculator` (Req 4.8), so every
        candidate carries an accurate Display_Unit count regardless of market.
        """
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
            generation_language=generation_language,
            angle_label=angle_label,
            display_width=_DISPLAY_WIDTH.text_width(source_copy),
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
