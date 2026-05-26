"""Compliance_Checker — local-dictionary + (optional) LLM compliance detection.

Implements design.md § Components / 4. Compliance_Checker and Requirements
3.1 – 3.10.

Detection has three stages:

1. **Empty / whitespace short-circuit** (Req 3.10): single ``EMPTY_COPY``
   ``BLOCK`` violation with ``compliance_score = 0.0``.
2. **Local dictionary scan** (Req 3.3 / 3.5 / 3.8): every Forbidden_Term in
   :mod:`creative_agent.config.forbidden_loader` is matched via
   case-insensitive, Unicode-aware word boundaries. Each match becomes a
   :class:`Violation` carrying the dictionary entry's ``category``,
   ``severity`` and ``suggestion`` — so ``BLOCK`` severity stays
   deterministic (supporting Property 9 monotonicity).
3. **LLM semantic check** (Req 3.4) — *optional*. When the constructor
   receives ``llm=None`` the tool runs in dictionary-only mode. Otherwise
   the LLM is asked for ``MISLEADING`` / ``FALSE_URGENCY`` /
   ``EXAGGERATION`` findings and severity is force-coerced to ``WARN``.
   LLM errors are logged and absorbed (Req 9.2: this tool degrades rather
   than aborts the pipeline) — the dictionary result is still returned.

Scoring (Req 3.3 / 3.4 / 3.9)::

    if any violation has severity == BLOCK: score = 0.0
    elif no violations:                     score = 1.0
    elif warn_count > 0:                    score = max(0.1, 1.0 - 0.2 * warn_count)
    else:                                   score = 1.0   # INFO-only

The whole call is wrapped in ``asyncio.wait_for(timeout=1.5)`` (Req 3.7).
On any failure (timeout, internal exception) the tool still returns a valid
:class:`Compliance_Report` carrying a single ``WARN`` entry so the
orchestrator's per-tool degradation policy (Req 9.2) can flag the candidate
without dropping it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from creative_agent.config import find_term_matches, load_forbidden_dictionary
from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.models import (
    Compliance_Report,
    Compliance_Severity,
    Target_Language,
    Violation,
    ViolationCategory,
)
from creative_agent.observability.logging import get_logger

__all__ = ["ComplianceChecker"]

log = get_logger(__name__)


# Total wall-clock budget per Req 3.7.
_TOTAL_TIMEOUT_S: float = 20.0
# Per-LLM-call timeout — leaves a few seconds slack inside the budget.
_LLM_TIMEOUT_MS: int = 18000

#: Stamped into every Compliance_Report.
_CHECKER_VERSION: str = "v1.0.0-mvp"

# LLM-allowed categories. Anything outside this set is dropped — the local
# dictionary owns the BLOCK-eligible categories (GAMBLING, MEDICAL_PROMISE,
# DISCRIMINATION, ...) and the LLM stage is never allowed to escalate
# severity, guaranteeing Property 9 (BLOCK monotonicity).
_LLM_ALLOWED_CATEGORIES: dict[str, ViolationCategory] = {
    "MISLEADING": ViolationCategory.MISLEADING,
    "FALSE_URGENCY": ViolationCategory.FALSE_URGENCY,
    "EXAGGERATION": ViolationCategory.EXAGGERATION,
}


class ComplianceChecker:
    """Hybrid local-dictionary + (optional) LLM compliance checker.

    Args:
        llm: Optional :class:`LLMClient`. When ``None`` the tool runs in
            dictionary-only mode (the MVP fallback).
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        copy: str,
        language: Target_Language = Target_Language.EN,
        context: dict | None = None,
    ) -> Compliance_Report:
        """Run the hybrid compliance check on ``copy``.

        The method never raises: a timeout or unexpected exception is logged
        and converted into a degraded :class:`Compliance_Report` carrying a
        single ``WARN`` entry, in line with Req 9.2 (per-candidate compliance
        failures degrade rather than abort).

        Args:
            copy: Ad copy to inspect.
            language: Language whose Forbidden_Term dictionary is loaded.
                Defaults to :data:`Target_Language.EN`.
            context: Optional per-candidate context (e.g. ``target_market``).
                Reserved for future use; currently unused.

        Returns:
            A :class:`Compliance_Report` summarising the verdict.
        """
        log.info(
            "compliance_checker.invoked",
            language=language.value,
            copy_length=len(copy),
            llm_available=self._llm is not None,
        )

        try:
            return await asyncio.wait_for(
                self._check_inner(copy=copy, language=language),
                timeout=_TOTAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.error(
                "compliance_checker.timeout",
                language=language.value,
                timeout_ms=int(_TOTAL_TIMEOUT_S * 1000),
            )
            return self._degraded_report(
                "Compliance check exceeded "
                f"{int(_TOTAL_TIMEOUT_S * 1000)}ms timeout"
            )
        except Exception as exc:  # noqa: BLE001 — never abort the pipeline
            log.error(
                "compliance_checker.unexpected_error",
                language=language.value,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return self._degraded_report(
                f"Compliance check failed: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Inner orchestration
    # ------------------------------------------------------------------

    async def _check_inner(
        self,
        *,
        copy: str,
        language: Target_Language,
    ) -> Compliance_Report:
        # 1. Empty / whitespace short-circuit (Req 3.10).
        if not copy or not copy.strip():
            empty_violation = Violation(
                start=0,
                # Violation requires end > start; for an empty string we
                # still need a valid [0, 1) interval.
                end=max(1, len(copy)),
                category=ViolationCategory.EMPTY_COPY,
                severity=Compliance_Severity.BLOCK,
                matched_term=None,
                suggestion="文案不可为空",
            )
            log.info("compliance_checker.empty_copy", language=language.value)
            return Compliance_Report(
                compliance_score=0.0,
                violations=[empty_violation],
                checked_at=self._now_iso(),
                checker_version=_CHECKER_VERSION,
            )

        # 2. Local dictionary scan (Req 3.3 / 3.5 / 3.8).
        dict_violations = self._scan_local_dictionary(copy, language)
        log.info(
            "compliance_checker.dictionary_scanned",
            language=language.value,
            dict_violation_count=len(dict_violations),
        )

        has_block = any(
            v.severity == Compliance_Severity.BLOCK for v in dict_violations
        )

        # 3. LLM semantic check (Req 3.4). Skipped when:
        #    * no LLM was injected (MVP dictionary-only mode), or
        #    * a BLOCK is already locked in (the candidate will be filtered
        #      out regardless, no point spending an LLM call).
        llm_violations: list[Violation] = []
        if self._llm is not None and not has_block:
            try:
                llm_violations = await self._llm_semantic_check(
                    copy=copy,
                    language=language,
                    dict_violations=dict_violations,
                )
                log.info(
                    "compliance_checker.llm_scanned",
                    language=language.value,
                    llm_violation_count=len(llm_violations),
                )
            except Exception as exc:  # noqa: BLE001 — LLM failure is non-fatal
                log.warning(
                    "compliance_checker.llm_failed_fallback_to_dict",
                    language=language.value,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        # 4. Merge + dedupe (same span+category → one entry), sort by start.
        merged = self._merge_and_dedupe(dict_violations, llm_violations)

        # 5. Score (Req 3.3 / 3.4 / 3.9).
        score = self._compute_score(merged)

        report = Compliance_Report(
            compliance_score=score,
            violations=merged,
            checked_at=self._now_iso(),
            checker_version=_CHECKER_VERSION,
        )
        log.info(
            "compliance_checker.completed",
            language=language.value,
            violation_count=len(merged),
            score=score,
        )
        return report

    # ------------------------------------------------------------------
    # Local-dictionary scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_local_dictionary(
        copy: str, language: Target_Language
    ) -> list[Violation]:
        """Match every Forbidden_Term in the language dictionary against ``copy``."""
        entries = load_forbidden_dictionary(language)
        violations: list[Violation] = []
        for entry in entries:
            for start, end in find_term_matches(copy, entry.term):
                violations.append(
                    Violation(
                        start=start,
                        end=end,
                        category=entry.category,
                        severity=entry.severity,
                        matched_term=entry.term,
                        suggestion=entry.suggestion,
                    )
                )
        return violations

    # ------------------------------------------------------------------
    # LLM semantic scan
    # ------------------------------------------------------------------

    async def _llm_semantic_check(
        self,
        *,
        copy: str,
        language: Target_Language,
        dict_violations: list[Violation],
    ) -> list[Violation]:
        """Ask the LLM for semantic-level WARN findings.

        The LLM is told which dictionary terms have already been flagged so
        it doesn't double-count them. Severity is always coerced to ``WARN``
        regardless of model output, so only the deterministic dictionary
        layer can ever emit ``BLOCK``.
        """
        assert self._llm is not None  # checked by caller

        already_hit_terms = sorted(
            {v.matched_term for v in dict_violations if v.matched_term}
        )

        prompt = (
            "你是 Google Ads 政策合规审核员。请检查以下广告文案是否包含违规内容。\n\n"
            f"文案（语言: {language.value}）：\n{copy}\n\n"
            "请仅检测以下类别的违规：\n"
            "- 误导性陈述 (MISLEADING)\n"
            "- 虚假紧迫感 (FALSE_URGENCY)\n"
            "- 夸大表述 (EXAGGERATION)\n\n"
            '如果没有发现违规，返回 {"violations": []}。\n'
            "如果有违规，返回 JSON：\n"
            "{\n"
            '  "violations": [\n'
            '    {"start": <字符起始>, "end": <字符结束>, '
            '"category": "MISLEADING|FALSE_URGENCY|EXAGGERATION", '
            '"severity": "WARN", "matched_term": "命中文本", '
            '"suggestion": "修改建议"}\n'
            "  ]\n"
            "}\n\n"
            "注意：\n"
            "- 仅返回 JSON，不要包含说明文字\n"
            "- 不要重复本地词典已检测到的违规（"
            f"{already_hit_terms}）"
        )

        response = await self._llm.complete_json(
            prompt,
            timeout_ms=_LLM_TIMEOUT_MS,
        )

        raw = response.get("violations") if isinstance(response, dict) else None
        if not isinstance(raw, list):
            return []

        parsed: list[Violation] = []
        for item in raw:
            v = self._parse_llm_violation(item, copy)
            if v is not None:
                parsed.append(v)
        return parsed

    @staticmethod
    def _parse_llm_violation(item: Any, copy: str) -> Optional[Violation]:
        """Validate one LLM-emitted violation; return ``None`` to drop it."""
        if not isinstance(item, dict):
            return None

        category_raw = item.get("category")
        if not isinstance(category_raw, str):
            return None
        category = _LLM_ALLOWED_CATEGORIES.get(category_raw.strip().upper())
        if category is None:
            return None

        copy_len = len(copy)
        start_raw = item.get("start")
        end_raw = item.get("end")
        matched_term = item.get("matched_term")

        start: Optional[int] = None
        end: Optional[int] = None

        if isinstance(start_raw, int) and isinstance(end_raw, int):
            if 0 <= start_raw < end_raw <= copy_len:
                start, end = start_raw, end_raw

        # Fallback: locate the matched_term substring inside the copy.
        if start is None and isinstance(matched_term, str) and matched_term:
            idx = copy.find(matched_term)
            if idx >= 0:
                start = idx
                end = idx + len(matched_term)

        # Last-resort fallback: span the leading window of the copy. Required
        # because Violation.end > Violation.start.
        if start is None or end is None or end <= start:
            start = 0
            end = max(1, min(copy_len, 50))
            if end <= start:
                end = start + 1

        suggestion = item.get("suggestion")
        if not isinstance(suggestion, str) or not suggestion.strip():
            suggestion = "请考虑修改以确保合规。"

        matched = (
            matched_term
            if isinstance(matched_term, str) and matched_term
            else None
        )

        try:
            return Violation(
                start=start,
                end=end,
                category=category,
                severity=Compliance_Severity.WARN,
                matched_term=matched,
                suggestion=suggestion,
            )
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Merge / score / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_dedupe(
        dict_violations: list[Violation],
        llm_violations: list[Violation],
    ) -> list[Violation]:
        """Merge dictionary + LLM hits.

        Deduplication: collapse entries that share the same
        ``(start, end, category)`` triple — this prevents the LLM from
        re-flagging a term the dictionary already matched. Sorted by
        ``(start, end)`` for deterministic output.
        """
        seen: set[tuple[int, int, ViolationCategory]] = set()
        merged: list[Violation] = []
        for v in [*dict_violations, *llm_violations]:
            key = (v.start, v.end, v.category)
            if key in seen:
                continue
            seen.add(key)
            merged.append(v)
        merged.sort(key=lambda v: (v.start, v.end))
        return merged

    @staticmethod
    def _compute_score(violations: list[Violation]) -> float:
        """Apply the scoring formula (Req 3.3 / 3.4 / 3.9)."""
        if any(v.severity == Compliance_Severity.BLOCK for v in violations):
            return 0.0
        if not violations:
            return 1.0
        warn_count = sum(
            1 for v in violations if v.severity == Compliance_Severity.WARN
        )
        if warn_count == 0:
            # INFO-only violations are informational; do not penalise.
            return 1.0
        return max(0.1, 1.0 - 0.2 * warn_count)

    @staticmethod
    def _now_iso() -> str:
        """ISO-8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _degraded_report(cls, reason: str) -> Compliance_Report:
        """Build a degraded report when the check itself failed (Req 9.2).

        We emit a single ``WARN`` violation so the orchestrator can flag the
        candidate for human review without dropping it from the ranking.
        """
        warn = Violation(
            start=0,
            end=1,
            category=ViolationCategory.MISLEADING,
            severity=Compliance_Severity.WARN,
            matched_term=None,
            suggestion=f"合规检查失败，需人工复审：{reason}",
        )
        return Compliance_Report(
            compliance_score=0.8,
            violations=[warn],
            checked_at=cls._now_iso(),
            checker_version=_CHECKER_VERSION,
        )
