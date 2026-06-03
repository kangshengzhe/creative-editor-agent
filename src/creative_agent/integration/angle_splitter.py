"""Angle_Splitter — decomposes selling points into distinct creative angles.

Implements design.md § Components and Interfaces / 3. Angle_Splitter and
Requirements 3.1, 3.2, 3.3, 3.8.

Behaviour summary
-----------------

* Uses the shared :class:`~creative_agent.llm.client.LLMClient` (the same
  client the Creative_Generator and Localization_Tool use) with a structured
  JSON decomposition prompt to turn free-form selling points (plus the
  campaign topic and target audience) into 4–8 distinct creative *angles*
  (Requirement 3.1).
* The LLM is steered towards a predefined angle taxonomy (convenience, price,
  speed, safety, quality, trust, exclusivity, social proof, custom). The
  ``custom`` slot lets the model coin a label when none of the canonical
  categories fit.
* When selling points are insufficient (fewer than ``min_angles``) or
  ``None`` / empty, the model is instructed to *infer* extra angles from the
  campaign topic and target audience until the minimum is reached
  (Requirements 3.2, 3.3). Inferred angles carry ``source="inferred"``.
* Output is clamped to ``[min_angles, max_angles]`` — surplus angles beyond
  ``max_angles`` are dropped; producing fewer than ``min_angles`` (or invalid
  JSON) triggers a retry.
* Decomposition is attempted up to 3 times (1 initial + 2 retries). When all
  attempts fail to yield at least ``min_angles`` angles, a
  :class:`~creative_agent.errors.codes.ToolFailureError` is raised so the
  caller (Creative_Generator) can fall back to single-prompt generation and
  surface the decomposition-failure warning (Requirement 3.8).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient
from creative_agent.observability.logging import get_logger

__all__ = ["Angle", "AngleSplitter"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Initial attempt + up to 2 retries (Requirement 3.8).
_MAX_ATTEMPTS: int = 3

#: Backoff between attempts, in seconds. Index 0 is unused (no sleep before
#: the first attempt); position N is the sleep before attempt N+1.
_RETRY_BACKOFF_S: tuple[float, ...] = (0.0, 0.100, 0.200)

#: Tokens budget for the JSON decomposition response.
_LLM_MAX_TOKENS: int = 1024

#: Sampling temperature — moderate so angles stay distinct without drifting
#: off the provided selling points / topic.
_LLM_TEMPERATURE: float = 0.5

#: Valid values for :attr:`Angle.source`.
_VALID_SOURCES: frozenset[str] = frozenset({"selling_point", "inferred"})

#: Predefined angle taxonomy surfaced to the LLM (Requirement 3.1). ``custom``
#: is a catch-all that lets the model coin a label outside the canonical set.
_ANGLE_TAXONOMY: tuple[str, ...] = (
    "convenience",
    "price",
    "speed",
    "safety",
    "quality",
    "trust",
    "exclusivity",
    "social proof",
    "custom",
)

_SYSTEM_PROMPT: str = (
    "You are a senior advertising strategist. Your job is to decompose a "
    "campaign brief into a set of distinct creative *angles* — each angle is "
    "a different value-proposition dimension the ad copy can lead with.\n\n"
    "You MUST respond with a single strict JSON object and nothing else "
    "(no Markdown fences, no commentary). The schema is:\n"
    '{"angles": [{"label": "<angle label>", "description": "<one short '
    'sentence>", "source": "selling_point" | "inferred"}, ...]}\n\n'
    "Rules:\n"
    "1. Return between 4 and 8 angles. Each angle MUST be distinct from the "
    "others (a different value-proposition category).\n"
    "2. Prefer labels from this taxonomy: convenience, price, speed, safety, "
    "quality, trust, exclusivity, social proof. Use the label \"custom\" "
    "(or a short custom phrase) only when none of the taxonomy categories "
    "fit.\n"
    "3. Angles derived directly from a provided selling point MUST set "
    '"source" to "selling_point". Angles you infer from the campaign topic '
    'or target audience MUST set "source" to "inferred".\n'
    "4. When fewer than 4 angles can be drawn from the selling points (or no "
    "selling points are provided), infer additional angles from the campaign "
    "topic and target audience until you reach at least 4."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Angle:
    """A single creative angle derived from a brief.

    Attributes:
        label: Short angle label, ideally from the predefined taxonomy
            (e.g. ``"convenience"``, ``"price"``, ``"speed"``) or a custom
            phrase when none fit.
        description: One-sentence description of the angle.
        source: Provenance of the angle — ``"selling_point"`` when derived
            directly from a provided selling point, or ``"inferred"`` when
            inferred from the campaign topic / target audience.
    """

    label: str
    description: str
    source: str


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class AngleSplitter:
    """Decomposes selling points into 4–8 distinct creative angles.

    Args:
        llm_client: Concrete :class:`LLMClient`; lifecycle owned by the
            caller (typically the orchestrator). The same client used by the
            Creative_Generator is reused here.
        min_angles: Minimum number of angles to return. Default 4
            (Requirement 3.1).
        max_angles: Maximum number of angles to return. Default 8
            (Requirement 3.1).
        timeout_ms: Per-LLM-call timeout in milliseconds passed through to the
            client.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        min_angles: int = 4,
        max_angles: int = 8,
        timeout_ms: int = 30000,
    ) -> None:
        if min_angles < 1:
            raise ValueError("min_angles must be >= 1")
        if max_angles < min_angles:
            raise ValueError("max_angles must be >= min_angles")
        self._llm = llm_client
        self._min_angles = min_angles
        self._max_angles = max_angles
        self._timeout_ms = timeout_ms
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose(
        self,
        selling_points: Optional[list[str]],
        campaign_topic: str,
        target_audience: Optional[str],
    ) -> list[Angle]:
        """Decompose the inputs into ``[min_angles, max_angles]`` angles.

        Args:
            selling_points: Free-form selling points; may be ``None`` or empty,
                in which case all angles are inferred from ``campaign_topic``
                and ``target_audience`` (Requirement 3.3).
            campaign_topic: The campaign theme / topic.
            target_audience: Free-form audience description; may be ``None``.

        Returns:
            A list of :class:`Angle` of length between ``min_angles`` and
            ``max_angles`` inclusive.

        Raises:
            ToolFailureError: When all 3 attempts (initial + 2 retries) fail
                to produce at least ``min_angles`` angles or valid JSON. The
                caller is expected to catch this and fall back to single-prompt
                generation (Requirement 3.8).
        """
        points = [p for p in (selling_points or []) if isinstance(p, str) and p.strip()]
        topic = (campaign_topic or "").strip() or "未指定"
        audience = (target_audience or "").strip() or "未指定"

        prompt = self._build_prompt(
            selling_points=points,
            campaign_topic=topic,
            target_audience=audience,
        )

        self._log.info(
            "angle_splitter.invoked",
            selling_point_count=len(points),
            min_angles=self._min_angles,
            max_angles=self._max_angles,
        )

        last_error: Optional[str] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # Backoff before retries (no sleep before the first attempt).
            if attempt > 1:
                backoff = _RETRY_BACKOFF_S[attempt - 1]
                if backoff > 0:
                    await asyncio.sleep(backoff)

            try:
                payload = await self._llm.complete_json(
                    prompt,
                    system=_SYSTEM_PROMPT,
                    max_tokens=_LLM_MAX_TOKENS,
                    temperature=_LLM_TEMPERATURE,
                    timeout_ms=self._timeout_ms,
                )
            except ToolFailureError as exc:
                # Includes invalid-JSON responses (Requirement 3.8).
                last_error = exc.message
                self._log.warning(
                    "angle_splitter.retry",
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=exc.message,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — defensive belt
                last_error = f"{type(exc).__name__}: {exc}"
                self._log.warning(
                    "angle_splitter.retry",
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    error=last_error,
                )
                continue

            angles = self._parse_angles(payload)

            if len(angles) >= self._min_angles:
                clamped = angles[: self._max_angles]
                self._log.info(
                    "angle_splitter.completed",
                    attempt=attempt,
                    count=len(clamped),
                )
                return clamped

            # Soft failure: too few angles — retry.
            last_error = (
                f"only {len(angles)} angle(s) returned, need {self._min_angles}"
            )
            self._log.warning(
                "angle_splitter.retry",
                attempt=attempt,
                max_attempts=_MAX_ATTEMPTS,
                error=last_error,
            )

        self._log.error(
            "angle_splitter.failed",
            attempts=_MAX_ATTEMPTS,
            last_error=last_error,
        )
        raise ToolFailureError(
            tool_name="Angle_Splitter",
            message=(
                f"Angle decomposition failed after {_MAX_ATTEMPTS} attempt(s); "
                f"last error: {last_error or 'unknown'}"
            ),
            details={
                "attempts": _MAX_ATTEMPTS,
                "last_error": last_error,
                "min_angles": self._min_angles,
            },
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        *,
        selling_points: list[str],
        campaign_topic: str,
        target_audience: str,
    ) -> str:
        """Build the user-message prompt for the decomposition call."""
        if selling_points:
            points_block = "\n".join(f"- {p}" for p in selling_points)
        else:
            points_block = (
                "(none provided — infer all angles from the campaign topic "
                "and target audience)"
            )

        taxonomy = ", ".join(_ANGLE_TAXONOMY)

        return (
            "Decompose the following campaign brief into distinct creative "
            "angles.\n\n"
            f"Campaign topic: {campaign_topic}\n"
            f"Target audience: {target_audience}\n"
            "Selling points:\n"
            f"{points_block}\n\n"
            f"Angle taxonomy to prefer: {taxonomy}.\n\n"
            f"Return between {self._min_angles} and {self._max_angles} "
            "distinct angles as strict JSON of the form "
            '{"angles": [{"label": "...", "description": "...", '
            '"source": "selling_point" | "inferred"}, ...]}.'
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_angles(payload: Any) -> list[Angle]:
        """Extract :class:`Angle` objects from the LLM JSON response.

        Strict path: ``{"angles": [{"label", "description", "source"}, ...]}``.
        A bare list of angle objects is also tolerated. Entries missing a
        usable ``label`` are dropped; an unknown / missing ``source`` defaults
        to ``"inferred"``; a missing ``description`` defaults to an empty
        string.
        """
        if isinstance(payload, list):
            raw_list: list[Any] = payload
        elif isinstance(payload, dict):
            cand = payload.get("angles")
            if isinstance(cand, list):
                raw_list = cand
            else:
                # No usable angle list — treated as a soft failure upstream.
                return []
        else:
            return []

        angles: list[Angle] = []
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                continue

            description = entry.get("description")
            description = description.strip() if isinstance(description, str) else ""

            source = entry.get("source")
            if not isinstance(source, str) or source.strip() not in _VALID_SOURCES:
                source = "inferred"
            else:
                source = source.strip()

            angles.append(
                Angle(
                    label=label.strip(),
                    description=description,
                    source=source,
                )
            )

        return angles
