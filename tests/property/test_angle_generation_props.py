"""Property-based tests for angle-based round-robin generation.

Feature: creative-localization-diversity.

Exercises the universal correctness properties from design.md for the
angle-based generation loop in
:class:`creative_agent.tools.creative_generator.CreativeGenerator`:

* **Property 7 — Round-robin angle generation** (Req 3.4): for N angles
  (4 <= N <= 8), the first N generation calls each target a *distinct* angle,
  requesting exactly 3 candidates per call, before any angle receives a second
  call.
* **Property 8 — Angle cycling prioritizes lowest count** (Req 3.5): once the
  round-robin phase completes, each subsequent (refill) call targets the angle
  with the fewest accepted candidates at that point, with a deterministic
  tie-break by the angle's original decomposition order.

Both properties are verified by reconstructing per-call angle targeting from
the recorded LLM calls. Two deterministic test doubles drive the generator:

* :class:`_StubAngleSplitter` — returns exactly N distinct angles in a fixed
  order, so the angle set under test is fully controlled.
* :class:`_RecordingAngleLLM` — a concrete :class:`LLMClient` whose
  ``complete_json`` returns 3 freshly-unique copies per call (so every produced
  candidate survives the generator's running dedup and is accepted) and records
  the targeted angle label parsed out of each prompt, along with the prompt
  itself and the copies it returned.

An English-primary market (``Target_Market.EN_GLOBAL``) is used so generation
stays on the standard English flow (no native-language translation step),
keeping the LLM double minimal and the angle accounting clean.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from creative_agent.integration.angle_splitter import Angle
from creative_agent.llm.client import LLMClient
from creative_agent.models.brief import Creative_Brief
from creative_agent.models.enums import (
    Creative_Type,
    Target_Market,
    Target_Platform,
)
from creative_agent.models.platform_spec import Platform_Spec
from creative_agent.tools.creative_generator import CreativeGenerator

# English-primary market → standard English generation flow (no native-language
# switching / translation), so the recording LLM double is the only collaborator
# that needs to respond.
_MARKET = Target_Market.EN_GLOBAL

# Exactly how many candidates each angle-focused LLM call requests (Req 3.4).
# Mirrors ``creative_generator._CANDIDATES_PER_ANGLE_CALL`` — kept as a local
# constant so the test asserts the contract rather than importing the private.
_CANDIDATES_PER_ANGLE_CALL = 3

# Extracts the focused angle label from an angle-call prompt. The generator
# wraps the angle label in corner brackets: 「<label>」.
_ANGLE_MARKER = re.compile(r"「([^」]+)」")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubAngleSplitter:
    """Angle_Splitter stub returning exactly ``n`` distinct angles in order.

    The CreativeGenerator only ever calls ``.decompose(...)`` on the injected
    splitter, so this minimal async stub fully controls the angle set under
    test. The returned order is the decomposition order the generator uses for
    round-robin (Req 3.4) and the deterministic tie-break (Req 3.5).
    """

    def __init__(self, labels: list[str]) -> None:
        self._angles = [
            Angle(
                label=label,
                description=f"Lead with {label}.",
                source="selling_point" if i % 2 == 0 else "inferred",
            )
            for i, label in enumerate(labels)
        ]

    async def decompose(
        self,
        selling_points: Optional[list[str]],
        campaign_topic: str,
        target_audience: Optional[str],
    ) -> list[Angle]:
        return list(self._angles)


class _RecordingAngleLLM(LLMClient):
    """Deterministic LLM double for angle-focused generation.

    Each :meth:`complete_json` call returns exactly 3 freshly-unique copies so
    every produced candidate survives the generator's running dedup and is
    accepted (the running ``seen_copies`` set never blocks a new copy). The
    call index is embedded in every copy so copies stay unique across calls.

    Every call is recorded with the targeted angle label (parsed from the
    prompt), the full prompt, and the copies returned, so the test can
    reconstruct per-call angle targeting and the accepted-count timeline.
    """

    def __init__(self, known_labels: list[str]) -> None:
        self._known = set(known_labels)
        self._call_index = 0
        #: One entry per ``complete_json`` call, in call order. Each entry has
        #: keys ``label`` (targeted angle), ``prompt`` (full user prompt), and
        #: ``copies`` (the list of copy strings returned).
        self.angle_calls: list[dict] = []

    def _detect_label(self, prompt: str) -> str:
        match = _ANGLE_MARKER.search(prompt)
        assert match is not None, (
            "angle-focused prompt did not contain a 「<label>」 marker; "
            f"prompt prefix={prompt[:120]!r}"
        )
        label = match.group(1)
        assert label in self._known, (
            f"prompt targeted unknown angle label {label!r}; "
            f"known labels={sorted(self._known)!r}"
        )
        return label

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> str:  # pragma: no cover - never exercised on the English flow
        raise NotImplementedError(
            "_RecordingAngleLLM.complete should not be called by angle generation"
        )

    async def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> dict:
        idx = self._call_index
        self._call_index += 1
        label = self._detect_label(prompt)
        # Unique-per-call copies: the call index guarantees no two calls (and
        # the variant index no two copies within a call) ever collide under the
        # generator's normalise()-based dedup, so all 3 are accepted.
        copies = [
            f"Headline c{idx}-v{j} for {label}"
            for j in range(_CANDIDATES_PER_ANGLE_CALL)
        ]
        self.angle_calls.append(
            {"label": label, "prompt": prompt, "copies": list(copies)}
        )
        return {"candidates": [{"copy": c} for c in copies]}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_labels(n: int) -> list[str]:
    """Return ``n`` distinct, non-overlapping angle labels in a fixed order."""
    return [f"angle-{i:02d}" for i in range(n)]


def _make_brief() -> Creative_Brief:
    """A valid English-global brief (English source → native generation off)."""
    return Creative_Brief(
        campaign_topic="Game top-up promotion",
        target_platform=Target_Platform.GOOGLE_ADS,
        target_market=_MARKET,
        creative_type=Creative_Type.HEADLINE,
        selling_points=["Fast top-ups", "Secure payments"],
        target_audience="Mobile gamers worldwide",
        source_language="en",
        brand_name="Coco",
    )


def _make_platform_spec() -> Platform_Spec:
    return Platform_Spec(
        platform=Target_Platform.GOOGLE_ADS,
        char_limits={
            Creative_Type.HEADLINE: 90,
            Creative_Type.DESCRIPTION: 200,
            Creative_Type.CTA: 30,
            Creative_Type.LONG_COPY: 300,
        },
        allowed_creative_types=[
            Creative_Type.HEADLINE,
            Creative_Type.DESCRIPTION,
            Creative_Type.CTA,
            Creative_Type.LONG_COPY,
        ],
    )


def _run_angles(
    *,
    n: int,
    min_count: int,
    accepted_angle_counts: Optional[dict[str, int]] = None,
) -> tuple[_RecordingAngleLLM, list[str], list]:
    """Drive angle generation for ``n`` angles; return (llm, labels, candidates).

    The recording LLM and stub splitter make the run fully deterministic, so
    the returned ``llm.angle_calls`` faithfully describe every per-angle call.
    """
    labels = _make_labels(n)
    llm = _RecordingAngleLLM(labels)
    splitter = _StubAngleSplitter(labels)
    generator = CreativeGenerator(
        llm,
        timeout_ms=60_000,
        angle_splitter=splitter,  # type: ignore[arg-type]
    )

    output = asyncio.run(
        generator.generate_with_angles(
            _make_brief(),
            _make_platform_spec(),
            min_count=min_count,
            request_id=f"req-angles-{n}",
            warnings=[],
            accepted_angle_counts=accepted_angle_counts,
        )
    )
    return llm, labels, output.candidates


# ---------------------------------------------------------------------------
# Property 7 — Round-robin angle generation (Req 3.4)
# ---------------------------------------------------------------------------


# Feature: creative-localization-diversity, Property 7: Round-robin angle generation
@settings(max_examples=100, deadline=None)
@given(n=st.integers(min_value=4, max_value=8))
def test_round_robin_angle_generation(n: int) -> None:
    """The first N generation calls each target a distinct angle with exactly
    3 candidates per call, before any angle receives a second call.

    ``min_count`` is set to ``3 * N + 1`` so the round-robin phase (which
    produces ``3 * N`` candidates) is forced to spill into a single refill
    call. That refill necessarily revisits an already-covered angle, which
    proves the first N calls were issued — one per distinct angle — *before*
    any angle's second call.

    Validates: Requirements 3.4
    """
    min_count = _CANDIDATES_PER_ANGLE_CALL * n + 1
    llm, labels, candidates = _run_angles(n=n, min_count=min_count)

    calls = llm.angle_calls
    # The round-robin phase must have issued at least one call per angle, plus
    # the forced refill call.
    assert len(calls) >= n + 1

    first_n = calls[:n]
    first_n_labels = [c["label"] for c in first_n]

    # The first N calls form a permutation of the N angle labels: every angle is
    # covered exactly once before any second call (Req 3.4).
    assert sorted(first_n_labels) == sorted(labels)
    assert len(set(first_n_labels)) == n

    # Each of those N calls requested exactly 3 candidates and produced 3
    # accepted copies (all unique → none dropped by dedup).
    for call in first_n:
        assert f"恰好 {_CANDIDATES_PER_ANGLE_CALL} 条" in call["prompt"]
        assert len(call["copies"]) == _CANDIDATES_PER_ANGLE_CALL

    # The (N+1)-th call is a second call to an already-covered angle, confirming
    # no angle was visited twice during the round-robin phase.
    assert calls[n]["label"] in set(labels)

    # Every produced candidate carries an angle label drawn from the angle set
    # (Req 3.6 attribution, exercised here in passing).
    assert candidates
    for cand in candidates:
        assert cand.angle_label in set(labels)


# ---------------------------------------------------------------------------
# Property 8 — Angle cycling prioritizes lowest count (Req 3.5)
# ---------------------------------------------------------------------------


# Feature: creative-localization-diversity, Property 8: Angle cycling prioritizes lowest count
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=4, max_value=8),
    use_seed=st.booleans(),
    raw_seeds=st.lists(st.integers(min_value=0, max_value=3), min_size=8, max_size=8),
)
def test_angle_cycling_prioritizes_lowest_count(
    n: int,
    use_seed: bool,
    raw_seeds: list[int],
) -> None:
    """Once the round-robin phase completes, each refill call targets the angle
    with the fewest accepted candidates, breaking ties by original angle order.

    ``min_count`` is set to ``2 * 3 * N`` so the loop must cycle well beyond the
    ``N x 3`` produced by the round-robin phase, forcing N additional refill
    calls. The accepted-count timeline is reconstructed from the ordered calls;
    for each refill call we assert its targeted angle was a current minimum and
    matches the deterministic (count, original-order) tie-break.

    Optionally seeds ``accepted_angle_counts`` (as the orchestrator does across
    refill rounds) to confirm cycling still honours the seeded balance.

    Validates: Requirements 3.5
    """
    labels = _make_labels(n)
    order_index = {label: i for i, label in enumerate(labels)}

    seed: Optional[dict[str, int]] = None
    if use_seed:
        seed = {label: raw_seeds[i] for i, label in enumerate(labels)}

    min_count = 2 * _CANDIDATES_PER_ANGLE_CALL * n
    llm, _labels, candidates = _run_angles(
        n=n, min_count=min_count, accepted_angle_counts=seed
    )

    calls = llm.angle_calls
    assert len(calls) >= n  # round-robin phase always issues N calls

    # --- Reconstruct the accepted-count timeline ----------------------------
    # Seeds mirror the generator: missing labels start at 0; seeds override.
    counts = {label: (seed.get(label, 0) if seed else 0) for label in labels}

    # Phase 1 (round-robin): one call per angle, in decomposition order.
    phase1 = calls[:n]
    assert sorted(c["label"] for c in phase1) == sorted(labels)
    for call in phase1:
        counts[call["label"]] += len(call["copies"])

    # Phase 2 (cycling): there must be at least one refill call to exercise the
    # lowest-count selection.
    refill_calls = calls[n:]
    assert refill_calls, "expected the loop to cycle beyond N x 3 candidates"

    for call in refill_calls:
        label = call["label"]
        # Deterministic selection the generator should have made: the angle with
        # the fewest accepted candidates, ties broken by original order.
        expected = min(counts, key=lambda lbl: (counts[lbl], order_index[lbl]))
        min_value = counts[expected]

        # The targeted angle held a (the) minimum count at selection time...
        assert counts[label] == min_value, (
            f"refill call targeted {label!r} (count {counts[label]}) but the "
            f"minimum accepted count was {min_value}"
        )
        # ...and matched the deterministic (count, original-order) tie-break.
        assert label == expected, (
            f"refill call targeted {label!r} but deterministic tie-break "
            f"selects {expected!r}"
        )

        counts[label] += len(call["copies"])

    # Every produced candidate is attributed to one of the angles (Req 3.6).
    assert candidates
    for cand in candidates:
        assert cand.angle_label in set(labels)
