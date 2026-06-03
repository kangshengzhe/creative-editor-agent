"""Property-based tests for the Semantic_Diversity_Checker.

Feature: creative-localization-diversity.

Exercises the universal correctness properties from design.md for
:class:`creative_agent.integration.semantic_diversity.SemanticDiversityChecker`.
Each property is tagged with its design property number and the requirement(s)
it validates.

The embedding model is replaced with a deterministic, dependency-free stub:
an injected ``embed_fn`` that maps each distinct candidate/accepted text key to
a Hypothesis-generated vector. This isolates the threshold-enforcement logic
under test from the heavy ``sentence-transformers`` dependency while keeping the
cosine computation real (the checker's own static ``cosine_similarity``).
"""

from __future__ import annotations

import asyncio

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from creative_agent.integration.semantic_diversity import SemanticDiversityChecker
from creative_agent.models import SemanticDiversityConfig

#: Distinct text keys for the candidate and the single accepted-pool member.
#: The injected ``embed_fn`` maps these to the generated vectors so the checker
#: embeds known vectors without touching any real embedding model.
_CANDIDATE_TEXT = "candidate-copy"
_ACCEPTED_TEXT = "accepted-copy"

#: Finite, non-subnormal floats in a moderate range. Subnormals are excluded so
#: vector norms cannot silently underflow to zero, keeping the generated
#: vectors well-conditioned for cosine similarity.
_components = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


@st.composite
def _vector_pairs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw two equal-length, non-zero embedding vectors (dimension 1–8)."""
    dim = draw(st.integers(min_value=1, max_value=8))
    vec_a = draw(st.lists(_components, min_size=dim, max_size=dim))
    vec_b = draw(st.lists(_components, min_size=dim, max_size=dim))
    return vec_a, vec_b


# Feature: creative-localization-diversity, Property 4: Cosine similarity threshold enforcement
@settings(max_examples=100)
@given(
    vectors=_vector_pairs(),
    threshold=st.floats(
        min_value=0.5,
        max_value=0.99,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ),
)
def test_cosine_similarity_threshold_enforcement(
    vectors: tuple[list[float], list[float]],
    threshold: float,
) -> None:
    """Property 4: Cosine similarity threshold enforcement.

    For any two embedding vectors and a configured Similarity_Threshold in
    [0.5, 0.99], the Semantic_Diversity_Checker rejects the candidate if and
    only if the cosine similarity between the candidate's vector and the
    accepted candidate's vector exceeds the threshold.

    A deterministic ``embed_fn`` maps the candidate/accepted text keys to the
    generated vectors, so the checker embeds exactly these vectors. The
    expected decision is computed with the checker's own static
    ``cosine_similarity`` to mirror the implementation's arithmetic exactly.

    **Validates: Requirements 2.3, 2.4**
    """
    candidate_vec, accepted_vec = vectors

    # Guard against zero-magnitude vectors: a zero vector yields a degenerate
    # cosine (0.0 by the implementation's convention) that does not meaningfully
    # exercise the threshold boundary.
    assume(any(component != 0.0 for component in candidate_vec))
    assume(any(component != 0.0 for component in accepted_vec))

    embeddings = {
        _CANDIDATE_TEXT: candidate_vec,
        _ACCEPTED_TEXT: accepted_vec,
    }

    checker = SemanticDiversityChecker(
        SemanticDiversityConfig(similarity_threshold=threshold),
        embed_fn=lambda text: embeddings[text],
    )

    expected_cosine = SemanticDiversityChecker.cosine_similarity(
        candidate_vec, accepted_vec
    )

    result = asyncio.run(
        checker.check_candidate(_CANDIDATE_TEXT, [_ACCEPTED_TEXT])
    )

    # The core biconditional: accepted iff similarity does NOT exceed threshold.
    assert result.accepted == (expected_cosine <= threshold)

    # Rejection occurs iff similarity strictly exceeds the threshold, and the
    # rejected pair must identify the offending (candidate, accepted) pair.
    if expected_cosine > threshold:
        assert result.accepted is False
        assert result.rejected_pair == (_CANDIDATE_TEXT, _ACCEPTED_TEXT)
    else:
        assert result.accepted is True
        assert result.rejected_pair is None


@st.composite
def _pool_with_duplicate_index(
    draw: st.DrawFn,
) -> tuple[list[str], int]:
    """Draw a pool of N distinct accepted texts (N in 1..8) and an index ``i``.

    The candidate under test is an exact duplicate of ``pool[i]``. ``unique=True``
    guarantees the pool members are pairwise distinct so each maps to its own
    one-hot basis vector (see :func:`test_full_pool_semantic_comparison`).
    """
    pool = draw(
        st.lists(
            st.text(min_size=1, max_size=24),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    duplicate_index = draw(st.integers(min_value=0, max_value=len(pool) - 1))
    return pool, duplicate_index


# Feature: creative-localization-diversity, Property 5: Full-pool semantic comparison
@settings(max_examples=100)
@given(pool_and_index=_pool_with_duplicate_index())
def test_full_pool_semantic_comparison(
    pool_and_index: tuple[list[str], int],
) -> None:
    """Property 5: Full-pool semantic comparison.

    For any pool of accepted candidates accumulated across refill rounds and a
    new candidate, the Semantic_Diversity_Checker compares the candidate against
    *every* accumulated accepted candidate — not just the latest batch entry.

    Construction: the accepted pool holds ``N`` (1..8) distinct texts. A
    deterministic ``embed_fn`` maps each distinct text to a one-hot basis vector
    keyed by its position in the stable pool ordering, so distinct texts are
    mutually orthogonal (cosine 0.0) and equal text → equal vector (cosine 1.0).
    The candidate is an exact duplicate of ``pool[i]`` for a generated index
    ``i``.

    Because only ``pool[i]`` shares the candidate's vector, the candidate is a
    semantic duplicate of exactly one — possibly the *oldest* (``i == 0``) —
    accepted member. If the checker only compared the candidate against the
    latest batch entry (``pool[-1]``), it would wrongly accept the candidate for
    every ``i != N - 1``. Asserting REJECTION for every ``i`` (including
    ``i == 0``) proves the whole accumulated pool is scanned, and the
    ``rejected_pair[1]`` must identify the duplicated member.

    **Validates: Requirements 2.2, 2.9**
    """
    pool, duplicate_index = pool_and_index
    candidate_text = pool[duplicate_index]

    # One-hot basis vector per distinct pool member: orthonormal so the cosine
    # between distinct texts is 0.0 and between equal texts (candidate vs its
    # duplicate) is exactly 1.0.
    dimension = len(pool)
    embeddings = {
        text: [1.0 if position == index else 0.0 for position in range(dimension)]
        for index, text in enumerate(pool)
    }

    checker = SemanticDiversityChecker(
        SemanticDiversityConfig(),
        embed_fn=lambda text: embeddings[text],
    )

    result = asyncio.run(checker.check_candidate(candidate_text, pool))

    # Rejection regardless of i proves the full accumulated pool was scanned:
    # had only pool[-1] been compared, candidates duplicating an earlier member
    # (i < N - 1) would have been accepted.
    assert result.accepted is False
    assert result.rejected_pair is not None
    # The offending accepted member is exactly the duplicated pool entry.
    assert result.rejected_pair[1] == pool[duplicate_index]
    assert result.rejected_pair[0] == candidate_text
    # An exact duplicate yields maximal cosine similarity.
    assert result.similarity_score == 1.0
