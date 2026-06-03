"""Semantic_Diversity_Checker — embedding-based semantic deduplication.

Implements design.md § Components and Interfaces / 2. Semantic_Diversity_Checker
and Requirements 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 2.9, 2.10.

Behaviour summary
-----------------

* Computes an Embedding_Vector for each candidate copy and the
  Cosine_Similarity between the new candidate and every accepted candidate in
  the pool (Requirements 2.1, 2.2, 2.9).
* Rejects the new candidate **iff** its highest Cosine_Similarity against any
  accepted candidate *exceeds* the configured Similarity_Threshold
  (Requirements 2.3, 2.4).
* Every rejection is logged with the candidate pair, the similarity score, and
  the applied threshold for observability (Requirement 2.6).
* When the embedding model is unavailable, or embedding computation exceeds the
  configured ``timeout_seconds``, the checker degrades to *text-dedup only*:
  it returns a :class:`DiversityResult` with ``fallback=True`` (and
  ``accepted=True`` — the candidate is not rejected on semantic grounds),
  logs a warning, and does **not** raise. The caller (Orchestrator) appends a
  request-level warning and, crucially, the global circuit-breaker failure
  counter is **not** incremented because no exception escapes this component
  (Requirement 2.8).

Embedding model
---------------

The embedding function is *pluggable*. Callers may inject any
``Callable[[str], Sequence[float]]`` via the ``embed_fn`` constructor argument
(e.g. a real sentence-transformers model, a cached client, or a deterministic
test stub). When no ``embed_fn`` is supplied, a default function is used that
lazily loads the configured ``sentence-transformers`` model
(``SemanticDiversityConfig.embedding_model``). If that package (or the model)
cannot be loaded, the default raises :class:`EmbeddingUnavailableError`, which
the checker translates into the text-dedup-only fallback described above. This
keeps the component importable and usable in environments without the heavy
ML dependency installed.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from creative_agent.models import SemanticDiversityConfig
from creative_agent.observability.logging import get_logger

__all__ = [
    "DiversityResult",
    "EmbeddingUnavailableError",
    "SemanticDiversityChecker",
]

#: An embedding function maps a text string to a numeric vector.
EmbedFn = Callable[[str], Sequence[float]]

#: Max characters of candidate text included in log records (avoids dumping
#: arbitrarily long copy into structured logs).
_LOG_TEXT_PREVIEW: int = 160


class EmbeddingUnavailableError(RuntimeError):
    """Raised by an embedding function when no model can produce a vector.

    The checker catches this internally and falls back to text-dedup only
    (Requirement 2.8); it is never propagated to the Orchestrator, so it does
    not contribute to the global circuit-breaker failure count.
    """


@dataclass
class DiversityResult:
    """Outcome of a single semantic-diversity check.

    Attributes:
        accepted: ``True`` if the candidate is semantically diverse enough to
            keep (or if the check degraded to fallback); ``False`` if it was
            rejected as a semantic duplicate.
        similarity_score: The highest Cosine_Similarity observed against the
            accepted pool, or ``None`` when no comparison was performed (empty
            pool) or the check fell back to text-dedup only.
        rejected_pair: ``(new_candidate, most_similar_accepted)`` when the
            candidate was rejected; ``None`` otherwise.
        fallback: ``True`` when the embedding model was unavailable or timed
            out and the checker degraded to *text-dedup only*. The Orchestrator
            uses this flag to append a request-level warning **without**
            incrementing the circuit breaker (Requirement 2.8).
    """

    accepted: bool
    similarity_score: Optional[float] = None
    rejected_pair: Optional[tuple[str, str]] = None
    fallback: bool = False


class SemanticDiversityChecker:
    """Embedding-based semantic deduplication for creative candidates.

    The checker holds no per-request state: the ``accepted_pool`` is passed in
    on every :meth:`check_candidate` call, letting the Orchestrator manage
    cross-round accumulation (Requirement 2.9). Embedding vectors are memoised
    in an internal cache keyed by text so repeated comparisons against the same
    accepted pool do not recompute embeddings.

    Args:
        config: :class:`SemanticDiversityConfig` supplying the
            Similarity_Threshold, ``timeout_seconds``, and embedding model name
            (Requirements 2.4, 2.8). Defaults are used when ``None``.
        embed_fn: Optional injectable embedding function
            ``Callable[[str], Sequence[float]]``. When ``None``, a default
            function backed by ``sentence-transformers`` is used (lazily
            loaded; see module docstring).
    """

    def __init__(
        self,
        config: Optional[SemanticDiversityConfig] = None,
        *,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self._config = config or SemanticDiversityConfig()
        self._threshold: float = self._config.similarity_threshold
        self._timeout_seconds: float = self._config.timeout_seconds
        self._model_name: str = self._config.embedding_model
        self._embed_fn: EmbedFn = embed_fn or self._default_embed_fn
        self._embedding_cache: dict[str, list[float]] = {}
        # Lazily-loaded sentence-transformers model for the default embed_fn.
        # Sentinel ``False`` means "not yet attempted"; ``None`` means "load
        # attempted and failed" so we don't retry on every call.
        self._model: object | None | bool = False
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Warm-up (one-time, outside the per-candidate timeout)
    # ------------------------------------------------------------------

    def warm_up(self) -> bool:
        """Eagerly load the embedding model and run one throwaway embedding.

        The first embedding call also loads the model (download + init), which
        for ``sentence-transformers`` takes several seconds — far longer than
        the per-candidate ``timeout_seconds`` (default 3s). If that first load
        happens *inside* :meth:`check_candidate`'s ``asyncio.wait_for`` budget,
        the initial candidate always times out and the whole request silently
        degrades to text-dedup only (Requirement 2.8 fallback) even though the
        model is perfectly usable.

        Calling :meth:`warm_up` once at process start (e.g. in ``server.py``)
        moves that cost out of the request path: by the time real candidates
        arrive the model is resident and each embedding is milliseconds, so the
        timeout only guards genuine slowness.

        Returns:
            ``True`` if the model is ready (semantic diversity is live);
            ``False`` if the model could not be loaded (the checker will run in
            text-dedup-only fallback mode). Never raises — a failed warm-up is
            logged and degrades gracefully.
        """
        try:
            # Embedding a tiny string forces the lazy model load + a real
            # forward pass, populating any internal caches.
            self._embed_fn("warmup")
        except EmbeddingUnavailableError as exc:
            self._log.warning(
                "semantic_diversity.warm_up_unavailable",
                embedding_model=self._model_name,
                detail=str(exc),
            )
            return False
        except Exception as exc:  # noqa: BLE001 — never let warm-up crash startup
            self._log.warning(
                "semantic_diversity.warm_up_failed",
                embedding_model=self._model_name,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

        self._log.info(
            "semantic_diversity.warm_up_ok",
            embedding_model=self._model_name,
        )
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_candidate(
        self, candidate_text: str, accepted_pool: list[str]
    ) -> DiversityResult:
        """Check whether ``candidate_text`` is semantically diverse.

        Computes the candidate's embedding and its Cosine_Similarity against
        every member of ``accepted_pool`` (Requirements 2.2, 2.9). The
        candidate is rejected iff its highest similarity *exceeds* the
        configured threshold (Requirements 2.3, 2.4).

        Embedding computation is bounded by ``timeout_seconds``; on timeout or
        when the embedding model is unavailable, the result is a fallback
        (``fallback=True``, ``accepted=True``) signalling text-dedup-only mode
        (Requirement 2.8). No exception is raised in that case.

        Args:
            candidate_text: The new candidate copy to evaluate.
            accepted_pool: All copies accepted so far in the current request
                (across refill rounds).

        Returns:
            A :class:`DiversityResult` describing the decision.
        """
        # Nothing to compare against → trivially diverse (no embedding needed).
        if not accepted_pool:
            return DiversityResult(
                accepted=True, similarity_score=None, rejected_pair=None
            )

        try:
            candidate_vec, pool_vecs = await asyncio.wait_for(
                self._gather_embeddings(candidate_text, accepted_pool),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._fallback(reason="timeout")
        except EmbeddingUnavailableError as exc:
            return self._fallback(reason="model_unavailable", detail=str(exc))

        # Find the most similar accepted candidate.
        best_score: float = -1.0
        best_text: Optional[str] = None
        for accepted_text, accepted_vec in pool_vecs:
            score = self.cosine_similarity(candidate_vec, accepted_vec)
            if score > best_score:
                best_score = score
                best_text = accepted_text

        # Reject iff similarity strictly exceeds the threshold (Req 2.3/2.4).
        if best_text is not None and best_score > self._threshold:
            self._log.info(
                "semantic_diversity.rejected",
                candidate=_preview(candidate_text),
                most_similar_accepted=_preview(best_text),
                similarity_score=round(best_score, 6),
                threshold=self._threshold,
            )
            return DiversityResult(
                accepted=False,
                similarity_score=best_score,
                rejected_pair=(candidate_text, best_text),
            )

        return DiversityResult(
            accepted=True,
            similarity_score=best_score if best_text is not None else None,
            rejected_pair=None,
        )

    def compute_embedding(self, text: str) -> list[float]:
        """Compute (and cache) the Embedding_Vector for ``text``.

        Args:
            text: The copy to embed.

        Returns:
            The embedding vector as a list of floats.

        Raises:
            EmbeddingUnavailableError: When the configured embedding model
                cannot produce a vector (e.g. the backing package is not
                installed). Callers that need graceful degradation should use
                :meth:`check_candidate`, which converts this into a fallback.
        """
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        vector = [float(x) for x in self._embed_fn(text)]
        self._embedding_cache[text] = vector
        return vector

    @staticmethod
    def cosine_similarity(
        vec_a: Sequence[float], vec_b: Sequence[float]
    ) -> float:
        """Return the Cosine_Similarity between two vectors.

        Pure-Python implementation (no numpy dependency). The result is clamped
        to ``[-1.0, 1.0]`` to absorb floating-point error. A zero-magnitude
        vector yields ``0.0`` (orthogonal by convention).

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity in ``[-1.0, 1.0]``.

        Raises:
            ValueError: When the vectors have different lengths.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                "cosine_similarity requires equal-length vectors; got "
                f"{len(vec_a)} and {len(vec_b)}"
            )
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
        # Clamp for float safety so downstream threshold comparisons are sane.
        if sim > 1.0:
            return 1.0
        if sim < -1.0:
            return -1.0
        return sim

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _gather_embeddings(
        self, candidate_text: str, accepted_pool: list[str]
    ) -> tuple[list[float], list[tuple[str, list[float]]]]:
        """Embed the candidate and every pool member (off the event loop).

        Each (potentially blocking) embedding call runs in a worker thread via
        :func:`asyncio.to_thread` so the surrounding :func:`asyncio.wait_for`
        can enforce ``timeout_seconds`` without blocking the event loop. Cached
        embeddings are returned directly without re-running the model.
        """
        candidate_vec = await asyncio.to_thread(self.compute_embedding, candidate_text)
        pool_vecs: list[tuple[str, list[float]]] = []
        for accepted_text in accepted_pool:
            vec = await asyncio.to_thread(self.compute_embedding, accepted_text)
            pool_vecs.append((accepted_text, vec))
        return candidate_vec, pool_vecs

    def _fallback(self, *, reason: str, detail: Optional[str] = None) -> DiversityResult:
        """Build a text-dedup-only fallback result and log a warning (Req 2.8).

        No exception is raised, so the global circuit-breaker counter is left
        untouched; the Orchestrator surfaces a request-level warning based on
        the ``fallback`` flag.
        """
        self._log.warning(
            "semantic_diversity.fallback",
            reason=reason,
            embedding_model=self._model_name,
            timeout_seconds=self._timeout_seconds,
            detail=detail,
        )
        return DiversityResult(
            accepted=True,
            similarity_score=None,
            rejected_pair=None,
            fallback=True,
        )

    def _default_embed_fn(self, text: str) -> list[float]:
        """Default embedding function backed by ``sentence-transformers``.

        Lazily loads the model named by the config. Raises
        :class:`EmbeddingUnavailableError` when the package or model cannot be
        loaded so the checker can degrade gracefully (Requirement 2.8).
        """
        model = self._load_default_model()
        try:
            vector = model.encode(text)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - any model error → unavailable
            raise EmbeddingUnavailableError(
                f"embedding model {self._model_name!r} failed to encode: {exc}"
            ) from exc
        return [float(x) for x in vector]

    def _load_default_model(self) -> object:
        """Lazily import and instantiate the sentence-transformers model.

        Caches the loaded model. A previous failed load (``self._model is None``)
        short-circuits to raising :class:`EmbeddingUnavailableError` without
        retrying the import on every call.
        """
        if self._model is None:
            raise EmbeddingUnavailableError(
                f"embedding model {self._model_name!r} previously failed to load"
            )
        if self._model is not False:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # noqa: BLE001 - ImportError or transitive failure
            self._model = None
            raise EmbeddingUnavailableError(
                "sentence-transformers is not available; install it or inject "
                f"an embed_fn (model={self._model_name!r}): {exc}"
            ) from exc

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:  # noqa: BLE001 - model download / load failure
            self._model = None
            raise EmbeddingUnavailableError(
                f"failed to load embedding model {self._model_name!r}: {exc}"
            ) from exc

        return self._model


def _preview(text: str) -> str:
    """Truncate ``text`` for inclusion in structured log records."""
    if len(text) <= _LOG_TEXT_PREVIEW:
        return text
    return text[:_LOG_TEXT_PREVIEW] + "…"
