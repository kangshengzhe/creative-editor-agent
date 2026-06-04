"""Production LLM client that talks to an OpenAI-compatible HTTP endpoint.

Configuration is read from environment variables (with ``.env`` auto-loaded
via :mod:`python_dotenv`), and may be overridden at construction time:

================================  =====================================
Variable                          Purpose
================================  =====================================
``TOKENPONY_API_KEY``             Bearer token for ``Authorization`` header.
``TOKENPONY_BASE_URL``            API root (e.g. ``https://.../v1``).
``TOKENPONY_MODEL``               Model name passed in the request body.
================================  =====================================

The client retries once on transient failures (network errors and HTTP 5xx
responses) and converts any terminal failure into a
:class:`~creative_agent.errors.codes.ToolFailureError` tagged with
``tool_name="LLMClient"``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient

# Load .env once at import time so that environment overrides are visible to
# both the default constructor and any direct ``os.environ`` lookups in the
# tools that build the client.
load_dotenv()

#: Default per-request timeout when the caller does not specify one (seconds).
_DEFAULT_TIMEOUT_S: float = 30.0

#: Suffix appended to ``complete_json`` prompts to coax the model into
#: returning bare JSON. Kept English-only because mimo-v2.5 sometimes
#: returns an empty completion when prompts mix Chinese and English.
_JSON_PROMPT_SUFFIX: str = (
    "\n\nRespond with valid JSON only. No markdown fences, no commentary, "
    "no surrounding text."
)

#: Recovery regex for models that wrap their JSON in ``\u0060\u0060\u0060json ... \u0060\u0060\u0060`` fences.
_JSON_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```",
    re.DOTALL,
)


class RealLLMClient(LLMClient):
    """OpenAI-compatible chat-completions client backed by ``httpx``.

    Args:
        api_key: Bearer token. Falls back to ``TOKENPONY_API_KEY``.
        base_url: API root URL (without trailing ``/chat/completions``).
            Falls back to ``TOKENPONY_BASE_URL``.
        model: Model identifier sent in the request body. Falls back to
            ``TOKENPONY_MODEL``.

    Raises:
        ValueError: When any of the three required settings is still missing
            after consulting environment variables.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv("TOKENPONY_API_KEY")
        resolved_base = base_url if base_url is not None else os.getenv("TOKENPONY_BASE_URL")
        resolved_model = model if model is not None else os.getenv("TOKENPONY_MODEL")

        missing: list[str] = []
        if not resolved_key:
            missing.append("TOKENPONY_API_KEY")
        if not resolved_base:
            missing.append("TOKENPONY_BASE_URL")
        if not resolved_model:
            missing.append("TOKENPONY_MODEL")
        if missing:
            raise ValueError(
                "RealLLMClient is missing required configuration: "
                + ", ".join(missing)
            )

        # ``resolved_*`` are guaranteed truthy after the check above; cast for
        # the benefit of static analysis.
        self._api_key: str = resolved_key  # type: ignore[assignment]
        self._base_url: str = resolved_base.rstrip("/")  # type: ignore[union-attr]
        self._model: str = resolved_model  # type: ignore[assignment]

        # Single pooled async client reused across ALL calls. Previously a new
        # httpx.AsyncClient was created and torn down per request, forcing a
        # fresh TCP + TLS handshake every time — with dozens of LLM calls per
        # generation (angles x candidates x pipeline) that handshake overhead
        # dominated latency. A shared client keeps connections alive (HTTP
        # keep-alive) so concurrent calls reuse the pool. Created lazily so the
        # client is bound to the running event loop.
        self._client: Optional[httpx.AsyncClient] = None
        # Generous connection pool so the orchestrator's asyncio.gather fan-out
        # (parallel angles / candidates / batch types) isn't serialized by a
        # small default limit.
        self._limits = httpx.Limits(
            max_connections=64,
            max_keepalive_connections=32,
            keepalive_expiry=30.0,
        )

    # ------------------------------------------------------------------
    # Public API (LLMClient)
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> str:
        payload = self._build_payload(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        data = await self._post_with_retry(payload, timeout_ms=timeout_ms)
        return self._extract_content(data)

    async def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> dict:
        # Some models (notably mimo-v2.5) occasionally reply with an empty
        # string for JSON-shaped prompts. Retry once with the SAME sampling
        # params — bumping temperature would push reasoning models past
        # their reasoning_tokens budget and make the empty-string outcome
        # more likely, not less.
        last_content: str = ""
        for attempt in range(2):
            payload = self._build_payload(
                prompt=prompt + _JSON_PROMPT_SUFFIX,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            data = await self._post_with_retry(payload, timeout_ms=timeout_ms)
            content = self._extract_content(data)
            last_content = content
            if content and content.strip():
                return self._parse_json(content)

        # Fall through: surface the empty-content failure as before so
        # callers see a consistent ToolFailureError.
        return self._parse_json(last_content)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        *,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared pooled client, creating it on first use.

        Lazy creation binds the client (and its connection pool) to the active
        event loop the first time a coroutine actually makes a call.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(limits=self._limits)
        return self._client

    async def aclose(self) -> None:
        """Close the pooled client and its connections. Safe to call repeatedly."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _post_with_retry(
        self,
        payload: dict[str, Any],
        *,
        timeout_ms: Optional[int],
    ) -> dict[str, Any]:
        timeout_s = (
            timeout_ms / 1000.0 if timeout_ms is not None else _DEFAULT_TIMEOUT_S
        )
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        client = self._get_client()

        last_exc: Optional[BaseException] = None
        # Two attempts total: initial + 1 retry on transient errors.
        for attempt in range(2):
            try:
                # Per-request timeout on the shared (pooled) client, so we keep
                # connection reuse while still honouring each call's budget.
                response = await client.post(
                    url, headers=headers, json=payload, timeout=timeout_s
                )
                if response.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"Upstream returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if attempt == 0:
                        continue
                    raise self._wrap_http_error(response, last_exc)
                if response.status_code >= 400:
                    # 4xx is non-retryable (bad key, bad request, etc.).
                    raise self._wrap_http_error(response, None)
                try:
                    return response.json()
                except ValueError as exc:
                    raise ToolFailureError(
                        tool_name="LLMClient",
                        message=(
                            "LLM endpoint returned non-JSON HTTP body: "
                            f"{response.text[:200]!r}"
                        ),
                        original_exception=exc,
                    ) from exc
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == 0:
                    continue
                raise ToolFailureError(
                    tool_name="LLMClient",
                    message=f"LLM transport error after retry: {exc}",
                    original_exception=exc,
                ) from exc

        # Defensive: the loop above either returns or raises.
        raise ToolFailureError(
            tool_name="LLMClient",
            message="LLM call failed without a specific error",
            original_exception=last_exc,
        )

    @staticmethod
    def _wrap_http_error(
        response: httpx.Response,
        original: Optional[BaseException],
    ) -> ToolFailureError:
        body_preview = response.text[:200] if response.text else ""
        return ToolFailureError(
            tool_name="LLMClient",
            message=(
                f"LLM endpoint returned HTTP {response.status_code}: "
                f"{body_preview!r}"
            ),
            original_exception=original,
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolFailureError(
                tool_name="LLMClient",
                message=f"LLM response missing choices[0].message.content: {data!r}",
                original_exception=exc,
            ) from exc

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as first_exc:
            match = _JSON_FENCE_RE.search(text)
            if match is None:
                raise ToolFailureError(
                    tool_name="LLMClient",
                    message=(
                        "LLM response is not valid JSON and contains no "
                        f"recognizable JSON block: {text[:200]!r}"
                    ),
                    original_exception=first_exc,
                ) from first_exc
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError as second_exc:
                raise ToolFailureError(
                    tool_name="LLMClient",
                    message=(
                        "LLM response contained a JSON fence but the inner "
                        f"payload is invalid: {match.group(1)[:200]!r}"
                    ),
                    original_exception=second_exc,
                ) from second_exc
        if not isinstance(parsed, dict):
            raise ToolFailureError(
                tool_name="LLMClient",
                message=(
                    "LLM JSON response is not an object/dict; got "
                    f"{type(parsed).__name__}"
                ),
            )
        return parsed
