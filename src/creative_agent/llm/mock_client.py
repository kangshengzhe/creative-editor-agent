"""In-memory mock LLM client for unit and property-based tests.

The mock matches a request to a pre-configured response by scanning the
prompt for any registered keyword (in registration order). It records every
call so tests can assert on the prompt that was sent, then returns either a
canned text/JSON value or raises a canned exception.

Typical usage::

    client = MockLLMClient()
    client.set_response("generate_creative", {"candidates": [...]})
    client.set_failure("translate_to_th", TimeoutError("simulated"))
    client.set_default_response("OK")

    text = await client.complete("...generate_creative for...")
    assert client.calls[-1]["is_json"] is False
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

from creative_agent.errors.codes import ToolFailureError
from creative_agent.llm.client import LLMClient

#: Type of values accepted as canned responses. ``dict`` is returned as-is by
#: :meth:`complete_json` and JSON-serialised by :meth:`complete`.
ResponseValue = Union[str, dict]


class MockLLMClient(LLMClient):
    """Deterministic LLM client driven by keyword → response mappings.

    Args:
        responses: Optional initial mapping of ``prompt_keyword`` → canned
            response. Insertion order is preserved and used for matching.
    """

    def __init__(
        self,
        responses: Optional[dict[str, ResponseValue]] = None,
    ) -> None:
        # Use a fresh dict so the caller's mapping cannot be mutated through us.
        self._responses: dict[str, ResponseValue] = dict(responses) if responses else {}
        self._failures: dict[str, Exception] = {}
        self._default_response: Optional[ResponseValue] = None
        #: Recorded calls. Each entry has keys
        #: ``prompt``, ``system``, ``max_tokens``, ``temperature``,
        #: ``timeout_ms``, ``is_json``.
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_response(self, prompt_keyword: str, response: ResponseValue) -> None:
        """Register or replace a response for prompts containing ``prompt_keyword``."""
        # Re-insert at the end so callers can override priority by re-registering.
        self._responses.pop(prompt_keyword, None)
        self._responses[prompt_keyword] = response

    def set_failure(self, prompt_keyword: str, exception: Exception) -> None:
        """Cause prompts containing ``prompt_keyword`` to raise ``exception``."""
        self._failures[prompt_keyword] = exception

    def set_default_response(self, response: ResponseValue) -> None:
        """Set the fallback response used when no keyword matches."""
        self._default_response = response

    def reset(self) -> None:
        """Clear all configured responses, failures, and recorded calls."""
        self._responses.clear()
        self._failures.clear()
        self._default_response = None
        self.calls.clear()

    # ------------------------------------------------------------------
    # LLMClient implementation
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
        self._record_call(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_ms=timeout_ms,
            is_json=False,
        )
        response = self._resolve(prompt)
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
        return response

    async def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout_ms: Optional[int] = None,
    ) -> dict:
        self._record_call(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_ms=timeout_ms,
            is_json=True,
        )
        response = self._resolve(prompt)
        if isinstance(response, dict):
            return response
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ToolFailureError(
                tool_name="LLMClient",
                message=(
                    "MockLLMClient was asked for JSON but the configured "
                    f"string response is not valid JSON: {response!r}"
                ),
                original_exception=exc,
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolFailureError(
                tool_name="LLMClient",
                message=(
                    "MockLLMClient JSON response did not parse to a dict; got "
                    f"{type(parsed).__name__}"
                ),
            )
        return parsed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_call(
        self,
        *,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        timeout_ms: Optional[int],
        is_json: bool,
    ) -> None:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout_ms": timeout_ms,
                "is_json": is_json,
            }
        )

    def _resolve(self, prompt: str) -> ResponseValue:
        # Failures take precedence over canned responses so a test can flip a
        # previously-successful prompt to fail without removing the response.
        for keyword, exc in self._failures.items():
            if keyword in prompt:
                raise exc
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        if self._default_response is not None:
            return self._default_response
        raise ToolFailureError(
            tool_name="LLMClient",
            message=(
                "MockLLMClient has no response configured for prompt "
                f"(prefix={prompt[:80]!r})"
            ),
        )
