"""``OpenAICompatibleLLM`` — the remote reply/summary model, over ``/chat/completions``.

The default target is LM Studio on ``http://localhost:1234/v1`` serving a Qwen
checkpoint, which is what the alice bench uses; any OpenAI-compatible endpoint works,
including a hosted one, because the only thing this client needs is
``POST {base_url}/chat/completions``.

Deliberate choices, all of them learned from the alice conversation bench:

* **No retries.** This model is off the admission path; a late reply is worse than a
  short silence, and the caller already has a typed error to log.
* **Non-thinking mode is a switch, not a hope.** Qwen3 emits ``<think>`` blocks unless
  the chat template is told otherwise. Set ``RT_AGENT_LLM_DISABLE_THINKING=1`` (or pass
  ``disable_thinking=True``) and the request carries
  ``{"chat_template_kwargs": {"enable_thinking": false}}``. Whatever the switch, any
  reasoning block that still comes back is stripped before the text is returned.
* **TLS is never disabled and ``trust_env`` stays on**, so an outbound proxy and its CA
  bundle keep working.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from rt_agent.contracts.protocols import ChatMessage
from rt_agent.llm.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsageError,
)

__all__ = [
    "BASE_URL_ENV",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DISABLE_THINKING_ENV",
    "MODEL_ENV",
    "NON_THINKING_EXTRA_BODY",
    "OpenAICompatibleLLM",
]

#: LM Studio's default OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "http://localhost:1234/v1"

BASE_URL_ENV = "RT_AGENT_LLM_BASE_URL"
MODEL_ENV = "RT_AGENT_LLM_MODEL"
DISABLE_THINKING_ENV = "RT_AGENT_LLM_DISABLE_THINKING"
DEFAULT_API_KEY_ENV = "RT_AGENT_LLM_API_KEY"

#: What a Qwen chat template needs to stay out of thinking mode.
NON_THINKING_EXTRA_BODY: Mapping[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

_CHAT_PATH = "/chat/completions"
_TRUE = frozenset({"1", "true", "yes", "on"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>", re.IGNORECASE)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """One-level-deep merge, so an explicit ``extra_body`` can extend the defaults."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks, including an unclosed trailing one."""
    without = _THINK_BLOCK.sub(" ", text)
    match = _OPEN_THINK.search(without)
    if match is not None:
        # An unclosed block means everything after it is reasoning, not an answer.
        without = without[: match.start()]
    return without.strip()


class OpenAICompatibleLLM:
    """A :class:`~rt_agent.contracts.protocols.ChatLLM` over an OpenAI-compatible server.

    ``base_url`` falls back to ``$RT_AGENT_LLM_BASE_URL`` and then to LM Studio's
    default; ``model`` falls back to ``$RT_AGENT_LLM_MODEL`` and is otherwise a
    configuration error, because no server can guess which checkpoint to load.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout_s: float = 15.0,
        extra_body: Mapping[str, Any] | None = None,
        connect_timeout_s: float = 3.0,
        temperature: float = 0.0,
        max_tokens: int = 160,
        disable_thinking: bool | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved_base = base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        self.base_url = resolved_base.rstrip("/")
        if not self.base_url:
            raise LLMConfigurationError("base_url must not be empty")
        resolved_model = model or os.environ.get(MODEL_ENV) or ""
        if not resolved_model:
            raise LLMConfigurationError(
                f"no model id: pass model=... or set ${MODEL_ENV} (for example the id "
                "LM Studio shows for the loaded checkpoint)"
            )
        self.model = resolved_model
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = min(float(connect_timeout_s), self.timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.disable_thinking = (
            _env_flag(DISABLE_THINKING_ENV) if disable_thinking is None else (disable_thinking)
        )
        defaults: Mapping[str, Any] = NON_THINKING_EXTRA_BODY if self.disable_thinking else {}
        self.extra_body: dict[str, Any] = _merge(defaults, extra_body or {})
        self._transport = transport
        self._client = client
        self._owns_client = client is None

    # -- plumbing -----------------------------------------------------------------

    @property
    def url(self) -> str:
        """The chat-completions endpoint this client posts to."""
        return f"{self.base_url}{_CHAT_PATH}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
                trust_env=True,
                transport=self._transport,
            )
        return self._client

    def build_body(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """The exact JSON body that :meth:`complete` would post. Useful in tests."""
        if not messages:
            raise LLMUsageError("a completion needs at least one message")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        body.update(self.extra_body)
        return body

    # -- the ChatLLM protocol -----------------------------------------------------

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's completion as plain text, or raise a typed error."""
        body = self.build_body(messages)
        client = self._get_client()
        try:
            async with asyncio.timeout(self.timeout_s):
                response = await client.post(self.url, json=body, headers=self._headers())
        except TimeoutError as error:
            raise LLMTimeoutError(f"chat completion exceeded {self.timeout_s:.1f}s") from error
        except httpx.TimeoutException as error:
            raise LLMTimeoutError(f"chat completion timed out: {error}") from error
        except httpx.HTTPError as error:
            raise LLMUnavailableError(f"chat completion transport failure: {error}") from error

        _raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as error:
            raise LLMProtocolError("chat completion response is not JSON") from error
        return parse_completion(payload)

    async def aclose(self) -> None:
        """Close the HTTP client, unless one was handed in."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        """Enter an ``async with`` block."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client on the way out."""
        await self.aclose()


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    detail = response.text[:300]
    if status in (401, 403):
        raise LLMAuthError(f"chat completion rejected the credentials ({status}): {detail}")
    if status in (400, 404, 422):
        raise LLMUsageError(f"chat completion request was refused ({status}): {detail}")
    if status in (429, 529):
        raise LLMRateLimitError(
            f"chat completion was rate limited ({status}): {detail}", _retry_after(response)
        )
    if status >= 500:
        raise LLMUnavailableError(f"chat completion server error ({status}): {detail}")
    raise LLMProtocolError(f"unexpected chat completion status {status}: {detail}")


def parse_completion(payload: Any) -> str:
    """Pull the assistant text out of a chat-completions body, fail-closed.

    Reasoning blocks are stripped here rather than trusted away, so a server that
    ignores the non-thinking switch still yields usable text.
    """
    if not isinstance(payload, dict):
        raise LLMProtocolError("chat completion body is not a JSON object")
    if payload.get("error"):
        raise LLMProtocolError(f"chat completion reported an error: {payload['error']!r}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProtocolError("chat completion returned no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMProtocolError("chat completion returned a malformed choice")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMProtocolError("chat completion choice has no message")
    content = message.get("content")
    if content is None:
        raise LLMEmptyResponseError("chat completion returned no content")
    if not isinstance(content, str):
        raise LLMProtocolError("chat completion content is not a string")
    text = strip_reasoning(content)
    if not text:
        raise LLMEmptyResponseError("chat completion returned only reasoning or whitespace")
    return text
