"""``OpenAICompatibleLLM`` against ``httpx.MockTransport``: no socket is ever opened.

What matters here is the shape of the request (an LM Studio / Qwen server is picky
about it, and the non-thinking switch is the difference between a spoken sentence and a
paragraph of reasoning) and that every failure mode arrives as a typed error.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rt_agent.contracts import ChatLLM, ChatMessage
from rt_agent.llm import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DISABLE_THINKING_ENV,
    MODEL_ENV,
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsageError,
    OpenAICompatibleLLM,
)

MESSAGES = [
    ChatMessage(role="system", content="You are Alice."),
    ChatMessage(role="user", content="S1: hello"),
]


def completion(text: str) -> dict[str, Any]:
    """A minimal well-formed chat-completions body."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "qwen3-8b",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
    }


class Recorder:
    """Captures the outgoing request and replies with a canned response."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.response = response or httpx.Response(200, json=completion("Hello there."))
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    @property
    def body(self) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.requests[-1].content)
        return parsed


def make_client(handler: Any, **kwargs: Any) -> OpenAICompatibleLLM:
    defaults: dict[str, Any] = {"base_url": "http://localhost:1234/v1", "model": "qwen3-8b"}
    defaults.update(kwargs)
    return OpenAICompatibleLLM(transport=httpx.MockTransport(handler), **defaults)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (BASE_URL_ENV, MODEL_ENV, DISABLE_THINKING_ENV, "RT_AGENT_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class TestConfiguration:
    def test_defaults_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://studio.local:1234/v1/")
        monkeypatch.setenv(MODEL_ENV, "qwen3-27b-mlx")
        client = OpenAICompatibleLLM()
        assert client.url == "http://studio.local:1234/v1/chat/completions"
        assert client.model == "qwen3-27b-mlx"

    def test_lm_studio_is_the_default_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "qwen3-8b")
        assert OpenAICompatibleLLM().base_url == DEFAULT_BASE_URL

    def test_a_missing_model_is_a_configuration_error(self) -> None:
        with pytest.raises(LLMConfigurationError):
            OpenAICompatibleLLM(base_url="http://localhost:1234/v1")

    def test_it_satisfies_the_chat_llm_protocol(self) -> None:
        assert isinstance(make_client(Recorder()), ChatLLM)


class TestRequestShape:
    async def test_body_carries_the_messages_verbatim(self) -> None:
        recorder = Recorder()
        client = make_client(recorder, max_tokens=99, temperature=0.0)
        await client.complete(MESSAGES)

        request = recorder.requests[-1]
        assert str(request.url) == "http://localhost:1234/v1/chat/completions"
        assert request.method == "POST"
        assert recorder.body == {
            "model": "qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are Alice."},
                {"role": "user", "content": "S1: hello"},
            ],
            "temperature": 0.0,
            "max_tokens": 99,
            "stream": False,
        }

    async def test_thinking_switch_is_off_by_default(self) -> None:
        recorder = Recorder()
        await make_client(recorder).complete(MESSAGES)
        assert "chat_template_kwargs" not in recorder.body

    async def test_thinking_switch_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DISABLE_THINKING_ENV, "1")
        recorder = Recorder()
        await make_client(recorder).complete(MESSAGES)
        assert recorder.body["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_thinking_switch_from_the_constructor(self) -> None:
        recorder = Recorder()
        await make_client(recorder, disable_thinking=True).complete(MESSAGES)
        assert recorder.body["chat_template_kwargs"] == {"enable_thinking": False}

    async def test_extra_body_merges_with_the_thinking_switch(self) -> None:
        recorder = Recorder()
        client = make_client(
            recorder,
            disable_thinking=True,
            extra_body={"top_p": 0.8, "chat_template_kwargs": {"extra": 1}},
        )
        await client.complete(MESSAGES)
        assert recorder.body["top_p"] == 0.8
        assert recorder.body["chat_template_kwargs"] == {"enable_thinking": False, "extra": 1}

    async def test_api_key_header_only_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = Recorder()
        client = make_client(recorder)
        await client.complete(MESSAGES)
        assert "authorization" not in recorder.requests[-1].headers

        monkeypatch.setenv("RT_AGENT_LLM_API_KEY", "sk-test")
        await client.complete(MESSAGES)
        assert recorder.requests[-1].headers["authorization"] == "Bearer sk-test"

    async def test_an_empty_message_list_is_refused(self) -> None:
        with pytest.raises(LLMUsageError):
            await make_client(Recorder()).complete([])


class TestResponseHandling:
    async def test_happy_path(self) -> None:
        text = await make_client(Recorder()).complete(MESSAGES)
        assert text == "Hello there."

    async def test_reasoning_is_stripped_even_when_the_switch_was_ignored(self) -> None:
        recorder = Recorder(
            httpx.Response(200, json=completion("<think>\nlong ponder\n</think>\n\nIt is four."))
        )
        assert await make_client(recorder).complete(MESSAGES) == "It is four."

    async def test_an_unclosed_reasoning_block_leaves_nothing(self) -> None:
        recorder = Recorder(httpx.Response(200, json=completion("<think> still thinking")))
        with pytest.raises(LLMEmptyResponseError):
            await make_client(recorder).complete(MESSAGES)

    @pytest.mark.parametrize(
        ("payload", "error"),
        [
            ({"choices": []}, LLMProtocolError),
            ({"error": {"message": "no model loaded"}}, LLMProtocolError),
            ({"choices": [{"message": {"role": "assistant"}}]}, LLMEmptyResponseError),
            ({"choices": [{"message": {"content": 7}}]}, LLMProtocolError),
        ],
    )
    async def test_malformed_bodies_are_typed_errors(
        self, payload: dict[str, Any], error: type[Exception]
    ) -> None:
        recorder = Recorder(httpx.Response(200, json=payload))
        with pytest.raises(error):
            await make_client(recorder).complete(MESSAGES)

    async def test_non_json_body(self) -> None:
        recorder = Recorder(httpx.Response(200, text="<html>nope</html>"))
        with pytest.raises(LLMProtocolError):
            await make_client(recorder).complete(MESSAGES)


class TestFailures:
    async def test_timeout_is_typed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(LLMTimeoutError):
            await make_client(handler, timeout_s=0.2).complete(MESSAGES)

    async def test_transport_failure_is_typed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMUnavailableError):
            await make_client(handler).complete(MESSAGES)

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (400, LLMUsageError),
            (401, LLMAuthError),
            (403, LLMAuthError),
            (404, LLMUsageError),
            (429, LLMRateLimitError),
            (500, LLMUnavailableError),
            (503, LLMUnavailableError),
        ],
    )
    async def test_http_status_mapping(self, status: int, error: type[Exception]) -> None:
        recorder = Recorder(httpx.Response(status, text="nope"))
        with pytest.raises(error):
            await make_client(recorder).complete(MESSAGES)

    async def test_rate_limit_keeps_retry_after(self) -> None:
        recorder = Recorder(httpx.Response(429, text="slow down", headers={"Retry-After": "2.5"}))
        with pytest.raises(LLMRateLimitError) as caught:
            await make_client(recorder).complete(MESSAGES)
        assert caught.value.retry_after_s == 2.5

    async def test_close_is_safe_for_an_injected_client(self) -> None:
        client = make_client(Recorder())
        await client.complete(MESSAGES)
        await client.aclose()
        await client.aclose()
