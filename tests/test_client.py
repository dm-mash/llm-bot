"""Tests for llm_bot.client.LLMClient using httpx.MockTransport."""

from __future__ import annotations

import pytest
import httpx

from llm_bot.client import (
    LLMClient,
    LLMRequestError,
    LLMRetryExhaustedError,
)
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import DetailListener, RequestDetails, ResponseDetails


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": "Hello from the LLM!"}}
            ]
        },
    )


class _RecordingListener:
    """DetailListener that collects emitted events for assertions."""

    def __init__(self) -> None:
        self.requests: list[RequestDetails] = []
        self.responses: list[ResponseDetails] = []

    def on_request(self, details: RequestDetails) -> None:
        self.requests.append(details)

    def on_response(self, details: ResponseDetails) -> None:
        self.responses.append(details)


def _make_client(handler, listener: _RecordingListener | None = None) -> LLMClient:
    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,  # no real sleeping in tests
    )
    transport = httpx.MockTransport(handler)
    return LLMClient(config, transport=transport, detail_listener=listener)


def test_send_prompt_returns_text():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        assert request.url == "https://example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = request.read().decode()
        assert '"model":"test-model"' in payload
        assert '"content":"Hi there"' in payload
        # No system prompt configured by default.
        assert '"role":"system"' not in payload
        return _ok_response()

    client = _make_client(handler)
    result = client.send_prompt("Hi there")

    assert result == "Hello from the LLM!"
    assert calls["count"] == 1


def test_system_prompt_is_prepended_when_configured():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        system_prompt="Respond with strict JSON: {\"answer\": string}",
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    payload = captured["payload"]
    assert '"role":"system"' in payload
    assert '"role":"user"' in payload
    # The system message must precede the user message.
    assert payload.index('"role":"system"') < payload.index('"role":"user"')
    assert "Respond with strict JSON" in payload
    assert '"content":"Hi there"' in payload


def test_max_response_words_adds_briefness_instruction():
    """Setting max_response_words must inject a briefness hint into the system prompt."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        max_response_words=50,
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    assert '"role":"system"' in captured["payload"]
    assert "не более примерно 50 слов" in captured["payload"]


def test_max_response_words_combines_with_existing_system_prompt():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        system_prompt="Respond as a helpful assistant.",
        max_response_words=20,
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    payload = captured["payload"]
    assert "Respond as a helpful assistant." in payload
    assert "не более примерно 20 слов" in payload


def test_default_system_prompt_is_prepended_to_specific_prompt():
    """default_system_prompt must be prepended to any other system prompt."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        default_system_prompt="Отвечай на языке запроса.",
        system_prompt="Respond as a helpful assistant.",
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    payload = captured["payload"]
    assert '"role":"system"' in payload
    assert "Отвечай на языке запроса." in payload
    assert "Respond as a helpful assistant." in payload
    # The default prompt must come before the specific one.
    assert payload.index("Отвечай на языке запроса.") < payload.index("Respond as a helpful assistant.")


def test_default_system_prompt_combines_with_specific_and_briefness():
    """default_system_prompt, system_prompt and the briefness hint are all joined."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        default_system_prompt="Отвечай на языке запроса.",
        system_prompt="Respond as a helpful assistant.",
        max_response_words=15,
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    payload = captured["payload"]
    assert "Отвечай на языке запроса." in payload
    assert "Respond as a helpful assistant." in payload
    assert "не более примерно 15 слов" in payload
    assert payload.index("Отвечай на языке запроса.") < payload.index("Respond as a helpful assistant.")


def test_max_tokens_is_never_sent():
    """The client must not rely on the unreliable API-level max_tokens parameter."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,
        max_response_words=50,
    )
    transport = httpx.MockTransport(handler)
    client = LLMClient(config, transport=transport)

    client.send_prompt("Hi there")

    assert "max_tokens" not in captured["payload"]


def test_retry_then_success():
    """A transient 503 followed by a 200 should succeed after one retry."""
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] < 3:
            return httpx.Response(503, text="Service Unavailable")
        return _ok_response()

    client = _make_client(handler)
    result = client.send_prompt("ping")

    assert result == "Hello from the LLM!"
    assert state["attempts"] == 3


def test_retries_exhausted_raises():
    """Persistent transient errors should raise LLMRetryExhaustedError."""
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        return httpx.Response(503, text="Service Unavailable")

    client = _make_client(handler)

    with pytest.raises(LLMRetryExhaustedError):
        client.send_prompt("ping")

    # initial attempt + max_retries retries = 3 total
    assert state["attempts"] == 3


def test_rate_limit_is_retried_then_success():
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] == 1:
            return httpx.Response(429, text="Too Many Requests")
        return _ok_response()

    client = _make_client(handler)
    assert client.send_prompt("ping") == "Hello from the LLM!"
    assert state["attempts"] == 2


def test_network_error_is_retried():
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise httpx.ConnectError("connection refused")
        return _ok_response()

    client = _make_client(handler)
    assert client.send_prompt("ping") == "Hello from the LLM!"
    assert state["attempts"] == 2


def test_permanent_http_error_raises_without_retry():
    """A 401 (auth failure) is permanent and must not be retried."""
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        return httpx.Response(401, text="Invalid API key")

    client = _make_client(handler)

    with pytest.raises(LLMRequestError, match="401"):
        client.send_prompt("ping")

    assert state["attempts"] == 1


def test_unexpected_response_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _make_client(handler)

    with pytest.raises(LLMRequestError, match="Unexpected response shape"):
        client.send_prompt("ping")


def test_config_overrides():
    base = LLMConfig(base_url="https://a", api_key="k", model="m")
    overridden = base.with_overrides(model="new-model")
    assert overridden.model == "new-model"
    assert overridden.base_url == "https://a"
    assert overridden.api_key == "k"
    # original untouched
    assert base.model == "m"


def test_detail_listener_receives_request_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello"}}
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )

    listener = _RecordingListener()
    client = _make_client(handler, listener)
    assert client.send_prompt("Hi there") == "Hello"

    assert len(listener.requests) == 1
    req = listener.requests[0]
    assert req.method == "POST"
    assert req.url == "https://example.test/v1/chat/completions"
    assert req.model == "test-model"
    assert req.payload["model"] == "test-model"
    assert req.payload["messages"] == [{"role": "user", "content": "Hi there"}]

    assert len(listener.responses) == 1
    resp = listener.responses[0]
    assert resp.status_code == 200
    assert resp.attempt == 1
    assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert resp.elapsed_ms is not None and resp.elapsed_ms >= 0


def test_detail_listener_reports_each_attempt():
    state = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["attempts"] += 1
        if state["attempts"] == 1:
            return httpx.Response(503, text="Service Unavailable")
        return _ok_response()

    listener = _RecordingListener()
    client = _make_client(handler, listener)
    assert client.send_prompt("ping") == "Hello from the LLM!"

    # 503 attempt then a successful 200 attempt.
    assert [r.status_code for r in listener.responses] == [503, 200]
    assert [r.attempt for r in listener.responses] == [1, 2]
    # Only one request detail is emitted per send_prompt.
    assert len(listener.requests) == 1


def test_detail_listener_usage_is_none_when_absent():
    listener = _RecordingListener()
    client = _make_client(lambda request: _ok_response(), listener)
    client.send_prompt("ping")

    assert len(listener.responses) == 1
    assert listener.responses[0].usage is None