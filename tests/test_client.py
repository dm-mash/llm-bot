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


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": "Hello from the LLM!"}}
            ]
        },
    )


def _make_client(handler) -> LLMClient:
    config = LLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        max_retries=2,
        retry_backoff=0.0,  # no real sleeping in tests
    )
    transport = httpx.MockTransport(handler)
    return LLMClient(config, transport=transport)


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