"""Tests for the GigaChat OAuth2 token provider."""

from __future__ import annotations

import base64

import httpx
import pytest

from llm_bot.client import LLMClient, LLMRequestError
from llm_bot.config import LLMConfig
from llm_bot.gigachat import GigaChatTokenProvider, build_gigachat_client


def _config(**kwargs) -> LLMConfig:
    defaults = {
        "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
        "model": "GigaChat-2-Max",
        "gigachat_client_id": "client-123",
        "gigachat_client_secret": "secret-abc",
        "timeout": 30.0,
    }
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def _oauth_response(token: str = "token-1", expires_in: int = 1800) -> httpx.Response:
    return httpx.Response(200, json={"access_token": token, "expires_in": expires_in})


class _Clock:
    """Controllable clock for testing token refresh."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_provider_fetches_token_with_client_credentials():
    clock = _Clock()
    oauth_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        oauth_calls["count"] += 1
        assert request.url == "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        assert request.headers["Content-Type"].startswith("application/x-www-form-urlencoded")
        assert request.headers["Authorization"] == "Basic " + base64.b64encode(
            b"client-123:secret-abc"
        ).decode()
        assert "RqUID" in request.headers
        body = request.read().decode()
        assert "grant_type=client_credentials" in body
        assert "scope=GIGACHAT_API_PERS" in body
        return _oauth_response()

    transport = httpx.MockTransport(handler)
    provider = GigaChatTokenProvider(_config(), transport=transport, time_fn=clock)

    assert provider.get_token() == "token-1"
    assert oauth_calls["count"] == 1


def test_provider_caches_token_within_expiry():
    clock = _Clock()
    oauth_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        oauth_calls["count"] += 1
        return _oauth_response(expires_in=1800)

    transport = httpx.MockTransport(handler)
    provider = GigaChatTokenProvider(_config(), transport=transport, time_fn=clock)

    assert provider.get_token() == "token-1"
    # Advance time but stay well before expiry.
    clock.now += 300
    assert provider.get_token() == "token-1"
    assert oauth_calls["count"] == 1


def test_provider_refreshes_after_expiry():
    clock = _Clock()
    oauth_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        oauth_calls["count"] += 1
        return _oauth_response(token=f"token-{oauth_calls['count']}", expires_in=1800)

    transport = httpx.MockTransport(handler)
    provider = GigaChatTokenProvider(_config(), transport=transport, time_fn=clock)

    assert provider.get_token() == "token-1"
    # Move past expiry (+ safety margin).
    clock.now += 2000
    assert provider.get_token() == "token-2"
    assert oauth_calls["count"] == 2


def test_provider_handles_expires_at_in_milliseconds():
    def handler(request: httpx.Request) -> httpx.Response:
        # expires_at given in milliseconds.
        return httpx.Response(200, json={"access_token": "tok", "expires_at": 1_001_800_000})

    transport = httpx.MockTransport(handler)
    clock = _Clock(start=1_000_000.0)
    provider = GigaChatTokenProvider(_config(), transport=transport, time_fn=clock)
    assert provider.get_token() == "tok"


def test_provider_raises_if_no_secret():
    config = _config(gigachat_client_secret="", gigachat_basic_auth="")
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="nope"))
    provider = GigaChatTokenProvider(config, transport=transport)

    with pytest.raises(LLMRequestError, match="client_secret"):
        provider.get_token()


def test_provider_raises_on_oauth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="invalid_grant")

    transport = httpx.MockTransport(handler)
    provider = GigaChatTokenProvider(_config(), transport=transport)

    with pytest.raises(LLMRequestError, match="400"):
        provider.get_token()


def test_build_gigachat_client_wires_token_provider():
    client = build_gigachat_client(_config())
    assert isinstance(client, LLMClient)
    assert isinstance(client._token_provider, GigaChatTokenProvider)