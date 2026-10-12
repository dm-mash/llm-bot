"""Tests for llm_bot.server: the HTTP service in front of an agent.

Everything runs offline: the app is exercised through httpx's ASGI transport and
the session factory returns a stub -- no stores, no model, no network.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_bot.client import ContextOverflowError, LLMRequestError
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.server import create_app
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.tokens import ChatResult, TokenUsage

URL = "/v1/chat/completions"


class _Engine:
    """A stand-in model: records calls and can sleep, fail or truncate."""

    def __init__(self, *, reply="reply-text", exc=None, sleep=0.0, truncated=False):
        self.reply = reply
        self.exc = exc
        self.sleep = sleep
        self.truncated = truncated
        self.calls: list[tuple[str, str]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def handle(self, session_id: str, text: str) -> ChatResult:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.calls.append((session_id, text))
        try:
            if self.sleep:
                time.sleep(self.sleep)
            if self.exc is not None:
                raise self.exc
            usage = TokenUsage(
                request_tokens=5,
                history_tokens=10,
                context_tokens=15,
                reply_tokens=7,
                total_tokens=22,
                context_window=8192,
            )
            return ChatResult(reply=self.reply, usage=usage, truncated=self.truncated)
        finally:
            with self._lock:
                self.concurrent -= 1


class _StubSession:
    def __init__(self, engine: _Engine, session_id: str) -> None:
        self.engine = engine
        self.session_id = session_id

    def chat_with_details(
        self, user_message: str, *, service_turn: bool = False
    ) -> ChatResult:
        return self.engine.handle(self.session_id, user_message)


def _app(engine: _Engine, **kwargs):
    return create_app(
        session_factory=lambda session_id: _StubSession(engine, session_id),
        agent_name="assistant",
        **kwargs,
    )


def _payload(text: str = "привет", session_id: str | None = None) -> dict:
    body = {"messages": [{"role": "user", "content": text}]}
    if session_id is not None:
        body["session_id"] = session_id
    return body


def _post(app, payload, **kwargs):
    with TestClient(app) as client:
        return client.post(URL, json=payload, **kwargs)


def test_health_reports_agent_and_queue():
    app = _app(_Engine(), parallel=2, health_probe=lambda: {"reachable": True})
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agent"] == "assistant"
    assert body["parallel"] == 2
    assert body["queue"] == {"queued": 0, "in_flight": 0}
    assert body["handled"] == 0
    assert body["errors"] == 0
    assert body["upstream"] == {"reachable": True}


def test_handled_counts_a_successful_turn_without_errors():
    app = _app(_Engine())
    assert _post(app, _payload("привет")).status_code == 200
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["handled"] == 1
    assert body["errors"] == 0


def test_a_failed_turn_counts_as_handled_and_errored():
    app = _app(_Engine(exc=LLMRequestError("upstream said no")))
    assert _post(app, _payload("привет")).status_code == 502
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["handled"] == 1
    assert body["errors"] == 1


def test_missing_or_wrong_token_is_rejected():
    app = _app(_Engine(), auth_token="secret")
    assert _post(app, _payload()).status_code == 401
    assert _post(app, _payload(), headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = _post(app, _payload(), headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_health_needs_no_token():
    app = _app(_Engine(), auth_token="secret")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_rate_limit_returns_429():
    app = _app(_Engine(), rate_limit=2)
    assert _post(app, _payload("a")).status_code == 200
    assert _post(app, _payload("b")).status_code == 200
    limited = _post(app, _payload("c"))
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_chat_returns_openai_shape_and_usage():
    engine = _Engine(reply="привет!")
    response = _post(_app(engine), _payload("здравствуй", session_id="s1"))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["session_id"] == "s1"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "привет!"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}
    assert engine.calls == [("s1", "здравствуй")]


def test_truncated_reply_marks_length():
    response = _post(_app(_Engine(truncated=True)), _payload())
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_context_overflow_returns_413():
    exc = ContextOverflowError("too big", context_tokens=9000, context_window=8192)
    response = _post(_app(_Engine(exc=exc)), _payload())
    assert response.status_code == 413
    assert "overflow" in response.json()["detail"].lower()


def test_missing_user_message_returns_400():
    response = _post(_app(_Engine()), {"messages": []})
    assert response.status_code == 400


def test_streaming_is_rejected():
    response = _post(_app(_Engine()), {**_payload(), "stream": True})
    assert response.status_code == 400


async def _gather(app, payloads):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.gather(
            *(client.post(URL, json=payload) for payload in payloads)
        )


def test_one_session_never_interleaves_turns():
    engine = _Engine(sleep=0.15)
    app = _app(engine, parallel=2)
    responses = asyncio.run(_gather(app, [_payload("a", "s"), _payload("b", "s")]))
    assert [r.status_code for r in responses] == [200, 200]
    assert engine.max_concurrent == 1


def test_overload_without_a_slot_returns_503():
    engine = _Engine(sleep=0.4)
    app = _app(engine, parallel=1, queue_timeout=0.05)
    responses = asyncio.run(_gather(app, [_payload("a", "s1"), _payload("b", "s2")]))
    assert sorted(r.status_code for r in responses) == [200, 503]


def test_request_timeout_returns_504():
    engine = _Engine(sleep=0.3)
    app = _app(engine, parallel=1, request_timeout=0.05)
    responses = asyncio.run(_gather(app, [_payload("a", "s")]))
    assert responses[0].status_code == 504


# --------------------------------------------------------------------------- #
# Wiring the real agent (no stub session), still offline via a mock transport
# --------------------------------------------------------------------------- #


class _StubStore:
    def __init__(self, config) -> None:
        self._config = config

    def get(self, name: str):
        return self._config

    def list(self):
        return [self._config.name]


def test_real_session_is_served_and_persisted(tmp_path):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "готово"}}]},
        )

    session_dir = str(tmp_path / "sessions")
    app = create_app(
        session_factory=lambda sid: make_session(
            sid,
            "assistant",
            model_store=_StubStore(
                ModelConfig(name="m", base_url="https://example.test/v1", api_key="k", model="gpt")
            ),
            agent_store=_StubStore(
                AgentConfig(name="assistant", model="m", system_prompt="Ты робот.")
            ),
            session_store=JsonSessionStore(session_dir),
            transport=httpx.MockTransport(handler),
            invariants=False,
        ),
        agent_name="assistant",
    )

    response = _post(app, _payload("привет", session_id="srv"))
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "готово"
    assert '"role":"system"' in captured["payload"]
    assert JsonSessionStore(session_dir).load("srv") == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "готово"},
    ]
