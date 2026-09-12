"""Tests for token accounting: counting, dialog growth, and context overflow."""

from __future__ import annotations

import httpx
import pytest

from llm_bot.agent import Agent
from llm_bot.client import (
    LLMClient,
    LLMConfig,
    ContextOverflowError,
    ContextTooLargeError,
    _classify_request_too_large,
)
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.tokens import (
    count_message_tokens,
    count_messages_tokens,
    estimate_tokens,
    merge_provider_usage,
    TokenUsage,
)


def _agent_config(*, max_tokens: int | None = 64) -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
        temperature=0.5,
        max_tokens=max_tokens,
    )


def _model_config(
    *,
    context_window: int | None = None,
    max_request_tokens: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
        context_window=context_window,
        max_request_tokens=max_request_tokens,
    )


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name):
        return self._config

    def list(self):
        return [self._config.name]


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name):
        return self._config

    def list(self):
        return [self._config.name]


def _session(transport, tmp_path, *, agent_cfg=None, model_cfg=None):
    agent_cfg = agent_cfg or _agent_config()
    model_cfg = model_cfg or _model_config()
    session_store = JsonSessionStore(str(tmp_path / "sessions"))
    return make_session(
        "s1",
        agent_cfg.name,
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(agent_cfg),
        session_store=session_store,
        transport=transport,
    )


# --- Estimation helpers ---------------------------------------------------- #


def test_estimate_tokens_is_deterministic_and_scales():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    # Longer text counts more tokens than shorter text.
    short = estimate_tokens("привет")
    long = estimate_tokens("привет " * 100)
    assert long > short


def test_count_message_includes_per_message_overhead():
    # 8 chars -> 2 content tokens + 4 overhead = 6.
    assert count_message_tokens({"role": "user", "content": "abcdefgh"}) == 6


def test_count_messages_sums_all():
    messages = [
        {"role": "system", "content": "abcd"},
        {"role": "user", "content": "abcd"},
        {"role": "assistant", "content": "abcd"},
    ]
    assert count_messages_tokens(messages) == 3 * (4 + 1)


# --- Merge with provider usage --------------------------------------------- #


def test_merge_provider_usage_overrides_estimates():
    usage = TokenUsage(
        request_tokens=5,
        history_tokens=20,
        context_tokens=25,
        reply_tokens=30,
        total_tokens=55,
        context_window=100,
        estimated=True,
    )
    merged = merge_provider_usage(
        usage,
        {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    )
    assert merged.context_tokens == 40
    assert merged.reply_tokens == 12
    assert merged.total_tokens == 52
    assert merged.estimated is False
    # Request/history breakdown preserved even when provider only gives totals.
    assert merged.request_tokens == 5
    assert merged.history_tokens == 20


def test_merge_provider_usage_noop_when_absent():
    usage = TokenUsage(context_tokens=7, context_window=10, estimated=True)
    assert merge_provider_usage(usage, None) is usage


# --- Dialog growth ---------------------------------------------------------- #


def test_context_tokens_grow_as_history_accumulates(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    session = _session(httpx.MockTransport(handler), tmp_path)

    first = session.chat_with_details("привет")
    second = session.chat_with_details("ещё вопрос")

    # The second turn re-sends the whole history, so its context is strictly
    # larger than the first turn's.
    assert second.usage.context_tokens > first.usage.context_tokens
    assert second.usage.history_tokens > first.usage.history_tokens
    # Both sends carried the full stack (system + growing history).
    assert captured["body"].count('"role"') >= 2

    # Request tokens reflect only the current user message.
    assert second.usage.request_tokens > 0
    assert second.usage.reply_tokens > 0


def test_reply_tokens_reported_from_provider_usage(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {
                    "prompt_tokens": 99,
                    "completion_tokens": 11,
                    "total_tokens": 110,
                },
            },
        )

    session = _session(httpx.MockTransport(handler), tmp_path)
    result = session.chat_with_details("вопрос")
    assert result.usage.estimated is False
    assert result.usage.context_tokens == 99
    assert result.usage.reply_tokens == 11
    assert result.usage.total_tokens == 110


# --- Overflow --------------------------------------------------------------- #


def test_overflow_raises_before_sending(tmp_path):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    # Tiny window: a clearly oversized message overflows.
    session = _session(
        httpx.MockTransport(handler),
        tmp_path,
        model_cfg=_model_config(context_window=20),
    )

    with pytest.raises(ContextOverflowError) as excinfo:
        session.chat_with_details("слишком длинное сообщение " * 10)

    # The exception carries the failing estimate and the configured window so
    # callers can report/recover without parsing the message text.
    assert excinfo.value.context_window == 20
    assert excinfo.value.context_tokens > 20

    # Nothing was sent to the model and nothing was persisted.
    assert sent == []
    assert session.history == []
    # The message was not appended either.
    assert session.last_usage is None


def test_overflow_boundary_fits(tmp_path):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    # Huge window: the message always fits.
    session = _session(
        httpx.MockTransport(handler),
        tmp_path,
        model_cfg=_model_config(context_window=100000),
    )
    result = session.chat_with_details("привет")
    assert result.usage.overflow is False
    assert sent  # it was actually sent


def test_overflow_flag_reflects_window(tmp_path):
    # Directly verify the flag on a usage snapshot without a network round-trip.
    usage = TokenUsage(
        context_tokens=150, context_window=100, estimated=True
    )
    assert usage.overflow is True
    assert usage.fill_percent == pytest.approx(150.0)


def test_agent_exposes_context_window():
    class _StubClient:
        def __init__(self, window):
            self.config = _model_config(context_window=window)

    agent = Agent(_agent_config(), client=_StubClient(4096))  # type: ignore[arg-type]
    assert agent.context_window == 4096


# --- Hard per-request size ceiling (e.g. Groq TPM 413) --------------------- #


def test_classify_groq_oversize_413():
    body = (
        '{"error":{"message":"Request too large for model `x` on tokens per '
        'minute (TPM): Limit 8000, Requested 8104, please reduce your message '
        'size","type":"tokens","code":"rate_limit_exceeded"}}'
    )
    exc = _classify_request_too_large(413, body)
    assert isinstance(exc, ContextTooLargeError)
    assert exc.status_code == 413
    assert exc.requested_tokens == 8104
    assert exc.limit_tokens == 8000


def test_classify_non_oversize_413_is_none():
    # A 413 that does not report requested>limit should NOT become a size error.
    assert _classify_request_too_large(413, "Payload too large") is None
    # A non-413 status is never classified as a size error.
    assert _classify_request_too_large(400, "bad request") is None


def test_chat_raises_context_too_large_on_groq_413(tmp_path):
    """A Groq-style 413 (requested > limit) becomes a non-retryable error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            413,
            json={
                "error": {
                    "message": (
                        "Request too large ... on tokens per minute (TPM): "
                        "Limit 8000, Requested 8104, please reduce ..."
                    ),
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    client = LLMClient(
        LLMConfig(
            base_url="https://example.test/v1", api_key="k", model="gpt"
        ),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ContextTooLargeError) as excinfo:
        client.chat([{"role": "user", "content": "hello"}])
    assert excinfo.value.requested_tokens == 8104
    assert excinfo.value.limit_tokens == 8000


def test_agent_preflight_blocks_over_account_ceiling(tmp_path):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    # Account ceiling far below the model window: the binding constraint.
    session = _session(
        httpx.MockTransport(handler),
        tmp_path,
        model_cfg=_model_config(
            context_window=100000, max_request_tokens=20
        ),
    )
    with pytest.raises(ContextTooLargeError) as excinfo:
        session.chat_with_details("слишком длинное сообщение " * 10)
    assert excinfo.value.limit_tokens == 20
    assert sent == []  # nothing was sent; refused up front
    assert session.history == []