"""Tests for the pluggable context-management strategies.

Covers the pure logic of :mod:`llm_bot.context_strategies` (sliding window,
sticky facts, branching) plus their integration through :class:`Session` using a
fake deterministic LLM that "sees" only what the strategy actually sends.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_bot.context_strategies import (
    Branching,
    SlidingWindow,
    StickyFacts,
    parse_facts,
)
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
        temperature=0.0,
        max_tokens=64,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
        context_window=1_000_000,
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


def _echo_handler(captured: list | None = None):
    """A fake LLM that records payloads and echoes a marker reply.

    The reply is constant (``"ok"``) — for recall-style checks we inspect
    ``captured`` request payloads directly rather than the reply.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    return handler


def _make_session(tmp_path, strategy=None, *, agent_cfg=None, session_id="s1"):
    """Build a session with the given pre-constructed strategy (or ``None``)."""
    captured = []
    store = JsonSessionStore(str(tmp_path / "sessions"))
    session = make_session(
        session_id,
        agent_cfg.name if agent_cfg else "assistant",
        model_store=_StubModelStore(_model_config()),
        agent_store=_StubAgentStore(agent_cfg or _agent_config()),
        session_store=store,
        transport=httpx.MockTransport(_echo_handler(captured)),
        strategy=strategy,
    )
    return session, captured


def _roles_of_last_request(captured) -> list[str]:
    return [m["role"] for m in captured[-1]["messages"]]


# --------------------------------------------------------------------------- #
# parse_facts
# --------------------------------------------------------------------------- #


def test_parse_facts_extracts_key_value_lines():
    text = "цель: создать бота\nограничения: без внешних сервисов\nмусорная строка"
    assert parse_facts(text, 20) == {
        "цель": "создать бота",
        "ограничения": "без внешних сервисов",
    }


def test_parse_facts_respects_max_and_prefers_newest():
    text = "\n".join(f"key{i}: v{i}" for i in range(5))
    facts = parse_facts(text, 3)
    assert len(facts) == 3
    # The last three keys survive; the oldest are dropped.
    assert list(facts) == ["key2", "key3", "key4"]


def test_parse_facts_lowercases_keys():
    assert parse_facts("Цель: бот", 10) == {"цель": "бот"}


# --------------------------------------------------------------------------- #
# Strategy 1: SlidingWindow
# --------------------------------------------------------------------------- #


def test_sliding_prepare_trims_request_but_keeps_history():
    s = SlidingWindow(window_size=4)
    for i in range(6):
        s.on_turn_end({"role": "user", "content": f"q{i}"}, f"a{i}")
    prepared = s.prepare({"role": "user", "content": "new"})
    # 12 recorded messages + the new user message = 13; only the last 4 go out.
    assert prepared.dropped == 9
    contents = [m["content"] for m in prepared.request_history]
    assert contents == ["a4", "q5", "a5", "new"]
    # Full history is still owned (not lost).
    assert len(s.history) == 12


def test_sliding_no_trim_below_window():
    s = SlidingWindow(window_size=10)
    s.on_turn_end({"role": "user", "content": "q0"}, "a0")
    prepared = s.prepare({"role": "user", "content": "q1"})
    assert prepared.dropped == 0
    assert len(prepared.request_history) == 3


def test_sliding_session_sends_only_window(tmp_path):
    strategy = SlidingWindow(window_size=4)
    session, captured = _make_session(tmp_path, strategy)
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")
    # The last request should contain the agent system prompt + the last 4 msgs
    # (dropped the earlier history), NOT the whole 16-message conversation.
    last = captured[-1]["messages"]
    user_contents = [m["content"] for m in last if m["role"] == "user"]
    assert len(user_contents) <= 4
    assert "вопрос 0" not in user_contents
    # Full history is still persisted on disk.
    assert len(session.history) == 16
    store = session._store
    assert len(store.load("s1")) == 16


def test_sliding_persists_and_resumes(tmp_path):
    strategy = SlidingWindow(window_size=4)
    session, _ = _make_session(tmp_path, strategy, session_id="sx")
    for i in range(6):
        session.chat_with_details(f"вопрос {i}")
    resumed, _ = _make_session(tmp_path, SlidingWindow(window_size=4), session_id="sx")
    assert len(resumed.history) == 12


# --------------------------------------------------------------------------- #
# Strategy 2: StickyFacts
# --------------------------------------------------------------------------- #


def test_sticky_facts_prepare_adds_prefix_block():
    strategy = StickyFacts(window_size=6, chat=lambda msgs: "цель: бот")
    strategy._facts = {"цель": "бот", "срок": "март"}
    prepared = strategy.prepare({"role": "user", "content": "hi"})
    assert len(prepared.prefix) == 1
    assert prepared.prefix[0]["role"] == "system"
    assert "цель: бот" in prepared.prefix[0]["content"]
    assert "срок: март" in prepared.prefix[0]["content"]


def test_sticky_facts_after_reply_updates_memory_via_chat():
    calls = []
    strategy = StickyFacts(
        window_size=6,
        chat=lambda msgs: (calls.append(msgs) or "цель: новый проект\nсрок: июнь"),
    )
    strategy.on_turn_end({"role": "user", "content": "ставим цель"}, "принято")
    assert strategy.facts == {"цель": "новый проект", "срок": "июнь"}
    assert len(calls) == 1
    # The transcript sent for fact-refresh contained the turn.
    assert "ставим цель" in calls[0][0]["content"]


def test_sticky_facts_survive_small_window_in_session(tmp_path):
    # window=4 so old turns drop out of the request, but facts persist via prefix.
    captured = []
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        # Distinguish the main-turn request from the facts-refresh call by the
        # presence of the agent system prompt ("Ты помощник").
        texts = " ".join(m.get("content", "") for m in payload["messages"])
        if "Ты помощник" in texts:
            captured.append(payload)
            content = "ok"
        else:
            # facts-refresh call: acknowledge current facts verbatim.
            content = "цель: сделать бота\nограничения: python"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": content}}]},
        )

    strategy = StickyFacts(window_size=4, chat=lambda msgs: "цель: сделать бота\nограничения: python")
    store = JsonSessionStore(str(tmp_path / "sessions"))
    session = make_session(
        "s1",
        "assistant",
        model_store=_StubModelStore(_model_config()),
        agent_store=_StubAgentStore(_agent_config()),
        session_store=store,
        transport=httpx.MockTransport(handler),
        strategy=strategy,
    )
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")

    # The final main-turn request still carries the durable facts prefix.
    last = captured[-1]["messages"]
    system_contents = [m["content"] for m in last if m["role"] == "system"]
    assert any("цель: сделать бота" in c for c in system_contents)
    # And the durable facts were persisted.
    assert session.strategy.facts["цель"] == "сделать бота"
    assert store.load_facts("s1")["ограничения"] == "python"


def test_sticky_facts_without_chat_keeps_memory_empty_but_works():
    strategy = StickyFacts(window_size=6, chat=None)
    strategy.on_turn_end({"role": "user", "content": "q"}, "a")
    assert strategy.facts == {}
    assert len(strategy.history) == 2


# --------------------------------------------------------------------------- #
# Strategy 3: Branching
# --------------------------------------------------------------------------- #


def test_branching_branch_creates_auto_checkpoint_and_becomes_active():
    s = Branching()
    s.on_turn_end({"role": "user", "content": "q0"}, "a0")
    s.on_turn_end({"role": "user", "content": "q1"}, "a1")
    s.branch("alt")
    assert s.current_branch == "alt"
    # New branch shares the trunk history (up to the checkpoint).
    assert s.branches["alt"] == s.branches["main"]
    # Continuing in 'alt' does not affect 'main'.
    s.on_turn_end({"role": "user", "content": "q-alt"}, "a-alt")
    assert len(s.branches["main"]) == 4
    assert len(s.branches["alt"]) == 6
    assert s.history[-2]["content"] == "q-alt"


def test_branching_switch_returns_to_a_sibling_branch():
    s = Branching()
    s.on_turn_end({"role": "user", "content": "q0"}, "a0")
    s.branch("b1")
    s.on_turn_end({"role": "user", "content": "in b1"}, "a1")
    s.switch("main")
    assert s.current_branch == "main"
    assert s.history[-2]["content"] == "q0"  # main stops at the checkpoint
    with pytest.raises(KeyError):
        s.switch("missing")


def test_branching_duplicate_name_rejected():
    s = Branching()
    s.branch("x")
    with pytest.raises(ValueError):
        s.branch("x")


def test_branching_session_persists_branches(tmp_path):
    strategy = Branching()
    session, _ = _make_session(tmp_path, strategy, session_id="sb")
    session.chat_with_details("общая идея")
    session.branch("web")
    session.chat_with_details("веб-вариант")

    # A fresh session on the same store resumes the branching state.
    resumed, _ = _make_session(tmp_path, Branching(), session_id="sb")
    assert set(resumed.strategy.branches) == {"main", "web"}
    assert resumed.strategy.current_branch == "web"
    # The 'web' branch holds the shared trunk (общая идея pair) + its own turn.
    assert len(resumed.history) == 4


def test_branching_session_isolates_sibling_branch(tmp_path):
    captured = []
    strategy = Branching()
    store = JsonSessionStore(str(tmp_path / "sessions"))
    session = make_session(
        "s1",
        "assistant",
        model_store=_StubModelStore(_model_config()),
        agent_store=_StubAgentStore(_agent_config()),
        session_store=store,
        transport=httpx.MockTransport(_echo_handler(captured)),
        strategy=strategy,
    )
    session.chat_with_details("q-trunk")
    session.branch("alt")
    session.chat_with_details("q-alt")
    # Switch back to the trunk branch and send another message there.
    session.switch_branch("main")
    session.chat_with_details("q-back-main")

    # The final request carries the trunk turns but NOT the sibling's divergent msg.
    contents = [m.get("content", "") for m in captured[-1]["messages"]]
    assert any("q-back-main" in c for c in contents)
    assert not any("q-alt" in c for c in contents)
    # The 'alt' branch kept its independent turn.
    assert any("q-alt" in m["content"] for m in session.strategy.branches["alt"])