"""Tests for the layered agent memory (llm_bot.memory / llm_bot.memory_store).

Covers:
* three layers stored separately;
* explicit choice of which layer a value goes to (write_to / read_from);
* short-term is in-memory and per-session;
* working persists per-session and is never shared across sessions;
* long-term persists per-agent and is isolated per-owner (privacy);
* prefix rendering order;
* explicit per-turn extraction (extract_memory);
* Session integration (memory prefix injection + short-term mirroring).
"""

from __future__ import annotations

import httpx
import pytest

from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.memory import (
    LongTermMemory,
    MemoryEntry,
    MemoryLayers,
    ShortTermMemory,
    WorkingMemory,
    extract_memory,
)
from llm_bot.memory_store import JsonMemoryStore
from llm_bot.stores import AgentConfig, ModelConfig


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "reply-text"}}]},
    )


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
        temperature=0.5,
        max_tokens=64,
    )


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name: str) -> ModelConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name: str) -> AgentConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


def _make_session(session_id, agent_cfg, transport, directory, owner="default",
                  auto_extract=True):
    model_cfg = ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )
    agent_store = _StubAgentStore(agent_cfg)
    model_store = _StubModelStore(model_cfg)
    session_store = JsonSessionStore(str(directory / "sessions"))
    memory_store = JsonMemoryStore(str(directory / "memory"))
    return make_session(
        session_id,
        agent_cfg.name,
        model_store=model_store,
        agent_store=agent_store,
        session_store=session_store,
        memory_store=memory_store,
        owner_id=owner,
        memory_auto_extract=auto_extract,
        transport=transport,
        invariants=False,
    )


def _make_layers(tmp_path, session="s1", agent="assistant", owner="alice"):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    layers = MemoryLayers(
        short=ShortTermMemory(),
        working=WorkingMemory(store=store, session_id=session),
        long=LongTermMemory(store=store, agent=agent, owner=owner),
    )
    return layers, store


# --------------------------------------------------------------------------- #
# Layer separation and explicit targeting
# --------------------------------------------------------------------------- #


def test_three_layers_are_separate_objects(tmp_path):
    layers, _ = _make_layers(tmp_path)
    assert layers.short is not None
    assert layers.working is not layers.short
    assert layers.long is not layers.working
    # distinct storage semantics: working is keyed by session, long by agent+owner
    assert layers.working._session_id == "s1"
    assert layers.long._owner == "alice"


def test_explicit_write_and_read_per_layer(tmp_path):
    layers, _ = _make_layers(tmp_path)
    # write explicitly to each durable layer
    layers.write_to("working", "goal", "build feature X")
    layers.write_to("long", "name", "Alice")
    # read explicitly from the chosen layer
    assert layers.read_from("working", "goal") == "build feature X"
    assert layers.read_from("long", "name") == "Alice"
    # a value written to working is NOT visible in long
    assert layers.read_from("long", "goal") is None
    # short cannot be written via write_to (it stores messages)
    with pytest.raises(ValueError):
        layers.write_to("short", "k", "v")


def test_named_helpers_target_the_intended_layer(tmp_path):
    layers, _ = _make_layers(tmp_path)
    layers.remember_working("deadline", "tomorrow")
    layers.remember_long_term("preference", "concise")
    assert layers.working.recall("deadline") == "tomorrow"
    assert layers.long.recall("preference") == "concise"
    assert layers.long.recall("deadline") is None
    assert layers.working.recall("preference") is None


# --------------------------------------------------------------------------- #
# Short-term: in-memory, per-session
# --------------------------------------------------------------------------- #


def test_short_term_is_in_memory_only_and_holds_dialog():
    short = ShortTermMemory()
    short.push({"role": "user", "content": "hi"})
    short.push({"role": "assistant", "content": "hello"})
    assert len(short) == 2
    assert short.recent(1) == [{"role": "assistant", "content": "hello"}]
    assert short.messages()[0] == {"role": "user", "content": "hi"}
    # no persistence: clear wipes everything
    short.clear()
    assert len(short) == 0


# --------------------------------------------------------------------------- #
# Working: per-session persistent, not shared
# --------------------------------------------------------------------------- #


def test_working_persists_for_same_session(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    w1 = WorkingMemory(store=store, session_id="s1")
    w1.remember("goal", "ship v1")
    # a new WorkingMemory for the SAME session reloads it
    w2 = WorkingMemory(store=store, session_id="s1")
    assert w2.recall("goal") == "ship v1"


def test_working_is_not_shared_between_sessions(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    WorkingMemory(store=store, session_id="s1").remember("goal", "task-a")
    other = WorkingMemory(store=store, session_id="s2")
    assert other.recall("goal") is None


# --------------------------------------------------------------------------- #
# Long-term: per-agent, per-owner isolated (privacy)
# --------------------------------------------------------------------------- #


def test_long_term_persists_for_same_agent_and_owner(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    LongTermMemory(store=store, agent="assistant", owner="alice").remember(
        "name", "Alice"
    )
    reloaded = LongTermMemory(store=store, agent="assistant", owner="alice")
    assert reloaded.recall("name") == "Alice"


def test_long_term_is_isolated_between_owners(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    LongTermMemory(store=store, agent="assistant", owner="alice").remember(
        "name", "Alice"
    )
    # Bob's session must NOT see Alice's durable profile
    bob = LongTermMemory(store=store, agent="assistant", owner="bob")
    assert bob.recall("name") is None


def test_long_term_is_isolated_between_agents(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    LongTermMemory(store=store, agent="assistant", owner="alice").remember(
        "name", "Alice"
    )
    other_agent = LongTermMemory(store=store, agent="translator", owner="alice")
    assert other_agent.recall("name") is None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_prefix_messages_order_long_then_working(tmp_path):
    layers, _ = _make_layers(tmp_path)
    layers.remember_long_term("name", "Alice")
    layers.remember_working("goal", "build")
    prefix = layers.prefix_messages()
    assert len(prefix) == 2
    # long-term first, then working
    assert "Долговременная память" in prefix[0]["content"]
    assert "Рабочая память" in prefix[1]["content"]
    assert "name: Alice" in prefix[0]["content"]
    assert "goal: build" in prefix[1]["content"]


def test_prefix_messages_omits_empty_layers(tmp_path):
    layers, _ = _make_layers(tmp_path)
    assert layers.prefix_messages() == []
    layers.remember_working("goal", "build")
    prefix = layers.prefix_messages()
    assert len(prefix) == 1
    assert "Рабочая память" in prefix[0]["content"]


# --------------------------------------------------------------------------- #
# Explicit per-turn extraction
# --------------------------------------------------------------------------- #


def _chat_returns_classifier_output(output):
    calls = []

    def chat(messages):
        calls.append(messages)
        return output

    chat.calls = calls  # type: ignore[attr-defined]
    return chat


def test_extract_memory_classifies_into_layers(tmp_path):
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {"goal": {"value": "build x", "evidence": "задача X"}}, '
        '"long_term": {"name": {"value": "Alice", "evidence": "меня зовут Алиса"}}}'
    )
    event = extract_memory(
        layers,
        {"role": "user", "content": "задача X, меня зовут Алиса"},
        "reply",
        chat,
    )
    assert event.working_written == ["goal"]
    assert event.long_term_written == ["name"]
    assert event.rejected == []
    assert event.verified_rate == 1.0
    assert event.recognized is True
    assert event.total_tokens >= 0
    assert layers.working.recall("goal") == "build x"
    assert layers.long.recall("name") == "Alice"


def test_a_malformed_classifier_reply_is_reported_briefly(tmp_path, caplog):
    """The reply runs to hundreds of characters and the reason it failed is the
    interesting part. One real reply missed a closing brace, and the log printed
    six invented facts about the customer instead — noise that reads like a
    memory bug."""
    layers, _ = _make_layers(tmp_path)
    broken = ('{"working": {}, "long_term": {"a": {"value": "от 14 лет", '
              '"evidence": "от 14 лет"}, "b": {"value": "6 месяцев", '
              '"evidence": "Сертификат действует 6 месяцев"}}')

    with caplog.at_level("WARNING"):
        event = extract_memory(
            layers, {"role": "user", "content": "полетать можно?"}, "ответ",
            _chat_returns_classifier_output(broken),
        )

    assert event.recognized is False
    assert event.working_written == [] and event.long_term_written == []
    (line,) = [r for r in caplog.records if "не разобран" in r.message]
    assert len(line.getMessage()) < 220
    assert "JSONDecodeError" in line.getMessage()
    assert len(layers.long) == 0


def test_extract_memory_ignores_garbage(tmp_path):
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output("not json at all")
    event = extract_memory(layers, {"role": "user", "content": "hi"}, "ok", chat)
    assert event.working_written == []
    assert event.long_term_written == []
    assert event.recognized is False
    assert len(layers.working) == 0
    assert len(layers.long) == 0


# --------------------------------------------------------------------------- #
# Session integration
# --------------------------------------------------------------------------- #


def test_session_injects_memory_prefix_and_mirrors_short_term(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    # auto_extract=False so the captured payload is the main chat request, not
    # the extra memory-classifier call.
    transport = httpx.MockTransport(handler)
    session = _make_session(
        "s1", _agent_config(), transport, tmp_path, auto_extract=False
    )

    # pre-seed durable memory before chatting
    assert session.memory is not None
    session.memory.remember_long_term("name", "Alice")
    session.memory.remember_working("goal", "build")

    result = session.chat("привет")
    assert result == "reply-text"

    # short-term mirrors the dialog
    assert session.memory.short.messages() == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "reply-text"},
    ]
    # the request includes the durable memory prefix
    payload = captured["payload"]
    assert "Долговременная память" in payload
    assert "Рабочая память" in payload
    assert "name: Alice" in payload
    assert "goal: build" in payload


def test_long_term_carries_across_sessions_same_owner(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    # auto_extract=False so the captured payload is the main chat request.
    transport = httpx.MockTransport(handler)
    s1 = _make_session(
        "s1", _agent_config(), transport, tmp_path, owner="alice", auto_extract=False
    )
    s1.memory.remember_long_term("name", "Alice")
    s1.chat("remember me")

    # a NEW session, same agent + owner, sees Alice's profile injected
    s2 = _make_session(
        "s2", _agent_config(), transport, tmp_path, owner="alice", auto_extract=False
    )
    assert s2.memory.long.recall("name") == "Alice"
    s2.chat("hi again")
    assert "name: Alice" in captured["payload"]


def test_long_term_not_leaked_to_other_owner_session(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response()

    transport = httpx.MockTransport(handler)
    s_alice = _make_session("sa", _agent_config(), transport, tmp_path, owner="alice")
    s_alice.memory.remember_long_term("name", "Alice")
    s_alice.chat("hi")

    s_bob = _make_session("sb", _agent_config(), transport, tmp_path, owner="bob")
    # Bob's long-term memory is empty; Alice's data is not visible
    assert s_bob.memory.long.recall("name") is None
    assert s_bob.memory.long.owner == "bob"


def test_memory_entries_preserve_source(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "memory"))
    w = WorkingMemory(store=store, session_id="s1")
    w.remember("goal", "build", source="derive")
    reloaded = WorkingMemory(store=store, session_id="s1")
    entry = reloaded.snapshot()["goal"]
    assert isinstance(entry, MemoryEntry)
    assert entry.source == "derive"
    assert entry.value == "build"


def test_session_auto_extract_fills_working_and_long(tmp_path):
    # First request -> main chat reply; second request -> memory classifier JSON.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _ok_response()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant",
                                 "content": (
                                     '{"working": {"goal": {"value": "build x", '
                                     '"evidence": "построй X"}}, '
                                     '"long_term": {"name": {"value": "Alice", '
                                     '"evidence": "меня зовут Алиса"}}}'
                                 )}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    session = _make_session("s1", _agent_config(), transport, tmp_path)

    session.chat("построй X, меня зовут Алиса")
    # Auto-extract wrote to working and long-term layers explicitly.
    assert session.memory.working.recall("goal") == "build x"
    assert session.memory.long.recall("name") == "Alice"
    # Events are recorded with token accounting.
    assert session.total_memory_extractions == 1
    assert session.last_memory_event is not None
    assert session.last_memory_event.working_written == ["goal"]
    assert session.last_memory_event.long_term_written == ["name"]
    assert session.last_memory_event.recognized is True
    assert session.total_memory_tokens >= 0


def test_session_auto_extract_can_be_disabled(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response()

    transport = httpx.MockTransport(handler)
    session = _make_session(
        "s1", _agent_config(), transport, tmp_path, auto_extract=False
    )
    session.chat("hi")
    assert session.memory_auto_extract is False
    assert session.total_memory_extractions == 0
    assert session.last_memory_event is None
    assert len(session.memory.working) == 0
    assert len(session.memory.long) == 0

# --------------------------------------------------------------------------- #
# Verification: only the user's own words become durable facts
# --------------------------------------------------------------------------- #


def test_a_fact_the_user_never_stated_is_dropped(tmp_path):
    """The observed failure: small talk produced «аллергия на орехи» and «две
    недели», neither of which anybody said. Once written to the long-term layer
    they are injected into every later turn, so the classifier's word is all
    that ever stands between a polite invention and a permanent fact.
    """
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {}, "long_term": {'
        '"allergies": {"value": "орехи", "evidence": "аллергия на орехи"}, '
        '"tenure": {"value": "две недели", "evidence": "работаю две недели"}}}'
    )
    event = extract_memory(
        layers, {"role": "user", "content": "привет, как дела?"}, "отлично", chat
    )

    assert event.long_term_written == []
    assert event.rejected == [
        ("long_term:allergies", "доказательства нет в словах пользователя"),
        ("long_term:tenure", "доказательства нет в словах пользователя"),
    ]
    assert event.verified_rate == 0.0
    assert len(layers.long) == 0


def test_an_assistants_own_conclusion_is_not_a_user_fact(tmp_path):
    """The reply may well be correct. It is still the assistant talking, and a
    correct answer is not something the user asserted about themselves."""
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {}, "long_term": {'
        '"allergies": {"value": "орехи", "evidence": "аллергия на орехи"}}}'
    )
    event = extract_memory(
        layers,
        {"role": "user", "content": "что посоветуешь?"},
        "У гостя аллергия на орехи, не советуйте миндаль.",
        chat,
    )

    assert event.long_term_written == []
    assert event.rejected[0][0] == "long_term:allergies"
    assert len(layers.long) == 0


def test_a_bare_value_with_no_evidence_is_dropped(tmp_path):
    """The old shape must not slip through as a silent downgrade: a classifier
    answering with plain strings has not verified anything."""
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {"goal": "build x"}, "long_term": {"name": "Alice"}}'
    )
    event = extract_memory(
        layers, {"role": "user", "content": "задача X"}, "reply", chat
    )

    assert event.working_written == []
    assert event.long_term_written == []
    assert [reason for _, reason in event.rejected] == ["нет доказательства"] * 2
    assert len(layers.working) == 0
    assert len(layers.long) == 0


def test_evidence_may_reach_across_line_breaks(tmp_path):
    """The quote is copied out of a multi-line reply; only whitespace and case
    may differ from the message the user actually sent."""
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {}, "long_term": {'
        '"budget": {"value": "400", "evidence": "БЮДЖЕТ   —\\nдо 400 рублей"}}}'
    )
    event = extract_memory(
        layers,
        {"role": "user", "content": "Бюджет —\nдо 400 рублей на всё вместе."},
        "ok",
        chat,
    )

    assert event.long_term_written == ["budget"]
    assert event.rejected == []
    assert layers.long.recall("budget") == "400"


def test_verified_rate_reports_a_mixed_round(tmp_path):
    """Precision is only useful on the rounds where something was proposed —
    a silent classifier must not look perfect."""
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {"goal": {"value": "party", "evidence": "я устраиваю"}}, '
        '"long_term": {"name": {"value": "Alice", "evidence": "Alice"}}}'
    )
    event = extract_memory(
        layers, {"role": "user", "content": "я устраиваю день рождения"}, "ok", chat
    )

    assert event.long_term_written == []
    assert event.verified_rate == 0.5


def test_a_stored_fact_points_back_at_the_words_it_came_from(tmp_path):
    """Provenance is what makes a stored fact arguable: without it the only way
    to answer «why does it think I said that?» is to delete everything."""
    layers, _ = _make_layers(tmp_path)
    chat = _chat_returns_classifier_output(
        '{"working": {}, "long_term": {'
        '"budget": {"value": "400", "evidence": "до 400 рублей"}}}'
    )
    event = extract_memory(
        layers,
        {"role": "user", "content": "Бюджет — до 400 рублей"},
        "ok",
        chat,
        turn=7,
    )

    entry = layers.long.snapshot()["budget"]
    assert entry.evidence == "до 400 рублей"
    assert entry.turn == 7
    assert entry.source == "derive"
    assert event.long_term_written == ["budget"]
    assert event.rejected == []


def test_the_memory_block_says_it_is_not_a_source(tmp_path):
    """Memory reads as one more context, and the model quotes from it.

    Measured on a 14-turn coffee dialogue: with memory on, every answer came
    back wrapping the customer's own stored words in «…» and pointing source
    numbers at them. The quote audit rejected all of it — the system did not
    accept the lie — but the model was producing it on every single turn, and
    the audit cannot stop it from trying. Memory is what the user said, not a
    document, and the block has to say so where the model reads it.
    """
    layers, _ = _make_layers(tmp_path)
    layers.remember_working("budget", "до 400", evidence="до 400 рублей", turn=3)

    block = layers.prefix_messages()[0]["content"]

    assert "не источник" in block.lower()
    assert "не цитируй" in block.lower()
    # The rule has to sit next to the data, not in a separate system message the
    # model may never attend to.
    assert "до 400" in block
    assert block.index("не цитируй") < block.index("до 400")


def test_a_stored_fact_is_tagged_with_the_turn_it_came_from(tmp_path):
    """Provenance pointed one turn into the future.

    The tag was read off ``_history``, a message list, not a turn counter: it
    read 1 on turn 1, then 3 on turn 2 — so at turn 2 memory already held facts
    attributed to turn 3, a turn the user had not reached. A fact you cannot
    point back at is worse than one without a number, because the number looks
    authoritative.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # main reply
            return _ok_response()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": (
                        '{"working": {"fact": {"value": "yes", '
                        '"evidence": "проверка"}}, "long_term": {}}'
                    )}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    session = _make_session("s1", _agent_config(), transport, tmp_path)

    session.chat("проверка раз")
    assert session.memory.working.snapshot()["fact"].turn == 1

    session.chat("проверка два")
    session.chat("проверка три")

    # The tag must be the turn the words were said on, and must not run ahead
    # of the conversation.
    assert session.memory.working.snapshot()["fact"].turn == 3
    turns = [e.working_written for e in session._memory_events]
    assert turns == [["fact"], ["fact"], ["fact"]]
