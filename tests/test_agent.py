"""Tests for llm_bot.agent.Agent and llm_bot.agent.Session."""

from __future__ import annotations

import httpx
import pytest

from llm_bot.agent import Agent
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig


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
    """Minimal ModelStore returning a fixed config, independent of YAML files."""

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


def test_agent_build_messages_prepends_system_prompt():
    agent = Agent(_agent_config(), client=object())  # type: ignore[arg-type]
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    messages = agent.build_messages(history)
    assert messages[0] == {"role": "system", "content": "Ты помощник."}
    assert messages[1:] == history


def test_agent_build_messages_without_system_prompt():
    config = AgentConfig(name="bare", model="openai")
    agent = Agent(config, client=object())  # type: ignore[arg-type]
    history = [{"role": "user", "content": "hello"}]
    assert agent.build_messages(history) == history


def _make_session(session_id, agent_cfg, transport, directory):
    model_cfg = ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )
    agent_store = _StubAgentStore(agent_cfg)
    model_store = _StubModelStore(model_cfg)
    session_store = JsonSessionStore(directory)
    return make_session(
        session_id,
        agent_cfg.name,
        model_store=model_store,
        agent_store=agent_store,
        session_store=session_store,
        transport=transport,
        invariants=False,
    )


def test_session_chat_grows_history_and_sends_full_stack(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    transport = httpx.MockTransport(handler)
    session = _make_session("s1", _agent_config(), transport, str(tmp_path / "sessions"))

    result = session.chat("привет")

    assert result == "reply-text"
    # user + assistant recorded in memory
    assert session.history == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "reply-text"},
    ]
    # persisted
    assert session._store.load("s1") == session.history

    # the stack sent to the LLM includes the system prompt and full history
    payload = captured["payload"]
    assert '"role":"system"' in payload
    assert '"role":"user"' in payload
    assert '"content":"привет"' in payload
    assert '"temperature":0.5' in payload
    assert '"max_tokens":64' in payload


def test_session_history_persists_across_instances(tmp_path):
    directory = str(tmp_path / "sessions")

    transport = httpx.MockTransport(lambda request: _ok_response())
    first = _make_session("s1", _agent_config(), transport, directory)
    first.chat("вопрос")

    second = _make_session("s1", _agent_config(), transport, directory)
    assert second.history == first.history


def test_two_sessions_share_agent_but_independent_histories(tmp_path):
    directory = str(tmp_path / "sessions")
    transport = httpx.MockTransport(lambda request: _ok_response())

    a = _make_session("a", _agent_config(), transport, directory)
    b = _make_session("b", _agent_config(), transport, directory)

    a.chat("только в a")
    assert a.history[0]["content"] == "только в a"
    assert b.history == []


def test_session_rejects_empty_message(tmp_path):
    transport = httpx.MockTransport(lambda request: _ok_response())
    session = _make_session("s1", _agent_config(), transport, str(tmp_path / "sessions"))
    with pytest.raises(ValueError):
        session.chat("   ")

# --------------------------------------------------------------------------- #
# What the retrieval is told the conversation is about
# --------------------------------------------------------------------------- #

PLANE_ANSWER = """Да, полёт на самолёте возможен. Сертификат «Чудеса на виражах» [4].
Источники (фрагменты из ответа):
[3] Чудеса на виражах.pdf:17-26 — Чудеса на виражах · chunk fixed_size-011-002
[4] Чудеса на виражах.pdf:1-10 — Чудеса на виражах · chunk fixed_size-011-000"""

ALTERNATIVES_ANSWER = """В предоставленных документах нет информации о минимальном возрасте.
* Дайвинг: от 14 лет [1]
* Прыжок в тандеме: от 16 лет [2]
Источники (фрагменты из ответа):
[1] Урок дайвинга_открытая вода.pdf:18-28 — дайвинг · chunk fixed_size-008-002
[2] Прыжок в тандеме.pdf:19-29 — тандем · chunk fixed_size-005-002"""


def test_a_decline_with_alternatives_does_not_become_the_subject(tmp_path):
    """Real dialogue, three turns, and the miss explained exactly.

    «а с какого возраста можно?» — the age limit genuinely is not in the plane
    document, so declining was right. The answer then listed what four *other*
    certificates say, and those citations were read as the subject. The next
    question was «а вес какой может быть?»; the weight limit *is* in the plane
    document, but by then the preference pointed at diving and tandem jumps, and
    the answer came back from those instead — including «Дайвинг: до 100 кг»,
    which no document in the corpus mentions.

    The mechanism reads the newest citations to work out the subject, and those
    two things are the same except exactly when the bot declines. The more useful
    a decline is, the faster the conversation is steered away from what it is
    about.
    """
    session = _make_session("s", _agent_config(), None, tmp_path)
    session._history = [
        {"role": "user", "content": "полетать на самолете можно?"},
        {"role": "assistant", "content": PLANE_ANSWER},
        {"role": "user", "content": "а с какого возраста можно?"},
        {"role": "assistant", "content": ALTERNATIVES_ANSWER},
    ]

    assert session._rag_subject() == ["Чудеса на виражах.pdf"]


def test_the_subject_still_follows_a_normal_answer(tmp_path):
    """The skip is scoped to declines. On an ordinary turn the newest citation is
    the best evidence of what is being discussed."""
    session = _make_session("s", _agent_config(), None, tmp_path)
    answered = ALTERNATIVES_ANSWER.replace(
        "В предоставленных документах нет информации о минимальном возрасте.",
        "Возраст участников — от 14 лет.",
    )
    assert "нет информации" not in answered
    session._history = [
        {"role": "assistant", "content": PLANE_ANSWER},
        {"role": "assistant", "content": answered},
    ]

    assert session._rag_subject() == [
        "Урок дайвинга_открытая вода.pdf", "Прыжок в тандеме.pdf",
    ]


def test_an_unverified_quote_does_not_warn_by_default(caplog):
    """Real terminal output: the quoted sentence, quoted back in full, as a
    WARNING — printed above the answer that already says
    «Формулировки проверены частично: 3 цитаты не совпали».

    The reader got the same fact twice, and the first copy was a screenful of
    prose in the middle of a chat. Which text failed is a diagnostic, so it lives
    behind -v; the count stays in the answer where the reader is already looking.
    """
    import inspect
    import logging

    from llm_bot import agent

    source = inspect.getsource(agent.Session)
    quoted = source[source.index("final.quotes.dropped"):]
    quoted = quoted[:quoted.index(")")] if ")" in quoted else quoted
    assert "logger.debug(" in quoted, quoted

    # And nothing in that block is still a warning.
    block = source[source.index("if final.quotes is not None and final.quotes.dropped"):
                   source.index("if final.grounding.status is")]
    assert "logger.warning" not in block
