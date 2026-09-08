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