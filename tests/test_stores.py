"""Tests for the storage backends: YAML model/agent stores and JSON session store."""

from __future__ import annotations

import os

import pytest

from llm_bot.json_session_store import JsonSessionStore
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore


MODELS_YAML = """\
models:
  openai:
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini
  local:
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: ""
    model: llama3.2
"""

AGENTS_YAML = """\
agents:
  assistant:
    model: openai
    system_prompt: "Ты помощник."
    temperature: 0.7
    max_tokens: 1024
  translator:
    model: openai
    system_prompt: "Переводи."
    temperature: 0.2
"""


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_yaml_model_store_resolves_env_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    path = _write(tmp_path, "models.yaml", MODELS_YAML)

    store = YamlModelStore(path)
    assert store.list() == ["local", "openai"]

    model = store.get("openai")
    assert model.name == "openai"
    assert model.base_url == "https://api.openai.com/v1"
    assert model.api_key == "secret-key"
    assert model.provider == "openai"

    local = store.get("local")
    assert local.api_key == ""
    assert local.model == "llama3.2"


def test_yaml_model_store_missing_env_resolves_empty(tmp_path):
    path = _write(tmp_path, "models.yaml", MODELS_YAML)
    store = YamlModelStore(path)
    model = store.get("openai")
    assert model.api_key == ""  # OPENAI_API_KEY not set in this test


def test_yaml_agent_store(tmp_path):
    path = _write(tmp_path, "agents.yaml", AGENTS_YAML)
    store = YamlAgentStore(path)
    assert store.list() == ["assistant", "translator"]

    agent = store.get("assistant")
    assert agent.model == "openai"
    assert agent.system_prompt == "Ты помощник."
    assert agent.temperature == 0.7
    assert agent.max_tokens == 1024


def test_yaml_store_unknown_key_raises(tmp_path):
    path = _write(tmp_path, "agents.yaml", AGENTS_YAML)
    store = YamlAgentStore(path)
    with pytest.raises(KeyError):
        store.get("nope")


def test_yaml_store_missing_file_raises(tmp_path):
    store = YamlModelStore(str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        store.list()


def test_json_session_store_roundtrip(tmp_path):
    store = JsonSessionStore(str(tmp_path / "sessions"))
    assert store.load("s1") == []

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    store.save("s1", history)
    assert store.load("s1") == history
    assert store.list() == ["s1"]


def test_json_session_store_invalid_id(tmp_path):
    store = JsonSessionStore(str(tmp_path / "sessions"))
    with pytest.raises(ValueError):
        store.save("../evil", [])