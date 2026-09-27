"""Agent + MCP integration: the agent calls an MCP tool and uses the result.

The LLM is mocked at the HTTP layer (``httpx.MockTransport``) with a scripted
sequence: first it answers with the ``{"call_tool": ...}`` directive, then —
after the tool result is fed back — with the final user-facing reply. The
MCP side is REAL: the notes server runs as a stdio child process, so the
full path (prompt -> directive -> MCP call -> result -> final answer) is
exercised end to end.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from llm_bot.agent import Session
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.mcp_tools import MCPToolBridge, MCPRouter
from llm_bot.stores import AgentConfig, ModelConfig

SERVER_MODULE = "llm_bot.mcp_servers.notes"


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


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник с доступом к CRM-заметкам.",
    )


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )


class _ScriptedLLM:
    """Mock transport returning canned replies in order, capturing payloads."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.payloads: list[dict] = []

    def as_transport(self) -> httpx.BaseTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.payloads.append(json.loads(request.read().decode()))
            reply = self._replies.pop(0)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": reply}}
                    ]
                },
            )

        return httpx.MockTransport(handler)


def _notes_router(db_path: Path) -> MCPRouter:
    """Router with one bridge to the real notes MCP server."""
    router = MCPRouter(
        [
            MCPToolBridge(
                "notes",
                command=sys.executable,
                args=["-m", SERVER_MODULE],
                env={**os.environ, "NOTES_DB": str(db_path)},
            )
        ]
    )
    router.refresh()
    return router


def _make_session(llm: _ScriptedLLM, tmp_path: Path, router: MCPRouter) -> Session:
    return make_session(
        "mcp-test",
        _agent_config().name,
        model_store=_StubModelStore(_model_cfg()),
        agent_store=_StubAgentStore(_agent_config()),
        session_store=JsonSessionStore(tmp_path / "sessions"),
        transport=llm.as_transport(),
        invariants=False,
        memory_store=None,
        mcp_router=router,
    )


def test_agent_calls_mcp_tool_and_uses_result(tmp_path):
    llm = _ScriptedLLM(
        [
            # Turn 1, request 1: the model decides it needs the tool.
            json.dumps(
                {
                    "call_tool": {
                        "name": "notes__add_note",
                        "arguments": {"text": "Позвонить клиенту", "tags": ["работа"]},
                    }
                },
                ensure_ascii=False,
            ),
            # Turn 1, request 2: final answer built from the tool result.
            "Готово: заметка «Позвонить клиенту» сохранена в CRM.",
        ]
    )
    router = _notes_router(tmp_path / "notes.json")
    session = _make_session(llm, tmp_path, router)

    result = session.chat_with_details("запиши заметку: позвонить клиенту, тег работа")

    # The final reply is the model's second answer (not the directive).
    assert "сохранена" in result.reply

    # The MCP round-trip is recorded with the correct routing.
    assert len(session.mcp_events) == 1
    event = session.mcp_events[0]
    assert event.ok and event.server == "notes" and event.tool == "add_note"
    assert "id=1" in event.result

    # The tool result actually reached the model: the second request carried
    # the directive echo + the result feedback, and the tools block was there.
    assert len(llm.payloads) == 2
    second = json.dumps(llm.payloads[1], ensure_ascii=False)
    assert "Результат инструмента notes__add_note" in second
    assert "Позвонить клиенту" in second
    first = json.dumps(llm.payloads[0], ensure_ascii=False)
    assert "call_tool" in first  # tools advertised in request 1

    # Only the FINAL reply lands in the persisted history.
    assert session.history[-1] == {"role": "assistant", "content": result.reply}
    assert any("запиши заметку" in m["content"] for m in session.history)


def test_agent_normal_reply_without_tool(tmp_path):
    llm = _ScriptedLLM(["Просто ответ без инструментов."])
    session = _make_session(llm, tmp_path, _notes_router(tmp_path / "notes.json"))

    result = session.chat_with_details("привет")

    assert result.reply == "Просто ответ без инструментов."
    assert session.mcp_events == []
    assert len(llm.payloads) == 1  # single request, no follow-up


def test_agent_executes_batch_of_tool_calls_in_one_round(tmp_path):
    """Multiple actions in one request -> ONE batch directive -> N MCP calls."""
    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "call_tool": [
                        {"name": "notes__add_note", "arguments": {"text": "первая"}},
                        {"name": "notes__add_note", "arguments": {"text": "вторая"}},
                        {"name": "notes__add_note", "arguments": {"text": "третья"}},
                    ]
                },
                ensure_ascii=False,
            ),
            "Все три заметки сохранены в CRM.",
        ]
    )
    session = _make_session(
        llm, tmp_path, _notes_router(tmp_path / "notes.json")
    )

    result = session.chat_with_details(
        "добавь заметки: поменять валюту, поменять масло, позвонить"
    )

    assert "сохранены" in result.reply
    # All three calls happened in a single round-trip.
    assert len(session.mcp_events) == 3
    assert [e.result for e in session.mcp_events if e.ok] == [
        "Заметка сохранена: id=1",
        "Заметка сохранена: id=2",
        "Заметка сохранена: id=3",
    ]
    # Exactly two LLM requests: directive round + final answer round.
    assert len(llm.payloads) == 2


def test_agent_handles_tool_failure(tmp_path):
    llm = _ScriptedLLM(
        [
            json.dumps(
                {"call_tool": {"name": "notes__missing_tool", "arguments": {}}}
            ),
            "Не удалось выполнить инструмент: такого нет.",
        ]
    )
    session = _make_session(llm, tmp_path, _notes_router(tmp_path / "notes.json"))

    result = session.chat_with_details("сделай что-нибудь")

    assert "Не удалось" in result.reply
    event = session.mcp_events[0]
    assert not event.ok
    assert "неизвестный инструмент" in event.error
