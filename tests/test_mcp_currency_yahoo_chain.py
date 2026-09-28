"""Agent composition test: the model chains the currency tools by itself.

One user prompt — «проанализируй курс доллара за 2025 год и сохрани отчёт» —
and the model builds the pipeline on its own: fetch_rates, then analyze_rates
(feeding the fetched JSON into ``data``), then save_report (feeding the
summary into ``data``). The LLM is scripted at the HTTP layer (canned replies
per round), while the MCP side is REAL: the currency server runs as a stdio
child process offline (``CURRENCY_FIXTURE``).

Verified here:
* automatic chain execution — three directive rounds without any extra
  prompts, bounded by the usual agent loop;
* correct data passing — each next directive's ``data`` argument carries the
  previous tool's exact output, which reached the model through the service
  feedback turn, and the stored report contains the expected numbers.
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
from llm_bot.api.currency_yahoo import analyze_rates
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.mcp_tools import MCPRouter, MCPToolBridge, ToolSpec
from llm_bot.stores import AgentConfig, ModelConfig

SERVER_MODULE = "llm_bot.mcp_servers.currency_yahoo"

SERIES = [
    {"date": "2025-01-01", "rate": 100.0},
    {"date": "2025-01-02", "rate": 101.0},
    {"date": "2025-01-03", "rate": 99.0},
    {"date": "2025-01-04", "rate": 102.5},
]
# Byte-exact outputs the server will produce (same dicts, same dumps).
FETCH_JSON = json.dumps(
    {
        "currency": "USD",
        "date_from": "2025-01-01",
        "date_to": "2025-12-31",
        "count": 4,
        "items": SERIES,
    },
    ensure_ascii=False,
)
SUMMARY_JSON = json.dumps(analyze_rates(SERIES, currency="USD"), ensure_ascii=False)

USER_PROMPT = "Проанализируй курс доллара за 2025 год и сохрани отчёт."
FINAL_REPLY = "Готово: курс USD за 2025 год проанализирован, отчёт сохранён."


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


def _currency_router(tmp_path: Path) -> MCPRouter:
    """Router with one bridge to the real currency MCP server (offline)."""
    fixture = tmp_path / "series.json"
    fixture.write_text(json.dumps(SERIES), encoding="utf-8")
    router = MCPRouter(
        [
            MCPToolBridge(
                "currency_yahoo",
                command=sys.executable,
                args=["-m", SERVER_MODULE],
                env={
                    **os.environ,
                    "CURRENCY_FIXTURE": str(fixture),
                    "CURRENCY_REPORTS_DB": str(tmp_path / "reports.json"),
                    "CURRENCY_SERIES_DIR": str(tmp_path / "series_out"),
                },
            )
        ]
    )
    router.refresh()
    return router


def _make_session(llm: _ScriptedLLM, tmp_path: Path, router: MCPRouter) -> Session:
    return make_session(
        "currency-chain-test",
        "assistant",
        model_store=_StubModelStore(
            ModelConfig(
                name="openai",
                base_url="https://example.test/v1",
                api_key="k",
                model="gpt",
            )
        ),
        agent_store=_StubAgentStore(
            AgentConfig(
                name="assistant",
                model="openai",
                system_prompt="Ты финансовый помощник.",
            )
        ),
        session_store=JsonSessionStore(tmp_path / "sessions"),
        transport=llm.as_transport(),
        invariants=False,
        memory_store=None,
        mcp_router=router,
    )


def test_agent_builds_the_chain_by_itself(tmp_path):
    """One prompt -> three rounds of directives -> report stored."""
    llm = _ScriptedLLM(
        [
            # Round 1: the model decides to fetch the rate series first.
            json.dumps(
                {
                    "call_tool": {
                        "name": "currency_yahoo__fetch_rates",
                        "arguments": {
                            "currency_code": "USD",
                            "date_from": "2025-01-01",
                            "date_to": "2025-12-31",
                        },
                    }
                },
                ensure_ascii=False,
            ),
            # Round 2: it feeds the EXACT fetch output into analyze_rates.
            json.dumps(
                {
                    "call_tool": {
                        "name": "currency_yahoo__analyze_rates",
                        "arguments": {"data": FETCH_JSON, "currency_code": "USD"},
                    }
                },
                ensure_ascii=False,
            ),
            # Round 3: it feeds the EXACT analyze output into save_report.
            json.dumps(
                {
                    "call_tool": {
                        "name": "currency_yahoo__save_report",
                        "arguments": {
                            "data": SUMMARY_JSON,
                            "title": "Курс USD за 2025",
                        },
                    }
                },
                ensure_ascii=False,
            ),
            # Round 4: the final user-facing answer.
            FINAL_REPLY,
        ]
    )
    session = _make_session(llm, tmp_path, _currency_router(tmp_path))

    result = session.chat_with_details(USER_PROMPT)

    # Automatic chain execution: exactly three real MCP calls in order.
    assert result.reply == FINAL_REPLY
    assert [(e.tool, e.ok) for e in session.mcp_events] == [
        ("fetch_rates", True),
        ("analyze_rates", True),
        ("save_report", True),
    ]
    assert all(e.server == "currency_yahoo" for e in session.mcp_events)

    # Data passing BETWEEN tools happens through the agent: each service
    # feedback turn carried the previous tool's raw output to the model, and
    # the next scripted directive embedded exactly that output.
    assert session.mcp_events[0].result == FETCH_JSON
    assert session.mcp_events[1].result == SUMMARY_JSON

    def _feedback(request: dict, marker: str) -> str:
        for message in request["messages"]:
            content = message.get("content", "")
            if message["role"] == "user" and marker in content:
                return content
        raise AssertionError(f"feedback not found: {marker}")

    fetch_feedback = _feedback(
        llm.payloads[1], "Результат инструмента currency_yahoo__fetch_rates"
    )
    assert FETCH_JSON in fetch_feedback
    analyze_feedback = _feedback(
        llm.payloads[2], "Результат инструмента currency_yahoo__analyze_rates"
    )
    assert SUMMARY_JSON in analyze_feedback

    # Four LLM requests total: three directive rounds + the final answer.
    assert len(llm.payloads) == 4

    # The pipeline really persisted the report with the expected numbers.
    stored = json.loads(
        (tmp_path / "reports.json").read_text(encoding="utf-8")
    )
    assert len(stored) == 1
    assert stored[0]["title"] == "Курс USD за 2025"
    assert stored[0]["report"]["end_rate"] == 102.5
    assert stored[0]["report"]["trend"] == "рост"


def test_prompt_block_advertises_chaining():
    """The router block tells the model HOW to build chains (one call at a
    time, previous result into the next arguments)."""

    class _StubBridge:
        name = "currency_yahoo"

        def list_tools(self):
            return [
                ToolSpec(
                    server="currency_yahoo",
                    name="fetch_rates",
                    description="Получить курс валюты к рублю за период",
                    parameters={"currency_code": {}, "date_from": {}, "date_to": {}},
                )
            ]

    router = MCPRouter([_StubBridge()])
    router.refresh()
    block = router.render_tools_block()
    assert "ЦЕПОЧКУ вызовов" in block
    assert "ПО ОДНОМУ" in block
    assert "в аргументы следующего инструмента" in block


def test_agent_builds_a_partial_chain_without_analysis(tmp_path):
    """«Выгрузи курс доллара за 2025 год и сохрани отчёт» -> fetch_rates
    + save_series, analyze_rates skipped.

    Download-and-save is ONE goal: without an analysis request (динамика,
    тренд, min/max) the «отчёт» is the dumped file, so save_series closes it.
    Nothing in the descriptions numbers the steps — save_series is chosen by
    its contract, which explicitly claims these phrases.
    """
    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "call_tool": {
                        "name": "currency_yahoo__fetch_rates",
                        "arguments": {
                            "currency_code": "USD",
                            "date_from": "2025-01-01",
                            "date_to": "2025-12-31",
                        },
                    }
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "call_tool": {
                        "name": "currency_yahoo__save_series",
                        "arguments": {
                            "data": FETCH_JSON,
                            "title": "Курс USD за 2025",
                        },
                    }
                },
                ensure_ascii=False,
            ),
            "Готово: курс USD за 2025 год выгружен в файл.",
        ]
    )
    session = _make_session(llm, tmp_path, _currency_router(tmp_path))

    result = session.chat_with_details(
        "Выгрузи курс доллара за 2025 год и сохрани отчёт."
    )

    assert [(e.tool, e.ok) for e in session.mcp_events] == [
        ("fetch_rates", True),
        ("save_series", True),
    ]
    assert session.mcp_events[0].result == FETCH_JSON
    assert "Ряд сохранён в файл" in session.mcp_events[1].result
    assert "выгружен в файл" in result.reply
    # Two directive rounds + the final answer — analyze_rates never called.
    assert len(llm.payloads) == 3

    dumped = json.loads(
        (tmp_path / "series_out" / "usd_2025-01-01_2025-12-31.json").read_text(
            encoding="utf-8"
        )
    )
    assert dumped["count"] == 4
    assert dumped["title"] == "Курс USD за 2025"
    assert dumped["items"][0]["rate"] == 100.0
