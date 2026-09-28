"""Cross-server MCP orchestration: ONE prompt drives ALL THREE servers.

The scripted LLM (canned replies per round at the HTTP layer) composes the
pipeline itself, while the MCP side is REAL: notes, scheduler and
currency_yahoo each run as a stdio child process, the currency one offline
(``CURRENCY_FIXTURE``). The scheduler's creation-time ``mcp_call`` validation
reads the same self-contained config through the ``MCP_CONFIG`` env override,
which is what makes the all-tmp-paths setup possible.

Flow under test (5 tool calls, 6 model replies, budget from ``max_rounds``):

    currency_yahoo__fetch_rates
      -> currency_yahoo__analyze_rates   (raw fetch output)
      -> currency_yahoo__save_report     (raw analysis output)
      -> notes__add_note                 (analysis numbers in the text)
      -> scheduler__schedule_task        (daily mcp_call collector)

Verified (the orchestration task's acceptance criteria):

* tool SELECTION — each server contributes exactly the tool its role needs;
* ROUTING — every call lands on the owning server (``event.server``);
* ORDER — the five calls happen strictly in pipeline order;
* DATA PASSING ACROSS SERVERS — the note text carries the analysis numbers
  that only the analyze_rates feedback turn could have delivered;
* SIDE EFFECTS — report, note and scheduled task really land in all three
  stores;
* budget boundaries — a smaller ``max_rounds`` caps the chain gracefully,
  and a single batch directive may mix servers in one round.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest
import yaml

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from llm_bot.api.currency_yahoo import analyze_rates
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.mcp_tools import (
    DEFAULT_MAX_ROUNDS,
    MCPRouter,
    MCPToolBridge,
    load_mcp_config,
    load_mcp_settings,
    router_from_config,
)
from llm_bot.stores import AgentConfig, ModelConfig

SERIES = [
    {"date": "2025-01-01", "rate": 100.0},
    {"date": "2025-01-02", "rate": 101.0},
    {"date": "2025-01-03", "rate": 99.0},
    {"date": "2025-01-04", "rate": 102.5},
]
# Byte-exact outputs the currency server will produce (same dicts, same dumps).
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
# The numbers below are extracted from the analysis summary — they can only
# have reached this directive through the analyze_rates feedback turn.
_SUMMARY = json.loads(SUMMARY_JSON)
NOTE_TEXT = (
    f"Курс USD за 2025: {_SUMMARY['start_rate']} -> {_SUMMARY['end_rate']} "
    f"({_SUMMARY['change_pct']}%, {_SUMMARY['trend']})."
)
FINAL_REPLY = (
    "Готово: анализ USD за 2025 сохранён, заметка добавлена, "
    "ежедневный сбор курса поставлен на 09:00."
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


def _write_shared_config(tmp_path: Path) -> dict:
    """Server map + ``max_rounds`` in ONE self-contained tmp config file.

    Returned as a dict so tests can build bridges with the merged process
    env; the file itself is what the scheduler child sees via ``MCP_CONFIG``.
    """
    config = {
        "max_rounds": 8,  # 6 model replies needed by the long flow
        "servers": {
            "notes": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.notes"],
                "env": {"NOTES_DB": str(tmp_path / "notes.json")},
            },
            "scheduler": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.scheduler"],
                "env": {
                    "SCHEDULER_DB": str(tmp_path / "scheduler.json"),
                    "MCP_CONFIG": str(tmp_path / "mcp.yaml"),
                },
            },
            "currency_yahoo": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.currency_yahoo"],
                "env": {
                    "CURRENCY_FIXTURE": str(tmp_path / "series.json"),
                    "CURRENCY_REPORTS_DB": str(tmp_path / "reports.json"),
                    "CURRENCY_SERIES_DIR": str(tmp_path / "series_out"),
                },
            },
        },
    }
    (tmp_path / "mcp.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    return config


def _orchestration_router(
    tmp_path: Path, *, max_rounds: int | None = None
) -> MCPRouter:
    """Router over all three REAL servers, self-contained in *tmp_path*."""
    fixture = tmp_path / "series.json"
    fixture.write_text(json.dumps(SERIES), encoding="utf-8")
    config = _write_shared_config(tmp_path)
    bridges = [
        MCPToolBridge(
            name,
            command=sys.executable,
            args=cfg["args"],
            env={**os.environ, **cfg["env"]},
        )
        for name, cfg in config["servers"].items()
    ]
    if max_rounds is None:
        # Same wiring the CLI uses: the config's max_rounds key decides.
        max_rounds = int(config.get("max_rounds", DEFAULT_MAX_ROUNDS))
    router = MCPRouter(bridges, max_rounds=max_rounds)
    router.refresh()
    return router


def _make_session(llm: _ScriptedLLM, tmp_path: Path, router: MCPRouter):
    return make_session(
        "mcp-orchestration-test",
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
                system_prompt="Ты помощник с доступом к отчётам, CRM и планировщику.",
            )
        ),
        session_store=JsonSessionStore(tmp_path / "sessions"),
        transport=llm.as_transport(),
        invariants=False,
        memory_store=None,
        mcp_router=router,
    )


def _directive(name: str, arguments: dict) -> str:
    return json.dumps({"call_tool": {"name": name, "arguments": arguments}},
                      ensure_ascii=False)


def _feedback(request: dict, marker: str) -> str:
    """The service feedback turn inside *request* carrying *marker*."""
    for message in request["messages"]:
        content = message.get("content", "")
        if message["role"] == "user" and marker in content:
            return content
    raise AssertionError(f"feedback not found: {marker}")


def _last_assistant(request: dict) -> str:
    """The latest assistant message inside *request* (a previous directive)."""
    assistants = [
        m.get("content", "")
        for m in request["messages"]
        if m["role"] == "assistant"
    ]
    if not assistants:
        raise AssertionError("no assistant message in request")
    return assistants[-1]


def test_long_flow_across_three_servers(tmp_path):
    """One prompt -> 5 tool calls over 3 servers, in strict pipeline order."""
    llm = _ScriptedLLM(
        [
            # Round 1: fetch the USD series.
            _directive(
                "currency_yahoo__fetch_rates",
                {
                    "currency_code": "USD",
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                },
            ),
            # Round 2: feed the EXACT fetch output into the analyzer.
            _directive(
                "currency_yahoo__analyze_rates",
                {"data": FETCH_JSON, "currency_code": "USD"},
            ),
            # Round 3: feed the EXACT analysis output into the report saver.
            _directive(
                "currency_yahoo__save_report",
                {"data": SUMMARY_JSON, "title": "Курс USD за 2025"},
            ),
            # Round 4: cross-server handoff — the note text carries the
            # analysis numbers delivered by the previous feedback turn.
            _directive(
                "notes__add_note",
                {"text": NOTE_TEXT, "tags": ["валюта", "отчёт"]},
            ),
            # Round 5: cross-server handoff — schedule a DAILY collector that
            # will re-run the currency fetch through mcp_call.
            _directive(
                "scheduler__schedule_task",
                {
                    "title": "Ежедневный сбор курса USD",
                    "action": "mcp_call",
                    "payload": {
                        "server": "currency_yahoo",
                        "tool": "fetch_rates",
                        "arguments": {
                            "currency_code": "USD",
                            "date_from": "2025-01-01",
                            "date_to": "2025-12-31",
                        },
                    },
                    "daily_at": "09:00",
                    "group": "финансы",
                },
            ),
            # Round 6: the final user-facing answer.
            FINAL_REPLY,
        ]
    )
    router = _orchestration_router(tmp_path)
    session = _make_session(llm, tmp_path, router)

    result = session.chat_with_details(
        "Проанализируй курс доллара за 2025 год, сохрани отчёт, добавь заметку "
        "с итогом и поставь ежедневный сбор курса на 09:00."
    )

    # SELECTION + ROUTING + ORDER: exactly five calls, each on its own
    # server, strictly in pipeline order, all successful.
    assert [(e.server, e.tool, e.ok) for e in session.mcp_events] == [
        ("currency_yahoo", "fetch_rates", True),
        ("currency_yahoo", "analyze_rates", True),
        ("currency_yahoo", "save_report", True),
        ("notes", "add_note", True),
        ("scheduler", "schedule_task", True),
    ]
    assert {e.server for e in session.mcp_events} == {
        "currency_yahoo", "notes", "scheduler",
    }
    assert result.reply == FINAL_REPLY

    # Six LLM requests: five directive rounds + the final answer — the
    # round budget (max_rounds=8 from the router) did not cut the flow.
    assert len(llm.payloads) == 6

    # The first request advertised ALL THREE servers' tools.
    first = json.dumps(llm.payloads[0], ensure_ascii=False)
    assert "currency_yahoo__fetch_rates" in first
    assert "notes__add_note" in first
    assert "scheduler__schedule_task" in first

    # DATA PASSING: the analyze feedback carried the raw summary to the
    # model, and the next directive embeds exactly its numbers.
    assert SUMMARY_JSON in _feedback(
        llm.payloads[2], "Результат инструмента currency_yahoo__analyze_rates"
    )
    # r4 (the note directive) is the response to payload[3]; it reaches the
    # next request as assistant history — payload[4] is where it appears.
    assert NOTE_TEXT in _last_assistant(llm.payloads[4])

    # SIDE EFFECT 1: the report landed in the currency store.
    stored = json.loads(
        (tmp_path / "reports.json").read_text(encoding="utf-8")
    )
    assert len(stored) == 1
    assert stored[0]["title"] == "Курс USD за 2025"
    assert stored[0]["report"]["end_rate"] == 102.5
    assert stored[0]["report"]["trend"] == "рост"

    # SIDE EFFECT 2: the note landed in the CRM with the analysis numbers.
    notes_db = json.loads(
        (tmp_path / "notes.json").read_text(encoding="utf-8")
    )
    assert notes_db == [
        {"id": 1, "text": NOTE_TEXT, "tags": ["валюта", "отчёт"]}
    ]

    # SIDE EFFECT 3: the daily collector landed in the scheduler store.
    scheduler_db = json.loads(
        (tmp_path / "scheduler.json").read_text(encoding="utf-8")
    )
    assert len(scheduler_db["tasks"]) == 1
    task = scheduler_db["tasks"][0]
    assert task["action"] == "mcp_call"
    assert task["status"] == "active"
    assert task["group"] == "финансы"
    assert task["schedule"] == {"kind": "daily", "at": "09:00"}
    assert task["payload"]["server"] == "currency_yahoo"
    assert task["payload"]["tool"] == "fetch_rates"
    assert task["payload"]["arguments"]["currency_code"] == "USD"


def test_batch_directive_mixes_servers_in_one_round(tmp_path):
    """One batch directive -> calls on TWO servers in a single round."""
    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "call_tool": [
                        {
                            "name": "notes__add_note",
                            "arguments": {"text": "перезагрузить роутер"},
                        },
                        {
                            "name": "scheduler__schedule_task",
                            "arguments": {
                                "title": "Напоминание о роутере",
                                "action": "reminder",
                                "payload": {"text": "перезагрузить роутер"},
                                "delay_seconds": 1800,
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            "Готово: заметка записана, напоминание поставлено на 30 минут.",
        ]
    )
    session = _make_session(
        llm, tmp_path, _orchestration_router(tmp_path)
    )

    result = session.chat_with_details(
        "Запиши заметку перезагрузить роутер и напомни о ней через 30 минут."
    )

    # Both servers were hit IN THE LISTED ORDER within one round.
    assert [(e.server, e.tool, e.ok) for e in session.mcp_events] == [
        ("notes", "add_note", True),
        ("scheduler", "schedule_task", True),
    ]
    assert "напоминание поставлено" in result.reply
    # A batch costs ONE round: two LLM requests total.
    assert len(llm.payloads) == 2
    # Both side effects persisted on their own servers.
    notes_db = json.loads(
        (tmp_path / "notes.json").read_text(encoding="utf-8")
    )
    assert [n["text"] for n in notes_db] == ["перезагрузить роутер"]
    scheduler_db = json.loads(
        (tmp_path / "scheduler.json").read_text(encoding="utf-8")
    )
    assert scheduler_db["tasks"][0]["action"] == "reminder"


def test_round_budget_caps_the_chain(tmp_path):
    """max_rounds=2 with a 3-call script: exactly 2 calls run, in order.

    The loop stops without a crash; the third directive never executes and
    remains the turn's final reply — the documented boundary behaviour that
    the ``max_rounds`` config key exists to raise.
    """
    third_directive = _directive(
        "scheduler__schedule_task",
        {
            "title": "Лишний шаг за пределами бюджета",
            "action": "reminder",
            "payload": {"text": "не должен выполниться"},
            "delay_seconds": 60,
        },
    )
    llm = _ScriptedLLM(
        [
            _directive(
                "notes__add_note", {"text": "первый шаг"}
            ),
            _directive(
                "currency_yahoo__fetch_rates",
                {
                    "currency_code": "USD",
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                },
            ),
            third_directive,
        ]
    )
    session = _make_session(
        llm, tmp_path, _orchestration_router(tmp_path, max_rounds=2)
    )

    result = session.chat_with_details("выполни три шага по цепочке")

    # The budget held: two calls, strictly ordered across two servers...
    assert [(e.server, e.tool, e.ok) for e in session.mcp_events] == [
        ("notes", "add_note", True),
        ("currency_yahoo", "fetch_rates", True),
    ]
    # ...the third directive never reached MCP (the store is created lazily
    # on the first write, so its absence proves no side effect)...
    assert not (tmp_path / "scheduler.json").exists()
    # ...three LLM requests happened (no extra round after the budget)...
    assert len(llm.payloads) == 3
    # ...and the leftover directive IS the final reply (boundary contract).
    assert result.reply == third_directive


def test_router_from_config_reads_max_rounds(tmp_path, monkeypatch):
    """``max_rounds`` flows from the YAML config into the router."""
    config = _write_shared_config(tmp_path)
    assert config["max_rounds"] == 8

    # The MCP_CONFIG env override points router_from_config at the same
    # self-contained file the scheduler child uses.
    monkeypatch.setenv("MCP_CONFIG", str(tmp_path / "mcp.yaml"))
    assert load_mcp_settings() == config
    assert sorted(load_mcp_config()) == [
        "currency_yahoo", "notes", "scheduler",
    ]

    router = router_from_config(["all"])
    assert router is not None
    assert router.max_rounds == 8
    assert "currency_yahoo__fetch_rates" in router.tools
    assert "notes__add_note" in router.tools
    assert "scheduler__schedule_task" in router.tools

    # Without the override the default location is used again (missing here,
    # so the explicit selection must fail loudly rather than read the tmp file).
    monkeypatch.setenv("MCP_CONFIG", str(tmp_path / "absent.yaml"))
    with pytest.raises(ValueError, match="конфигурация MCP не найдена"):
        router_from_config(["notes"])


def test_parser_survives_planning_text_before_directive():
    """Live-model finding: the reply prefixes the directive with planning
    text containing brace fragments — the naive first-{..last-} slice used
    to produce invalid JSON and silently dropped the directive. The parser
    must find the first VALID JSON object carrying call_tool."""
    reply = (
        "We need to: analyze USD rate, save report, add note, schedule."
        " First call fetch_rates."
        ' payload {"currency_code":"USD"}, daily_at "09:00".'
        ' Then: {"call_tool": {"name": "notes__add_note",'
        ' "arguments": {"text": "итог"}}}'
    )
    directives = MCPRouter.parse_directives(reply)
    assert [d.name for d in directives] == ["notes__add_note"]
    assert directives[0].arguments == {"text": "итог"}
    # Plain forms keep working: single directive, fenced, and batch.
    assert [
        d.name for d in MCPRouter.parse_directives(_directive("notes__list_notes", {}))
    ] == ["notes__list_notes"]
    fenced = "```json\n" + _directive("notes__find_notes", {"query": "корм"}) + "\n```"
    assert [
        d.name for d in MCPRouter.parse_directives(fenced)
    ] == ["notes__find_notes"]
    batch = json.dumps(
        {
            "call_tool": [
                {"name": "notes__add_note", "arguments": {"text": "a"}},
                {"name": "notes__find_notes"},
            ]
        },
        ensure_ascii=False,
    )
    assert [d.name for d in MCPRouter.parse_directives(batch)] == [
        "notes__add_note",
        "notes__find_notes",
    ]
    assert MCPRouter.parse_directives("Обычный ответ без JSON.") == []
    assert MCPRouter.parse_directives('{"other": {"a": 1}}') == []


def test_parser_repairs_unbalanced_directive_json():
    """Live finding (user CLI run): the model dropped the outer closing
    brace and left a stray ']' — {\"call_tool\": {...}}] — the naive parse
    failed, the chain stalled and the raw JSON leaked to the user as the
    reply. The bracket-repair fallback must recover the directive."""
    malformed = json.dumps(
        {
            "call_tool": {
                "name": "currency_yahoo__analyze_rates",
                "arguments": {
                    "data": [
                        {"date": "2026-01-02", "rate": 79.0009},
                        {"date": "2026-01-14", "rate": 78.7472},
                    ],
                    "currency_code": "USD",
                },
            }
        },
        ensure_ascii=False,
    )
    # Recreate the exact malformation: drop ONE outer '}', append ']'.
    broken = malformed[:-1] + "]"
    assert broken.endswith("}]")
    directives = MCPRouter.parse_directives(broken)
    assert [d.name for d in directives] == ["currency_yahoo__analyze_rates"]
    assert directives[0].arguments["currency_code"] == "USD"
    assert len(directives[0].arguments["data"]) == 2
    # A well-formed reply still parses to the SAME directive (repair is a
    # fallback only, normal parsing is untouched).
    assert [
        d.name for d in MCPRouter.parse_directives(malformed)
    ] == ["currency_yahoo__analyze_rates"]
    # Non-directive text mentioning call_tool never becomes a directive.
    assert MCPRouter.parse_directives('расскажи про call_tool без JSON') == []


FABRICATED = (
    "Результат инструмента currency_yahoo__analyze_rates:\n"
    '{"currency":"USD","summary":{"min_rate":71.2,"max_rate":87.3,'
    '"average_rate":78.5,"trend":"upward"},"details":{"points":24}}'
)


def test_fabricated_tool_result_gets_corrected(tmp_path):
    """Live finding: instead of the analyze directive the model returned a
    FABRICATED «Результат инструмента …» message (mimicking the service
    format, real directive hidden in reasoning). The loop must recognize the
    impersonation, push back with a corrective service turn and let the model
    redo the missed step — the step must then REALLY execute."""
    llm = _ScriptedLLM(
        [
            # Round 1: fetch the USD series (clean directive).
            _directive(
                "currency_yahoo__fetch_rates",
                {
                    "currency_code": "USD",
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                },
            ),
            # Round 2: NO directive — a fabricated tool result instead.
            FABRICATED,
            # Round 3 (after the correction): the model redoes the step.
            _directive(
                "currency_yahoo__analyze_rates",
                {"data": FETCH_JSON, "currency_code": "USD"},
            ),
            # Round 4: the final user-facing answer.
            FINAL_REPLY,
        ]
    )
    router = _orchestration_router(tmp_path)
    session = _make_session(llm, tmp_path, router)

    result = session.chat_with_details(
        "Проанализируй курс доллара за 2025 год и сохрани отчёт."
    )

    # The missed step was ACTUALLY executed after the correction — not
    # substituted by the fabricated aggregates.
    assert [(e.server, e.tool, e.ok) for e in session.mcp_events] == [
        ("currency_yahoo", "fetch_rates", True),
        ("currency_yahoo", "analyze_rates", True),
    ]
    assert len(llm.payloads) == 4
    assert result.reply == FINAL_REPLY

    # The corrective turn quotes the fabricated reply as assistant history
    # and explains that only the system reports tool results.
    third = llm.payloads[2]
    assert FABRICATED in _last_assistant(third)
    corrections = [
        m["content"]
        for m in third["messages"]
        if m["role"] == "user" and "Выдавать выдуманные результаты" in m["content"]
    ]
    assert len(corrections) == 1

    # The REAL analyze feedback followed the correction.
    assert SUMMARY_JSON in _feedback(
        llm.payloads[3], "Результат инструмента currency_yahoo__analyze_rates"
    )

    # Durable history keeps only the final answer, not the impersonation.
    assert session.history[-1]["content"] == FINAL_REPLY
    assert FABRICATED not in session.history[-1]["content"]


def test_fabricated_reply_at_budget_edge_is_delivered(tmp_path):
    """Documented boundary: when the round budget is exhausted, a fabricated
    reply stands (same as a leftover directive) — no correction is spent."""
    llm = _ScriptedLLM(
        [
            _directive(
                "currency_yahoo__fetch_rates",
                {
                    "currency_code": "USD",
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                },
            ),
            FABRICATED,
        ]
    )
    router = _orchestration_router(tmp_path, max_rounds=2)
    session = _make_session(llm, tmp_path, router)

    result = session.chat_with_details("Проанализируй курс USD за 2025.")

    # Only the first (real) call executed; the fabricated reply was delivered
    # as-is because the budget had no room for a corrective round.
    assert [(e.server, e.tool, e.ok) for e in session.mcp_events] == [
        ("currency_yahoo", "fetch_rates", True)
    ]
    assert len(llm.payloads) == 2
    assert result.reply == FABRICATED
    # No corrective turn was injected anywhere.
    assert all(
        "Выдавать выдуманные результаты" not in m.get("content", "")
        for payload in llm.payloads
        for m in payload["messages"]
    )
