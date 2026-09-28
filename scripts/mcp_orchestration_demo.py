#!/usr/bin/env python3
"""Live demo: one prompt drives ALL THREE MCP servers (orchestration).

1. Fetches the USD-2025 daily rate series once (Yahoo Finance, no key; or a
   ``--fixture`` file offline) and caches it into a self-contained tmp dir;
2. builds ONE router over the three own MCP servers — currency_yahoo, notes,
   scheduler — sharing a tmp ``mcp.yaml`` (the scheduler server receives its
   location through the ``MCP_CONFIG`` env override so its ``mcp_call``
   creation-time validation passes without ``data/mcp.yaml``);
3. runs the deterministic cross-server chain directly through the router
   (no LLM): fetch_rates -> analyze_rates -> save_report -> notes.add_note
   (analysis numbers in the text) -> scheduler.schedule_task (daily
   ``mcp_call`` collector) — the data-passing check;
4. gives a real LLM session ONE user prompt asking for the whole pipeline
   and lets the agent build and run it by itself over multiple tool rounds
   (budget: the config's ``max_rounds`` key) — the automatic-execution check.

Writes a markdown log to results/mcp_orchestration_demo.md.

Examples:
    python scripts/mcp_orchestration_demo.py
    python scripts/mcp_orchestration_demo.py --fixture data/cbr_fixture.json
    python scripts/mcp_orchestration_demo.py --out results/mcp_orchestration_demo.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.api import currency_yahoo
from llm_bot.client import LLMError
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.mcp_tools import MCPToolBridge, MCPRouter
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = "results/mcp_orchestration_demo.md"
USER_PROMPT = (
    "Проанализируй курс доллара за 2025 год, сохрани отчёт, добавь заметку "
    "с итогом и поставь ежедневный сбор курса на 09:00."
)
# The round budget lives in the config (top-level max_rounds key): the flow
# needs 6 model replies (5 tool rounds + the final answer).
MAX_ROUNDS = 8


def build_router(tmp: Path, series_path: Path) -> MCPRouter:
    """One router over all three own MCP servers, self-contained in *tmp*.

    The currency server runs offline on the cached series; the scheduler
    server validates ``mcp_call`` targets against the SAME tmp config via
    the ``MCP_CONFIG`` env override.
    """
    config = {
        "max_rounds": MAX_ROUNDS,
        "servers": {
            "notes": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.notes"],
                "env": {"NOTES_DB": str(tmp / "notes.json")},
            },
            "scheduler": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.scheduler"],
                "env": {
                    "SCHEDULER_DB": str(tmp / "scheduler.json"),
                    "MCP_CONFIG": str(tmp / "mcp.yaml"),
                },
            },
            "currency_yahoo": {
                "command": sys.executable,
                "args": ["-m", "llm_bot.mcp_servers.currency_yahoo"],
                "env": {
                    "CURRENCY_FIXTURE": str(series_path),
                    "CURRENCY_REPORTS_DB": str(tmp / "reports.json"),
                    "CURRENCY_SERIES_DIR": str(tmp / "series_out"),
                },
            },
        },
    }
    config_path = tmp / "mcp.yaml"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The scheduler child reads this same file during mcp_call validation.
    os.environ["MCP_CONFIG"] = str(config_path)

    bridges = [
        MCPToolBridge(
            name,
            command=sys.executable,
            args=cfg["args"],
            env={**os.environ, **cfg["env"]},
        )
        for name, cfg in config["servers"].items()
    ]
    router = MCPRouter(bridges, max_rounds=MAX_ROUNDS)
    router.refresh()
    return router


def _log_event(log: list[str], event) -> None:
    status = "OK" if event.ok else "FAILED"
    body = (event.result or event.error).replace("\n", " ")
    body = body if len(body) <= 220 else body[:220] + "…"
    line = f"- `{event.server}__{event.tool}` **{status}**: `{body}`"
    log.append(line)
    print(f"    [mcp] {event.server}__{event.tool} {status}: {body}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="assistant")
    parser.add_argument(
        "--fixture",
        default="",
        help="use this pre-saved series file instead of the Yahoo API",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--prompt",
        default="",
        help="override the one user prompt given to the agent stage",
    )
    args = parser.parse_args()
    prompt = args.prompt.strip() or USER_PROMPT

    tmp = Path(tempfile.mkdtemp(prefix="orchestration_demo_"))
    series_path = tmp / "series.json"

    # 1. Get the series once: real API, or the provided fixture offline.
    try:
        series = currency_yahoo.fetch_rates(
            "USD", "2025-01-01", "2025-12-31",
            fixture=args.fixture or None,
        )
        source = args.fixture or "Yahoo Finance chart API (без ключа)"
    except currency_yahoo.CurrencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    series_path.write_text(
        json.dumps(series, ensure_ascii=False), encoding="utf-8"
    )

    router = build_router(tmp, series_path)
    tools = sorted(router.tools)
    print(f"MCP tools ({len(tools)}, 3 servers): {tools}")
    if not tools:
        print("error: no MCP server exposed any tool", file=sys.stderr)
        return 1

    log: list[str] = [
        "# Демо: оркестрация — один промпт ведёт цепочку через три MCP-сервера",
        "",
        f"Источник данных: `{source}`",
        f"Точек в ряде: {len(series)} ({series[0]['date']} .. {series[-1]['date']})",
        f"Бюджет раундов: `max_rounds: {MAX_ROUNDS}` (ключ конфига mcp.yaml)",
        "",
        "## 1. Детерминированная кросс-серверная цепочка через MCP (без LLM)",
        "",
        "fetch_rates -> analyze_rates -> save_report (currency_yahoo)"
        " -> add_note (notes) -> schedule_task (scheduler)",
        "",
    ]

    # 2. Deterministic chain through the router — proves cross-server data
    #    passing: each stage consumes the previous tool's RAW output.
    fetched = router.call_tool(
        "currency_yahoo__fetch_rates",
        {"currency_code": "USD", "date_from": "2025-01-01",
         "date_to": "2025-12-31"},
    )
    _log_event(log, fetched)
    analyzed = router.call_tool(
        "currency_yahoo__analyze_rates",
        {"data": fetched.result, "currency_code": "USD"},
    )
    _log_event(log, analyzed)
    saved = router.call_tool(
        "currency_yahoo__save_report",
        {"data": analyzed.result, "title": "Курс USD за 2025"},
    )
    _log_event(log, saved)
    summary = json.loads(analyzed.result)
    note_text = (
        f"Курс USD за 2025: {summary['start_rate']} -> {summary['end_rate']} "
        f"({summary['change_pct']}%, {summary['trend']})."
    )
    noted = router.call_tool(
        "notes__add_note",
        {"text": note_text, "tags": ["валюта", "отчёт"]},
    )
    _log_event(log, noted)
    scheduled = router.call_tool(
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
    )
    _log_event(log, scheduled)
    if not all(e.ok for e in (fetched, analyzed, saved, noted, scheduled)):
        print("error: deterministic chain failed", file=sys.stderr)
        return 1

    log += [
        "",
        "Заметка (числа пришли из анализа через MCP): «" + note_text + "»",
        "",
        "## 2. Агент сам ведёт всю цепочку (один промпт, реальная модель)",
        "",
        f"Промпт: «{prompt}»",
        "",
        "```",
    ]

    # 3. One prompt, the agent runs the whole cross-server flow by itself.
    session = make_session(
        "mcp-orchestration-demo",
        args.agent,
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(),
        invariants=False,
        mcp_router=router,
    )
    print(f"\n>>> {prompt}")
    try:
        result = session.chat_with_details(prompt)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.reply)
    for event in session.mcp_events:
        status = "OK" if event.ok else "FAILED"
        body = (event.result or event.error).replace("\n", " ")
        body = body if len(body) <= 160 else body[:160] + "…"
        print(f"    [mcp] {event.server}__{event.tool} {status}: {body}")
    log.append(result.reply)
    log.append("```")
    log.append("")
    log.append("Вызовы агента (по порядку):")
    log.append("")
    for event in session.mcp_events:
        _log_event(log, event)
    log.append("")
    servers_touched = sorted({e.server for e in session.mcp_events})
    log.append(
        f"Вызовов инструментов: {len(session.mcp_events)}; "
        f"серверов задействовано: {len(servers_touched)} ({', '.join(servers_touched)})"
    )
    log.append("")

    # 4. Dump all three stores — prove every server really did its part.
    def _dump(path: Path, missing: str) -> str:
        return (
            json.dumps(
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else missing,
                ensure_ascii=False, indent=2,
            )
        )

    log += [
        "## 3. Хранилища после прогона",
        "",
        "### currency_yahoo (отчёты)",
        "",
        "```json",
        _dump(tmp / "reports.json", "[]"),
        "```",
        "",
        "### notes (CRM)",
        "",
        "```json",
        _dump(tmp / "notes.json", "[]"),
        "```",
        "",
        "### scheduler (фоновые задачи)",
        "",
        "```json",
        _dump(tmp / "scheduler.json", {"tasks": [], "results": []}),
        "```",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(log), encoding="utf-8")
    print(f"\nlog written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
