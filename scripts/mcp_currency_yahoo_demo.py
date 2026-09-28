#!/usr/bin/env python3
"""Live demo: the agent composes the currency pipeline by itself.

1. Fetches the USD-2025 daily rate series once (Yahoo Finance, no key; or a
   ``--fixture`` file offline) and caches it, so both stages below run on the
   same data without repeating network calls;
2. runs the deterministic chain fetch_rates -> analyze_rates -> save_report
   directly through the MCP router (no LLM) — the data-passing check;
3. gives a real LLM session ONE user prompt — «проанализируй курс доллара за
   2025 год и сохрани отчёт» — and lets the agent build and run the same
   chain by itself over multiple tool rounds — the automatic-execution check.

Writes a markdown log to results/mcp_currency_yahoo_demo.md.

Examples:
    python scripts/mcp_currency_yahoo_demo.py
    python scripts/mcp_currency_yahoo_demo.py --prompt "Выгрузи курс доллара за 2025 год и сохрани отчёт."
    python scripts/mcp_currency_yahoo_demo.py --fixture data/cbr_fixture.json --out results/mcp_currency_yahoo_demo.md
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
from llm_bot.mcp_tools import MCPRouter, MCPToolBridge
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.currency_yahoo"
DEFAULT_OUT = "results/mcp_currency_yahoo_demo.md"
USER_PROMPT = "Проанализируй курс доллара за 2025 год и сохрани отчёт."


def build_router(series_path: Path, reports_path: Path) -> MCPRouter:
    """Router with the currency_yahoo MCP server in offline fixture mode."""
    router = MCPRouter(
        [
            MCPToolBridge(
                "currency_yahoo",
                command=sys.executable,
                args=["-m", SERVER_MODULE],
                env={
                    **os.environ,
                    "CURRENCY_FIXTURE": str(series_path),
                    "CURRENCY_REPORTS_DB": str(reports_path),
                },
            )
        ]
    )
    router.refresh()
    return router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="assistant")
    parser.add_argument(
        "--fixture",
        default="",
        help="use this pre-saved series file instead of the CBR API",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--prompt",
        default="",
        help="override the one user prompt given to the agent stage",
    )
    args = parser.parse_args()
    prompt = args.prompt.strip() or USER_PROMPT

    tmp = Path(tempfile.mkdtemp(prefix="currency_demo_"))
    series_path = tmp / "series.json"
    reports_path = tmp / "reports.json"

    # 1. Get the series once: real CBR API, or the provided fixture offline.
    try:
        series = currency_yahoo.fetch_rates(
            "USD", "2025-01-01", "2025-12-31",
            fixture=args.fixture or None,
        )
        source = args.fixture or "Yahoo Finance chart API (без ключа)"
    except currency_yahoo.CurrencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    series_path.write_text(json.dumps(series, ensure_ascii=False), encoding="utf-8")

    router = build_router(series_path, reports_path)
    tools = sorted(router.tools)
    print(f"MCP tools: {tools}")
    if not tools:
        print(
            "error: currency_yahoo MCP server did not expose any tools",
            file=sys.stderr,
        )
        return 1

    log: list[str] = [
        "# Демо: агент сам строит цепочку currency_yahoo (fetch -> analyze -> save)",
        "",
        f"Источник данных: `{source}`",
        f"Точек в ряде: {len(series)} ({series[0]['date']} .. {series[-1]['date']})",
        "",
        "## 1. Детерминированная цепочка через MCP (без LLM)",
        "",
    ]

    # 2. Deterministic chain through the router — proves data passing.
    events = []
    fetched = router.call_tool(
        "currency_yahoo__fetch_rates",
        {"currency_code": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31"},
    )
    events.append(fetched)
    analyzed = router.call_tool(
        "currency_yahoo__analyze_rates",
        {"data": fetched.result, "currency_code": "USD"},  # raw previous output
    )
    events.append(analyzed)
    saved = router.call_tool(
        "currency_yahoo__save_report",
        {"data": analyzed.result, "title": "Курс USD за 2025"},  # raw previous output
    )
    events.append(saved)
    for event in events:
        status = "OK" if event.ok else "FAILED"
        print(f"[mcp] {event.server}__{event.tool} {status}")
        body = event.result or event.error
        log.append(f"- `{event.tool}` **{status}**: `{body[:220]}{'…' if len(body) > 220 else ''}`")
    if not all(e.ok for e in events):
        print("error: deterministic chain failed", file=sys.stderr)
        return 1

    summary = json.loads(analyzed.result)
    log += [
        "",
        f"Итог анализа: {summary['count']} точек, "
        f"start {summary['start_rate']} -> end {summary['end_rate']}, "
        f"{summary['change_pct']}% ({summary['trend']})",
        "",
        "## 2. Агент сам строит цепочку (один промпт, реальная модель)",
        "",
        f"Промпт: «{prompt}»",
        "",
        "```",
    ]

    # 3. One prompt, the agent runs the whole chain by itself.
    session = make_session(
        "mcp-currency-yahoo-demo",
        args.agent,
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(),
        invariants=False,
        mcp_router=router,
    )
    print(f">>> {prompt}")
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
        status = "OK" if event.ok else "FAILED"
        body = (event.result or event.error).replace("\n", " ")
        body = body if len(body) <= 220 else body[:220] + "…"
        log.append(f"- `{event.server}__{event.tool}` **{status}**: `{body}`")
    log.append("")
    log.append(f"Раундов LLM с директивами: {len(session.mcp_events)}")
    log.append("")

    # 4. Prove the report really landed in the store.
    stored = json.loads(reports_path.read_text(encoding="utf-8"))
    log += [
        "## 3. Хранилище отчётов после прогона",
        "",
        "```json",
        json.dumps(stored, ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    Path(args.out).write_text("\n".join(log), encoding="utf-8")
    print(f"\nlog written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
