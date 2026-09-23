#!/usr/bin/env python3
"""Live demo: the agent reaches the mock-CRM through its own MCP server.

Runs a two-turn dialog against a real LLM with the MCP tools attached:

1. "запиши заметку ..."  -> the model must emit the call_tool directive, the
   session executes add_note over MCP, and the model confirms using the result;
2. "найди заметку ..."   -> find_notes is called the same way and the answer
   quotes the stored note.

Afterwards the JSON database is dumped, proving the write really happened on
the API side (not in the model's imagination).

Examples:
    python scripts/mcp_notes_demo.py
    python scripts/mcp_notes_demo.py --agent assistant --out results/mcp_notes_demo.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMError
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.mcp_tools import MCPToolBridge, MCPRouter
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "scripts" / "notes_mcp_server.py"
DEFAULT_DB = ROOT / "data" / "notes_demo.json"
DEFAULT_OUT = "results/mcp_notes_demo.md"


def build_router(db_path: Path) -> MCPRouter:
    """Router with the notes MCP server pointed at *db_path*."""
    router = MCPRouter(
        [
            MCPToolBridge(
                "notes",
                command=sys.executable,
                args=[str(SERVER)],
                env={**os.environ, "NOTES_DB": str(db_path)},
            )
        ]
    )
    router.refresh()
    return router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="assistant")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()  # a clean slate makes the demo reproducible

    router = build_router(db_path)
    tools = sorted(router.tools)
    print(f"MCP tools: {tools}")
    if not tools:
        print("error: notes MCP server did not expose any tools", file=sys.stderr)
        return 1

    session = make_session(
        "mcp-notes-demo",
        args.agent,
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(),
        invariants=False,
        mcp_router=router,
    )

    turns = [
        "Запиши в CRM заметку «купить корм коту» с тегом «покупки».",
        "Выведи список всех заметок, которые сейчас лежат в CRM.",
    ]
    log: list[dict] = []
    for text in turns:
        print(f"\n>>> {text}")
        try:
            result = session.chat_with_details(text)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.reply)
        for event in session.mcp_events:
            status = "OK" if event.ok else "FAILED"
            print(
                f"    [mcp] {event.server}__{event.tool} {status}: "
                f"{event.result or event.error}"
            )
        log.append(
            {
                "user": text,
                "reply": result.reply,
                "mcp": [
                    {
                        "tool": e.tool,
                        "server": e.server,
                        "ok": e.ok,
                        "result": e.result,
                        "error": e.error,
                    }
                    for e in session.mcp_events
                ],
                "usage": {
                    "context_tokens": result.usage.context_tokens,
                    "reply_tokens": result.usage.reply_tokens,
                },
            }
        )
        session._mcp_events.clear()  # only report each turn's own events

    db_state = (
        json.loads(db_path.read_text(encoding="utf-8"))
        if db_path.exists()
        else []
    )
    print("\nnotes DB after the demo:")
    print(json.dumps(db_state, ensure_ascii=False, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# MCP notes demo — live transcript\n\n```json\n"
        + json.dumps({"turns": log, "notes_db": db_state}, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
