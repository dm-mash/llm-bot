#!/usr/bin/env python3
"""Smoke-check the notes MCP server: list tools and call each one.

Starts the ``llm_bot.mcp_servers.notes`` module as a child process, performs
the MCP
handshake, prints the tool catalog and exercises add/list/find against a
temporary database. Prints PASS/FAIL per check and exits non-zero on failure.

    python scripts/check_notes_mcp.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER_MODULE = "llm_bot.mcp_servers.notes"


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "NOTES_DB": str(Path(tmp) / "notes.json")}
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULE],
            cwd=str(ROOT),
            env=env,
        )
        failures: list[str] = []

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                print(f"server   : {init.serverInfo.name} {init.serverInfo.version}")
                print(f"protocol : {init.protocolVersion}")

                catalog = await session.list_tools()
                names = {t.name for t in catalog.tools}
                print(f"tools    : {sorted(names)}")
                if {"add_note", "list_notes", "find_notes"} <= names:
                    print("PASS     : tool registration + schemas")
                else:
                    failures.append("catalog mismatch")

                schemas_ok = all(t.inputSchema.get("properties") for t in catalog.tools)
                print(f"PASS     : input parameter schemas" if schemas_ok
                      else "FAIL     : empty schema")
                if not schemas_ok:
                    failures.append("schema empty")

                added = await session.call_tool(
                    "add_note",
                    {"text": "Купить хлеб", "tags": ["покупки"]},
                )
                print(f"add_note -> {added.content[0].text}")
                if "id=1" not in added.content[0].text:
                    failures.append("add_note result")

                await session.call_tool(
                    "add_note", {"text": "Отпуск в Сочи", "tags": ["отпуск"]}
                )
                found = await session.call_tool(
                    "find_notes", {"query": "сочи"}
                )
                print(f"find_notes -> {found.content[0].text!r}")
                if "Отпуск в Сочи" not in found.content[0].text:
                    failures.append("find_notes result")

                listed = await session.call_tool("list_notes", {})
                print(f"list_notes -> {len(listed.content[0].text.splitlines())} lines")
                if listed.content[0].text.count("id=") != 2:
                    failures.append("list_notes count")

                tagged = await session.call_tool(
                    "list_notes", {"tag": "покупки"}
                )
                if "Купить хлеб" in tagged.content[0].text:
                    print("PASS     : tag filter")
                else:
                    failures.append("tag filter")

        if failures:
            print(f"FAILED: {failures}")
            return 1
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
