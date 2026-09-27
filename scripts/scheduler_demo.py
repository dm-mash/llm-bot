#!/usr/bin/env python3
"""Live verification scenario for the scheduler (results are pasted into
results/mcp_scheduler_verification.md). Uses a temp store, no network."""

import anyio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.scheduler"
DAEMON = ROOT / "scripts" / "scheduler_daemon.py"


def _text(result):
    return result.content[0].text


async def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sched_verify_"))
    env = {**os.environ, "SCHEDULER_DB": str(tmp / "scheduler.json")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(ROOT),
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            catalog = await session.list_tools()
            print("== 1. Каталог инструментов ==")
            print(", ".join(sorted(t.name for t in catalog.tools)))

            print("\n== 2. Создание задач (reminder / mcp_call / llm_summary) ==")
            r1 = await session.call_tool(
                "schedule_task",
                {
                    "title": "Напоминание: пересобрать заметки",
                    "action": "reminder",
                    "payload": {"text": "проверь CRM"},
                    "delay_seconds": 1,
                },
            )
            print(_text(r1))
            r2 = await session.call_tool(
                "schedule_task",
                {
                    "title": "Периодический снимок CRM",
                    "action": "mcp_call",
                    "payload": {
                        "server": "notes",
                        "tool": "list_notes",
                        "arguments": {},
                    },
                    "every_seconds": 3600,
                    "group": "данные",
                },
            )
            print(_text(r2))
            r3 = await session.call_tool(
                "schedule_task",
                {
                    "title": "Утренний дайджест",
                    "action": "llm_summary",
                    "payload": {"sources": "all"},
                    "daily_at": "09:00",
                },
            )
            print(_text(r3))

            bad = await session.call_tool(
                "schedule_task", {"title": "битая", "action": "reminder"}
            )
            print("Задача без расписания отвергнута:", bad.isError)

            print("\n== 3. Список задач ==")
            print(_text(await session.call_tool("list_tasks", {})))

    print("\n== 4. Демон --once (отдельный процесс) ==")
    import time

    time.sleep(1.5)  # let the reminder's delay elapse
    proc = subprocess.run(
        [sys.executable, str(DAEMON), "--once", "--db", str(tmp / "scheduler.json")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    for line in proc.stderr.splitlines():
        if "выполнено" in line or "✔" in line or "✖" in line:
            print(line)

    import json

    data = json.loads((tmp / "scheduler.json").read_text(encoding="utf-8"))
    print("Задача once ->", data["tasks"][0]["status"])
    for res in data["results"]:
        print(f"Результат: ok={res['ok']} :: {res['summary'][:120]}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("\n== 5. Агрегированный результат (task_digest) ==")
            print(_text(await session.call_tool("task_digest", {"last_n": 1})))


if __name__ == "__main__":
    anyio.run(main)
