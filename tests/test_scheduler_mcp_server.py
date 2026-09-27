"""End-to-end tests for the scheduler MCP server + daemon (real processes).

Each test spawns the ``llm_bot.mcp_servers.scheduler`` module through the
same MCP client stack the agent uses, covering the task requirements directly:
tool
registration with typed schemas, delayed/periodic task creation persisted to
JSON, and the aggregated-result view (``task_digest``). The daemon smoke test
runs ``scripts/scheduler_daemon.py --once`` as a subprocess against the same
store, proving the two processes cooperate through the JSON file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.scheduler"
DAEMON = ROOT / "scripts" / "scheduler_daemon.py"

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def _params(tmp_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(ROOT),
        env={
            **os.environ,
            "SCHEDULER_DB": str(tmp_path / "scheduler.json"),
        },
    )


async def _with_session(tmp_path, action):
    async with stdio_client(_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


def _text(result) -> str:
    return result.content[0].text


async def test_server_lists_tools_with_schemas(tmp_path):
    """Registration + parameter descriptions."""

    async def action(session: ClientSession):
        return await session.list_tools()

    catalog = await _with_session(tmp_path, action)
    tools = {t.name: t for t in catalog.tools}
    assert {
        "schedule_task",
        "list_tasks",
        "cancel_task",
        "pause_task",
        "resume_task",
        "get_task_results",
        "task_digest",
    } <= set(tools)

    sched = tools["schedule_task"]
    props = set(sched.inputSchema.get("properties", {}))
    assert {"title", "action"} <= props
    assert {"title", "action"} <= set(sched.inputSchema.get("required", []))
    assert "task_id" in tools["cancel_task"].inputSchema.get("required", [])


async def test_schedule_task_roundtrip_and_error(tmp_path):
    """Create → list → digest; invalid input surfaces as an MCP error."""

    async def action(session: ClientSession):
        created = await session.call_tool(
            "schedule_task",
            {
                "title": "Напоминание про чай",
                "action": "reminder",
                "payload": {"text": "завари чай"},
                "delay_seconds": 30,
            },
        )
        listed = await session.call_tool("list_tasks", {})
        bad = await session.call_tool("schedule_task", {"title": "x", "action": "reminder"})
        invented = await session.call_tool(
            "schedule_task",
            {
                "title": "Курс доллара",
                "action": "collect_usd_rate",  # invented name, empty payload
                "every_seconds": 3600,
            },
        )
        return created, listed, bad, invented

    created, listed, bad, invented = await _with_session(tmp_path, action)

    assert not created.isError
    assert "id=" in _text(created)
    assert "Напоминание про чай" in _text(listed)
    assert bad.isError  # no schedule selector → must be a visible error
    assert invented.isError  # invented action name → rejected at creation
    assert "mcp_call" in _text(invented)


async def test_cancelled_task_disappears_from_active(tmp_path):
    async def action(session: ClientSession):
        created = await session.call_tool(
            "schedule_task",
            {
                "title": "Временная",
                "action": "reminder",
                "payload": {"text": "..."},
                "every_seconds": 60,
            },
        )
        task_id = _text(created).split("id=")[1].split(",")[0].strip()
        await session.call_tool("cancel_task", {"task_id": task_id})
        active = await session.call_tool("list_tasks", {"status": "active"})
        cancelled = await session.call_tool("list_tasks", {"status": "cancelled"})
        return task_id, active, cancelled

    task_id, active, cancelled = await _with_session(tmp_path, action)
    assert task_id not in _text(active)
    assert task_id in _text(cancelled)


async def test_digest_reports_task_and_unknown_id_errors(tmp_path):
    async def action(session: ClientSession):
        await session.call_tool(
            "schedule_task",
            {
                "title": "Сбор",
                "action": "reminder",
                "payload": {"text": "данные"},
                "every_seconds": 60,
                "group": "данные",
            },
        )
        digest = await session.call_tool("task_digest", {"last_n": 2})
        by_group = await session.call_tool("task_digest", {"group": "данные"})
        miss = await session.call_tool("get_task_results", {"task_id": "нету"})
        return digest, by_group, miss

    digest, by_group, miss = await _with_session(tmp_path, action)
    assert "Сбор" in _text(digest)
    assert "Сбор" in _text(by_group)
    assert miss.isError


# ---------------------------------------------------------------------------
# Daemon smoke test: two real processes over one JSON store
# ---------------------------------------------------------------------------


def test_daemon_once_executes_due_task(tmp_path):
    """schedule (via MCP server) → run daemon --once → digest shows the run."""
    db = tmp_path / "scheduler.json"
    env = {**os.environ, "SCHEDULER_DB": str(db)}

    async def create():
        async with stdio_client(_params(tmp_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "schedule_task",
                    {
                        "title": "Через секунду",
                        "action": "reminder",
                        "payload": {"text": "сработало"},
                        "delay_seconds": 0.05,
                    },
                )
                assert not result.isError, _text(result)
                return _text(result)

    import anyio

    anyio.run(create)

    # The daemon is a separate process sharing the store via SCHEDULER_DB.
    proc = subprocess.run(
        [sys.executable, str(DAEMON), "--once", "--db", str(db)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "выполнено задач: 1" in proc.stderr

    data = json.loads(db.read_text(encoding="utf-8"))
    assert len(data["results"]) == 1
    assert data["results"][0]["ok"] is True
    assert "НАПОМИНАНИЕ" in data["results"][0]["summary"]
    assert "сработало" in data["results"][0]["summary"]
    # The once-task is done after its single run.
    assert data["tasks"][0]["status"] == "done"

    # A second --once pass has nothing due.
    proc2 = subprocess.run(
        [sys.executable, str(DAEMON), "--once", "--db", str(db)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert "выполнено задач: 0" in proc2.stderr
