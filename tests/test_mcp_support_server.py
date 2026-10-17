"""End-to-end tests for the support MCP server (real stdio child process).

Each test spawns ``llm_bot.mcp_servers.support`` through the same MCP client
stack the agent uses, so the day-33 requirement is covered directly: the desk is
reachable as tools, with typed parameters and text results. No mocks — the
server really runs, over a copy of the shipped demo desk.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.support"
DESK = ROOT / "support.example.json"

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def desk(tmp_path: Path) -> Path:
    path = tmp_path / "support.json"
    shutil.copy(DESK, path)
    return path


def _params(desk: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(ROOT),
        env={**os.environ, "SUPPORT_DB": str(desk)},
    )


async def _with_session(desk, action):
    async with stdio_client(_params(desk)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


def _text(result) -> str:
    return "".join(getattr(part, "text", "") for part in result.content)


async def test_server_lists_tools_with_schemas(desk):
    async def action(session: ClientSession):
        return await session.list_tools()

    catalog = await _with_session(desk, action)
    tools = {t.name: t for t in catalog.tools}
    assert {"get_ticket", "get_user", "list_tickets", "search_tickets"} <= set(tools)
    assert "ticket_id" in tools["get_ticket"].inputSchema.get("properties", {})
    assert {"user_id", "status"} <= set(
        tools["list_tickets"].inputSchema.get("properties", {})
    )
    assert "query" in tools["search_tickets"].inputSchema.get("properties", {})


async def test_get_ticket_returns_the_conversation(desk):
    async def action(session: ClientSession):
        return await session.call_tool("get_ticket", {"ticket_id": "T-5001"})

    text = _text(await _with_session(desk, action))
    assert "T-5001" in text
    assert "open" in text
    assert "Переписка:" in text
    # The customer line travels with the ticket: an id alone says nothing about
    # who is waiting.
    assert "anna@example.com" in text


async def test_get_user_returns_the_plan(desk):
    async def action(session: ClientSession):
        return await session.call_tool("get_user", {"user_id": "U-1002"})

    text = _text(await _with_session(desk, action))
    assert "dmitry@example.com" in text
    assert "business" in text


async def test_list_tickets_filters_by_user(desk):
    async def action(session: ClientSession):
        return await session.call_tool("list_tickets", {"user_id": "U-1001"})

    text = _text(await _with_session(desk, action))
    assert "T-5001" in text and "T-5003" in text
    assert "T-5002" not in text


async def test_search_tickets_finds_by_message_text(desk):
    async def action(session: ClientSession):
        return await session.call_tool("search_tickets", {"query": "резервные коды"})

    text = _text(await _with_session(desk, action))
    assert "T-5002" in text


async def test_an_unknown_id_reports_back_instead_of_failing(desk):
    # The tool returns the reason as text: the model must be able to read that
    # the id is wrong, not just see a transport error.
    async def action(session: ClientSession):
        return await session.call_tool("get_ticket", {"ticket_id": "T-0000"})

    text = _text(await _with_session(desk, action))
    assert "не найден" in text.lower()


async def test_a_missing_desk_file_is_explained(desk, tmp_path):
    async def action(session: ClientSession):
        return await session.call_tool("list_tickets", {})

    params = _params(tmp_path / "absent.json")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            text = _text(await session.call_tool("list_tickets", {}))

    assert "support.example.json" in text