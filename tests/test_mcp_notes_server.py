"""End-to-end tests for the notes MCP server (real stdio child process).

Each test spawns ``scripts/notes_mcp_server.py`` through the same MCP client
stack the agent uses, so the three task requirements are covered directly:
tool registration, input-parameter schemas, and result return.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER = (
    Path(__file__).resolve().parent.parent / "scripts" / "notes_mcp_server.py"
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def _params(tmp_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env={**os.environ, "NOTES_DB": str(tmp_path / "notes.json")},
    )


async def _with_session(tmp_path, action):
    async with stdio_client(_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


async def test_server_lists_tools_with_schemas(tmp_path):
    """Registration + parameter descriptions (task bullets 1 and 2)."""

    async def action(session: ClientSession):
        return await session.list_tools()

    catalog = await _with_session(tmp_path, action)
    tools = {t.name: t for t in catalog.tools}
    assert {"add_note", "list_notes", "find_notes"} <= set(tools)

    add = tools["add_note"]
    assert set(add.inputSchema.get("properties", {})) >= {"text", "tags"}
    assert "text" in add.inputSchema.get("required", [])
    assert set(tools["find_notes"].inputSchema.get("required", [])) == {"query"}


async def test_server_returns_tool_results(tmp_path):
    """add -> find -> list round-trip returns real data (task bullet 3)."""

    async def action(session: ClientSession):
        added = await session.call_tool(
            "add_note", {"text": "Купить хлеб", "tags": ["покупки"]}
        )
        await session.call_tool(
            "add_note", {"text": "Отпуск в Сочи", "tags": ["отпуск"]}
        )
        found = await session.call_tool("find_notes", {"query": "сочи"})
        listed = await session.call_tool("list_notes", {"tag": "покупки"})
        return added, found, listed

    added, found, listed = await _with_session(tmp_path, action)

    assert not added.isError
    assert "id=1" in added.content[0].text
    assert "Отпуск в Сочи" in found.content[0].text
    assert "Купить хлеб" in listed.content[0].text
    assert listed.content[0].text.count("id=") == 1


async def test_server_error_result_on_bad_input(tmp_path):
    """Invalid input surfaces as an MCP error result, not a crash."""

    async def action(session: ClientSession):
        return await session.call_tool("add_note", {"text": "   "})

    result = await _with_session(tmp_path, action)
    assert result.isError
