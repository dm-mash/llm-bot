"""End-to-end tests for the git MCP server (real stdio child process).

Spawns ``llm_bot.mcp_servers.git`` through the same MCP client stack the agent
uses and points ``GIT_REPO`` at a throwaway repository, so tool registration,
input schemas and real results are all covered without touching the project's
own checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.git"

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    for args in (
        ("-c", "init.defaultBranch=main", "init"),
        ("add", "-A"),
        ("commit", "-m", "init"),
    ):
        subprocess.run(["git", *args], cwd=str(tmp_path), env=env,
                       capture_output=True, check=True)
    return tmp_path


def _params(repo: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(ROOT),
        env={**os.environ, "GIT_REPO": str(repo)},
    )


async def _with_session(repo, action):
    async with stdio_client(_params(repo)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


async def test_server_lists_tools_with_schemas(git_repo):
    async def action(session: ClientSession):
        return await session.list_tools()

    catalog = await _with_session(git_repo, action)
    tools = {t.name: t for t in catalog.tools}
    assert {"current_branch", "list_files", "changed_files", "diff"} <= set(tools)
    assert "pattern" in tools["list_files"].inputSchema.get("properties", {})
    assert {"ref", "added_only"} <= set(
        tools["changed_files"].inputSchema.get("properties", {})
    )
    assert {"ref", "staged", "path"} <= set(
        tools["diff"].inputSchema.get("properties", {})
    )


async def test_server_returns_branch_and_files(git_repo):
    async def action(session: ClientSession):
        branch = await session.call_tool("current_branch", {})
        files = await session.call_tool("list_files", {})
        return branch, files

    branch, files = await _with_session(git_repo, action)

    assert not branch.isError
    assert branch.content[0].text.strip() == "main"
    assert not files.isError
    assert "a.txt" in files.content[0].text


async def test_server_diff_truncation_names_files_it_hid(git_repo):
    """The truncation marker is followed by the complete changed-file list.

    ``a.txt`` produces a huge diff so the body is cut; ``untracked.txt`` never
    appears in the diff at all. Both must be named after the marker, or the
    model reports "two files changed" about a ten-file working tree.
    """
    (git_repo / "a.txt").write_text("changed\n" * 3000, encoding="utf-8")
    (git_repo / "untracked.txt").write_text("u\n", encoding="utf-8")

    async def action(session: ClientSession):
        return await session.call_tool("diff", {})

    result = await _with_session(git_repo, action)

    text = result.content[0].text
    assert not result.isError
    assert "усечён" in text
    assert "a.txt" in text
    assert "untracked.txt" in text
