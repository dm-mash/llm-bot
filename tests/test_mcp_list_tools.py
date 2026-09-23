"""MCP connection tests: the client must connect and list the tool catalog.

Spawns the real ``mcp-server-time`` reference server (installed via
``requirements.txt``) as a child process and drives the same code path as
``scripts/mcp_list_tools.py``:

* the ``initialize`` handshake succeeds,
* ``tools/list`` returns the expected catalog with valid schemas.

No network and no LLM API is involved — the server runs locally on stdio.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

# The mcp-server-time reference server ships a console script; fall back to
# ``python -m mcp_server_time`` when only the package is importable.
HAS_SCRIPT = shutil.which("mcp-server-time") is not None
HAS_MODULE = importlib.util.find_spec("mcp_server_time") is not None

pytestmark = pytest.mark.skipif(
    not (HAS_SCRIPT or HAS_MODULE),
    reason="mcp-server-time is not installed",
)

# Tools the reference server is expected to expose.
EXPECTED_TOOLS = {"get_current_time", "convert_time"}


def _server_params() -> StdioServerParameters:
    if HAS_SCRIPT:
        return StdioServerParameters(command="mcp-server-time")
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server_time"])


async def _fetch_tools() -> tuple[str, dict]:
    """Open a session, handshake and return (server name, {tool: has_schema})."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()  # MCP handshake
            result = await session.list_tools()
            tools = {t.name: bool(t.inputSchema) for t in result.tools}
            return init.serverInfo.name, tools


def test_mcp_connection_and_tool_list():
    """Connection is established and the expected tool catalog is returned."""
    pytest.importorskip("anyio")
    import anyio

    server_name, tools = anyio.run(_fetch_tools)

    # Handshake gave us a server identity.
    assert server_name, "initialize must report a server name"

    # The tool catalog matches the reference server's contract.
    assert EXPECTED_TOOLS <= set(tools), f"missing tools: {EXPECTED_TOOLS - set(tools)}"

    # Every advertised tool carries an input schema (JSON Schema object).
    for name, has_schema in tools.items():
        assert has_schema, f"tool {name} has no input schema"


def test_script_smoke_run():
    """The deliverable script itself runs and exits 0 against the local server."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "mcp_list_tools.py"
    assert script.is_file(), f"missing deliverable script: {script}"

    import subprocess

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"
    assert "protocol :" in proc.stdout, "handshake info not printed"
    assert "get_current_time" in proc.stdout, "tool list not printed"
