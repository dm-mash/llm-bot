#!/usr/bin/env python3
"""Connect to an MCP server over stdio or HTTP and list its available tools.

This is the deliverable for the "MCP connection" task: a minimal client that

1. establishes an MCP connection (``initialize`` handshake),
2. requests the tool catalog (``tools/list``),
3. prints protocol version, server identity and every tool with its
   input-argument schema.

Transport is chosen by the flags:

* ``--command <cmd> [args...]`` — spawn a local server as a child process and
  talk to it over stdin/stdout (the default uses the ``mcp-server-time``
  reference server installed with this project).
* ``--url <endpoint>``          — connect to a remote server via the Streamable
  HTTP transport (e.g. ``https://mcp.deepwiki.com/mcp``).

Exit code is 0 when the connection and the tool listing succeed, 1 otherwise —
so the script doubles as a smoke check.

Examples:
    python scripts/mcp_list_tools.py
    python scripts/mcp_list_tools.py --command mcp-server-fetch
    python scripts/mcp_list_tools.py --url https://mcp.deepwiki.com/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

# Default local reference server: installed via requirements.txt (mcp-server-time).
DEFAULT_COMMAND = "mcp-server-time"
DEFAULT_URL = "https://mcp.deepwiki.com/mcp"


async def list_tools_stdio(command: str, args: list[str] | None = None) -> None:
    """Spawn *command* as an MCP server on stdio and print its tool list."""
    params = StdioServerParameters(command=command, args=args or [])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _handshake_and_list(session)


async def list_tools_http(url: str) -> None:
    """Connect to an MCP server at *url* over Streamable HTTP and print tools."""
    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await _handshake_and_list(session)


async def _handshake_and_list(session: ClientSession) -> None:
    """Run initialize + tools/list on an open session and render the result."""
    init = await session.initialize()  # the MCP handshake itself
    print(f"protocol : {init.protocolVersion}")
    print(f"server   : {init.serverInfo.name} {init.serverInfo.version}")
    if init.instructions:
        print(f"about    : {init.instructions.strip()}")

    result = await session.list_tools()
    print(f"tools    : {len(result.tools)}")
    for tool in result.tools:
        print(f"\n- {tool.name}")
        if tool.description:
            desc = " ".join(tool.description.split())
            print(f"    {desc}")
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name, spec in props.items():
            req = "required" if name in required else "optional"
            typ = spec.get("type", "?")
            print(f"    arg {name}: {typ} ({req})")


def _resolve_default_command() -> tuple[str, list[str]]:
    """Return the command/args for the default reference time server."""
    if shutil.which(DEFAULT_COMMAND):
        return DEFAULT_COMMAND, []
    # Fallback: run the installed package as a module.
    return sys.executable, ["-m", "mcp_server_time"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--command",
        nargs="+",
        metavar=("CMD", "ARG"),
        help="spawn a local MCP server: command plus optional arguments",
    )
    group.add_argument(
        "--url",
        help="connect to a remote MCP server over Streamable HTTP",
    )
    ns = parser.parse_args()

    try:
        if ns.command:
            command, args = ns.command[0], ns.command[1:]
            asyncio.run(list_tools_stdio(command, args))
        elif ns.url:
            asyncio.run(list_tools_http(ns.url))
        else:
            command, args = _resolve_default_command()
            asyncio.run(list_tools_stdio(command, args))
    except Exception as exc:  # connection/handshake failure -> non-zero exit
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
