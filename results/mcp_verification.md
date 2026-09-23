# MCP connection — verification report

Task: connect to an MCP server and print the list of available tools.

Deliverables:

- [`scripts/mcp_list_tools.py`](../scripts/mcp_list_tools.py) — the client
  (stdio **and** Streamable HTTP transports).
- [`tests/test_mcp_list_tools.py`](../tests/test_mcp_list_tools.py) — pytest
  coverage of the connection + tool catalog.
- [`requirements.txt`](../requirements.txt) — added `mcp` (official Python SDK)
  and `mcp-server-time` (reference server used as the local target).

## What the client does

1. Spawns `mcp-server-time` as a child process (stdio transport) or opens a
   Streamable HTTP connection (`--url`).
2. Performs the MCP `initialize` handshake (protocol version + server info).
3. Calls `tools/list` and prints every tool with its description and the
   required/optional arguments parsed from the input JSON Schema.

## Run log

```
$ .venv/bin/python scripts/mcp_list_tools.py
protocol : 2025-11-25
server   : mcp-time 1.30.0
tools    : 2

- get_current_time
    Get current time in a specific timezone
    arg timezone: string (required)

- convert_time
    Convert time between timezones
    arg source_timezone: string (required)
    arg time: string (required)
    arg target_timezone: string (required)
```

Remote HTTP target (no API key needed):

```
$ .venv/bin/python scripts/mcp_list_tools.py --url https://mcp.deepwiki.com/mcp
protocol : 2025-11-25
server   : DeepWiki 2.14.3
about    : DeepWiki MCP provides AI-powered documentation for GitHub repositories.
...
tools    : 3   # read_wiki_structure, read_wiki_contents, ask_wiki_question
```

## Test results

```
$ .venv/bin/python -m pytest tests/test_mcp_list_tools.py -v
tests/test_mcp_list_tools.py::test_mcp_connection_and_tool_list PASSED
tests/test_mcp_list_tools.py::test_script_smoke_run PASSED
============================== 2 passed in 1.64s ===============================

$ .venv/bin/python -m pytest -q
305 passed in 2.41s        # full suite, nothing broken
```

Checks performed:

- [x] connection is established (`initialize` returns protocol `2025-11-25`
      and server identity),
- [x] the tool list is returned correctly (`get_current_time`,
      `convert_time`; every tool has an input schema),
- [x] script exit code is 0; failures (bad command, unreachable URL) exit 1,
- [x] tests skip cleanly when `mcp-server-time` is not installed.

## Ready-made MCP servers to connect to

| Server | Install / endpoint | Tools |
| --- | --- | --- |
| `mcp-server-time` | `pip install mcp-server-time` (default here) | `get_current_time`, `convert_time` |
| `mcp-server-fetch` | `pip install mcp-server-fetch` | `fetch` (URL → markdown) |
| `mcp-server-git` | `pip install mcp-server-git` | `git_status`, `git_log`, `git_diff`, … |
| `mcp-server-memory` | `pip install mcp-server-memory` | knowledge-graph memory tools |
| `@modelcontextprotocol/server-filesystem` | `npx` (Node.js) | sandboxed file read/write |
| DeepWiki | `https://mcp.deepwiki.com/mcp` | repo documentation Q&A |
| Context7 | `https://mcp.context7.com/mcp` | library documentation lookup |
