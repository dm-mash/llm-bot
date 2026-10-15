# MCP

## MCP connection

The project includes a minimal **MCP (Model Context Protocol) client** that
connects to a ready-made MCP server and prints its tool catalog
([`scripts/mcp_list_tools.py`](../scripts/mcp_list_tools.py), covered by
[`tests/test_mcp_list_tools.py`](../tests/test_mcp_list_tools.py); verification
log in [`results/mcp_verification.md`](../results/mcp_verification.md)).

```bash
# Local stdio server (default): the mcp-server-time reference server
python scripts/mcp_list_tools.py
# protocol : 2025-11-25
# server   : mcp-time 1.30.0
# tools    : 2
#   - get_current_time   (arg timezone: string, required)
#   - convert_time       (args source_timezone, time, target_timezone)

# Another local stdio server
python scripts/mcp_list_tools.py --command mcp-server-fetch

# Remote server over Streamable HTTP
python scripts/mcp_list_tools.py --url https://mcp.deepwiki.com/mcp
```

How it works: the client spawns the server as a child process and speaks MCP
over stdin/stdout (`stdio_client`), or opens a Streamable HTTP connection
(`streamablehttp_client`); then it performs the `initialize` handshake and
calls `tools/list`. Exit code 0 = connection OK. Dependencies: `mcp`
(official SDK) and `mcp-server-time` — both in [`requirements.txt`](../requirements.txt).

Ready-made servers to point the client at:

| Server | How to run / endpoint | Tools |
| --- | --- | --- |
| `mcp-server-time` | `pip install mcp-server-time` (default) | `get_current_time`, `convert_time` |
| `mcp-server-fetch` | `pip install mcp-server-fetch` | `fetch` (URL → markdown) |
| `mcp-server-git` | `pip install mcp-server-git` | `git_status`, `git_log`, `git_diff`, … |
| `mcp-server-memory` | `pip install mcp-server-memory` | knowledge-graph memory tools |
| `@modelcontextprotocol/server-filesystem` | `npx @modelcontextprotocol/server-filesystem <dir>` (Node.js) | sandboxed file read/write |
| DeepWiki | `https://mcp.deepwiki.com/mcp` | repo documentation Q&A |
| Context7 | `https://mcp.context7.com/mcp` | library documentation lookup |

## Own MCP server: notes-CRM + agent tool use

The project ships its **own MCP server** around a small API — a mock-CRM of
notes kept in a JSON file — and an agent integration that calls it
(verification report: [`results/mcp_notes_verification.md`](../results/mcp_notes_verification.md)).

- [`llm_bot/api/notes.py`](../llm_bot/api/notes.py) — the mock-CRM API
  (`add_note` / `list_notes` / `find_notes`) over `NOTES_DB`
  (default `data/notes.json`).
- [`llm_bot/mcp_servers/notes.py`](../llm_bot/mcp_servers/notes.py) — FastMCP
  stdio server registering the three tools with typed parameter schemas and
  text results (run as `python -m llm_bot.mcp_servers.notes`).
- [`llm_bot/mcp_tools.py`](../llm_bot/mcp_tools.py) — `MCPToolBridge` (one
  connection per server) and `MCPRouter` (aggregated catalog, `server__tool`
  namespacing, per-server failure isolation).

Agent integration uses a **prompt protocol** (the transport speaks plain
chat-completions): the router injects a system block advertising the tools;
when the model needs one it replies `{"call_tool": {"name": ..., "arguments":
{...}}}` — or a batch `{"call_tool": [{...}, ...]}` when one request needs
several actions. The session executes the call(s) over MCP, feeds the
results back as a service turn and the model produces the final answer
(rounds bounded by `mcp_max_rounds`; a batch costs one round). Each
round-trip is recorded in `session.mcp_events` and printed as `[mcp] ...`
lines on stderr. MCP is opt-in: without `--mcp` the feature is off even when
`data/mcp.yaml` exists, and an explicit `--mcp` with a broken/missing setup
fails with a clean one-line error instead of a traceback.

```bash
cp mcp.example.yaml data/mcp.yaml    # server list (stdio command or HTTP url)

python -m llm_bot --list-mcp                                   # catalog check
python -m llm_bot --agent assistant --mcp notes \
    "запиши заметку: купить корм коту, тег покупки"            # one-shot use
python scripts/mcp_notes_demo.py                               # live 2-turn demo
```

Adding more MCP servers is config-only — new entries in `data/mcp.yaml`
(e.g. `mcp-server-time` or `https://mcp.deepwiki.com/mcp`) selected with
`--mcp notes,time` / `--mcp all`. The bundled `git` server below is registered
the same way.

## MCP orchestration: one prompt, several servers

With several servers registered in `data/mcp.yaml`, the router advertises
them all in one tools block (`server__tool` namespacing) and the agent
composes **cross-server chains** by itself: analysis tools from
`currency_yahoo`, note-taking from `notes`, background scheduling from
`scheduler`. Each round-trip is recorded in `session.mcp_events` with its
own `server` attribution, so selection, routing and call order are all
observable.

```bash
python -m llm_bot --agent assistant --mcp all \
    "проанализируй курс доллара за 2025 год, сохрани отчёт, добавь заметку \
     с итогом и поставь ежедневный сбор курса на 09:00"
python scripts/mcp_orchestration_demo.py   # live: deterministic cross-server chain + agent stage
```

Long flows need more tool rounds than the default budget allows. The budget
lives in the MCP config as a top-level key next to `servers:`:

```yaml
# mcp.yaml — max model replies per turn in the tool loop
# (each round = one model reply; a batch of calls costs one round).
# Default: 4 — enough for a 3-call chain plus the final answer.
max_rounds: 8
servers:
  notes:          ...
  scheduler:      ...
  currency_yahoo: ...
```

Two more pieces make multi-server setups self-contained (tests, demos, the
daemon):

* the config location can be overridden with the `MCP_CONFIG` environment
  variable — the scheduler server validates `mcp_call` targets against the
  same file the session uses, without a hardcoded `data/mcp.yaml`;
* the round budget flows config → `MCPRouter.max_rounds` → the session's
  tool loop; when the budget runs out mid-chain the loop stops cleanly (the
  pending directive is left unexecuted).

Tests: `tests/test_mcp_orchestration.py` — a scripted-LLM long flow over
three REAL stdio servers (`fetch_rates → analyze_rates → save_report →
notes__add_note → scheduler__schedule_task`: strict order, per-server
routing, cross-server data passing, side effects in all three stores), a
batch directive mixing two servers in one round, the round-budget boundary,
and the config→router wiring.
