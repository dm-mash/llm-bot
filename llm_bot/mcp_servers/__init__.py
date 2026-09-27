"""Own MCP (FastMCP) servers of the bot.

Run each server with ``python -m llm_bot.mcp_servers.<name>`` from the
project root (so ``llm_bot`` is importable and relative ``data/`` paths in
env-based configuration resolve correctly). The servers speak MCP over
stdio: ONLY protocol frames go to stdout, diagnostics to stderr.

The actual logic lives in :mod:`llm_bot.api` — each server here is a thin
wrapper registering tools with typed schemas.
"""
