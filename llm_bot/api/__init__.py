"""Domain APIs — backends wrapped by the MCP servers in ``llm_bot.mcp_servers``.

Each module here holds the actual business logic (storage, computation,
external calls) so it can be unit-tested without spawning MCP processes;
the servers in :mod:`llm_bot.mcp_servers` are thin FastMCP wrappers.
"""
