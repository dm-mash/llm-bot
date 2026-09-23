"""Tests for MCPToolBridge / MCPRouter (no real server needed).

The bridge is exercised against a fake session via monkeypatched internals;
the router logic (namespacing, directive parsing, routing, error isolation)
is covered with stub bridges.
"""

from __future__ import annotations

import pytest

from llm_bot.mcp_tools import (
    MCPDirective,
    MCPEvent,
    MCPToolBridge,
    MCPToolError,
    MCPRouter,
    ToolSpec,
    load_mcp_config,
    router_from_config,
)


class StubBridge(MCPToolBridge):
    """A bridge that never opens a connection; catalog/calls come from maps."""

    def __init__(self, name, tools, fail_on=()):
        super().__init__(name, command="stub")
        self._tools = tools
        self._fail_on = set(fail_on)
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return [
            ToolSpec(server=self.name, name=n, description=f"{n} desc", parameters=p)
            for n, p in self._tools.items()
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self._fail_on:
            raise MCPToolError(f"{name} exploded")
        return f"{self.name}:{name} ok"


# -- bridge --------------------------------------------------------------------


def test_bridge_requires_command_or_url():
    with pytest.raises(ValueError):
        MCPToolBridge("broken")


def test_bridge_list_tools_normalizes(monkeypatch):
    bridge = MCPToolBridge("x", command="noop")

    class FakeTool:
        name = "t"
        description = "  does things  "
        inputSchema = {"properties": {"a": {"type": "string"}}}

    class FakeResult:
        tools = [FakeTool()]

    async def fake_run(action):
        class FakeSession:
            async def list_tools(self):
                return FakeResult()

        return await action(FakeSession())

    monkeypatch.setattr(bridge, "_run", fake_run)
    specs = bridge.list_tools()
    assert specs[0].server == "x"
    assert specs[0].name == "t"
    assert specs[0].description == "does things"
    assert "a" in specs[0].parameters


# -- router: catalog & namespacing ----------------------------------------------


def test_router_namespaces_and_merges_catalogs():
    router = MCPRouter(
        [
            StubBridge("notes", {"find_notes": {"query": {"type": "string"}}}),
            StubBridge("time", {"find_notes": {}, "now": {}}),
        ]
    )
    router.refresh()
    assert set(router.tools) == {
        "notes__find_notes",
        "time__find_notes",
        "time__now",
    }
    block = router.render_tools_block()
    assert "notes__find_notes" in block
    assert '{"call_tool"' in block


def test_router_empty_block_when_no_tools():
    router = MCPRouter([StubBridge("empty", {})])
    router.refresh()
    assert router.render_tools_block() == ""


# -- router: directive parsing --------------------------------------------------


def test_parse_directives_accepts_plain_json():
    reply = '{"call_tool": {"name": "notes__add_note", "arguments": {"text": "x"}}}'
    assert MCPRouter.parse_directives(reply) == [
        MCPDirective(name="notes__add_note", arguments={"text": "x"})
    ]


def test_parse_directives_accepts_batch():
    reply = (
        '{"call_tool": ['
        '{"name": "notes__add_note", "arguments": {"text": "a"}},'
        '{"name": "notes__add_note", "arguments": {"text": "b"}},'
        '{"name": "notes__list_notes", "arguments": {}}]}'
    )
    directives = MCPRouter.parse_directives(reply)
    assert [d.name for d in directives] == [
        "notes__add_note",
        "notes__add_note",
        "notes__list_notes",
    ]


def test_parse_directives_tolerates_fence_and_prose():
    reply = 'Вот вызов:\n```json\n{"call_tool": {"name": "t", "arguments": {}}}\n```'
    directives = MCPRouter.parse_directives(reply)
    assert len(directives) == 1 and directives[0].name == "t"


def test_parse_directives_skips_malformed_batch_entries():
    reply = '{"call_tool": [{"name": "ok"}, {"arguments": {}}, "junk", {"name": 42}]}'
    directives = MCPRouter.parse_directives(reply)
    assert [d.name for d in directives] == ["ok"]


def test_parse_directives_rejects_normal_text():
    assert MCPRouter.parse_directives("Обычный ответ пользователю.") == []
    assert MCPRouter.parse_directives('{"other": 1}') == []
    assert MCPRouter.parse_directives("{broken json") == []
    assert MCPRouter.parse_directives('{"call_tool": "junk"}') == []


# -- router: execution & isolation -----------------------------------------------


def test_router_routes_call_to_owning_bridge():
    notes = StubBridge("notes", {"find_notes": {}})
    time = StubBridge("time", {"now": {}})
    router = MCPRouter([notes, time])
    router.refresh()

    event = router.call_tool("notes__find_notes", {"query": "x"})
    assert event.ok and event.result == "notes:find_notes ok"
    assert notes.calls == [("find_notes", {"query": "x"})]
    assert time.calls == []


def test_router_isolates_failure_and_unknown_tool():
    notes = StubBridge("notes", {"boom": {}}, fail_on=("boom",))
    router = MCPRouter([notes])
    router.refresh()

    failed = router.call_tool("notes__boom", {})
    assert not failed.ok and "exploded" in failed.error

    unknown = router.call_tool("nope", {})
    assert not unknown.ok and "неизвестный инструмент" in unknown.error
    # Router itself never raised — the session turn can continue.


def test_router_lists_servers_in_order():
    router = MCPRouter(
        [StubBridge("b", {}), StubBridge("a", {})]
    )
    assert router.server_names() == ["b", "a"]


# -- config loading ---------------------------------------------------------------


def test_load_mcp_config_missing_file(tmp_path):
    assert load_mcp_config(tmp_path / "absent.yaml") == {}


def test_router_from_config_selects_and_validates(tmp_path):
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "servers:\n"
        "  notes:\n"
        "    command: python\n"
        "    args: [srv.py]\n"
        "    env:\n"
        "      NOTES_DB: data/notes.json\n"
        "  time:\n"
        "    command: mcp-server-time\n",
        encoding="utf-8",
    )

    # Nothing selected + unreachable servers -> no router (feature off).
    assert router_from_config([], path=config) is None

    with pytest.raises(ValueError):
        router_from_config(["nope"], path=config)


def test_router_from_config_explicit_flag_with_missing_config_fails(tmp_path):
    """`--mcp notes` without a config must fail loudly, not silently."""
    with pytest.raises(ValueError, match="конфигурация MCP не найдена"):
        router_from_config(["notes"], path=tmp_path / "absent.yaml")


def test_router_from_config_silent_when_flag_absent(tmp_path):
    """No --mcp flag + no config = feature off (silent, by design)."""
    assert router_from_config(None, path=tmp_path / "absent.yaml") is None


def test_router_from_config_dead_server_fails_loudly(tmp_path):
    """An explicit selection whose servers expose no tools must fail."""
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "servers:\n"
        "  broken:\n"
        "    command: python\n"
        "    args: [-c, \"import sys; sys.exit(2)\"]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="не отдают ни одного инструмента"):
        router_from_config(["broken"], path=config)
