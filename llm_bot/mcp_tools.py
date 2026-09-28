"""MCP tool integration: one bridge per server, one router for all of them.

``MCPToolBridge`` owns the connection to a single MCP server (stdio child
process or Streamable HTTP endpoint), fetches its tool catalog and performs
tool calls. ``MCPRouter`` aggregates several bridges behind one facade: it
renders a single system-prompt block describing every available tool and
routes qualified calls (``server__tool``) to the right bridge, isolating a
failing server so the rest keep working.

Because the project's :class:`~llm_bot.client.LLMClient` speaks plain
chat-completions (no native ``tools`` parameter), the agent integrates tools
through a **prompt protocol**: the router's block instructs the model to reply
with ``{"call_tool": {"name": ..., "arguments": {...}}}`` when a tool is
needed. :meth:`MCPRouter.parse_directive` recognizes that reply; the session
then executes the call and feeds the result back as a service turn.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

# Default config location (project convention: data/ holds runtime YAML).
DEFAULT_MCP_CONFIG = Path("data") / "mcp.yaml"

# Default tool-loop budget for a session built around a router: max model
# replies per turn (a batch of calls in ONE reply costs one round). The
# config's top-level ``max_rounds`` key overrides it for long cross-server
# chains (see MCPRouter.max_rounds).
DEFAULT_MAX_ROUNDS = 4


def _config_path(path: Path | None = None) -> Path:
    """Resolve the config location: explicit path > ``$MCP_CONFIG`` > default.

    The env override lets every process of a distributed setup — the CLI,
    the scheduler MCP server validating ``mcp_call`` targets, and the
    daemon's bridge factory — point at ONE config (tests and demos use it
    to stay self-contained without touching ``data/``).
    """
    return path or Path(os.environ.get("MCP_CONFIG") or DEFAULT_MCP_CONFIG)


class MCPToolError(RuntimeError):
    """Raised when a tool call fails on the MCP side."""


@dataclass(frozen=True)
class ToolSpec:
    """A normalized description of one MCP tool (for the prompt block)."""

    server: str
    name: str
    description: str
    parameters: dict  # JSON Schema "properties" mapping


@dataclass(frozen=True)
class MCPDirective:
    """A parsed tool-call directive from the model's reply."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class MCPEvent:
    """Diagnostics of one tool round-trip (mirrors ``MemoryEvent``)."""

    tool: str
    server: str
    ok: bool
    result: str = ""
    error: str = ""


class MCPToolBridge:
    """Connection to a single MCP server: catalog + tool calls.

    Either *command* (stdio child process) or *url* (Streamable HTTP) must be
    given. Each connection lives only for the duration of one operation —
    the child process is spawned per call and shut down cleanly afterwards,
    which keeps the bridge stateless and restart-safe.
    """

    def __init__(
        self,
        name: str,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command and not url:
            raise ValueError(
                f"MCP-сервер {name!r}: укажите command (stdio) или url (HTTP)."
            )
        self.name = name
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env

    # -- connection helpers ---------------------------------------------------

    @staticmethod
    def _resolve_command(command: str) -> str:
        """Map bare ``python``/``python3`` to the running interpreter.

        The bot almost always runs inside a venv; a config entry
        ``command: python`` would otherwise resolve to the system Python,
        which usually lacks the ``mcp`` package and every project import.
        """
        if Path(command).name in ("python", "python3"):
            return sys.executable
        return command

    def _stdio_params(self) -> StdioServerParameters:
        assert self.command is not None
        return StdioServerParameters(
            command=self._resolve_command(self.command),
            args=self.args,
            env={**self.env} if self.env else None,
        )

    async def _run(self, action):
        """Open a session and run *action* (async callable) against it."""
        if self.command:
            async with stdio_client(self._stdio_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await action(session)
        assert self.url is not None
        async with streamablehttp_client(self.url) as (read, write, _sid):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await action(session)

    def _run_sync(self, action):
        """Run :meth:`_run` on a fresh event loop (sync call sites)."""
        import anyio

        return anyio.run(self._run, action)

    # -- public API -----------------------------------------------------------

    def list_tools(self) -> list[ToolSpec]:
        """Fetch the server's tool catalog, normalized to :class:`ToolSpec`."""

        async def _fetch(session: ClientSession) -> list[ToolSpec]:
            result = await session.list_tools()
            specs = []
            for tool in result.tools:
                schema = tool.inputSchema or {}
                specs.append(
                    ToolSpec(
                        server=self.name,
                        name=tool.name,
                        description=(tool.description or "").strip(),
                        parameters=schema.get("properties", {}),
                    )
                )
            return specs

        return self._run_sync(_fetch)

    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute *name* with *arguments* and return the text result.

        Raises :class:`MCPToolError` when the server reports an error or the
        tool result is marked ``isError``.
        """

        async def _call(session: ClientSession) -> str:
            result = await session.call_tool(name, arguments or {})
            parts = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(text)
            text = "\n".join(parts).strip()
            if result.isError or not text:
                raise MCPToolError(text or "(пустой результат инструмента)")
            return text

        try:
            return self._run_sync(_call)
        except MCPToolError:
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised as MCPToolError
            # anyio wraps child-process failures into an ExceptionGroup;
            # unwrap so the user sees the root cause, not the container.
            # Duck-typed: builtin BaseExceptionGroup exists only on 3.11+,
            # while anyio's backport defines the same ``exceptions`` attribute.
            cause = exc
            while getattr(cause, "exceptions", None):
                cause = cause.exceptions[0]
            raise MCPToolError(f"{type(cause).__name__}: {cause}") from cause


class MCPRouter:
    """Facade over several :class:`MCPToolBridge` instances.

    * renders ONE system-prompt block from every bridge's catalog,
    * namespaces tool names (``server__tool``) to avoid clashes,
    * routes parsed directives to the owning bridge,
    * isolates server failures per call.
    """

    def __init__(
        self,
        bridges: list[MCPToolBridge],
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self._bridges = {b.name: b for b in bridges}
        # Tool-loop budget for sessions built around this router: how many
        # model replies one turn may consume (a batch of calls in ONE reply
        # costs one round). Long cross-server chains need more than the
        # default; the config's top-level ``max_rounds`` key overrides it.
        self.max_rounds = max(1, int(max_rounds))
        # Qualified name -> (bridge, tool); built lazily on first render.
        self._tools: dict[str, tuple[MCPToolBridge, ToolSpec]] = {}

    # -- catalog --------------------------------------------------------------

    def refresh(self) -> None:
        """Re-fetch every bridge's catalog (skips unreachable servers)."""
        self._tools.clear()
        for bridge in self._bridges.values():
            try:
                for spec in bridge.list_tools():
                    qualified = f"{bridge.name}__{spec.name}"
                    self._tools[qualified] = (bridge, spec)
            except Exception as exc:  # noqa: BLE001 - isolation by design
                print(
                    f"[mcp] сервер {bridge.name!r} недоступен: {exc}",
                    file=sys.stderr,
                )

    @property
    def tools(self) -> dict[str, ToolSpec]:
        """Qualified-name -> spec mapping (call :meth:`refresh` first)."""
        return {q: spec for q, (_, spec) in self._tools.items()}

    # -- prompt protocol ------------------------------------------------------

    def render_tools_block(self) -> str:
        """Render the system block advertising tools and the call directive.

        Empty string when no tools are available (the block is then not
        injected at all and the model never sees the protocol).
        """
        if not self._tools:
            return ""
        lines = [
            "Доступные внешние инструменты (MCP). Их результаты — единственный"
            " источник фактов о внешних данных; не выдумывай результаты.",
            "",
        ]
        for qualified, (_, spec) in sorted(self._tools.items()):
            params = []
            if spec.parameters:
                shown = ", ".join(spec.parameters.keys())
                params.append(f"параметры: {shown}")
            desc = spec.description.splitlines()[0] if spec.description else ""
            lines.append(f"- {qualified} — {desc} {'; '.join(params)}".rstrip())
        lines += [
            "",
            "Если для ответа нужен инструмент, ответь ТОЛЬКО валидным JSON"
            " без пояснений:",
            '  {"call_tool": {"name": "<имя>", "arguments": {...}}}',
            "Если действий несколько (например, просят добавить три заметки),"
            " перечисли их в массиве:",
            '  {"call_tool": [{"name": ..., "arguments": ...}, ...]}',
            "Для сложной задачи строй ЦЕПОЧКУ вызовов: инструменты могут"
            " стыковаться друг с другом (в описаниях сказано, что принимает"
            " каждый). Выполняй их ПО ОДНОМУ: верни директиву только для"
            " следующего шага, получишь результат служебным сообщением —"
            " передай его в аргументы следующего инструмента без изменений,"
            " затем переходи к шагу после него.",
            "Директива — это ОБЫЧНЫЙ ТЕКСТ ответа: никакого специального"
            " синтаксиса вызова инструментов, никакой служебной разметки"
            " вроде tool_call-токенов. Если намерен вызвать инструмент —"
            " верни только JSON-директиву из формата выше.",
            "Никогда не выдумывай результаты и не отвечай, будто действие"
            " выполнено, без реального вызова инструмента.",
            "Служебные сообщения «Результат инструмента …» пишет ТОЛЬКО"
            " система после реального вызова — не воспроизводи этот формат"
            " в своих ответах.",
            "Если подходящего инструмента для запроса нет — честно скажи об"
            " этом обычным текстом. Не подменяй запрос другой задачей:"
            " например, сбор данных нельзя заменять напоминанием или"
            " выдуманным действием.",
            "Если инструмент не нужен — отвечай обычным текстом, JSON не",
            "используй.",
        ]
        return "\n".join(lines)

    @staticmethod
    def parse_directives(reply: str) -> list[MCPDirective]:
        """Parse the tool-call directive in the model's reply.

        Accepts both forms documented in the prompt block: a single call
        (``{"call_tool": {...}}``) and a batch (``{"call_tool": [...]}``).
        Returns an empty list when the reply is not a directive (a normal
        user-facing answer). Malformed entries inside a batch are skipped.

        The directive is located by scanning for the FIRST *valid* JSON
        object that carries ``call_tool`` (``json.JSONDecoder.raw_decode``),
        not by slicing from the first ``{`` to the last ``}``: live models
        prepend planning text before the directive, and that text can
        contain brace fragments which break the naive slice.

        When nothing parses but the reply clearly tries to be a directive
        (it mentions ``call_tool``), a best-effort bracket repair runs: it
        drops closers with no matching opener and closes the unclosed
        brackets. Live finding it fixes: ``{"call_tool": {...}}]`` — the
        model dropped the outer closing brace and left a stray ``]``, the
        chain stalled and the raw JSON leaked to the user. (A bracket
        inside a JSON string literal would be dropped too — accepted,
        because the alternative is losing the whole directive.)
        """
        text = reply.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        directives = MCPRouter._scan_directive_json(text)
        if not directives and "call_tool" in text:
            repaired = MCPRouter._close_unbalanced_brackets(text)
            if repaired != text:
                directives = MCPRouter._scan_directive_json(repaired)
        return directives

    @staticmethod
    def _scan_directive_json(text: str) -> list[MCPDirective]:
        """Return directives from the first valid call_tool JSON in *text*."""
        decoder = json.JSONDecoder()
        idx = text.find("{")
        while idx != -1:
            try:
                data, _end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
                continue
            if isinstance(data, dict) and "call_tool" in data:
                call = data["call_tool"]
                if isinstance(call, dict):
                    call = [call]
                directives: list[MCPDirective] = []
                if isinstance(call, list):
                    for entry in call:
                        if isinstance(entry, dict) and isinstance(
                            entry.get("name"), str
                        ):
                            arguments = entry.get("arguments")
                            directives.append(
                                MCPDirective(
                                    name=entry["name"],
                                    arguments=(
                                        arguments
                                        if isinstance(arguments, dict)
                                        else {}
                                    ),
                                )
                            )
                return directives
            idx = text.find("{", idx + 1)
        return []

    @staticmethod
    def _close_unbalanced_brackets(text: str) -> str:
        """Drop closers with no matching opener; close what stays open."""
        pairs = {")": "(", "]": "[", "}": "{"}
        closers = {opener: closer for closer, opener in pairs.items()}
        stack: list[str] = []
        out: list[str] = []
        for ch in text:
            if ch in "([{":
                stack.append(ch)
                out.append(ch)
            elif ch in ")]}":
                if stack and stack[-1] == pairs[ch]:
                    stack.pop()
                    out.append(ch)
                # a mismatched closer is dropped (stray ']' and friends)
            else:
                out.append(ch)
        out.extend(closers[c] for c in reversed(stack))
        return "".join(out)

    # -- execution ------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> MCPEvent:
        """Execute a (possibly qualified) tool call, never raising.

        The outcome is always an :class:`MCPEvent`; failures carry the error
        text in ``event.error`` so the model can see and explain them.
        """
        if name not in self._tools:
            return MCPEvent(
                tool=name,
                server="?",
                ok=False,
                error=f"неизвестный инструмент: {name}",
            )
        bridge, spec = self._tools[name]
        try:
            text = bridge.call_tool(spec.name, arguments)
        except Exception as exc:  # noqa: BLE001 - isolation by design
            return MCPEvent(
                tool=spec.name, server=bridge.name, ok=False, error=str(exc)
            )
        return MCPEvent(tool=spec.name, server=bridge.name, ok=True, result=text)

    def server_names(self) -> list[str]:
        """Configured server names, in definition order."""
        return list(self._bridges)


# -- config loading (project convention: data/*.yaml) -------------------------


def load_mcp_settings(path: Path | None = None) -> dict:
    """Read the whole MCP config document from *path*.

    When *path* is omitted, the location comes from the ``MCP_CONFIG``
    environment variable (if set), falling back to ``data/mcp.yaml``.

    Returns the raw mapping (top-level keys: ``servers``, ``max_rounds``);
    a missing file yields ``{}``.
    """
    config_path = _config_path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def load_mcp_config(path: Path | None = None) -> dict[str, dict]:
    """Read the MCP server map from *path* (default ``data/mcp.yaml``).

    Returns ``{name: {command|url, args?, env?}}``; missing file -> ``{}``.
    """
    servers = load_mcp_settings(path).get("servers", {})
    return servers if isinstance(servers, dict) else {}


def router_from_config(
    names: list[str] | None = None, path: Path | None = None
) -> MCPRouter | None:
    """Build an :class:`MCPRouter` from the YAML config.

    *names* selects servers (``None``/``["all"]`` = every configured one).

    Failure policy:

    * the ``--mcp`` flag is absent (``names is None``) -> ``None``: MCP is
      OFF by default, even when a config file exists (opt-in by design, so
      unrelated tests/sessions never get a tools block injected);
    * no flag AND no config -> ``None`` (plain no-op);
    * an EXPLICIT ``--mcp ...`` selection with a missing/empty config, an
      unknown server name, or servers that expose no tools -> ``ValueError``
      (the user asked for MCP; failing silently would look like the model
      ignoring tools, so the process must not start).
    """
    config_path = _config_path(path)
    if names is None:
        return None
    servers = load_mcp_config(config_path)
    if not servers:
        raise ValueError(
            f"Указан --mcp, но конфигурация MCP не найдена или пуста: "
            f"{config_path}. Скопируйте mcp.example.yaml в data/mcp.yaml "
            "и перечислите нужные серверы."
        )
    if names is None or [n.lower() for n in names] == ["all"]:
        selected = list(servers)
    else:
        selected = []
        for name in names:
            key = name.strip()
            if key not in servers:
                raise ValueError(
                    f"MCP-сервер {key!r} не найден в {config_path}. "
                    f"Доступны: {', '.join(servers)}"
                )
            selected.append(key)
    if not selected:
        return None
    bridges = [
        MCPToolBridge(
            name,
            command=str(cfg.get("command") or "") or None,
            args=[str(a) for a in cfg.get("args", [])],
            url=cfg.get("url"),
            env={str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
        )
        for name, cfg in ((n, servers[n]) for n in selected)
    ]
    settings = load_mcp_settings(config_path)
    raw_rounds = settings.get("max_rounds")
    router = MCPRouter(
        bridges,
        max_rounds=(
            DEFAULT_MAX_ROUNDS if raw_rounds is None else int(raw_rounds)
        ),
    )
    router.refresh()
    if not router.tools:
        if names is not None:
            raise ValueError(
                f"MCP-сервер(ы) {', '.join(selected)} не отдают ни одного "
                "инструмента (не запускаются?). Проверьте их командой "
                "python -m llm_bot --list-mcp."
            )
        return None
    return router
