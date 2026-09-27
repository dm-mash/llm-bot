"""Action executors for scheduled tasks (used by the daemon).

Three actions cover the task's requirements:

* ``mcp_call`` — call **any tool of any configured MCP server**; this makes new
  data sources (currency rates, series episodes) a config change in
  ``data/mcp.yaml``, not daemon code;
* ``reminder`` — the result is simply the payload text; the daemon "delivers"
  it by writing it into the results journal (the user reads it later via
  ``task_digest``);
* ``llm_summary`` — aggregates recent results of the source tasks (all / a
  group / explicit ids) and asks the LLM for a human summary; when no LLM is
  configured it degrades to a deterministic summary so tests never need the
  network.

Every executor receives the task plus a small run context and returns a
:class:`~llm_bot.scheduler.TaskResult`. Raising is allowed — the daemon
converts exceptions into failed results, isolating a bad task from the loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Protocol

from llm_bot.mcp_tools import MCPToolBridge
from llm_bot.scheduler import (
    ACTION_LLM_SUMMARY,
    ACTION_MCP_CALL,
    ACTION_REMINDER,
    ScheduledTask,
    SchedulerError,
    TaskResult,
    iso,
    replace,
    utc_now,
)

# Result sources considered "fresh" for summaries (per source task).
SUMMARY_TAIL_RESULTS = 20
# Cap of each source summary in the material given to the LLM.
SUMMARY_RESULT_CHARS = 500


class Summarizer(Protocol):
    """Anything that can turn a prompt into a reply (LLMClient.chat-compatible)."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class ActionContext:
    """Everything an action may need besides the task itself.

    * *store* — read access to results of other tasks (for ``llm_summary``);
    * *bridge_factory* — builds an :class:`MCPToolBridge` by server name
      (injectable so tests can fake MCP without spawning processes);
    * *chat_factory* — returns an LLM ``chat`` callable or ``None`` when no
      LLM is configured (then summaries degrade to deterministic ones).
    """

    def __init__(
        self,
        store,
        bridge_factory: Callable[[str], MCPToolBridge] | None = None,
        chat_factory: Callable[[], Summarizer | None] | None = None,
    ) -> None:
        self.store = store
        self._bridge_factory = bridge_factory or _default_bridge_factory
        self._chat_factory = chat_factory or (lambda: None)


# ---------------------------------------------------------------------------
# mcp_call
# ---------------------------------------------------------------------------


def run_mcp_call(task: ScheduledTask, ctx: ActionContext) -> TaskResult:
    """Execute ``payload.server`` → ``payload.tool`` with ``payload.arguments``.

    Uses the same bridge stack as the chat agent, so the daemon and the user's
    session see identical tools from one ``data/mcp.yaml``.
    """
    payload = task.payload
    server = str(payload.get("server", "")).strip()
    tool = str(payload.get("tool", "")).strip()
    if not server or not tool:
        raise ValueError(
            "payload задачи mcp_call должен содержать 'server' и 'tool'."
        )
    # The model sometimes reuses the chat-qualified name ("notes__list_notes")
    # as the bare tool name — strip the matching server prefix.
    if tool.startswith(f"{server}__"):
        tool = tool.split("__", 1)[1]
    arguments = payload.get("arguments") or {}
    bridge = ctx._bridge_factory(server)
    text = bridge.call_tool(tool, arguments)
    return TaskResult(
        task_id=task.id,
        run_at=iso(utc_now()),
        ok=True,
        summary=f"{server}__{tool}: {text}"[:2000],
        details={"server": server, "tool": tool, "arguments": arguments},
    )


def _default_bridge_factory(server: str) -> MCPToolBridge:
    """Build a bridge from ``data/mcp.yaml`` for *server* (config-driven)."""
    from llm_bot.mcp_tools import load_mcp_config

    config = load_mcp_config()
    if server not in config:
        raise ValueError(
            f"MCP-сервер {server!r} не найден в data/mcp.yaml. "
            f"Доступны: {', '.join(config) or '—'}"
        )
    cfg = config[server]
    return MCPToolBridge(
        server,
        command=str(cfg.get("command") or "") or None,
        args=[str(a) for a in cfg.get("args", [])],
        url=cfg.get("url"),
        env={str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
    )


# ---------------------------------------------------------------------------
# reminder
# ---------------------------------------------------------------------------


def run_reminder(task: ScheduledTask, ctx: ActionContext) -> TaskResult:
    """A reminder fires into the results journal — nothing else to execute."""
    # The model sometimes names the text "message" instead of "text" —
    # accept both rather than silently reminding with the title only.
    text = (
        str(task.payload.get("text") or task.payload.get("message") or "")
    ).strip() or task.title
    return TaskResult(
        task_id=task.id,
        run_at=iso(utc_now()),
        ok=True,
        summary=f"НАПОМИНАНИЕ: {text}",
        details={"text": text},
    )


# ---------------------------------------------------------------------------
# llm_summary
# ---------------------------------------------------------------------------


def resolve_sources(
    task: ScheduledTask, ctx: ActionContext
) -> list[ScheduledTask]:
    """Tasks whose results the summary aggregates.

    ``payload.sources``: ``"all"`` / missing → every task with results,
    a list of group names, or a list of task ids.
    """
    sources = task.payload.get("sources", "all")
    tasks = ctx.store.list_tasks()
    tasks = [t for t in tasks if t.id != task.id]
    if sources == "all" or sources is None:
        return tasks
    if isinstance(sources, str):
        wanted = sources.strip()
        by_group = [t for t in tasks if t.group == wanted]
        by_id = [t for t in tasks if t.id == wanted]
        return by_group or by_id
    if isinstance(sources, list):
        keys = [str(s).strip() for s in sources]
        return [
            t for t in tasks if t.group in keys or t.id in keys
        ]
    raise ValueError(f"Непонятный payload.sources: {sources!r}")


def render_material(task: ScheduledTask, ctx: ActionContext) -> str:
    """Plain-text material for the summary: recent results of the sources."""
    blocks = []
    for src in resolve_sources(task, ctx):
        results = ctx.store.list_results(task_id=src.id)
        if not results:
            continue
        lines = [f"Задача «{src.title}»" + (f" [{src.group}]" if src.group else "")]
        for res in results[-SUMMARY_TAIL_RESULTS:]:
            mark = "ок" if res.ok else "ОШИБКА"
            text = (res.summary or "").replace("\n", " ")
            if len(text) > SUMMARY_RESULT_CHARS:
                text = text[: SUMMARY_RESULT_CHARS - 1] + "…"
            lines.append(f"  - [{mark} {res.run_at}] {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_SUMMARY_SYSTEM = (
    "Ты — аналитик. Ниже — накопленные результаты фоновых задач агента. "
    "Составь краткую сводку для человека: главное изменение, цифры, "
    "аномалии и ошибки. Без воды, только факты из материала."
)

_DETERMINISTIC_HEAD = (
    "Сводка (LLM не настроен, детерминированный режим):"
)


def run_llm_summary(task: ScheduledTask, ctx: ActionContext) -> TaskResult:
    """Aggregate source results; LLM when configured, deterministic otherwise."""
    material = render_material(task, ctx)
    if not material:
        return TaskResult(
            task_id=task.id,
            run_at=iso(utc_now()),
            ok=True,
            summary="Сводка: пока нет ни одного результата задач-источников.",
            details={"sources": 0},
        )
    summarizer = ctx._chat_factory()
    if summarizer is not None:
        try:
            reply = summarizer.chat(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": material},
                ]
            )
            return TaskResult(
                task_id=task.id,
                run_at=iso(utc_now()),
                ok=True,
                summary=str(reply).strip(),
                details={"mode": "llm", "material_chars": len(material)},
            )
        except Exception as exc:  # degrade, not crash: summary still produced
            fallback_note = f"LLM недоступен ({exc}); детерминированный режим."
    else:
        fallback_note = ""
    # Deterministic degradation: counters + tail of raw summaries.
    sources = resolve_sources(task, ctx)
    lines = [line for line in (fallback_note, _DETERMINISTIC_HEAD) if line]
    for src in sources:
        results = ctx.store.list_results(task_id=src.id)
        if not results:
            continue
        ok_n = sum(1 for r in results if r.ok)
        lines.append(f"▪ {src.title}: прогонов {len(results)}, ок {ok_n}")
        last = results[-1]
        text = (last.summary or "").replace("\n", " ")[:200]
        lines.append(f"  последний: {text}")
    return TaskResult(
        task_id=task.id,
        run_at=iso(utc_now()),
        ok=True,
        summary="\n".join(lines),
        details={"mode": "deterministic", "material_chars": len(material)},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ACTIONS: dict[str, Callable[[ScheduledTask, ActionContext], TaskResult]] = {
    ACTION_MCP_CALL: run_mcp_call,
    ACTION_REMINDER: run_reminder,
    ACTION_LLM_SUMMARY: run_llm_summary,
}

KNOWN_ACTIONS = ", ".join(sorted(ACTIONS))


def validate_action(
    action: str,
    payload: dict | None = None,
    known_servers: list[str] | None = None,
    known_tools: list[str] | None = None,
) -> None:
    """Reject broken task definitions at creation time, not in the daemon.

    Live beta findings this prevents (each previously burned the task's
    retries before dying as ``failed``):

    * invented action names (``collect_usd_rate``) — checked against the
      registry / coercible forms;
    * invented MCP server names (``server: "default"``) — checked against
      ``data/mcp.yaml``, or the explicit *known_servers* list (tests).

    Raises :class:`SchedulerError` with an actionable message either way.
    """
    normalized = (action or "").strip().lower()
    data = payload or {}
    rescuable = (
        normalized in ACTIONS
        or ("server" in data and "tool" in data)
        or "__" in normalized
        or any(word in normalized for word in ("remind", "напомин"))
    )
    if not rescuable:
        raise SchedulerError(
            f"Неизвестное действие {action!r}. Доступны: {KNOWN_ACTIONS}. "
            "Чтобы вызвать инструмент MCP, используйте action='mcp_call' "
            "с payload {server, tool, arguments}."
        )
    # The mcp_call target must be a real configured server.
    server = str(
        data.get("server")
        or (normalized.split("__", 1)[0] if "__" in normalized else "")
        or ""
    ).strip()
    if server:
        if known_servers is None:
            from llm_bot.mcp_tools import load_mcp_config

            known_servers = sorted(load_mcp_config())
        names = sorted(known_servers)
        if server not in names:
            raise SchedulerError(
                f"MCP-сервер {server!r} не найден в data/mcp.yaml. "
                f"Доступны: {', '.join(names) or '—'}."
            )
        # The tool must exist on that server. Listing spawns the server, so
        # tests may pass *known_tools*; if the server is temporarily down we
        # skip this check rather than block scheduling — run-time retries
        # will surface real failures.
        raw_tool = str(
            data.get("tool")
            or (normalized.split("__", 1)[1] if "__" in normalized else "")
            or ""
        ).strip()
        tool = (
            raw_tool.split("__", 1)[1]
            if raw_tool.startswith(f"{server}__")
            else raw_tool
        )
        if tool:
            if known_tools is None:
                try:
                    bridge = _default_bridge_factory(server)
                    known_tools = [spec.name for spec in bridge.list_tools()]
                except Exception:  # noqa: BLE001 - skip, do not block
                    return
            if tool not in known_tools:
                raise SchedulerError(
                    f"У сервера {server!r} нет инструмента {raw_tool!r}. "
                    f"Доступны: {', '.join(sorted(known_tools)) or '—'}. "
                    "В payload.tool передавайте короткое имя инструмента "
                    "без префикса сервера."
                )


def _fold_arguments(payload: dict) -> dict:
    """Nest flat tool arguments under ``payload['arguments']``.

    Live beta finding: the model writes ``{server, tool, text: …}`` — or just
    ``{text: …}`` when the action itself is ``server__tool`` — instead of
    nesting the arguments. Every key besides server/tool/arguments is treated
    as a tool argument; explicit ``arguments`` values win over folded ones.
    """
    result = dict(payload)
    extra = {
        k: v for k, v in result.items() if k not in ("server", "tool", "arguments")
    }
    if not extra:
        return result
    raw = result.get("arguments")
    arguments = dict(raw) if isinstance(raw, dict) else {}
    for key, value in extra.items():
        arguments.setdefault(key, value)
    result["arguments"] = arguments
    for key in extra:
        del result[key]
    return result


def coerce_action(task: ScheduledTask) -> ScheduledTask:
    """Return *task* with its action normalized to a registry name.

    Real-world beta finding: the model naturally names the action after the
    tool it wants to call (``add_note``) or after the verb (``напоминание``)
    instead of the three registry names. Normalize what we can; the task is
    returned unchanged when the action is already known or unrecognizable.

    * payload already carries ``server`` + ``tool`` → ``mcp_call``;
    * a qualified ``server__tool`` used as the action itself → ``mcp_call``
      (payload gets server/tool filled in);
    * flat tool arguments are folded into ``payload['arguments']``;
    * Russian/English reminder synonyms → ``reminder``.
    """
    action = (task.action or "").strip().lower()
    if action in ACTIONS:
        return task
    payload = dict(task.payload or {})
    if "tool" in payload and "server" in payload:
        return replace(
            task, action=ACTION_MCP_CALL, payload=_fold_arguments(payload)
        )
    if "__" in action:  # qualified "server__tool" as the action itself
        server, tool = action.split("__", 1)
        payload.setdefault("server", server)
        payload.setdefault("tool", tool)
        return replace(
            task, action=ACTION_MCP_CALL, payload=_fold_arguments(payload)
        )
    if any(word in action for word in ("remind", "напомин")):
        return replace(task, action=ACTION_REMINDER, payload=payload)
    return task


def run_action(task: ScheduledTask, ctx: ActionContext) -> TaskResult:
    """Dispatch a task to its executor; unknown action → failed result."""
    task = coerce_action(task)
    executor = ACTIONS.get(task.action)
    if executor is None:
        return TaskResult(
            task_id=task.id,
            run_at=iso(utc_now()),
            ok=False,
            summary=(
                f"Неизвестное действие {task.action!r}. Доступны: {KNOWN_ACTIONS}. "
                "Чтобы вызвать инструмент MCP, используйте действие 'mcp_call' "
                "с payload {server, tool, arguments}."
            ),
        )
    return executor(task, ctx)
