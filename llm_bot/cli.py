"""Console entry point for the LLM client and agent-based chat.

Usage examples:
    python -m llm_bot --agent assistant                 # interactive chat
    python -m llm_bot --agent assistant --session my1   # resume session
    python -m llm_bot --agent assistant --profile dev   # personalized chat
    python -m llm_bot --agent translator "Hello"        # one-shot via agent
    python -m llm_bot --list-agents                     # show available agents
    python -m llm_bot --list-profiles                   # show available profiles
    python -m llm_bot --show-profile dev                # inspect one profile

Legacy (single direct call, no agent):
    python -m llm_bot "Hello, who are you?"
    python -m llm_bot --model llama3.2 "Tell me a joke"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from llm_bot.agent import Session
from llm_bot.client import LLMClient, LLMError
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import (
    DetailListener,
    RequestDetails,
    ResponseDetails,
)
from llm_bot.factory import make_session
from llm_bot.gigachat import build_gigachat_client
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.memory_store import JsonMemoryStore
from llm_bot.profiles import profile_prompt_block
from llm_bot.task_state import TaskStage
from llm_bot.yaml_stores import (
    YamlAgentStore,
    YamlModelStore,
    YamlProfileStore,
)


def _sanitize_text(text: str) -> str:
    """Replace lone surrogates so text read from stdin is safely UTF-8 encodable.

    CPython decodes stdin with the ``surrogateescape`` error handler: any byte
    that is not valid UTF-8 (e.g. a continuation byte left over when Backspace
    splits a multi-byte character while editing) becomes a lone surrogate code
    point. Surrogates live fine in memory but crash httpx's JSON serialization
    with ``UnicodeEncodeError``. Mapping them to ``U+FFFD`` keeps the message
    sendable.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-bot",
        description="Chat with an LLM agent or send a single prompt to an "
        "OpenAI-compatible API.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="The prompt text. If omitted and --agent is set, starts an "
        "interactive chat.",
    )

    # Agent / session options.
    parser.add_argument(
        "--agent",
        default=None,
        metavar="NAME",
        help="Name of an agent defined in data/agents.yaml (references a model "
        "from data/models.yaml).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="[legacy] Override the model identifier when NOT using --agent.",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Session id for persisting/continuing a conversation history "
        "(stored in data/sessions/). A new one is generated if omitted.",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List the names of all defined agents and exit.",
    )
    parser.add_argument(
        "--strategy",
        choices=("sliding", "facts", "branching"),
        default=None,
        help="Context-management strategy for this session: sliding window / "
        "sticky facts / branching. Takes precedence over the agent's configured "
        "context_strategy.",
    )
    parser.add_argument(
        "--window-messages",
        type=int,
        default=None,
        metavar="N",
        help="Sliding-window size in MESSAGES kept in each request for "
        "--strategy sliding/facts (overrides the agent's context_window_messages).",
    )
    parser.add_argument(
        "--owner",
        default="default",
        metavar="ID",
        help="Owner/user namespace for long-term memory isolation (privacy). "
        "Different owners of the same agent never share profile/decisions/knowledge.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Personalization profile from data/profiles.yaml (style, format, "
        "constraints). Applied on top of the agent config at composition time; "
        "memory is not affected.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List the names of all defined profiles and exit.",
    )
    parser.add_argument(
        "--show-profile",
        default=None,
        metavar="NAME",
        help="Show one profile's settings and exit.",
    )
    parser.add_argument(
        "--task-state",
        action="store_true",
        help="Enable the task state machine (stage planning/execution/validation/"
        "done + pause) for this session; the stage/step/expected-action block is "
        "injected into the system context on every turn. Takes precedence over "
        "the agent's configured task_state.",
    )
    parser.add_argument(
        "--no-task-detect",
        action="store_true",
        help="Disable automatic task detection (the small LLM call after each "
        "turn that recognizes task setup, stage hints and pause/resume phrases).",
    )

    # Legacy direct-call options (used only without --agent).
    parser.add_argument(
        "--base-url",
        default=None,
        help="[legacy] Override the LLM API base URL (else LLM_BASE_URL or default).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="[legacy] Override the API key (else LLM_API_KEY).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="[legacy] Optional system prompt sent before the user prompt.",
    )
    parser.add_argument(
        "--max-response-words",
        type=int,
        default=None,
        help="[legacy] Target maximum length of the reply, in words.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gigachat"),
        default="openai",
        help="[legacy] Provider auth mode for the direct-call path.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print request/response details (URL, model, payload, token usage) "
        "to stderr.",
    )
    return parser


class _DetailPrinter:
    """Print request/response details to stderr, keeping stdout clean.

    Implements the :class:`~llm_bot.diagnostics.DetailListener` protocol so it
    plugs straight into :class:`~llm_bot.client.LLMClient`.
    """

    def on_request(self, details: RequestDetails) -> None:
        print(f"[details] {details.method} {details.url}", file=sys.stderr)
        print(f"[details] model: {details.model}", file=sys.stderr)
        body = json.dumps(details.payload, ensure_ascii=False, indent=2)
        print("[details] request body:", file=sys.stderr)
        for line in body.splitlines():
            print(f"           {line}", file=sys.stderr)

    def on_response(self, details: ResponseDetails) -> None:
        elapsed = (
            f" in {details.elapsed_ms:.0f}ms" if details.elapsed_ms is not None else ""
        )
        print(
            f"[details] attempt {details.attempt}: HTTP {details.status_code}{elapsed}",
            file=sys.stderr,
        )
        if details.usage:
            parts = " ".join(f"{k}={v}" for k, v in details.usage.items())
            print(f"[details] usage: {parts}", file=sys.stderr)
        if details.body:
            print("[details] response body:", file=sys.stderr)
            body = json.dumps(details.body, ensure_ascii=False, indent=2)
            for line in body.splitlines():
                print(f"           {line}", file=sys.stderr)


def _new_session_id(agent_name: str) -> str:
    """Generate a fresh, filesystem-safe session id for an agent."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{agent_name}-{stamp}"


def _print_resume_info(session: Session) -> None:
    """Print context info when resuming an existing session, if any.

    Called right after opening an existing session. Reports the number of
    messages currently in the history and, depending on the active context-
    management strategy:

    * ``sliding``  — the sliding-window size;
    * ``facts``    — the number of sticky facts in durable memory (plus window);
    * ``branching``— the list of dialogue branches with the current one marked.

    Also reports the rolling-compression summary size (in characters) when a
    summary exists. Uses ``getattr`` so lightweight fakes/tests that only
    implement ``chat`` do not need to expose compression/strategy internals.
    """
    history = getattr(session, "history", None) or []
    summary = getattr(session, "summary", "") or ""
    strategy = getattr(session, "strategy", None)
    memory = getattr(session, "memory", None)

    if not history and not summary and strategy is None and memory is None:
        return

    parts = [f"сообщений в истории: {len(history)}"]
    if summary:
        parts.append(f"summary: {len(summary)} симв.")

    if strategy is not None:
        name = getattr(strategy, "name", "") or "unknown"
        window = getattr(strategy, "window_size", None)
        if name == "sliding":
            if window is not None:
                parts.append(f"окно: {window} сообщ.")
        elif name == "facts":
            facts = getattr(strategy, "facts", None) or {}
            if window is not None:
                parts.append(f"окно: {window} сообщ.")
            parts.append(f"фактов: {len(facts)}")
        elif name == "branching":
            branches = getattr(strategy, "branches", None) or {}
            current = getattr(strategy, "current_branch", "")
            rendered = ", ".join(
                f"*{b}*" if b == current else b for b in branches
            )
            parts.append(f"ветки [{current}]: {rendered}")

    if memory is not None:
        owner = getattr(memory.long, "owner", "")
        parts.append(f"owner={owner}")
        parts.append(f"память: working={len(memory.working)} long={len(memory.long)}")

    print("[session] " + ", ".join(parts), file=sys.stderr)


def _print_usage(session: Session) -> None:
    """Print token accounting for the session's most recent turn to stderr.

    Uses ``getattr`` so lightweight fakes/tests that only implement ``chat`` do
    not need to also expose ``last_usage``.
    """
    usage = getattr(session, "last_usage", None)
    if usage is None:
        return
    parts = [
        f"request={usage.request_tokens}",
        f"history={usage.history_tokens}",
        f"context={usage.context_tokens}",
        f"reply={usage.reply_tokens}",
        f"total={usage.total_tokens}",
    ]
    if usage.context_window:
        parts.append(f"limit={usage.context_window}")
        if usage.fill_percent is not None:
            parts.append(f"fill={usage.fill_percent:.0f}%")
    source = "estimated" if usage.estimated else "provider"
    if usage.overflow:
        parts.append("OVERFLOW")
    print(f"[tokens ({source})] " + " ".join(parts), file=sys.stderr)


def _print_compression(session: Session) -> None:
    """Print a service message to stderr when the history was just compressed.

    Reports how many messages were folded into the summary, the estimated token
    space they freed in the context, and the history length before -> after.
    Uses ``getattr`` so lightweight fakes/tests that only implement ``chat`` do
    not need to expose compression internals.
    """
    event = getattr(session, "last_compression_event", None)
    if event is None or event.messages_folded <= 0:
        return
    parts = [
        f"свёрнуто {event.messages_folded} сообщений",
        f"-{event.folded_tokens} токенов контекста",
        f"история {event.history_before}->{event.history_after}",
        f"summary {event.summary_chars} симв.",
    ]
    total = getattr(session, "total_compressions", None)
    if total:
        parts.append(f"(всего сжатий: {total})")
    print("[compression] " + ", ".join(parts), file=sys.stderr)


def _print_memory(session: Session) -> None:
    """Print a service message to stderr when memory was just auto-extracted.

    Reports how many facts were written to the working vs long-term layers and
    the tokens the classifier call cost. Uses ``getattr`` so lightweight fakes
    that only implement ``chat`` do not need to expose memory internals.
    """
    memory = getattr(session, "memory", None)
    if memory is None:
        return
    event = getattr(session, "last_memory_event", None)
    if event is None:
        return
    written = len(event.working_written) + len(event.long_term_written)
    parts = [
        f"память: working={len(memory.working)} long={len(memory.long)}",
        f"добавлено фактов: {written}",
        f"токены классификации: {event.total_tokens}",
    ]
    total = getattr(session, "total_memory_extractions", 0)
    if total:
        parts.append(f"(всего ходов с извлечением: {total})")
    print("[memory] " + ", ".join(parts), file=sys.stderr)


def _run_agent_chat(
    agent_name: str,
    *,
    session_id: str | None,
    prompt: str | None,
    detail_listener: DetailListener | None,
    strategy_override: str | None = None,
    window_messages: int | None = None,
    owner_id: str = "default",
    profile_name: str | None = None,
    task_state: bool | None = None,
    task_auto_detect: bool = True,
) -> int:
    """Run an agent-based session; either one shot or an interactive loop."""
    agent_store = YamlAgentStore()
    model_store = YamlModelStore()
    session_store = JsonSessionStore()
    memory_store = JsonMemoryStore()
    session_id = session_id or _new_session_id(agent_name)

    session = make_session(
        session_id,
        agent_name,
        model_store=model_store,
        agent_store=agent_store,
        session_store=session_store,
        memory_store=memory_store,
        owner_id=owner_id,
        detail_listener=detail_listener,
        strategy_override=strategy_override,
        window_messages=window_messages,
        profile=profile_name,
        task_state=task_state,
        task_auto_detect=task_auto_detect,
    )

    # When resuming an existing conversation, surface how much context is loaded.
    _print_resume_info(session)

    if prompt is not None:
        try:
            result = session.chat_with_details(prompt)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.reply)
        _print_usage(session)
        _print_compression(session)
        _print_memory(session)
        return 0

    return _interactive_loop(session)


_HISTORY_COMMANDS = {"/history", "/история"}
_BRANCH_COMMANDS = {"/branch", "/switch", "/branches"}
_TASK_COMMANDS = {
    "/task", "/задача",
}

# Words offered by tab-completion in the interactive prompt.
_COMMAND_WORDS = [
    "/history",
    "/история",
    "/branch",
    "/switch",
    "/branches",
    "/task",
    "/задача",
    "exit",
    "quit",
]


def _command_completer():
    """Build (lazily) a prompt_toolkit completer for slash commands.

    Built on first use so that importing this module never requires
    prompt_toolkit to be importable — it is only needed when actually editing
    on an interactive terminal.
    """
    from prompt_toolkit.completion import WordCompleter

    return WordCompleter(_COMMAND_WORDS, ignore_case=True)


def _print_history(session: Session) -> None:
    """Print the session's stored message history, if any."""
    history = session.history
    if not history:
        print("(history is empty for this session)", file=sys.stderr)
        return
    print(f"History of session '{session.session_id}':")
    for i, msg in enumerate(history, start=1):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        label = "you" if role == "user" else role
        print(f"  {i}. [{label}] {content}")
    print()


def _read_input(
    prompt: str = "> ",
    history: Any | None = None,
) -> str:
    """Read one line of user input.

    Uses prompt_toolkit when stdin is an interactive terminal so that editing
    works on whole Unicode characters rather than raw bytes. This avoids the
    classic problem where Backspace splits a multi-byte UTF-8 character and
    leaves a lone continuation byte (decoded to a surrogate by CPython's
    ``surrogateescape``), which would otherwise crash JSON serialization.

    When *history* is provided it is used for Up/Down recall within the current
    session (in-memory only, never written to disk). The completer suggests
    slash commands (``/history``, ``/история``) and ``exit``/``quit``.

    Falls back to the built-in ``input()`` when stdin is piped/redirected (e.g.
    in tests or when a prompt is fed via a pipe) or prompt_toolkit is missing.
    In that fallback, *history* and completion are ignored.

    Raises:
        EOFError: On end-of-input (Ctrl-D).
        KeyboardInterrupt: On Ctrl-C.
    """
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import prompt as pt_prompt
        except ImportError:  # pragma: no cover - prompt_toolkit is a dependency
            pass
        else:
            kwargs: dict[str, Any] = {"completer": _command_completer()}
            if history is not None:
                kwargs["history"] = history
            try:
                return pt_prompt(prompt, **kwargs)
            except EOFError:
                raise
            except KeyboardInterrupt:
                raise
    return input(prompt)


def _handle_branch_command(session: Session, raw: str) -> bool:
    """Handle a branching slash-command; returns True when the input was a command.

    Supports ``/branch <name>`` (fork the current position into a new branch),
    ``/switch <name>`` (make a branch active) and ``/branches`` (list them). Only
    meaningful when the session uses the ``branching`` strategy; otherwise prints
    a hint and returns True so the text is not sent as a chat message.
    """
    cmd = raw.split(None, 1)
    name = cmd[0].lower()
    if name not in _BRANCH_COMMANDS:
        return False
    strategy = getattr(session, "strategy", None)
    if strategy is None or getattr(strategy, "name", "") != "branching":
        print("[branches] активна не стратегия branching; используйте "
              "--strategy branching.", file=sys.stderr)
        return True

    if name == "/branches":
        print(f"[branches] текущая ветка: {strategy.current_branch}", file=sys.stderr)
        for b, hist in strategy.branches.items():
            marker = " *" if b == strategy.current_branch else ""
            print(f"[branches]   {b}{marker} ({len(hist)} сообщ.)", file=sys.stderr)
        return True

    arg = cmd[1].strip() if len(cmd) > 1 else ""
    if name == "/branch":
        if not arg:
            print("[branch] укажите имя: /branch <name>", file=sys.stderr)
            return True
        try:
            session.branch(arg)
        except ValueError as exc:
            print(f"[branch] {exc}", file=sys.stderr)
            return True
        print(f"[branch] создана ветка '{arg}' (checkpoint текущей позиции).",
              file=sys.stderr)
        return True

    # /switch
    if not arg:
        print("[switch] укажите имя: /switch <name>", file=sys.stderr)
        return True
    try:
        session.switch_branch(arg)
    except KeyError:
        print(f"[switch] ветка '{arg}' не найдена. /branches — список.",
              file=sys.stderr)
        return True
    print(f"[switch] переключено на ветку '{arg}'.", file=sys.stderr)
    return True


def _print_task_status(session: Session) -> None:
    """Print the current task state (stage / pause / step / action)."""
    state = session.task_state
    if state is None:
        print("[task] машина состояния задачи не включена "
              "(запустите с --task-state).", file=sys.stderr)
        return
    if not state.description and state.stage is TaskStage.PLANNING \
            and not state.step and not state.log:
        print("[task] задача не начата. /task start <описание> — начать.",
              file=sys.stderr)
        return
    stage = state.stage.value
    if state.is_paused and state.paused_from is not None:
        stage = f"{stage} (пауза; до паузы — {state.paused_from.value})"
    print(f"[task] этап: {stage}", file=sys.stderr)
    if state.description:
        print(f"[task] задача: {state.description}", file=sys.stderr)
    if state.step:
        print(f"[task] шаг: {state.step}", file=sys.stderr)
    if state.expected_action:
        print(f"[task] ожидаемое действие: {state.expected_action}", file=sys.stderr)
    for entry in state.log[-3:]:
        print(f"[task]   • {entry}", file=sys.stderr)


def _task_auto_turn(session: Session, stage: TaskStage) -> None:
    """Send one service chat turn so the model reacts to a stage change now.

    Called after ``/task start`` / ``next`` / ``resume``: without it the model
    would only learn about the new stage when the user sends their next
    message. The reply is printed like an ordinary chat answer. Failures are
    reported to stderr but never abort the command.
    """
    service = (
        f"Этап задачи изменился на '{stage.value}'. Действуй по актуальному "
        "состоянию задачи (см. блок состояния)."
    )
    try:
        reply = session.chat(service)
    except LLMError as exc:
        print(f"[task] авто-ход не удался: {exc}", file=sys.stderr)
        return
    print(f"[task] авто-ход (этап '{stage.value}'):", file=sys.stderr)
    print(reply)


def _handle_task_command(session: Session, raw: str) -> bool:
    """Handle a task-state slash-command; True when the input was a command.

    Supports ``/task`` (status), ``/task start <описание>``, ``/task step <текст>``,
    ``/task action <текст>``, ``/task next``, ``/task pause``, ``/task resume``
    and ``/task reset``. Manual commands take priority over auto-detection and
    edit the machine directly; every change is persisted immediately.
    """
    cmd = raw.split(None, 1)
    if cmd[0].lower() not in _TASK_COMMANDS:
        return False
    task = session.task
    if task is None:
        print("[task] машина состояния задачи не включена "
              "(запустите с --task-state).", file=sys.stderr)
        return True
    arg = cmd[1].strip() if len(cmd) > 1 else ""
    sub = arg.split(None, 1)
    name = sub[0].lower() if sub else ""
    rest = sub[1].strip() if len(sub) > 1 else ""

    try:
        if not name or name in {"status", "статус"}:
            _print_task_status(session)
        elif name in {"start", "начать"}:
            if not rest:
                print("[task] укажите описание: /task start <описание>",
                      file=sys.stderr)
            else:
                state = task.start(rest)
                print(f"[task] задача запущена (этап planning): {rest}",
                      file=sys.stderr)
                _task_auto_turn(session, state.stage)
        elif name in {"step", "шаг"}:
            if not rest:
                print("[task] укажите текст: /task step <текст>", file=sys.stderr)
            else:
                task.set_step(rest)
                print(f"[task] шаг: {rest}", file=sys.stderr)
        elif name in {"action", "действие"}:
            if not rest:
                print("[task] укажите текст: /task action <текст>",
                      file=sys.stderr)
            else:
                task.set_expected_action(rest)
                print(f"[task] ожидаемое действие: {rest}", file=sys.stderr)
        elif name in {"next", "далее"}:
            state = task.next_stage()
            print(f"[task] этап: {state.stage.value}", file=sys.stderr)
            _task_auto_turn(session, state.stage)
        elif name in {"pause", "пауза"}:
            state = task.pause()
            print(f"[task] пауза (был этап '{state.paused_from.value if state.paused_from else '?'}'); "
                  "после перезапуска достаточно сказать «продолжай».",
                  file=sys.stderr)
            # No auto-turn on pause: the model must not be prompted while paused.
        elif name in {"resume", "продолжить", "продолжай"}:
            state = task.resume()
            print(f"[task] продолжаем на этапе '{state.stage.value}'.",
                  file=sys.stderr)
            _task_auto_turn(session, state.stage)
        elif name in {"reset", "сброс"}:
            task.reset()
            print("[task] состояние задачи сброшено.", file=sys.stderr)
        else:
            print("[task] неизвестная подкоманда. Доступны: status, start, step, "
                  "action, next, pause, resume, reset.", file=sys.stderr)
    except ValueError as exc:
        print(f"[task] {exc}", file=sys.stderr)
    return True


def _task_paused_gate(session: Session, raw: str) -> bool:
    """Return True when the input was consumed by the pause gate.

    While the task is paused, ordinary chat input is NOT sent to the model
    (hard-gate mode): the user is told to resume first. Control commands and
    slash-commands still work.
    """
    task = getattr(session, "task", None)
    if task is None or not task.state.is_paused:
        return False
    print(
        "[task] задача на паузе — сообщение не отправлено. "
        "/task resume — продолжить; /task — статус.",
        file=sys.stderr,
    )
    return True


def _interactive_loop(session: Session) -> int:
    """Run an interactive REPL-style chat against a session."""
    print(f"Starting chat with agent '{session.agent.name}' "
          f"(session {session.session_id}). Type 'exit' or Ctrl-D to quit. "
          "Use /history to see past messages.",
          file=sys.stderr)
    # In-memory input history scoped to this run/session so Up/Down recall only
    # what was typed here; nothing is persisted to disk.
    history = None
    if sys.stdin.isatty():
        from prompt_toolkit.history import InMemoryHistory

        history = InMemoryHistory()
    try:
        while True:
            try:
                raw = _sanitize_text(_read_input("> ", history=history)).strip()
            except EOFError:
                break
            if not raw:
                continue
            user = raw.lower()
            if user in {"exit", "quit"}:
                break
            if user in _HISTORY_COMMANDS:
                _print_history(session)
                continue
            if _handle_branch_command(session, raw):
                continue
            if _handle_task_command(session, raw):
                continue
            # Hard pause gate: while the task is paused, ordinary messages are
            # NOT sent to the model until the user resumes explicitly.
            if _task_paused_gate(session, raw):
                continue
            try:
                print(session.chat(raw))
            except LLMError as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
            _print_usage(session)
            _print_compression(session)
            _print_memory(session)
    except KeyboardInterrupt:
        pass
    return 0


def _run_legacy(args: argparse.Namespace) -> int:
    """Original single-prompt behaviour, kept for backward compatibility."""
    prompt = args.prompt_text

    config = LLMConfig.from_env().with_overrides(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        max_response_words=args.max_response_words,
    )

    detail_listener = _DetailPrinter() if args.details else None

    if args.provider == "gigachat":
        client = build_gigachat_client(config, detail_listener=detail_listener)
    else:
        client = LLMClient(config, detail_listener=detail_listener)

    try:
        response = client.send_prompt(prompt)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0


def _resolve_prompt(args: argparse.Namespace, *, interactive_allowed: bool) -> str | None:
    """Get the prompt from arguments or, for interactive use, return ``None``."""
    if args.prompt:
        return " ".join(args.prompt).strip()

    # Agent path: no argument means interactive chat.
    if interactive_allowed and args.agent:
        return None

    # Legacy path: read from stdin.
    if sys.stdin.isatty():
        prompt = _read_interactive()
    else:
        data = sys.stdin.read()
        prompt = _sanitize_text(data).strip()

    if not prompt:
        raise ValueError("No prompt provided. Pass it as an argument, pipe it via "
                         "stdin, or use --agent for an interactive chat.")
    return prompt


def _read_interactive() -> str:
    """Read a prompt from an interactive terminal, one line at a time.

    An empty line (just Enter) signals the end of input, so a single plain
    Enter without any text is treated as "no prompt".
    """
    lines: list[str] = []
    for raw in sys.stdin:
        line = _sanitize_text(raw).rstrip("\n")
        if line.strip() == "" and lines:
            break
        if line.strip() == "":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _print_profiles() -> int:
    """Print detailed information about every defined profile."""
    profile_store = YamlProfileStore()
    try:
        names = profile_store.list()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not names:
        print("No profiles defined.", file=sys.stderr)
        return 0

    for name in names:
        profile = profile_store.get(name)
        print(f"[{name}]")
        if profile.style:
            print(f"  style       : {profile.style}")
        if profile.format:
            print(f"  format      : {profile.format}")
        if profile.expertise:
            print(f"  expertise   : {profile.expertise}")
        if profile.language:
            print(f"  language    : {profile.language}")
        if profile.max_response_words is not None:
            print(f"  max words   : {profile.max_response_words}")
        if profile.temperature is not None:
            print(f"  temperature : {profile.temperature}")
        if profile.forbidden_topics:
            print(f"  forbidden   : {', '.join(profile.forbidden_topics)}")
        if profile.interests:
            print(f"  interests   : {', '.join(profile.interests)}")
        if profile.extra_instructions:
            print(f"  extra       : {profile.extra_instructions!r}")
        print()
    return 0


def _show_profile(name: str) -> int:
    """Print one profile's settings and the prompt block it generates."""
    profile_store = YamlProfileStore()
    try:
        profile = profile_store.get(name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError:
        try:
            available = ", ".join(profile_store.list()) or "(none)"
        except FileNotFoundError:
            available = "(profiles.yaml not found)"
        print(
            f"error: unknown profile '{name}'. Available: {available}.",
            file=sys.stderr,
        )
        return 2

    print(f"[{profile.name}]")
    for field_name in (
        "style",
        "format",
        "expertise",
        "language",
        "max_response_words",
        "temperature",
    ):
        value = getattr(profile, field_name)
        if value is not None:
            print(f"  {field_name.replace('_', ' ')}: {value}")
    if profile.forbidden_topics:
        print(f"  forbidden topics: {', '.join(profile.forbidden_topics)}")
    if profile.interests:
        print(f"  interests: {', '.join(profile.interests)}")
    if profile.extra_instructions:
        print(f"  extra instructions: {profile.extra_instructions!r}")

    block = profile_prompt_block(profile)
    if block:
        print("\nGenerated prompt block:")
        for line in block.splitlines():
            print(f"  {line}")
    return 0


def _print_agents() -> int:
    """Print detailed information about every defined agent."""
    agent_store = YamlAgentStore()
    model_store = YamlModelStore()
    names = agent_store.list()
    if not names:
        print("No agents defined.", file=sys.stderr)
        return 0

    for name in names:
        agent = agent_store.get(name)
        print(f"[{name}]")
        print(f"  model profile : {agent.model}")
        try:
            model = model_store.get(agent.model)
            print(f"  provider      : {model.provider}")
            print(f"  api model     : {model.model or '(default)'}")
            print(f"  base url      : {model.base_url or '(default)'}")
        except (KeyError, FileNotFoundError):
            print(f"  provider      : <unknown model '{agent.model}'>")
        if agent.system_prompt:
            print(f"  system prompt : {agent.system_prompt!r}")
        if agent.default_system_prompt:
            print(f"  default sys   : {agent.default_system_prompt!r}")
        parts = []
        if agent.temperature is not None:
            parts.append(f"temperature={agent.temperature}")
        if agent.max_tokens is not None:
            parts.append(f"max_tokens={agent.max_tokens}")
        if agent.max_response_words is not None:
            parts.append(f"max_response_words={agent.max_response_words}")
        if parts:
            print(f"  generation    : {', '.join(parts)}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_agents:
        return _print_agents()

    if args.list_profiles:
        return _print_profiles()

    if args.show_profile:
        return _show_profile(args.show_profile)

    if args.profile and not args.agent:
        print(
            "error: --profile requires --agent (profiles personalize an "
            "agent's behaviour).",
            file=sys.stderr,
        )
        return 2

    detail_listener = _DetailPrinter() if args.details else None

    if args.agent:
        try:
            prompt = _resolve_prompt(args, interactive_allowed=True)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return _run_agent_chat(
            args.agent,
            session_id=args.session,
            prompt=prompt,
            detail_listener=detail_listener,
            strategy_override=args.strategy,
            window_messages=args.window_messages,
            owner_id=args.owner,
            profile_name=args.profile,
            task_state=True if args.task_state else None,
            task_auto_detect=not args.no_task_detect,
        )

    # Legacy path.
    args.prompt_text = " ".join(args.prompt).strip()
    try:
        args.prompt_text = _resolve_prompt(args, interactive_allowed=False)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _run_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())