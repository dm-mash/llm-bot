"""Console entry point for the LLM client and agent-based chat.

Usage examples:
    python -m llm_bot --agent assistant                 # interactive chat
    python -m llm_bot --agent assistant --session my1   # resume session
    python -m llm_bot --agent translator "Hello"        # one-shot via agent
    python -m llm_bot --list-agents                     # show available agents

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
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore


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


def _run_agent_chat(
    agent_name: str,
    *,
    session_id: str | None,
    prompt: str | None,
    detail_listener: DetailListener | None,
) -> int:
    """Run an agent-based session; either one shot or an interactive loop."""
    agent_store = YamlAgentStore()
    model_store = YamlModelStore()
    session_store = JsonSessionStore()
    session_id = session_id or _new_session_id(agent_name)

    session = make_session(
        session_id,
        agent_name,
        model_store=model_store,
        agent_store=agent_store,
        session_store=session_store,
        detail_listener=detail_listener,
    )

    if prompt is not None:
        try:
            print(session.chat(prompt))
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    return _interactive_loop(session)


_HISTORY_COMMANDS = {"/history", "/история"}

# Words offered by tab-completion in the interactive prompt.
_COMMAND_WORDS = ["/history", "/история", "exit", "quit"]


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
            try:
                print(session.chat(raw))
            except LLMError as exc:
                print(f"error: {exc}", file=sys.stderr)
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