#!/usr/bin/env python3
"""Demonstrate how tokens grow across a dialog and what breaks on overflow.

Runs three scenarios against a fake (in-memory) LLM so it works offline and is
fully deterministic:

    * SHORT   — a few brief turns; context is tiny, plenty of headroom.
    * LONG    — many verbose turns; the context fills up and the cost creeps
                toward the model's context window.
    * OVERFLOW — so many turns that the assembled history exceeds the context
                window; the agent refuses to send the request and raises
                ``ContextOverflowError`` (this is what "breaks").

For each turn it prints the same breakdown the agent computes internally:
request / history / context / reply tokens and how full the window is.

The model's context window is shrunk via the ``--context-window`` flag (default
1200) so the growth and the overflow become visible with a handful of turns
instead of tens of thousands of tokens.

Examples:
    python scripts/token_demo.py
    python scripts/token_demo.py --turns 6
    python scripts/token_demo.py --context-window 400 --turns 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from llm_bot.agent import Session
from llm_bot.client import ContextOverflowError, ContextTooLargeError
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    count_message_tokens,
    count_messages_tokens,
)


def _reply_handler(length: int):
    """Return an httpx handler that echoes a padded reply with token usage.

    The mock reports ``prompt_tokens`` using the *same* deterministic estimator
    the agent uses for its pre-flight overflow check, and ``completion_tokens``
    from the reply we are about to emit. That way the ``ctx``/``fill%`` shown in
    the table always agree with the guard: the overflow fires exactly when the
    estimate crosses the context window, and the error message's token count
    matches the visible ``ctx`` column.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        prompt_tokens = count_messages_tokens(payload.get("messages", []))
        content = "ok " * length
        completion_tokens = count_message_tokens(
            {"role": "assistant", "content": content}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    return handler


def _build_session(
    context_window: int,
    directory: str,
    turns_so_far: int,
    *,
    max_request_tokens: int | None = None,
) -> Session:
    """Build a Session backed by a fake transport and a tiny context window.

    ``max_request_tokens`` optionally sets an account-tier per-request ceiling so
    the demo can show the "refused because too large for the account" case.
    """
    model_cfg = ModelConfig(
        name="demo",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt-demo",
        context_window=context_window,
        max_request_tokens=max_request_tokens,
    )
    agent_cfg = AgentConfig(
        name="demo",
        model="demo",
        system_prompt="Ты краткий помощник. Отвечай коротко.",
        temperature=0.0,
        max_tokens=64,
    )

    class _AgentStore:
        def get(self, name):  # noqa: ANN001
            return agent_cfg

        def list(self):
            return [agent_cfg.name]

    class _ModelStore:
        def get(self, name):  # noqa: ANN001
            return model_cfg

        def list(self):
            return [model_cfg.name]

    session_store = JsonSessionStore(directory)
    session_id = f"demo-{context_window}-{turns_so_far}"
    return make_session(
        session_id,
        agent_cfg.name,
        model_store=_ModelStore(),
        agent_store=_AgentStore(),
        session_store=session_store,
        transport=httpx.MockTransport(_reply_handler(16)),
    )


def _header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def _row(session: Session, turn: int, *, note: str = "") -> None:
    u = session.last_usage
    if u is None:
        print(f"  turn {turn}: (no usage)")
        return
    limit = u.context_window or DEFAULT_CONTEXT_WINDOW
    fill = f"{u.fill_percent:.0f}%" if u.fill_percent is not None else "?"
    print(
        f"  turn {turn:>2}: req={u.request_tokens:<4} hist={u.history_tokens:<5} "
        f"ctx={u.context_tokens:<5} reply={u.reply_tokens:<4} "
        f"total={u.total_tokens:<6} fill={fill:>4} / limit {limit}"
        + (f"   <-- {note}" if note else "")
    )


def _run_long_dialog(directory: str, context_window: int, turns: int) -> None:
    """A verbose dialog that grows each turn, showing cost climbing."""
    _header("LONG dialog — cost grows as history accumulates")
    session = _build_session(context_window, directory, 0)
    base = ("Это очередной довольно длинный запрос с кучей слов для заполнения "
            "контекста модели новыми токенами каждый раз.")
    for i in range(1, turns + 1):
        prompt = base + (" и ещё чуть больше текста сверху." * i)
        try:
            session.chat_with_details(prompt)
        except ContextOverflowError as exc:
            print(f"  turn {i}: OVERFLOW at request time — {exc}")
            break
        _row(session, i)
    print()


def _run_overflow(directory: str, context_window: int) -> None:
    """Grow until the context window is exceeded; show exactly what breaks."""
    _header("OVERFLOW — history exceeds the context window")
    session = _build_session(context_window, directory, 999)
    chunk = "слово " * 40  # ~10 tokens per message chunk
    turn = 0
    while True:
        turn += 1
        try:
            session.chat_with_details(chunk)
        except ContextOverflowError as exc:
            # The table above stopped at the LAST SUCCESSFUL turn. This turn was
            # blocked before sending, so we read the failing estimate directly
            # from the exception rather than from ``session.last_usage``.
            limit = exc.context_window
            fill = (exc.context_tokens / limit) * 100.0
            print(f"  turn {turn}: request BLOCKED — would have been "
                  f"ctx={exc.context_tokens} tokens, fill={fill:.0f}% > limit {limit}")
            print(f"    {exc}")
            print("\n  That is the failure: the agent refuses to talk to the model,")
            print("  nothing is sent, and the caller must trim history or start anew.")
            break
        _row(session, turn)
    print()


def _run_account_limit(
    directory: str,
    context_window: int,
    max_request_tokens: int,
) -> None:
    """Show a per-request size ceiling that binds below the model window."""
    _header("ACCOUNT LIMIT — per-request ceiling smaller than the model window")
    session = _build_session(
        context_window,
        directory,
        500,
        max_request_tokens=max_request_tokens,
    )
    chunk = "слово " * 40
    turn = 0
    while True:
        turn += 1
        try:
            session.chat_with_details(chunk)
        except ContextTooLargeError as exc:
            limit = exc.limit_tokens or max_request_tokens
            requested = exc.requested_tokens or 0
            fill = (requested / limit) * 100.0 if limit else 0.0
            print(f"  turn {turn}: request BLOCKED by account ceiling — "
                  f"{requested} > {limit} ({fill:.0f}%)")
            print(f"    {exc}")
            print("\n  Note: the model context window is larger, but the account")
            print("  tier refuses any single request above the ceiling. Retrying the")
            print("  same payload can never succeed — only shrinking it helps.")
            break
        _row(session, turn)
    print()


def _run_short(directory: str, context_window: int) -> None:
    """A couple of brief turns; tiny footprint, lots of headroom."""
    _header("SHORT dialog — small footprint, plenty of headroom")
    session = _build_session(context_window, directory, 1)
    for i, prompt in enumerate(["Привет!", "Что нового?", "Пока."], start=1):
        session.chat_with_details(prompt)
        _row(session, i)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-window", type=int, default=1200,
                        help="Model context window (tokens) for the demo.")
    parser.add_argument("--turns", type=int, default=4,
                        help="Number of turns in the LONG scenario.")
    parser.add_argument("--max-request-tokens", type=int, default=None,
                        help="Account-tier per-request token ceiling for the "
                             "ACCOUNT LIMIT scenario (default: half the window).")
    parser.add_argument("--directory", default=None,
                        help="Where to persist demo sessions (default: temp dir).")
    args = parser.parse_args(argv)

    import tempfile

    directory = args.directory or tempfile.mkdtemp(prefix="token-demo-")
    context_window = args.context_window
    max_request_tokens = args.max_request_tokens or max(1, context_window // 2)
    print(f"Using context window = {context_window} tokens, "
          f"account ceiling = {max_request_tokens} tokens.\n")

    _run_short(directory, context_window)
    _run_long_dialog(directory, context_window, args.turns)
    _run_account_limit(directory, context_window, max_request_tokens)
    _run_overflow(directory, context_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())