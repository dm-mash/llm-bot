"""Console entry point for the LLM client.

Usage examples:
    python -m llm_bot "Hello, who are you?"
    echo "Summarize this" | python -m llm_bot
    python -m llm_bot --model llama3.2 "Tell me a joke"
    python -m llm_bot --base-url http://localhost:11434/v1 "Hi"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from llm_bot.client import LLMClient, LLMError
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import (
    DetailListener,
    RequestDetails,
    ResponseDetails,
)
from llm_bot.gigachat import build_gigachat_client


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-bot",
        description="Send a prompt to an OpenAI-compatible LLM API and print the response.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="The prompt text. If omitted, the prompt is read from stdin.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the LLM API base URL (else LLM_BASE_URL or default).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the API key (else LLM_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model identifier (else LLM_MODEL).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt sent before the user prompt (e.g. to request "
        "a specific JSON response schema). Overrides LLM_SYSTEM_PROMPT.",
    )
    parser.add_argument(
        "--max-response-words",
        type=int,
        default=None,
        help="Target maximum length of the reply, in words. Added to the system "
        "prompt as a briefness instruction (else LLM_MAX_RESPONSE_WORDS). "
        "Leave unset for no limit.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gigachat"),
        default="openai",
        help="Provider auth mode. 'gigachat' exchanges GIGACHAT_CLIENT_SECRET "
        "for an OAuth2 token before calling the API.",
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


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Get the prompt from CLI arguments or, if empty, from stdin."""
    if args.prompt:
        return " ".join(args.prompt).strip()

    # In a real terminal, read interactively until an empty line (or EOF).
    # This lets users type/paste a prompt and finish it with a blank line,
    # instead of hanging on sys.stdin.read() while waiting for EOF.
    if sys.stdin.isatty():
        prompt = _read_interactive()
    else:
        # Piped / redirected input: consume the whole stream.
        data = sys.stdin.read()
        prompt = data.strip()

    if not prompt:
        raise ValueError("No prompt provided. Pass it as an argument or pipe it via stdin.")
    return prompt


def _read_interactive() -> str:
    """Read a prompt from an interactive terminal, one line at a time.

    An empty line (just Enter) signals the end of input, so a single plain
    Enter without any text is treated as "no prompt".
    """
    lines: list[str] = []
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if line.strip() == "" and lines:
            # A blank line after some content ends the prompt.
            break
        if line.strip() == "":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        prompt = _resolve_prompt(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())