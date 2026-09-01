"""Console entry point for the LLM client.

Usage examples:
    python -m llm_bot "Hello, who are you?"
    echo "Summarize this" | python -m llm_bot
    python -m llm_bot --model llama3.2 "Tell me a joke"
    python -m llm_bot --base-url http://localhost:11434/v1 "Hi"
"""

from __future__ import annotations

import argparse
import logging
import sys

from llm_bot.client import LLMClient, LLMError
from llm_bot.config import LLMConfig
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
    )

    if args.provider == "gigachat":
        client = build_gigachat_client(config)
    else:
        client = LLMClient(config)

    try:
        response = client.send_prompt(prompt)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())