#!/usr/bin/env python3
"""A minimal client for the llm-bot HTTP service (``scripts/serve_llm.py``).

It plays the part of an external user: sends an OpenAI-style chat request and
prints the reply. Handy to prove the client/server split, to test from a second
device, and to simulate several simultaneous connections.

    # one question
    python scripts/chat_client.py --token secret -m "привет"

    # interactive
    python scripts/chat_client.py --token secret

    # two simultaneous connections (stability smoke test)
    python scripts/chat_client.py --token secret --concurrent 2 -m "привет"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client for the llm-bot HTTP service.")
    parser.add_argument("--url", default=os.environ.get("SERVE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.environ.get("SERVE_TOKEN"))
    parser.add_argument(
        "--session-id", default=None, help="Conversation id (default: per process)."
    )
    parser.add_argument("-m", "--message", default=None, help="Send one message and exit.")
    parser.add_argument(
        "--count", type=int, default=1, help="Repeat the message N times (sequential)."
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Send this many requests at once (uses --message).",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    return parser


def ask(client: httpx.Client, args, message: str, session_id: str) -> str:
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    response = client.post(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": message}], "session_id": session_id},
    )
    if response.status_code != 200:
        raise SystemExit(f"HTTP {response.status_code}: {response.text}")
    data = response.json()
    if args.json:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return data["choices"][0]["message"]["content"]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session_id = args.session_id or f"cli-{os.getpid()}"

    with httpx.Client(timeout=args.timeout) as client:
        if args.message is None:
            print("Interactive mode. Ctrl-D to quit.")
            while True:
                try:
                    message = input("> ").strip()
                except EOFError:
                    print()
                    break
                if not message:
                    continue
                print(ask(client, args, message, session_id))
            return 0

        if args.concurrent > 1:
            with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
                futures = [
                    pool.submit(ask, client, args, args.message, f"{session_id}-{i}")
                    for i in range(args.concurrent)
                ]
                for i, future in enumerate(futures, 1):
                    print(f"[{i}] {future.result()}")
            return 0

        for _ in range(args.count):
            print(ask(client, args, args.message, session_id))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
