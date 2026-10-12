#!/usr/bin/env python3
"""Run the self-hosted HTTP service for the local LLM agent.

Typical split deployment: this runs on the frontend machine (the one with the
external address) and points ``data/models.yaml`` at Ollama on the GPU host over
the LAN, e.g. ``base_url: http://192.168.1.50:11434/v1``. Clients then talk only
to this process; Ollama's port stays on the internal network.

    # frontend machine, one worker (see llm_bot/server.py on why single-process)
    SERVE_TOKEN=secret python scripts/serve_llm.py \\
        --host 0.0.0.0 --port 8000 --agent assistant \\
        --rag-index data/kb_emb/index_structure.json --parallel 1 --rate-limit 30

    # another device
    curl -s http://<frontend>:8000/v1/chat/completions \\
        -H "Authorization: Bearer secret" -H "Content-Type: application/json" \\
        -d '{"messages":[{"role":"user","content":"привет"}]}'
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from llm_bot.factory import build_retriever, make_session  # noqa: E402
from llm_bot.json_session_store import JsonSessionStore  # noqa: E402
from llm_bot.memory_store import JsonMemoryStore  # noqa: E402
from llm_bot.rag import Retriever  # noqa: E402
from llm_bot.server import create_app  # noqa: E402
from llm_bot.stores import ModelConfig  # noqa: E402
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local LLM agent over HTTP.")
    parser.add_argument("--host", default=os.environ.get("SERVE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVE_PORT", "8000")))
    parser.add_argument(
        "--agent",
        default=os.environ.get("SERVE_AGENT", "assistant"),
        help="Agent name from data/agents.yaml.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SERVE_TOKEN"),
        help="Bearer token required from clients. Also read from $SERVE_TOKEN.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=int(os.environ.get("SERVE_PARALLEL", "1")),
        help="Maximum concurrent turns; the rest queue.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=int(os.environ.get("SERVE_RATE_LIMIT", "0")),
        help="Requests per minute per client; 0 disables the limiter.",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=float(os.environ.get("SERVE_QUEUE_TIMEOUT", "300")),
        help="Seconds a request may wait for a slot before 503; <=0 waits forever.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("SERVE_REQUEST_TIMEOUT", "0")) or None,
        help="Optional ceiling on a single turn, seconds.",
    )
    parser.add_argument(
        "--rag-index",
        default=os.environ.get("SERVE_RAG_INDEX"),
        help="Path to a JSON index for RAG (e.g. data/kb_emb).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=int(os.environ.get("SERVE_RAG_TOP_K", "8")),
    )
    parser.add_argument(
        "--rag-rerank",
        action="store_true",
        help="Enable the cross-encoder rerank stage (CPU-heavy; off by default).",
    )
    parser.add_argument("--log-level", default=os.environ.get("SERVE_LOG_LEVEL", "info"))
    return parser


def _strip_v1(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed[:-3].rstrip("/") if trimmed.endswith("/v1") else trimmed


def make_ollama_probe(model: ModelConfig):
    """Best-effort Ollama probe for ``/health``: is the host up, is the model loaded."""
    root = _strip_v1(model.base_url)

    def probe() -> dict:
        info: dict = {"base_url": model.base_url, "model": model.model}
        try:
            with httpx.Client(timeout=5.0) as client:
                tags = client.get(f"{root}/api/tags")
                if tags.status_code != 200:
                    info["note"] = "no /api/tags here (not an Ollama host?)"
                    return info
                names = [m.get("name", "") for m in tags.json().get("models", [])]
                info["reachable"] = True
                info["model_present"] = any(
                    n == model.model or n.split(":")[0] == model.model.split(":")[0]
                    for n in names
                )
                ps = client.get(f"{root}/api/ps")
                if ps.status_code == 200:
                    loaded = [m.get("name", "") for m in ps.json().get("models", [])]
                    info["loaded_models"] = loaded
                    info["model_loaded"] = model.model in loaded
        except httpx.HTTPError as exc:
            info["reachable"] = False
            info["error"] = str(exc)
        return info

    return probe


def _warm_retriever(retriever: Retriever) -> None:
    """Load the embedding model before the first request needs it.

    Best-effort: a failed warm-up just means the first query pays the load, so
    the error is reported and swallowed. ``Retriever.embedder`` serializes the
    load, so a request that arrives mid-warm-up waits instead of loading twice.
    """
    started = time.monotonic()
    try:
        retriever.embedder  # triggers the lazy model load
    except Exception as exc:  # noqa: BLE001 - warm-up must never take the server down
        print(f"[warn] RAG warm-up failed: {exc}", file=sys.stderr, flush=True)
    else:
        print(
            f"RAG: embedding model ready in {time.monotonic() - started:.1f}s",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    model_store = YamlModelStore()
    agent_store = YamlAgentStore()
    session_store = JsonSessionStore()
    memory_store = JsonMemoryStore()

    agent_config = agent_store.get(args.agent)
    model = model_store.get(agent_config.model)

    # One Retriever for the whole process. A fresh one per session would reload
    # the embedding model — tens of CPU seconds — on every new dialog, and the
    # model is only ever read, so sharing it is safe. Built here so a missing
    # index fails at startup, and passed by reference into every session.
    retriever = None
    if args.rag_index:
        retriever = build_retriever(
            args.rag_index,
            context_window=model.context_window,
            top_k=args.rag_top_k,
            rerank=args.rag_rerank,
        )

    def session_factory(session_id: str):
        return make_session(
            session_id,
            args.agent,
            model_store=model_store,
            agent_store=agent_store,
            session_store=session_store,
            memory_store=memory_store,
            retriever=retriever,
        )

    app = create_app(
        session_factory=session_factory,
        agent_name=args.agent,
        auth_token=args.token,
        parallel=args.parallel,
        rate_limit=args.rate_limit or None,
        queue_timeout=args.queue_timeout,
        request_timeout=args.request_timeout,
        health_probe=make_ollama_probe(model),
    )

    auth = "token required" if args.token else "NO AUTH (set SERVE_TOKEN)"
    print(
        f"Serving agent '{args.agent}' -> {model.base_url} ({model.model})",
        flush=True,
    )
    print(
        f"Listen http://{args.host}:{args.port}  parallel={args.parallel}  {auth}",
        flush=True,
    )
    if retriever is not None:
        print("RAG: warming the embedding model in the background", flush=True)
        threading.Thread(
            target=_warm_retriever,
            args=(retriever,),
            name="rag-warmup",
            daemon=True,
        ).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
