"""A self-hosted HTTP service in front of a local LLM agent.

This exposes a :class:`~llm_bot.agent.Session` over an OpenAI-compatible
``/v1/chat/completions`` endpoint, plus a tiny browser chat page and a
``/health`` probe. It is meant to run on a separate "frontend" machine that
reaches Ollama over the LAN: the model host stays a dumb GPU box, while auth,
session state, rate limiting and the concurrency queue live here.

Why a wrapper instead of exposing Ollama directly:

* Ollama has no auth, sessions, RAG or rate limiting -- that is this project's
  code, and it should run where the clients point, not on the GPU host;
* the LLM client is synchronous and :class:`Session` is not thread-safe, so the
  blocking call is bounced to a worker thread, bounded by one asyncio semaphore
  (the "queue") plus one lock per session -- a single session never interleaves
  its own turns;
* the per-model limits in :class:`~llm_bot.config.LLMConfig` (``context_window``,
  ``max_request_tokens``, ``max_tokens``) already refuse an oversized request
  before it leaves the process, so a bad client cannot overflow the model.

``parallel`` > 1 serves several sessions concurrently; Ollama may still
serialise them, but the queue gives deterministic overload behaviour (503)
instead of an unbounded pile-up. The service is single-process by design --
rate-limit counters and the session cache are per-process -- so run it with one
uvicorn worker.
"""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from typing import Any, Callable, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from llm_bot.client import (
    ContextOverflowError,
    ContextTooLargeError,
    LLMError,
    LLMRequestError,
    LLMTruncatedError,
)
from llm_bot.invariants import InvariantViolationError
from llm_bot.tokens import ChatResult

DEFAULT_QUEUE_TIMEOUT = 300.0
DEFAULT_AGENT = "assistant"

#: Anything a :class:`~llm_bot.agent.Session` needs for one turn. Typed as a
#: Protocol so tests can pass a stand-in without building real stores.
class SessionLike(Protocol):
    def chat_with_details(self, user_message: str, *, service_turn: bool = False) -> ChatResult: ...


#: Builds (or loads) a session for the given id. Called from a worker thread.
SessionFactory = Callable[[str], SessionLike]

#: Optional probe: returns a JSON-serialisable dict describing the upstream.
HealthProbe = Callable[[], dict[str, Any]]


class RateLimiter:
    """A per-key token bucket. Thread-safety is not needed: handlers run on one
    event loop, so a plain dict is enough."""

    def __init__(self, per_minute: int, burst: int | None = None) -> None:
        if per_minute <= 0:
            raise ValueError("rate_limit must be positive")
        self.rate = per_minute / 60.0
        self.burst = float(burst if burst is not None else per_minute)
        self._state: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> float | None:
        """Consume one token. Return ``None`` when allowed, else seconds to wait."""
        now = time.monotonic()
        tokens, last = self._state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / self.rate
        self._state[key] = (tokens - 1.0, now)
        return None


def create_app(
    *,
    session_factory: SessionFactory,
    agent_name: str = DEFAULT_AGENT,
    auth_token: str | None = None,
    parallel: int = 1,
    rate_limit: int | None = None,
    rate_burst: int | None = None,
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT,
    request_timeout: float | None = None,
    health_probe: HealthProbe | None = None,
    title: str = "llm-bot",
) -> FastAPI:
    """Assemble the FastAPI app.

    Args:
        session_factory: Builds/loads a session by id.
        agent_name: Reported in responses and ``/health``.
        auth_token: When set, every endpoint except ``/health`` requires
            ``Authorization: Bearer <token>``. ``None`` disables auth (local dev).
        parallel: Maximum concurrent turns; further requests queue.
        rate_limit: Requests per minute per client (token or IP). ``None`` = off.
        rate_burst: Bucket size; defaults to ``rate_limit``.
        queue_timeout: Seconds a request may wait for a slot before 503.
            Defaults to a cold RAG turn's worth of time; ``<= 0`` waits
            indefinitely.
        request_timeout: Optional ceiling on a single turn; over it -> 504.
        health_probe: Optional upstream check for ``/health``.
        title: App title.
    """
    if parallel < 1:
        raise ValueError("parallel must be >= 1")

    limiter = RateLimiter(rate_limit, rate_burst) if rate_limit else None
    semaphore = asyncio.Semaphore(parallel)
    locks: dict[str, asyncio.Lock] = {}
    sessions: dict[str, SessionLike] = {}
    state = {"queued": 0, "in_flight": 0, "handled": 0, "errors": 0}

    def _session_lock(session_id: str) -> asyncio.Lock:
        lock = locks.get(session_id)
        if lock is None:
            lock = locks[session_id] = asyncio.Lock()
        return lock

    def _session(session_id: str) -> SessionLike:
        session = sessions.get(session_id)
        if session is None:
            session = sessions[session_id] = session_factory(session_id)
        return session

    def _require_auth(request: Request) -> None:
        if auth_token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, auth_token):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _rate_key(request: Request) -> str:
        if auth_token is not None:
            return request.headers.get("authorization", "")
        return request.client.host if request.client else "unknown"

    def _apply_rate_limit(request: Request) -> None:
        if limiter is None:
            return
        wait = limiter.check(_rate_key(request))
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(max(1, int(wait + 0.999)))},
            )

    async def _run_turn(session_id: str, text: str) -> ChatResult:
        async with _session_lock(session_id):
            state["queued"] += 1
            try:
                if queue_timeout and queue_timeout > 0:
                    await asyncio.wait_for(semaphore.acquire(), timeout=queue_timeout)
                else:
                    await semaphore.acquire()
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail="Server busy, retry shortly.",
                    headers={"Retry-After": "5"},
                )
            finally:
                state["queued"] -= 1
            state["in_flight"] += 1
            failed = True
            try:
                session = _session(session_id)
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(
                    None, lambda: session.chat_with_details(text)
                )
                if request_timeout and request_timeout > 0:
                    try:
                        result = await asyncio.wait_for(future, timeout=request_timeout)
                    except asyncio.TimeoutError:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Upstream timed out after {request_timeout:g}s.",
                        ) from None
                else:
                    result = await future
                failed = False
                return result
            finally:
                state["in_flight"] -= 1
                state["handled"] += 1
                if failed:
                    state["errors"] += 1
                semaphore.release()

    app = FastAPI(title=title)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok",
            "agent": agent_name,
            "parallel": parallel,
            "queue": {"queued": state["queued"], "in_flight": state["in_flight"]},
            "handled": state["handled"],
            "errors": state["errors"],
        }
        if health_probe is not None:
            try:
                body["upstream"] = health_probe()
            except Exception as exc:  # pragma: no cover - defensive
                body["status"] = "degraded"
                body["upstream"] = {"error": str(exc)}
        return body

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        _require_auth(request)
        _apply_rate_limit(request)
        payload = await _read_json(request)
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported.")
        text = _last_user_message(payload)
        session_id = _session_id(payload, request)
        try:
            result = await _run_turn(session_id, text)
        except HTTPException:
            raise
        except ContextOverflowError as exc:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Context overflow: {exc.context_tokens} tokens > "
                    f"window {exc.context_window}. Start a new session or trim history."
                ),
            ) from exc
        except ContextTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except InvariantViolationError as exc:
            raise HTTPException(status_code=400, detail=exc.refusal) from exc
        except LLMTruncatedError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except LLMRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return JSONResponse(_openai_response(agent_name, session_id, result))

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _CHAT_PAGE

    return app


async def _read_json(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")
    return payload


def _last_user_message(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            raise HTTPException(status_code=400, detail="The user message is empty.")
    raise HTTPException(status_code=400, detail="No user message found in 'messages'.")


def _session_id(payload: dict[str, Any], request: Request) -> str:
    value = payload.get("session_id") or request.headers.get("x-session-id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "default"


def _openai_response(agent_name: str, session_id: str, result: ChatResult) -> dict[str, Any]:
    usage = result.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": agent_name,
        "session_id": session_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.reply},
                "finish_reason": "length" if result.truncated else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.context_tokens,
            "completion_tokens": usage.reply_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


_CHAT_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llm-bot</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #log { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; height: 50vh; overflow-y: auto; white-space: pre-wrap; }
  .u { color: #0a58ca; } .a { color: #146c43; }
  form { display: flex; gap: .5rem; margin-top: 1rem; }
  input[type=text] { flex: 1; padding: .5rem; }
  input[type=password] { width: 10rem; }
  button { padding: .5rem 1rem; }
</style>
</head>
<body>
<h1>Локальная LLM</h1>
<div id="log">Введите вопрос и нажмите «Отправить».</div>
<form id="f">
  <input type="text" id="q" placeholder="Ваш вопрос…" autocomplete="off" required>
  <input type="password" id="t" placeholder="token" title="Bearer token">
  <button type="submit">Отправить</button>
</form>
<script>
const log = document.getElementById('log');
let sid = sessionStorage.getItem('sid');
if (!sid) { sid = 'web-' + Math.random().toString(36).slice(2, 10); sessionStorage.setItem('sid', sid); }
document.getElementById('t').value = localStorage.getItem('token') || '';
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  const token = document.getElementById('t').value.trim();
  if (!q) return;
  localStorage.setItem('token', token);
  document.getElementById('q').value = '';
  log.textContent += '\\n\\nВы: ' + q + '\\nLLM: …';
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
      body: JSON.stringify({ messages: [{ role: 'user', content: q }], session_id: sid })
    });
    const data = await res.json();
    const reply = res.ok ? data.choices[0].message.content : ('ошибка ' + res.status + ': ' + (data.detail || ''));
    log.textContent = log.textContent.replace(/\\nLLM: …$/, '\\nLLM: ' + reply);
  } catch (err) {
    log.textContent = log.textContent.replace(/\\nLLM: …$/, '\\nLLM: сеть недоступна');
  }
  log.scrollTop = log.scrollHeight;
});
</script>
</body>
</html>
"""
