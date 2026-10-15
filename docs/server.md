# Self-hosted HTTP service (local LLM as a service)

`llm_bot/server.py` + `scripts/serve_llm.py` expose an agent over an
OpenAI-compatible `POST /v1/chat/completions`, plus a browser chat page (`GET /`)
and a `GET /health` probe. The goal is to hand the model to other people and
devices without giving them the GPU host: **Ollama stays on the internal network,
the service is the only thing that faces the outside.**

```
client ── HTTPS + Bearer ──► [frontend: scripts/serve_llm.py]  (agent + RAG + auth + queue + limits)
                                   │  OpenAI-compatible, http over LAN
                                   ▼
                             [GPU host: ollama]  (OLLAMA_HOST=0.0.0.0:11434, firewall: frontend only)
```

Why a wrapper instead of exposing Ollama directly: Ollama has no auth, sessions,
RAG or rate limiting — that is this project's code, and it belongs on the machine
the clients point at, not on the GPU box. The client is synchronous and a session
is not thread-safe, so each turn runs in a worker thread, bounded by one asyncio
semaphore (the queue) plus one lock per session.

## Run it

```bash
pip install -r requirements.txt            # adds fastapi + uvicorn

# frontend machine, one worker (the queue, sessions and counters are per-process)
SERVE_TOKEN=secret python scripts/serve_llm.py \
    --host 0.0.0.0 --port 8000 --agent assistant \
    --rag-index data/kb_emb/index_structure.json --parallel 1 --rate-limit 30
```

`data/models.yaml` on the frontend points at the model host over the LAN, e.g.
`base_url: http://<gpu-host>:11434/v1` (with a generous `timeout` — a 7B on a weak
GPU can take tens of seconds per turn).

## Make Ollama reachable on the LAN

On the GPU host, bind Ollama to all interfaces and let only the frontend through:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
# firewall: allow tcp/11434 from the frontend's IP only
```

Ollama has no authentication, so its port must never face the internet — only the
service does.

## Putting it on the network

Both paths need no code changes:

* **Direct** (real public IP / domain): bind the service to `0.0.0.0` and, if you
  want TLS, put a terminator in front (`caddy reverse-proxy`, `nginx`).
* **Tunnel** (CGNAT, no port forwarding): `cloudflared tunnel --url http://127.0.0.1:8000`
  on the frontend gives a public HTTPS URL with no router changes.

## Call it

```bash
# curl
curl -s http://<frontend>:8000/v1/chat/completions \
  -H "Authorization: Bearer $SERVE_TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"привет"}],"session_id":"demo"}'

# bundled client: one question, interactive, or two simultaneous connections
python scripts/chat_client.py --token secret -m "привет"
python scripts/chat_client.py --token secret
python scripts/chat_client.py --token secret --concurrent 2 -m "привет"
```

The `session_id` (body field or `X-Session-Id` header) selects the conversation;
omitting it uses one shared `default` session. The page at `/` asks for the token
once and keeps a per-tab session.

## Limits and stability

* **Auth** — `SERVE_TOKEN`; every route but `/health` needs `Authorization: Bearer`.
* **Concurrency** — `--parallel N` turns at once, the rest wait in an asyncio
  queue. Turns for the *same* session always run one at a time (a lock per
  session), so a dialog never interleaves. Waiting past `--queue-timeout` → `503`;
  the default (300 s) is sized for a cold RAG turn, which can spend a minute
  loading the embedder before the model even starts.
* **Rate limit** — `--rate-limit` requests/minute per client → `429` + `Retry-After`.
* **Context / length** — the per-model `context_window`, `max_request_tokens` and
  `max_tokens` still apply; an oversized request is refused with `413` *before* it
  reaches the model. A reply cut by the token limit returns `finish_reason: "length"`.
* **Per-turn ceiling** — `--request-timeout` returns `504` for a runaway turn.

`GET /health` reports upstream reachability and whether the model is loaded
(`/api/tags`, `/api/ps`), the live queue depth, and the counters `handled`
(turns that reached the model) and `errors` (a subset: those that failed). The
counters are in-memory, per-process, and reset on restart — as is the rate-limit
state and the session cache.

## Verify

```bash
# two simultaneous connections
python scripts/chat_client.py --token secret --concurrent 2 -m "привет"

# stability + limits: watch the status codes
seq 1 20 | xargs -P 5 -I{} curl -s -o /dev/null -w "%{http_code}\n" \
  http://<frontend>:8000/v1/chat/completions \
  -H "Authorization: Bearer secret" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"привет"}]}' | sort | uniq -c
```

Run a single uvicorn worker: the rate-limit counters and the session cache live in
the process. Tests: `tests/test_server.py` (offline — stub session + ASGI transport).
