# llm-bot

A provider-agnostic Python client for OpenAI-compatible LLM APIs, extended with
an **agent layer**. Beyond single prompts, it supports multiple LLM providers,
named agents (roles), and persistent multi-turn chat sessions that keep their
own message history.

Architecture layers:

- [`LLMClient`](llm_bot/client.py) — transport: HTTP, retries, auth, payload.
- [`Agent`](llm_bot/agent.py) — a role: system prompt, temperature, max tokens,
  and a reference to a model profile.
- [`Session`](llm_bot/agent.py) — a conversation with an agent; owns its message
  history and persists it to disk.
- Storage backends ([`stores.py`](llm_bot/stores.py)) — pluggable repositories
  for models, agents and sessions (YAML/JSON today, a database later).

This design keeps the console UI ([`cli.py`](llm_bot/cli.py)) thin and makes it
trivial to add a **web interface** later without touching the logic.

## Features

- Provider-agnostic — works with OpenAI, Ollama, LM Studio, LocalAI, GigaChat,
  and any other server exposing the `/v1/chat/completions` endpoint.
- **Multiple providers** — model credentials live in `data/models.yaml`, so you
  can define any number of LLM profiles.
- **Named agents** — roles defined in `data/agents.yaml` (system prompt,
  temperature, max tokens) that reference a model profile.
- **Personalization profiles** — orchestration config in `data/profiles.yaml`
  (style, format, expertise, constraints) composed onto an agent at session
  creation (`--profile developer`); memory is never touched by profiles
  (see [Personalization](docs/personalization.md)).
- **Persistent chat sessions** — each conversation keeps its own history in
  `data/sessions/*.json` and survives process restarts.
- Automatic retries with exponential backoff for transient failures
  (timeouts, network errors, HTTP 429 / 5xx).
- Clear error handling via custom exception types.
- Console entry point: interactive chat (`--agent`) or a single prompt.
- `--details` flag to print request/response diagnostics (URL, model, formed
  request body, token usage, per-attempt timing) to stderr.
- Built-in GigaChat support with automatic OAuth2 token acquisition and refresh.
- **Context compression** — rolling-summary keeps long dialogs inside the token
  budget: recent turns stay verbatim, older ones fold into a running summary that
  is injected at the front of each request (see [compression](docs/context.md)).
- **Pluggable context strategies** — three alternative ways to manage context
  *without* summary, all selectable per-agent or per-session (`--strategy`):
  sliding window / sticky facts (key-value durable memory) / branching
  (see [context strategies](docs/context.md)).
- **Invariants (hard constraints)** — owner-declared architecture/decision/stack/
  business rules stored separately from the dialog (`data/invariants.yaml` +
  per-session entries); injected into every request and enforced by a
  deterministic regex gate BEFORE anything is sent, with self-explaining
  refusals (see [invariants](docs/context.md)).
- **Task state machine** — the work on a task is formalized as a finite-state
  machine (stage `planning → execution → validation → done` + `paused`, current
  step, expected action). The stage is recognized from the dialog by a small
  LLM call or driven manually via `/task` commands; the snapshot is persisted
  with the session and injected into the system context on every turn, so
  after a pause or a process restart a bare «продолжай» is enough
  (see [the task state machine](docs/context.md)).
- **Comparison scripts** — reproducible experiments (`scripts/compare_compression.py`,
  `scripts/compare_context_strategies.py`) comparing answer quality and token spend
  across strategies.
- Pytest suite using `httpx.MockTransport` (no network needed).

## Project structure

```
llm-bot/
├── llm_bot/
│   ├── __init__.py            # package exports
│   ├── config.py              # LLMConfig — settings from env vars
│   ├── diagnostics.py         # RequestDetails/ResponseDetails + listener protocol
│   ├── client.py              # LLMClient — transport + retries + errors + detail events
│   ├── compress.py            # ContextCompressor + CompressionSettings (rolling summary)
│   ├── context_strategies.py  # SlidingWindow / StickyFacts / Branching (pluggable strategies)
│   ├── task_state.py          # TaskStateMachine: formal task stage/step/action FSM
│   ├── invariants.py          # Invariant/InvariantRegistry: hard constraints + gate
│   ├── profiles.py            # apply_profile: compose ProfileConfig onto AgentConfig
│   ├── stores.py              # Repository interfaces (ModelStore/AgentStore/ProfileStore/SessionStore)
│   ├── yaml_stores.py         # YamlModelStore / YamlAgentStore / YamlProfileStore (data/*.yaml)
│   ├── json_session_store.py  # JsonSessionStore (data/sessions/*.json)
│   ├── agent.py               # Agent (role) + Session (conversation with history)
│   ├── factory.py             # Wiring: assemble client/agent/session from stores
│   ├── gigachat.py            # GigaChat OAuth2 token provider + factory
│   ├── cli.py                 # console entry point (interactive chat / single prompt)
│   ├── server.py              # FastAPI app: self-hosted HTTP service in front of an agent
│   └── __main__.py            # enables `python -m llm_bot`
├── scripts/
│   ├── compare_compression.py       # token/quality experiment: with vs without compression
│   ├── compare_context_strategies.py# strategies on a shared "ТЗ" scenario
│   ├── index_documents.py           # local document index: chunking + embeddings + benchmark
│   ├── serve_llm.py                 # run the HTTP service (scripts/chat_client.py is the client)
│   └── ...                          # other comparison/demo scripts
├── tests/
│   ├── test_client.py         # LLMClient tests (mocked transport)
│   ├── test_server.py         # HTTP service tests (stub session, ASGI transport)
│   ├── test_agent.py          # Agent + Session tests
│   ├── test_compress.py       # context-compression tests
│   ├── test_task_state.py     # task state machine tests
│   ├── test_compare_compression.py
│   ├── test_index_documents.py # document index (offline, fake embedder)
│   ├── test_rerank.py          # second retrieval stage (offline, fake encoder)
│   ├── test_stores.py         # YAML/JSON store tests
│   └── test_gigachat.py       # GigaChat token provider tests
├── results/                   # experiment reports (markdown)
├── docs/                      # detailed documentation (this README is the overview)
├── plans/                     # design/planning notes
├── dialogues/                 # scripted dialogues for scripts/eval_dialogue.py
├── models.example.yaml        # template -> copy to data/models.yaml
├── agents.example.yaml        # template -> copy to data/agents.yaml
├── invariants.example.yaml    # template -> copy to data/invariants.yaml
├── index_queries.example.yaml # template -> hand-written benchmark queries for the index
├── rag_questions.example.yaml       # control questions for the repo + coffee-shop corpora
├── rag_questions.yaml                # control questions for the knowledge-base corpus
├── requirements.txt
├── .env.example
└── README.md

data/                          # runtime data — DO NOT COMMIT (see .gitignore)
├── models.yaml                # real provider credentials
├── agents.yaml                # real agent definitions
└── sessions/                  # per-session chat histories (*.json)
```

## Installation

Requires Python 3.10+.

```bash
cd llm-bot
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Quickstart

Create the runtime files from the templates (they are **not** committed):

```bash
mkdir -p data/sessions
cp models.example.yaml data/models.yaml
cp agents.example.yaml data/agents.yaml
cp .env.example .env
```

Chat with an agent (history persists in `data/sessions/` and resumes with the
same `--session` id):

```bash
python -m llm_bot --agent assistant
python -m llm_bot --agent translator "Good morning"      # one-shot
```

Inside the interactive chat: `/history` prints the transcript, `/help <вопрос>`
asks about the project itself, `exit`/`quit`/Ctrl-D leave. All flags and modes
are in [docs/usage.md](docs/usage.md).

## Documentation

The README stays an overview; the detailed docs live in [`docs/`](docs/).

- [Configuration](docs/configuration.md) — env vars, `data/models.yaml`, agents, sessions.
- [Usage](docs/usage.md) — CLI flags and modes, JSON format, diagnostics.
- [Personalization](docs/personalization.md) — profiles as orchestration config.
- [Programmatic use](docs/programmatic.md) — assembling clients and sessions in code.
- [MCP](docs/mcp.md) — connection, the notes-CRM server, multi-server orchestration.
- [Developer assistant `/help`](docs/help.md) — answering about the project itself.
- [Currency pipeline](docs/currency.md) — agent-composed tool chains.
- [Scheduler](docs/scheduler.md) — background tasks 24/7.
- [Context, strategies, tasks, invariants, compression](docs/context.md) — token accounting and the context machinery.
- [Benchmarks](docs/benchmarks.md) — comparing prompt strategies and model tiers.
- [Retrieval and RAG](docs/retrieval.md) — the local index and answering from it.
- [Self-hosted HTTP service](docs/server.md) — serving an agent over HTTP.

- [Code review](docs/code-review.md)
- [Support assistant](docs/support.md) — product questions from the FAQ plus the customer's ticket.

## Tests

Run the mocked test suite (no live API required):

```bash
python -m pytest tests/ -v
```

Coverage includes: successful request, retry-then-success, rate limiting,
network errors, exhausted retries, permanent HTTP errors, unexpected response
shapes, config overrides, and the detail-listener events (request url/model/payload,
token usage, per-attempt reporting).
