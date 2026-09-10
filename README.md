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
- **Persistent chat sessions** — each conversation keeps its own history in
  `data/sessions/*.json` and survives process restarts.
- Automatic retries with exponential backoff for transient failures
  (timeouts, network errors, HTTP 429 / 5xx).
- Clear error handling via custom exception types.
- Console entry point: interactive chat (`--agent`) or a single prompt.
- `--details` flag to print request/response diagnostics (URL, model, formed
  request body, token usage, per-attempt timing) to stderr.
- Built-in GigaChat support with automatic OAuth2 token acquisition and refresh.
- Pytest suite using `httpx.MockTransport` (no network needed).

## Project structure

```
llm-bot/
├── llm_bot/
│   ├── __init__.py            # package exports
│   ├── config.py              # LLMConfig — settings from env vars
│   ├── diagnostics.py         # RequestDetails/ResponseDetails + listener protocol
│   ├── client.py              # LLMClient — transport + retries + errors + detail events
│   ├── stores.py              # Repository interfaces (ModelStore/AgentStore/SessionStore)
│   ├── yaml_stores.py         # YamlModelStore / YamlAgentStore (data/*.yaml)
│   ├── json_session_store.py  # JsonSessionStore (data/sessions/*.json)
│   ├── agent.py               # Agent (role) + Session (conversation with history)
│   ├── factory.py             # Wiring: assemble client/agent/session from stores
│   ├── gigachat.py            # GigaChat OAuth2 token provider + factory
│   ├── cli.py                 # console entry point (interactive chat / single prompt)
│   └── __main__.py            # enables `python -m llm_bot`
├── tests/
│   ├── test_client.py         # LLMClient tests (mocked transport)
│   ├── test_agent.py          # Agent + Session tests
│   ├── test_stores.py         # YAML/JSON store tests
│   └── test_gigachat.py       # GigaChat token provider tests
├── models.example.yaml        # template -> copy to data/models.yaml
├── agents.example.yaml        # template -> copy to data/agents.yaml
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

## Configuration

Copy `.env.example` to `.env` and edit it:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

| Variable             | Default                 | Description                                              |
| -------------------- | ----------------------- | -------------------------------------------------------- |
| `LLM_BASE_URL`       | `https://api.openai.com/v1` | Base URL of the chat-completions API                |
| `LLM_API_KEY`        | *(empty)*               | API key (leave empty for local providers)                |
| `LLM_MODEL`          | `gpt-4o-mini`           | Model identifier                                         |
| `LLM_MAX_RETRIES`    | `3`                     | Retries for transient failures                           |
| `LLM_RETRY_BACKOFF`  | `1.0`                   | Base backoff seconds (exponential: `backoff * 2^n`)      |
| `LLM_TIMEOUT`        | `30`                    | Request timeout in seconds                               |
| `LLM_SYSTEM_PROMPT`  | *(empty)*           | Optional system prompt sent before the user prompt (e.g. to request a JSON response format) |
| `LLM_DEFAULT_SYSTEM_PROMPT` | *(empty)*   | Optional **base** system prompt that is always prepended to any other system prompt (`LLM_SYSTEM_PROMPT`, expert roles, etc.). Set it to keep replies in the user's language by default. Leave empty to disable. |
| `LLM_MAX_RESPONSE_WORDS` | *(empty)*           | Target maximum reply length in words; adds a briefness instruction to the system prompt (empty = no limit). Per-invocation via `--max-response-words`. |
| `LLM_MAX_TOKENS`         | *(empty)*           | Hard cap on generated tokens per reply, enforced by the API (`max_tokens` in the payload). Empty = provider default. |

### Example: use a local Ollama server

```bash
# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=
LLM_MODEL=llama3.2
```

### Example: GigaChat (Sber)

GigaChat exposes an OpenAI-compatible endpoint, but requires an OAuth2 token
exchange before each session. The client handles this automatically via
`--provider gigachat`: it exchanges your credentials for an access token, caches
it, and refreshes it when it expires (~30 min).

```bash
# .env
LLM_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
LLM_MODEL=GigaChat-2-Max
GIGACHAT_CLIENT_ID=your-client-id
GIGACHAT_CLIENT_SECRET=your-client-secret
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

Then run with:

```bash
python -m llm_bot --provider gigachat "Привет! Расскажи о себе"
```

> **Note:** if your Sber credentials only provide a `client_secret` (no separate
> client id), set `GIGACHAT_CLIENT_SECRET` and leave `GIGACHAT_CLIENT_ID` empty,
> or provide a pre-encoded `GIGACHAT_BASIC_AUTH="Basic ..."` if your variant
> differs from `base64(client_id:client_secret)`.

## Models, agents and sessions

The agent layer uses three separate definitions:

1. **Models** (`data/models.yaml`, key `models`) — provider credentials only
   (`base_url`, `api_key`, `model`, optional GigaChat OAuth fields). No behaviour.
2. **Agents** (`data/agents.yaml`, key `agents`) — a role: which model to use,
   the system prompt, temperature and `max_tokens`. Many agents can share one
   model profile while differing in behaviour.
3. **Sessions** (`data/sessions/*.json`) — a conversation with an agent, owning
   its own message history. Sessions are created at runtime, not defined in YAML.

Create the real files from the templates (they are **not** committed):

```bash
mkdir -p data/sessions
cp models.example.yaml data/models.yaml
cp agents.example.yaml data/agents.yaml
cp .env.example .env
```

Secrets in `data/models.yaml` are referenced as `${ENV_VAR}` and resolved from
the environment / `.env`, so real keys never need to be committed. Keep `data/`
out of version control (it is listed in `.gitignore`).

Example `data/models.yaml`:

```yaml
models:
  openai-gpt4o:
    provider: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini
  ollama-local:
    provider: openai
    base_url: http://localhost:11434/v1
    api_key: ""
    model: llama3.2
```

Example `data/agents.yaml` (two agents sharing the same model profile):

```yaml
agents:
  translator:
    model: openai-gpt4o
    system_prompt: "Переводи с русского на английский и обратно."
    temperature: 0.2
    max_tokens: 512
  critic:
    model: openai-gpt4o
    system_prompt: "Давай строгий критический разбор текста."
    temperature: 0.9
```

List available agents:

```bash
python -m llm_bot --list-agents
```

## Usage

### Interactive chat with an agent

Start an interactive chat with the `assistant` agent (history is saved to
`data/sessions/` and resumes with the same `--session` id):

```bash
python -m llm_bot --agent assistant
python -m llm_bot --agent assistant --session my-conversation   # resume
```

Inside the interactive chat, type `/history` (or `/история`) to print all
previous messages of the current session; `exit`/`quit`/Ctrl-D leave the chat.

A single one-shot turn via an agent:

```bash
python -m llm_bot --agent translator "Good morning"
```

### Legacy: single direct call

Prompt as an argument:

```bash
python -m llm_bot "Explain what an API is in one sentence."
```

Prompt piped via stdin:

```bash
echo "Summarize this text..." | python -m llm_bot
```

Override settings per-invocation:

```bash
python -m llm_bot --base-url http://localhost:11434/v1 --model llama3.2 "Hello"
python -m llm_bot --model gpt-4o-mini --verbose "Tell me a joke"
```

### Inspecting request/response details

Pass `--details` to see exactly what is sent and how the API responds. Details go
to **stderr**, so the reply on stdout stays clean:

```bash
python -m llm_bot --details "Hello"
```

```
[details] POST https://api.openai.com/v1/chat/completions
[details] model: gpt-4o-mini
[details] request body:
           {
             "model": "gpt-4o-mini",
             "messages": [
               {"role": "user", "content": "Hello"}
             ]
           }
[details] attempt 1: HTTP 200 in 320ms
[details] usage: prompt_tokens=8 completion_tokens=5 total_tokens=13
```

- The printed URL is where the request goes; the model and the fully formed
  request body are shown verbatim (the `Authorization` header is never printed,
  so API keys stay safe).
- On retries, each attempt is reported (`attempt 1`, `attempt 2`, ...) with its
  HTTP status and round-trip time.
- Token usage appears only when the provider reports it (OpenAI does; some local
  servers omit it).

### Enforcing a specific JSON response format

Use a system prompt to tell the model exactly how to format its reply. It can be
set per-invocation with `--system-prompt` or globally via `LLM_SYSTEM_PROMPT`:

```bash
python -m llm_bot \
  --system-prompt 'Отвечай строго в формате JSON со схемой {"answer": string, "summary": string}' \
  "Summarize the provided text"
```

When no system prompt is set, no `system` message is included in the request.

Exit codes:

- `0` — success
- `1` — the LLM call failed (after retries)
- `2` — invalid usage (no prompt provided)

## Tests

Run the mocked test suite (no live API required):

```bash
python -m pytest tests/ -v
```

Coverage includes: successful request, retry-then-success, rate limiting,
network errors, exhausted retries, permanent HTTP errors, unexpected response
shapes, config overrides, and the detail-listener events (request url/model/payload,
token usage, per-attempt reporting).

## Programmatic use

### Via an agent session

Assemble an agent from the stores and chat with it. The session keeps its own
history (persisted to `data/sessions/`), so multi-turn context is automatic:

```python
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

session = make_session(
    "my-session",                     # session id (resume with the same id)
    "translator",                     # agent name from data/agents.yaml
    model_store=YamlModelStore(),
    agent_store=YamlAgentStore(),
    session_store=JsonSessionStore(),
)

reply = session.chat("Good morning")  # returns only the text
print(reply)
print(session.history)                # [user, assistant, ...] full transcript
```

An `Agent` is a stateless role; many sessions can share it. History lives on the
session, so different topics or users get independent conversations.

### Direct low-level call (`LLMClient`)

For a single prompt without any agent/session state:

```python
from llm_bot.client import LLMClient, LLMError
from llm_bot.config import LLMConfig

config = LLMConfig.from_env()          # reads .env / environment
client = LLMClient(config)

try:
    answer = client.send_prompt("Hello!")
    print(answer)
except LLMError as exc:
    print("Failed:", exc)
```

To send a pre-built message stack (e.g. a conversation history assembled by the
caller), use [`LLMClient.chat(messages)`](llm_bot/client.py) instead of
`send_prompt`.

For GigaChat, use the factory which wires up the OAuth2 token provider for you:

```python
from llm_bot.config import LLMConfig
from llm_bot.gigachat import build_gigachat_client

config = LLMConfig.from_env()
client = build_gigachat_client(config)  # auto obtains + refreshes the token
print(client.send_prompt("Привет!"))
```

To capture request/response details programmatically, pass a `detail_listener`
that implements the `DetailListener` protocol (URL, model, payload on
`on_request`; status, token usage, timing, attempt on `on_response`):

```python
from llm_bot.client import LLMClient
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import RequestDetails, ResponseDetails

class Printer:
    def on_request(self, d: RequestDetails) -> None:
        print(d.method, d.url, d.model, d.payload)
    def on_response(self, d: ResponseDetails) -> None:
        print(d.status_code, d.usage, d.elapsed_ms, d.attempt)

client = LLMClient(LLMConfig.from_env(), detail_listener=Printer())
print(client.send_prompt("Hello"))
```

## Comparing prompt strategies

[`scripts/compare_methods.py`](scripts/compare_methods.py) runs one task through
**four prompting strategies** and prints a comparison table, so you can see how
much the approach affects the answer:

1. **Прямой ответ** — the raw task, no extra instructions.
2. **Решай пошагово** — the task plus a "solve step by step" instruction.
3. **Сначала промпт → потом решение** — first ask the model to draft a good
   prompt, then solve using that drafted prompt (two calls).
4. **Группа экспертов** — a panel of roles (analyst, engineer, critic); each
   expert solves the task **independently** with its role description sent as the
   system prompt and the task text as the user prompt (one call per role).

The script builds on `LLMClient`, so **provider and model come from your `.env`**
(`LLMConfig`), exactly like the rest of the project. No extra config needed.

```bash
# Run both built-in tasks with all 4 methods
.venv/bin/python scripts/compare_methods.py

# A single built-in task
.venv/bin/python scripts/compare_methods.py --task-index chickens
.venv/bin/python scripts/compare_methods.py --task-index fizzbuzz

# Your own task (+ expected answer to enable automatic scoring)
.venv/bin/python scripts/compare_methods.py \
  --task "What is 7*6+9?" --expected 51

# Use GigaChat (auth comes from GIGACHAT_* env vars)
.venv/bin/python scripts/compare_methods.py --provider gigachat

# Save the report as markdown or JSON
.venv/bin/python scripts/compare_methods.py --out results/comparison.md

# Run only some methods
.venv/bin/python scripts/compare_methods.py --methods direct,experts

# Keep each reply brief (soft word limit, same as the CLI's --max-response-words)
.venv/bin/python scripts/compare_methods.py --max-response-words 50

# Print per-request details (URL, model, payload, token usage) to stderr,
# same as the CLI's --details
.venv/bin/python scripts/compare_methods.py --details
```

The built-in logical tasks have a known **canonical answer**, so the script can
flag which methods got it right. Scoring is **language-agnostic**: when the
expected answer contains numbers, the script checks that the same numbers appear
in the reply (e.g. the English `"Chickens: 16, Cows: 6"` correctly matches the
Russian canonical answer `"6 коров, 16 кур"`). For expected answers without
numbers it falls back to a word-overlap check. The report shows every prompt and
response, a verdict column, and — for the multi-step methods — the intermediate
outputs.

## Comparing models by tier (weak / medium / strong)

[`scripts/compare_models.py`](scripts/compare_models.py) runs the **same request**
through several models on the same OpenAI-compatible endpoint (by default a weak,
a medium and a strong model) and reports, per model:

- **latency** — total wall-clock response time in milliseconds;
- **tokens** — `prompt` / `completion` / `total` from the provider's `usage`;
- **cost** — estimated USD price via a small price table (0 for free/local models);
- **quality** — automatic correctness score against an expected answer when one is
  known (same language-agnostic scoring as `compare_methods`), otherwise manual.

The **base URL / provider** come from your `.env` (`LLMConfig`) exactly like the
rest of the project; the models themselves are passed explicitly, since comparing
tiers means calling *different* models on the same endpoint.

```bash
# Run all built-in tasks through the default weak/medium/strong ladder on Groq
.venv/bin/python scripts/compare_models.py --out results/

# A single built-in task, markdown or JSON report
.venv/bin/python scripts/compare_models.py --task-index chickens --out results/compare_models_chickens.md
.venv/bin/python scripts/compare_models.py --task-index chickens --out results/compare_models_chickens.json

# Your own task (+ expected answer to enable automatic quality scoring)
.venv/bin/python scripts/compare_models.py --task "What is 7*6+9?" --expected 51

# Explicit model set
.venv/bin/python scripts/compare_models.py \
  --models openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b

# Prices for a paid provider (USD per 1M input/output tokens); repeatable
.venv/bin/python scripts/compare_models.py --price "gpt-4o=2.50,10.00"

# Use GigaChat (auth comes from GIGACHAT_* env vars)
.venv/bin/python scripts/compare_models.py --provider gigachat

# Per-request details (URL, payload, token usage) to stderr, like the CLI's --details
.venv/bin/python scripts/compare_models.py --details
```

The default model ladder is tailored to this project's Groq account
(`openai/gpt-oss-20b` → `openai/gpt-oss-120b` → `qwen/qwen3.8-27b`). If a model id
isn't available on your account you'll get a `404` — pass your own ids with
`--models`. An example report is in
[`results/compare_models_analysis.md`](results/compare_models_analysis.md).

## Adding a web interface

Because the request logic is isolated in `LLMClient`, adding a web UI is mostly
just adding a route that calls it. Below is a minimal [FastAPI](https://fastapi.tiangolo.com/)
example to illustrate how easy the integration is:

```python
# web.py (new file)
from fastapi import FastAPI
from pydantic import BaseModel

from llm_bot.client import LLMClient, LLMError
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import RequestDetails, ResponseDetails

app = FastAPI()
client = LLMClient(LLMConfig.from_env())


class PromptRequest(BaseModel):
    prompt: str


class RequestLog:
    """Reuse the same detail events the CLI uses — here we collect them."""

    def __init__(self) -> None:
        self.details: list[dict] = []

    def on_request(self, d: RequestDetails) -> None:
        self.details.append({"request": {"url": d.url, "model": d.model, "payload": d.payload}})

    def on_response(self, d: ResponseDetails) -> None:
        self.details.append({"response": {"status": d.status_code, "usage": d.usage}})


@app.post("/chat")
def chat(req: PromptRequest):
    log = RequestLog()
    client = LLMClient(LLMConfig.from_env(), detail_listener=log)
    try:
        reply = client.send_prompt(req.prompt)
        return {"reply": reply, "details": log.details}
    except LLMError as exc:
        return {"error": str(exc)}
```

Steps to go web-ready:

1. Add `fastapi` and `uvicorn` to `requirements.txt`.
2. Create `web.py` like the example above.
3. Run with `uvicorn web:app --reload`.

You can then call it with:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello"}'
```

No changes to `LLMClient` are required — the same instance, configuration, and
error handling are reused by both the console and the web layer. For heavier web
usage you may want to keep a single long-lived `LLMClient` (instead of creating
one per request) and, optionally, add an async variant using `httpx.AsyncClient`.