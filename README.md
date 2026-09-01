# llm-bot

A provider-agnostic Python client for OpenAI-compatible LLM APIs. It sends a
prompt to the API, receives the model's response, and prints it to the console.

The core request logic lives in a reusable service layer
([`LLMClient`](llm_bot/client.py)) that is deliberately kept separate from the
console UI ([`cli.py`](llm_bot/cli.py)). This makes it trivial to add a **web
interface** later without touching the request logic (see
[Adding a web interface](#adding-a-web-interface)).

## Features

- Provider-agnostic — works with OpenAI, Ollama, LM Studio, LocalAI, and any
  other server exposing the `/v1/chat/completions` endpoint.
- Configuration via environment variables / `.env` (no hard-coded keys).
- Automatic retries with exponential backoff for transient failures
  (timeouts, network errors, HTTP 429 / 5xx).
- Clear error handling via custom exception types.
- Console entry point accepting a prompt as an argument or via stdin.
- Built-in GigaChat support (`--provider gigachat`) with automatic OAuth2 token
  acquisition and refresh.
- Pytest suite using `httpx.MockTransport` (no network needed).

## Project structure

```
llm-bot/
├── llm_bot/
│   ├── __init__.py      # package exports
│   ├── config.py        # LLMConfig — settings from env vars
│   ├── client.py        # LLMClient — request logic + retries + errors
│   ├── gigachat.py      # GigaChat OAuth2 token provider + factory
│   ├── cli.py           # console entry point
│   └── __main__.py      # enables `python -m llm_bot`
├── tests/
│   ├── test_client.py   # pytest tests (mocked transport)
│   └── test_gigachat.py # GigaChat token provider tests
├── requirements.txt
├── .env.example
└── README.md
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

## Usage

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
shapes, and config overrides.

## Programmatic use

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

For GigaChat, use the factory which wires up the OAuth2 token provider for you:

```python
from llm_bot.config import LLMConfig
from llm_bot.gigachat import build_gigachat_client

config = LLMConfig.from_env()
client = build_gigachat_client(config)  # auto obtains + refreshes the token
print(client.send_prompt("Привет!"))
```

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

app = FastAPI()
client = LLMClient(LLMConfig.from_env())


class PromptRequest(BaseModel):
    prompt: str


@app.post("/chat")
def chat(req: PromptRequest):
    try:
        return {"reply": client.send_prompt(req.prompt)}
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