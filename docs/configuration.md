# Configuration and data files

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

Besides `base_url`/`api_key`/`model`, a profile may set `context_window`
(input budget for overflow checks) and `timeout` (per-request seconds, falling
back to `LLM_TIMEOUT`). A local model partly offloaded to CPU needs the larger
timeout: 30 s is fine for an API, not for a 7B.

Caveat for Ollama: the OpenAI-compatible `/v1/chat/completions` endpoint only
honours `temperature`, `max_tokens`, `top_p`, `seed`, `stop` and the penalties.
It silently **drops** generation options such as `repeat_penalty` and `num_ctx`
(and ignores a top-level `num_ctx`); those work only through the native
`/api/chat` or a `Modelfile`. So `context_window` here governs *our* overflow
check, not the server's `num_ctx` — the server keeps its own default.

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
