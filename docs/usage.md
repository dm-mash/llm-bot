# Usage

## Interactive chat with an agent

Start an interactive chat with the `assistant` agent (history is saved to
`data/sessions/` and resumes with the same `--session` id):

```bash
python -m llm_bot --agent assistant
python -m llm_bot --agent assistant --session my-conversation   # resume
```

Inside the interactive chat, type `/history` (or `/история`) to print all
previous messages of the current session; `/help <вопрос>` (or `/помощь`) asks
the assistant about the project itself, using whatever project context the
session was launched with (see [the developer assistant](help.md)); `exit`/`quit`/Ctrl-D
leave the chat.

When you resume an existing session (with `--session`), the CLI prints a
`[session]` line to stderr with how much context is loaded: the number of
messages in the history and, depending on the active context strategy, the
sliding-window size (`окно`), the sticky-fact count (`фактов`), or the list of
dialogue branches with the current one marked (`ветки [<current>]: ...`). A
rolling-compression summary is reported as `summary: N симв.`.

A single one-shot turn via an agent:

```bash
python -m llm_bot --agent translator "Good morning"
```

A personalized turn — same agent, different behaviour via a profile
(see [profiles](personalization.md)):

```bash
python -m llm_bot --agent assistant --profile student "Объясни рекурсию"
python -m llm_bot --agent assistant --profile developer --session dev1   # resumes a session too
```

## Legacy: single direct call

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

## Inspecting request/response details

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

## Enforcing a specific JSON response format

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
