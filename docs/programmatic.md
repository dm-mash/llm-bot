# Programmatic use

## Via an agent session

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

## Direct low-level call (`LLMClient`)

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
caller), use [`LLMClient.chat(messages)`](../llm_bot/client.py) instead of
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
