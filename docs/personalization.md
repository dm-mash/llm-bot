# Profiles (personalization)

Profiles answer "HOW should the agent behave for THIS user/task" — the third
config layer. The separation of concerns is strict:

| Layer | File | Answers |
|---|---|---|
| Transport | `data/models.yaml` | which API and how to call it |
| Role | `data/agents.yaml` | who the agent is (role, defaults) |
| **Profile** | `data/profiles.yaml` | how to answer THIS user (style, format, constraints) |
| Memory | `data/memory/` | what was learned from the dialog (facts, decisions) |

A profile is **orchestration config, not memory**: it is declared in YAML like
agents and models, applied once at session creation by
[`apply_profile`](../llm_bot/profiles.py) (system-prompt directives +
`max_response_words`/`temperature` overrides), and never written to or read
from the memory layers.

Memory auto-extraction is deliberately conservative: the classifier may store
only what the **user** asserted, never the assistant's own conclusions and never
the contents of retrieved documents. This matters most when RAG is on. A
grounded answer can still be wrong, and an unverified claim that reaches
`long_term` becomes durable — it outranks the RAG block in the message stack
(memory is injected before it) and survives the session, so one confabulation
would contradict the documents on every later turn, including the next session.

Create the file from the template:

```bash
cp profiles.example.yaml data/profiles.yaml
```

Then personalize a chat (the launch command you already use, plus one flag):

```bash
python -m llm_bot --agent assistant --profile developer   # terse, technical, code-first
python -m llm_bot --agent assistant --profile student     # step-by-step, friendly
python -m llm_bot --agent assistant                       # unchanged, no profile
```

Inspect profiles without chatting:

```bash
python -m llm_bot --list-profiles            # all profiles with settings
python -m llm_bot --show-profile developer   # one profile + its generated prompt block
```

Unknown profile names fail fast with the list of available ones — the same
behaviour as unknown agents.

Verify that different profiles give observably different answers to the SAME
question (writes `results/compare_profiles.md` + `.json`):

```bash
python scripts/compare_profiles.py --profiles developer,student,child
python scripts/compare_profiles.py --include-none   # add a no-profile baseline row
```

Programmatic composition (no CLI needed):

```python
from llm_bot.profiles import apply_profile
from llm_bot.stores import AgentConfig, ProfileConfig
from llm_bot.yaml_stores import YamlProfileStore

agent = AgentConfig(name="assistant", model="gpt4o",
                    system_prompt="Ты полезный помощник.")
profile = YamlProfileStore().get("developer")
personalized = apply_profile(agent, profile)
```
