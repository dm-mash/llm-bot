# Benchmarks

## Comparing prompt strategies

[`scripts/compare_methods.py`](../scripts/compare_methods.py) runs one task through
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

[`scripts/compare_models.py`](../scripts/compare_models.py) runs the **same request**
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
[`results/compare_models_analysis.md`](../results/compare_models_analysis.md).
