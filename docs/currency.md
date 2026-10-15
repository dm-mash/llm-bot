# Currency pipeline: the agent composes the tool chain

Rate tools over daily exchange rates (USD/EUR/GBP → RUB, Yahoo Finance,
keyless) where **the agent builds the chain itself** from a single prompt.
There is no step numbering — every tool's first description line states its
data contract (what it consumes, what it returns, where to pass the
result), so different tasks produce different chains: «проанализируй курс
доллара за 2025 год и сохрани отчёт» becomes `fetch_rates` →
`analyze_rates` → `save_report` (one directive per round, the previous
result arriving as a service turn and passed on verbatim), while «получи
курс и выгрузи в файл» becomes just `fetch_rates` → `save_series`
(verification report:
[`results/mcp_currency_yahoo_verification.md`](../results/mcp_currency_yahoo_verification.md)).

- [`llm_bot/api/currency_yahoo.py`](../llm_bot/api/currency_yahoo.py) — the
  backend: `fetch_rates` (offline `CURRENCY_FIXTURE` mode included),
  `analyze_rates` (period, min/max, mean, change, trend), idempotent
  `save_report` (atomic writes, content-hash dedup) and `save_series` (raw
  dump to a JSON file named by currency+period); the tools accept the
  payload as a JSON string or an already-parsed object.
- [`llm_bot/mcp_servers/currency_yahoo.py`](../llm_bot/mcp_servers/currency_yahoo.py) —
  FastMCP stdio server (`python -m llm_bot.mcp_servers.currency_yahoo`);
  each tool's **data contract lives in the first docstring line** — the
  router advertises only that line to the model.
- [`llm_bot/mcp_tools.py`](../llm_bot/mcp_tools.py) — `render_tools_block`
  instructs the model to chain calls one at a time and to forward the
  previous tool's result unchanged.

```bash
cp mcp.example.yaml data/mcp.yaml    # adds the "currency_yahoo" server entry
python -m llm_bot --agent assistant --mcp currency_yahoo \
    "проанализируй курс доллара за 2025 год и сохрани отчёт"
python -m llm_bot --agent assistant --mcp currency_yahoo \
    "получи курс евро за 2025 год и выгрузи ряд в файл"   # no analyze_rates
python scripts/mcp_currency_yahoo_demo.py  # live demo: deterministic chain + one-prompt agent run
```

The live LLM leg needs a provider quota that fits two rounds of a
~24-point series (Groq's free 8000 TPM returns 429) and a model that
follows text protocols — models tuned for native tool-calling (e.g. the
`liquid` agent's) reply in Hermes `tool_call` syntax that the plain-chat
protocol cannot parse. The chain mechanics are proven by scripted-LLM
tests in [`tests/test_mcp_currency_yahoo_chain.py`](../tests/test_mcp_currency_yahoo_chain.py).
