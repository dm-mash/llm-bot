#!/usr/bin/env python3
"""Compare context retention and token cost with vs without history compression.

Runs a long dialog against a **fake, deterministic LLM** (via
``httpx.MockTransport``), so it works offline and produces stable numbers:

* each turn plants several distinct *facts* into the conversation;
* the final turn asks the model to recall every fact it can see in its context;
* the mock answers the recall question by echoing exactly the facts present in
  the request ``messages`` — a faithful model of "what the model can see".

The same dialog is run twice:

1. **plain**   — no compression: every turn resends the full history.
2. **compressed** — rolling-summary compression enabled (keep-last / block).
   Old turns are folded into a running summary; the summarizer is simulated to
   retain a configurable fraction (``--retention``) of the facts, modelling the
   real-world information loss that compression can introduce.

For each run we report:

* the **recall quality** = fraction of all planted facts the model can still see
  in the final turn's context;
* the **token spend** = total prompt tokens across main turns, plus the extra
  tokens spent on the summarization calls themselves (the "cost of compressing"),
  so the reader sees whether compression actually saves tokens net of its own
  overhead.

Examples:
    python scripts/compare_compression.py
    python scripts/compare_compression.py --turns 12 --retention 0.7
    python scripts/compare_compression.py --out results/compression_analysis.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from llm_bot.compress import CompressionSettings, summarize_prompt
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig


# --------------------------------------------------------------------------- #
# Fact encoding / extraction (the fake model's "world").
# --------------------------------------------------------------------------- #

_FACT_RE = re.compile(r"ФАКТ (\d+):\s*([^\n]+)")

# Descriptions for the planted facts (cycled if there are more facts than these).
_FACT_TEXT = [
    "имя клиента Алиса",
    "бюджет проекта 5000 долларов",
    "срок сдачи 15 марта",
    "команда из трёх человек",
    "предпочтение — лаконичные ответы",
    "язык общения — русский",
    "дедлайн по контракту — июнь",
    "рабочая среда — Linux",
]


def plant_fact(fact_id: int) -> str:
    """Return a user message that plants a single fact with id *fact_id*."""
    description = _FACT_TEXT[fact_id % len(_FACT_TEXT)]
    return f"Запомни факт: ФАКТ {fact_id}: {description}"


def facts_in_text(text: str) -> dict[int, str]:
    """Extract ``{id: description}`` from any text containing ``ФАКТ n: ...`` lines."""
    return {int(i): desc for i, desc in _FACT_RE.findall(text)}


def facts_in_messages(messages: list[dict[str, str]]) -> dict[int, str]:
    """Union of all facts mentioned across a message stack (context + prompts)."""
    merged: dict[int, str] = {}
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            merged.update(facts_in_text(content))
    return merged


def _is_summarization_request(messages: list[dict[str, str]]) -> bool:
    """Detect the internal summarization call by its prompt marker."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith(
                summarize_prompt("", "").splitlines()[0][:30]
            ):
                return True
    return False


def _is_recall_question(messages: list[dict[str, str]]) -> bool:
    last_user = ""
    for msg in messages:
        if msg.get("role") == "user":
            last_user = msg.get("content", "") or ""
    low = last_user.lower()
    return "перечисли" in low or "все факты" in low


def _retain_facts(facts: dict[int, str], retention: float) -> dict[int, str]:
    """Keep a deterministic subset of *facts* to model summarization loss.

    Retention is applied pseudo-randomly but deterministically (hash of the fact
    id), so the script is reproducible: roughly ``retention`` of the facts are
    kept, always preserving the *lowest* ids first (older = safer to keep).
    """
    ordered = sorted(facts.items())  # lowest ids first
    keep_count = max(0, int(round(len(ordered) * retention)))
    # Always keep the oldest ``keep_count`` facts.
    return dict(ordered[:keep_count])


def build_handler(retention: float):
    """Return an httpx handler simulating the LLM for the comparison dialog.

    * Summarization calls return a summary retaining ``retention`` of the facts
      visible in the prompt (existing summary + new block).
    * The final recall question echoes every fact visible in the request context.
    * Any other turn just acknowledges.

    The returned handler exposes a ``counters`` dict with ``summary_requests``
    and ``summary_prompt_tokens`` so the script can bill the extra cost of
    compression separately from the main conversation.
    """
    from llm_bot.tokens import count_messages_tokens

    counters = {"summary_requests": 0, "summary_prompt_tokens": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        messages = payload.get("messages", [])

        if _is_summarization_request(messages):
            counters["summary_requests"] += 1
            counters["summary_prompt_tokens"] += count_messages_tokens(messages)
            prompt = messages[-1].get("content", "")
            visible = facts_in_text(prompt)
            retained = _retain_facts(visible, retention)
            summary = "\n".join(
                f"ФАКТ {k}: {v}" for k, v in sorted(retained.items())
            ) or "(пусто)"
            return _reply(summary, messages)

        if _is_recall_question(messages):
            visible = facts_in_messages(messages)
            answer = "\n".join(
                f"ФАКТ {k}: {v}" for k, v in sorted(visible.items())
            ) or "(фактов не видно)"
            return _reply(answer, messages)

        return _reply("принято", messages)

    handler.counters = counters  # type: ignore[attr-defined]
    return handler


def _reply(content: str, messages: list[dict[str, str]]) -> httpx.Response:
    """Build a chat-completions response with deterministic token usage."""
    from llm_bot.tokens import count_message_tokens, count_messages_tokens

    prompt_tokens = count_messages_tokens(messages)
    completion_tokens = count_message_tokens(
        {"role": "assistant", "content": content}
    )
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


# --------------------------------------------------------------------------- #
# Running one dialog.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class DialogResult:
    mode: str                      # "plain" | "compressed"
    turns: int
    facts_per_turn: int
    total_facts: int
    recalled: int                  # facts visible in the final recall context
    recall_quality: float          # recalled / total_facts
    main_context_tokens: int       # sum of prompt tokens over main turns
    summary_context_tokens: int    # sum of prompt tokens over summarization calls
    total_tokens: int
    summary_requests: int
    settings: CompressionSettings | None = None


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="demo",
        system_prompt="Ты помощник.",
        temperature=0.0,
        max_tokens=128,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="demo",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt-demo",
        context_window=1_000_000,
    )


def _run_dialog(
    *,
    turns: int,
    facts_per_turn: int,
    retention: float,
    compression: CompressionSettings | None,
    directory: str,
    mode: str,
) -> DialogResult:
    handler = build_handler(retention)
    transport = httpx.MockTransport(handler)

    class _AgentStore:
        def get(self, name):
            return _agent_config()

        def list(self):
            return [_agent_config().name]

    class _ModelStore:
        def get(self, name):
            return _model_config()

        def list(self):
            return [_model_config().name]

    session_store = JsonSessionStore(directory)
    session = make_session(
        f"cmp-{mode}",
        "assistant",
        model_store=_ModelStore(),
        agent_store=_AgentStore(),
        session_store=session_store,
        transport=transport,
        compression=compression,
    )

    fact_id = 0
    main_tokens = 0
    for _ in range(turns):
        for _ in range(facts_per_turn):
            session.chat_with_details(plant_fact(fact_id))
            fact_id += 1
            main_tokens += session.last_usage.context_tokens
        # A short filler turn keeps the dialog reading naturally between facts.
        session.chat_with_details("Продолжай.")
        main_tokens += session.last_usage.context_tokens

    # The final recall question.
    result = session.chat_with_details(
        "Вопрос: перечисли все факты из разговора."
    )
    main_tokens += result.usage.context_tokens

    # Count facts the model could see in the final request context.
    recall_answer = result.reply
    recalled = len(facts_in_text(recall_answer))
    summary_tokens = handler.counters["summary_prompt_tokens"]

    return DialogResult(
        mode=mode,
        turns=turns,
        facts_per_turn=facts_per_turn,
        total_facts=fact_id,
        recalled=recalled,
        recall_quality=recalled / max(fact_id, 1),
        main_context_tokens=main_tokens,
        summary_context_tokens=summary_tokens,
        total_tokens=main_tokens + summary_tokens,
        summary_requests=handler.counters["summary_requests"],
        settings=compression,
    )


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #

def render_text(plain: DialogResult, compressed: DialogResult) -> str:
    lines = [
        f"Диалог: {plain.turns} реплик × {plain.facts_per_turn} факта "
        f"(всего {plain.total_facts} фактов)",
        f"Параметры сжатия: keep-last={compressed.settings.keep_last}, "
        f"block={compressed.settings.block_size}, "
        f"retention={_retention_label()}",
        "",
        f"{'Режим':<12} {'Recall':<8} {'Качество':<10} {'Промпт-токены':<16} "
        f"{'Суммаризация':<14} {'Всего':<10}",
        f"{'-'*12} {'-'*8} {'-'*10} {'-'*16} {'-'*14} {'-'*10}",
    ]
    for r in (plain, compressed):
        lines.append(
            f"{r.mode:<12} {r.recalled:<8}/{r.total_facts} "
            f"{r.recall_quality * 100:>5.1f}% "
            f"{r.main_context_tokens:<16} "
            f"{r.summary_context_tokens:<14} "
            f"{r.main_context_tokens + r.summary_context_tokens:<10}"
        )
    savings = _percent_saved(plain, compressed)
    lines.extend(
        [
            "",
            f"Экономия токенов (промпт): {savings:.1f}%",
            f"Сводных запросов на суммаризацию: {compressed.summary_requests}",
            "",
            "Примечание: 'Recall' — сколько фактов модель могла увидеть в контексте",
            "последнего запроса. При retention<1 сжатие теряет часть старых фактов.",
        ]
    )
    return "\n".join(lines)


_RETENTION: float = 1.0


def _retention_label() -> str:
    return f"{_RETENTION:.2f}"


def _percent_saved(plain: DialogResult, compressed: DialogResult) -> float:
    plain_total = plain.main_context_tokens + plain.summary_context_tokens
    comp_total = compressed.main_context_tokens + compressed.summary_context_tokens
    if plain_total <= 0:
        return 0.0
    return (1.0 - comp_total / plain_total) * 100.0


def render_markdown(plain: DialogResult, compressed: DialogResult) -> str:
    saved = _percent_saved(plain, compressed)
    c = compressed
    return (
        "# Сравнение: без сжатия vs со сжатием истории\n\n"
        f"- Реплик в диалоге: **{plain.turns}**, фактов на реплику: "
        f"**{plain.facts_per_turn}**, всего фактов: **{plain.total_facts}**\n"
        f"- Сжатие: keep-last={c.settings.keep_last}, "
        f"block={c.settings.block_size}, retention={_retention_label()}\n\n"
        "| Режим | Recall | Качество | Промпт-токены | Токены суммаризации | Всего |\n"
        "|---|---|---:|---:|---:|---:|\n"
        f"| Без сжатия | {plain.recalled}/{plain.total_facts} | "
        f"{plain.recall_quality * 100:.1f}% | {plain.main_context_tokens} | 0 | "
        f"{plain.main_context_tokens} |\n"
        f"| Со сжатием | {c.recalled}/{c.total_facts} | "
        f"{c.recall_quality * 100:.1f}% | {c.main_context_tokens} | "
        f"{c.summary_context_tokens} | "
        f"{c.main_context_tokens + c.summary_context_tokens} |\n\n"
        f"**Экономия промпт-токенов:** {saved:.1f}%  "
        f"(суммарных вызовов суммаризации: {c.summary_requests})\n"
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=12,
                        help="Number of conversation turns (default: 12).")
    parser.add_argument("--facts-per-turn", type=int, default=2,
                        help="Facts planted per turn (default: 2).")
    parser.add_argument("--keep-last", type=int, default=6,
                        help="Compression keep-last N (default: 6).")
    parser.add_argument("--block", type=int, default=14,
                        help="Compression block size M (default: 14).")
    parser.add_argument("--retention", type=float, default=1.0,
                        help="Fraction of facts the simulated summarizer retains "
                             "(0..1, default: 1.0 = lossless).")
    parser.add_argument("--out", default=None,
                        help="Optional .md file to write the markdown report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    global _RETENTION
    args = build_parser().parse_args(argv)
    _RETENTION = args.retention

    settings = CompressionSettings(keep_last=args.keep_last, block_size=args.block)
    directory = tempfile.mkdtemp(prefix="cmp-compress-")

    print(f"Диалог: {args.turns} реплик, {args.facts_per_turn} факта на реплику, "
          f"retention={args.retention:.2f}")
    print(f"Сжатие: keep-last={settings.keep_last}, block={settings.block_size}\n")

    plain = _run_dialog(
        turns=args.turns,
        facts_per_turn=args.facts_per_turn,
        retention=args.retention,
        compression=None,
        directory=directory,
        mode="plain",
    )
    compressed = _run_dialog(
        turns=args.turns,
        facts_per_turn=args.facts_per_turn,
        retention=args.retention,
        compression=settings,
        directory=directory,
        mode="compressed",
    )

    print(render_text(plain, compressed))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_markdown(plain, compressed), encoding="utf-8")
        print(f"\nСохранено: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())