#!/usr/bin/env python3
"""Compare context-management strategies on one shared "ТЗ" scenario.

Runs the exact same dialog ("собираем ТЗ", ~12-15 messages) against a **fake,
deterministic LLM** (via ``httpx.MockTransport``), so it works offline and
produces stable, reproducible numbers.

Each turn plants one requirement of the TOR into the conversation; the final turn
asks the model to list every requirement. The fake model answers by echoing
exactly the requirements it can see in the request it received — a faithful model
of "what the strategy actually sent", i.e. what the real model could answer from.

Strategies compared (all share one code path through ``Session``):

* **full**      — the whole history is sent every turn (baseline).
* **summary**   — rolling-summary compression (keep-last / block).
* **sliding**   — only the last ``N`` messages are sent.
* **facts**     — a durable key/value facts block + the last ``N`` messages.
* **branching** — the dialog continues in one branch (full history, no loss);
                  its real value is exploratory UX, not token savings.

For each we report:

* **recall** — fraction of all planted requirements the model could still see in
  the final request's context (answer quality + stability);
* **prompt tokens** — sum of the main-turn request contexts;
* **overhead tokens** — extra LLM tokens spent keeping the strategy state fresh
  (facts refresh / summarization calls);
* **total tokens** — prompt + overhead.

Examples:
    python scripts/compare_context_strategies.py
    python scripts/compare_context_strategies.py --turns 12 --window 6
    python scripts/compare_context_strategies.py --out results/context_strategies_analysis.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from llm_bot.compress import CompressionSettings, summarize_prompt
from llm_bot.context_strategies import Branching, SlidingWindow, StickyFacts
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig


# --------------------------------------------------------------------------- #
# The shared "ТЗ" scenario
# --------------------------------------------------------------------------- #

_REQ_RE = re.compile(r"ТРЕБОВАНИЕ (\d+):\s*(.+)")

_REQ_DESC = [
    "имя проекта «Алиса»",
    "цель — автоматизировать приём заявок",
    "бюджет 5000 долларов",
    "срок сдачи 15 марта",
    "команда из трёх человек",
    "стек — Python и PostgreSQL",
    "ограничение — только внутренние сервисы",
    "требуется веб-интерфейс",
    "доступ по ролям",
    "логирование всех операций",
    "уведомления по почте",
    "экспорт отчётов в CSV",
]


def plant_requirement(fact_id: int) -> str:
    """Return a user message planting requirement *fact_id*."""
    desc = _REQ_DESC[fact_id % len(_REQ_DESC)]
    return f"Добавь требование к ТЗ: ТРЕБОВАНИЕ {fact_id}: {desc}"


def request_text(messages: list[dict[str, str]]) -> str:
    """Concatenate all message content (the context the model can see)."""
    return "\n".join(
        msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
    )


def visible_descs(messages: list[dict[str, str]], all_descs: set[str]) -> set[str]:
    """Which requirement descriptions are present anywhere in the request context.

    Substring matching is a faithful, simple model of "the model knows a fact when
    its content is present in context" — it works whether the requirement arrives
    verbatim in the history, in a summary, or in a rendered sticky-facts block.
    """
    text = request_text(messages)
    return {d for d in all_descs if d in text}


def _is_recall_question(messages: list[dict[str, str]]) -> bool:
    last_user = ""
    for msg in messages:
        if msg.get("role") == "user":
            last_user = msg.get("content", "") or ""
    low = last_user.lower()
    return "перечисли все требования" in low


def _is_summarization_request(messages: list[dict[str, str]]) -> bool:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith(
                summarize_prompt("", "").splitlines()[0][:30]
            ):
                return True
    return False


# --------------------------------------------------------------------------- #
# Fake LLM handler for main-turn requests (transport-level).
# --------------------------------------------------------------------------- #


def build_handler(all_descs: set[str]):
    """Return an httpx handler that echoes requirements visible in the request.

    * Recall question -> list every planted requirement whose description text
      appears somewhere in the request.
    * Summarization prompt -> produce a summary retaining all visible requirements.
    * Anything else -> short acknowledgement.
    """
    from llm_bot.tokens import count_message_tokens, count_messages_tokens

    counters = {"summary_prompt_tokens": 0}

    def _echo(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        messages = payload.get("messages", [])
        visible = visible_descs(messages, all_descs)
        prompt_tokens = count_messages_tokens(messages)
        if _is_recall_question(messages):
            body = "\n".join(f"ТРЕБОВАНИЕ: {d}" for d in sorted(visible)) or (
                "(требований не видно)"
            )
        elif _is_summarization_request(messages):
            counters["summary_prompt_tokens"] += prompt_tokens
            body = "\n".join(f"ТРЕБОВАНИЕ: {d}" for d in sorted(visible)) or "(пусто)"
        else:
            body = "принято"
        completion_tokens = count_message_tokens(
            {"role": "assistant", "content": body}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": body}}
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    _echo._counters = counters  # type: ignore[attr-defined]
    return _echo


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class StrategyResult:
    mode: str
    turns: int
    total_reqs: int
    recalled: int
    recall_quality: float
    main_tokens: int
    overhead_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.main_tokens + self.overhead_tokens


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="demo",
        system_prompt="Ты помогаешь собрать ТЗ.",
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


def _facts_chat(accumulated: dict[int, str]):
    """chat callable for StickyFacts: extracts requirements from a turn into facts."""

    def chat(request: list[dict[str, str]]) -> str:
        prompt = request[-1].get("content", "")
        for req_id, desc in _REQ_RE.findall(prompt):
            accumulated[int(req_id)] = desc.strip()
        return "\n".join(f"req{i}: {d}" for i, d in sorted(accumulated.items()))

    return chat


def _run_dialog(
    *,
    turns: int,
    window: int,
    mode: str,
    directory: str,
) -> StrategyResult:
    all_descs = {_REQ_DESC[i % len(_REQ_DESC)] for i in range(turns)}
    handler = build_handler(all_descs)
    transport = httpx.MockTransport(handler)
    session_store = JsonSessionStore(directory)

    common = dict(
        model_store=_ModelStore(),
        agent_store=_AgentStore(),
        session_store=session_store,
        transport=transport,
    )

    strategy = None
    compression = None
    facts_accum = {}
    if mode == "sliding":
        strategy = SlidingWindow(window)
    elif mode == "facts":
        strategy = StickyFacts(window, max_facts=50, chat=_facts_chat(facts_accum))
    elif mode == "branching":
        strategy = Branching(window_size=None)
    elif mode == "summary":
        compression = CompressionSettings(keep_last=window, block_size=window * 2)

    session = make_session(
        f"cmp-{mode}",
        "assistant",
        strategy=strategy,
        compression=compression,
        **common,
    )

    main_tokens = 0
    for i in range(turns):
        result = session.chat_with_details(plant_requirement(i))
        main_tokens += result.usage.context_tokens

    recall = session.chat_with_details("Вопрос: перечисли все требования из разговора.")
    main_tokens += recall.usage.context_tokens

    # The fake model echoed what it could see; count those we actually planted.
    recalled = len(visible_descs([{"role": "assistant", "content": recall.reply}], all_descs))
    overhead = 0
    if mode == "facts":
        overhead = session.strategy.extra_tokens
    elif mode == "summary":
        overhead = handler._counters["summary_prompt_tokens"]

    return StrategyResult(
        mode=mode,
        turns=turns,
        total_reqs=turns,
        recalled=recalled,
        recall_quality=recalled / max(turns, 1),
        main_tokens=main_tokens,
        overhead_tokens=overhead,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_text(results: list[StrategyResult], window: int, turns: int) -> str:
    lines = [
        f"Сценарий: собираем ТЗ ({turns} реплик, по одному требованию на реплику).",
        f"Окно (для sliding/facts/summary): {window} сообщений.",
        "",
        f"{'Стратегия':<10} {'Recall':<10} {'Качество':<9} {'Промпт':<8} "
        f"{'Overhead':<9} {'Всего':<8}",
        f"{'-'*10} {'-'*10} {'-'*9} {'-'*8} {'-'*9} {'-'*8}",
    ]
    for r in results:
        lines.append(
            f"{r.mode:<10} {r.recalled:<6}/{r.total_reqs} "
            f"{r.recall_quality * 100:>5.1f}% "
            f"{r.main_tokens:<8} {r.overhead_tokens:<9} {r.total_tokens:<8}"
        )
    return "\n".join(lines)


def render_markdown(results: list[StrategyResult], window: int, turns: int) -> str:
    rows = []
    for r in results:
        rows.append(
            f"| {r.mode} | {r.recalled}/{r.total_reqs} | {r.recall_quality * 100:.1f}% | "
            f"{r.main_tokens} | {r.overhead_tokens} | {r.total_tokens} |"
        )
    return (
        "# Сравнение стратегий управления контекстом (без summary)\n\n"
        f"- Сценарий: **собираем ТЗ**, {turns} реплик, по одному требованию на "
        f"реплику (всего требований: {turns}).\n"
        f"- Окно для sliding/facts/summary: **{window} сообщений**.\n"
        "- Фейковая детерминированная LLM: на финальный вопрос «перечисли все "
        "требования» отвечает только тем, что реально видит в контексте запроса.\n\n"
        "| Стратегия | Recall | Качество | Промпт-токены | Overhead | Всего |\n"
        "|---|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        "### Заметки\n\n"
        "- **full** — эталон качества (всё видно), но самый дорогой и растёт без "
        "ограничений.\n"
        "- **summary** — дешевле full, но суммаризация может терять часть деталей "
        "и сама стоит токенов (overhead).\n"
        "- **sliding** — очень дёшево и предсказуемо, НО при малом окне модель "
        "перестаёт видеть ранние требования (низкий recall) — данные не теряются "
        "на диске, но в ответе их нет.\n"
        "- **facts** — durable-память сохраняет требования даже при малом окне "
        "(высокий recall), цена — дополнительный LLM-вызов на обновление фактов "
        "(overhead).\n"
        "- **branching** — шлёт полную историю ветки (recall 100%), поэтому по "
        "токенам как full; его ценность — возможность исследовать альтернативные "
        "ветки ТЗ без потери исходного пути, а не экономия токенов.\n"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=12,
                        help="Number of scenario turns (default: 12).")
    parser.add_argument("--window", type=int, default=6,
                        help="Sliding-window size in messages (default: 6).")
    parser.add_argument("--out", default=None,
                        help="Optional .md file to write the markdown report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = tempfile.mkdtemp(prefix="cmp-strategies-")
    modes = ["full", "summary", "sliding", "facts", "branching"]

    print(f"Сценарий: собираем ТЗ ({args.turns} реплик), окно={args.window} сообщ.\n")
    results = [
        _run_dialog(turns=args.turns, window=args.window, mode=m, directory=directory)
        for m in modes
    ]

    print(render_text(results, args.window, args.turns))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            render_markdown(results, args.window, args.turns), encoding="utf-8"
        )
        print(f"\nСохранено: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())