#!/usr/bin/env python3
"""Compare weak / medium / strong models on the SAME request.

Runs one (or several) tasks through a list of models — by default a "weak", a
"medium" and a "strong" tier on the same OpenAI-compatible endpoint — and reports,
per model:

  * latency       - wall-clock response time (ms) for the whole call
  * tokens        - prompt / completion / total tokens, when the provider reports them
  * cost          - estimated price in USD, using a small built-in price table
                    (per 1M input / 1M output tokens). Free/local models cost 0.
  * quality       - automatic correctness score against an expected answer when one
                    is known (same language-agnostic scoring as compare_methods),
                    otherwise left for manual review.

It uses the project's service layer (``llm_bot.client.LLMClient``). The **provider /
base URL** come from the environment / ``.env`` (see ``LLMConfig``), so any
OpenAI-compatible endpoint works (OpenAI, Groq, Ollama, LM Studio, LocalAI,
GigaChat, ...). The models themselves are passed explicitly, since comparing tiers
means calling *different* models on the same endpoint.

Default model set (a weak/medium/strong ladder available on this project's Groq
account):
    openai/gpt-oss-20b     (weak)
    openai/gpt-oss-120b    (medium)
    qwen/qwen3.8-27b       (strong)

Examples:
    python scripts/compare_models.py
    python scripts/compare_models.py --task-index chickens
    python scripts/compare_models.py \
        --models openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b
    python scripts/compare_models.py --task "What is 7*6+9?" --expected 51
    python scripts/compare_models.py --provider gigachat
    python scripts/compare_models.py --out results/models.md
    python scripts/compare_models.py --out results/models.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/compare_models.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMClient, LLMError
from llm_bot.cli import _DetailPrinter
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import DetailListener, RequestDetails, ResponseDetails
from llm_bot.gigachat import build_gigachat_client


# --------------------------------------------------------------------------- #
# Built-in tasks (same style as scripts/compare_methods.py).
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    title: str
    text: str
    expected_answer: str | None = None  # used for objective scoring when known


BUILTIN_TASKS: list[Task] = [
    Task(
        id="chickens",
        title="Куры и коровы",
        text=(
            "На ферме есть куры и коровы. Всего 22 головы и 56 ног. "
            "Сколько кур и сколько коров на ферме?"
        ),
        expected_answer="6 коров, 16 кур",
    ),
    Task(
        id="fizzbuzz",
        title="FizzBuzz до 30",
        text=(
            "Выведи результат классической задачи FizzBuzz для чисел от 1 до 30 "
            "включительно: если число делится на 3 — 'Fizz', на 5 — 'Buzz', "
            "на 15 — 'FizzBuzz', иначе само число."
        ),
        expected_answer="FizzBuzz",
    ),
    Task(
        id="story",
        title="Короткая история",
        text=(
            "Напиши короткую историю (3–5 предложений) про потерявшегося "
            "робота, который однажды утром нашёл на крыше городскую голубку."
        ),
        expected_answer=None,
    ),
    Task(
        id="logic",
        title="Логическая задача",
        text=(
            "Реши логическую задачу: На улице стоят пять складов."
            "Каменщик работает в зеленом складе."
            "У портного есть леопард."
            "На желтом складе едят котлету."
            "Электрик ест хлеб."
            "Желтый склад стоит сразу справа от синего склада."
            "Тот, кто пьет коньяк разводит пуму."
            "В белом складе пьют вино."
            "В центральном складе едят пельмени."
            "Плотник работает на первом складе."
            "Сосед того, кто пьет водку, держит медведя."
            "На складе по соседству с тем, в котором держат тигра, пьют вино."
            "Тот, кто пьет виски, ест макароны."
            "Программист пьет ром."
            "Плотник работает рядом с красным складом."
            "Вопрос - Кто ест пиццу? Выведи только Профессию, без рассуждений."
        ),
        expected_answer=None, # "Плотник" не подойдёт, т.к. сравнение не строгое
    ),
]

# Default ladder of weak / medium / strong models on Groq.
# NOTE: the exact ids depend on what your account/endpoint exposes. These were
# chosen from the models available on this project's Groq account:
#   openai/gpt-oss-20b       (weak)
#   openai/gpt-oss-120b      (medium)
#   qwen/qwen3.8-27b         (strong)
# Pass your own with --models to change the set or the endpoint (base URL comes
# from .env / --base-url).
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",   # weak
    "openai/gpt-oss-120b",  # medium
    "qwen/qwen3.8-27b",     # strong
]

# Tier captions shown in reports, keyed by order of appearance.
_TIER_LABELS = ["слабая", "средняя", "сильная"]


# --------------------------------------------------------------------------- #
# Cost table. Prices are USD per 1,000,000 tokens (input / output).
# Zero = free / local. Providers not listed cost 0 (unknown -> estimate 0).
# Override per model with --price "model=input_price,output_price".
# --------------------------------------------------------------------------- #

def _default_price_table() -> dict[str, tuple[float, float]]:
    """USD per 1M tokens (input, output). Models not listed default to 0."""
    return {
        # OpenAI (approximate public pricing as of writing; adjust if needed)
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "o1": (15.00, 60.00),
        "o1-mini": (1.10, 4.40),
        # Groq models are served via a free/public API; cost is 0 unless you run
        # a paid plan. Local/Ollama models are always free.
    }


_DEFAULT_PRICES = _default_price_table()


# --------------------------------------------------------------------------- #
# Client construction.
# --------------------------------------------------------------------------- #

def _build_client(
    config: LLMConfig,
    provider: str,
    detail_listener: DetailListener | None = None,
) -> LLMClient:
    if provider == "gigachat":
        return build_gigachat_client(config, detail_listener=detail_listener)
    return LLMClient(config, detail_listener=detail_listener)


# --------------------------------------------------------------------------- #
# A listener that captures latency + token usage for ONE call.
# --------------------------------------------------------------------------- #

class _MultiListener(DetailListener):
    """Fan out detail events to several listeners (e.g. --details + measurement)."""

    def __init__(self, listeners: list[DetailListener]) -> None:
        self._listeners = listeners

    def on_request(self, details: RequestDetails) -> None:
        for listener in self._listeners:
            listener.on_request(details)

    def on_response(self, details: ResponseDetails) -> None:
        for listener in self._listeners:
            listener.on_response(details)


class _CallRecorder(DetailListener):
    """Collects timing and token usage from a single ``send_prompt`` call.

    ``elapsed_ms`` is the total wall-clock time of the whole call, measured
    around ``client.send_prompt(...)`` (including retries), so it reflects the
    real perceived latency. Token usage is taken from the last successful HTTP
    attempt reported via ``on_response``.
    """

    def __init__(self) -> None:
        self.usage: dict[str, int] | None = None
        self.first_byte_ms: float | None = None

    def on_request(self, details: RequestDetails) -> None:
        self._start = time.monotonic()

    def on_response(self, details: ResponseDetails) -> None:
        if details.usage:
            self.usage = details.usage
        if self.first_byte_ms is None and details.elapsed_ms is not None:
            self.first_byte_ms = details.elapsed_ms


def _run_model(
    make_client: Callable[[str, DetailListener | None], LLMClient],
    model: str,
    task_text: str,
    expected: str | None,
) -> "ModelResult":
    """Run one task through one model; returns the measured result.

    ``make_client(model, detail_listener)`` builds a fresh :class:`LLMClient`
    configured to use *model* on the shared base URL / provider, passing the
    recorder as the detail listener so timing + token usage are captured
    without touching private attributes.
    """
    recorder = _CallRecorder()
    client = make_client(model, recorder)

    started = time.monotonic()
    try:
        answer = client.send_prompt(task_text).strip()
    finally:
        elapsed_ms = (time.monotonic() - started) * 1000.0

    correct, summary = _score(answer, expected)
    return ModelResult(
        model=model,
        response=answer,
        elapsed_ms=elapsed_ms,
        first_byte_ms=recorder.first_byte_ms,
        usage=recorder.usage or {},
        correct=correct,
        score_summary=summary,
    )


# --------------------------------------------------------------------------- #
# Scoring / analysis helpers (mirrors compare_methods.compare_temperature).
# --------------------------------------------------------------------------- #

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparisons."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> set[str]:
    """Return the set of integer tokens appearing in *text*."""
    nums: set[str] = set()
    for token in re.findall(r"\d[\d\s.,]*", text):
        cleaned = re.sub(r"[^\d]", "", token)
        if cleaned:
            nums.add(cleaned)
    return nums


def _score(actual: str, expected: str | None) -> tuple[bool, str]:
    """Return (correct, summary) based on the expected answer when provided."""
    if not expected:
        return True, "нет эталона — оцени вручную"
    expected_nums = _extract_numbers(expected)
    if expected_nums:
        actual_nums = _extract_numbers(actual)
        missing = expected_nums - actual_nums
        if not missing:
            return True, f"числа совпадают с эталоном ({expected})"
        return False, f"нет чисел {sorted(missing)} из эталона ({expected})"
    expected_words = _normalize(expected).split()
    hits = [w for w in expected_words if w in _normalize(actual)]
    ratio = len(hits) / max(len(expected_words), 1)
    if ratio >= 0.6:
        return True, f"содержит эталон ({expected})"
    return False, f"не совпадает с эталоном ({expected})"


def _word_count(text: str) -> int:
    return len(_normalize(text).split())


# --------------------------------------------------------------------------- #
# Cost estimation.
# --------------------------------------------------------------------------- #

def _price_for(model: str, price_overrides: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Look up (input_per_m, output_per_m) USD for a model.

    User ``--price`` overrides take precedence; otherwise the built-in default
    table is consulted; unknown models cost 0. Matches both the exact model id
    and its bare name (e.g. ``provider/gpt-4o`` → ``gpt-4o``).
    """
    if model in price_overrides:
        return price_overrides[model]
    bare = model.split("/")[-1]
    if bare in price_overrides:
        return price_overrides[bare]
    if model in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[model]
    if bare in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[bare]
    return (0.0, 0.0)


def _estimate_cost(
    usage: dict[str, int],
    price: tuple[float, float],
) -> float:
    """USD cost of *usage* given (input_per_m, output_per_m) prices."""
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    input_cost = prompt * price[0] / 1_000_000
    output_cost = completion * price[1] / 1_000_000
    return input_cost + output_cost


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class ModelResult:
    model: str
    response: str
    elapsed_ms: float
    first_byte_ms: float | None
    usage: dict[str, int]
    correct: bool
    score_summary: str
    cost_usd: float = 0.0  # filled in after pricing is resolved

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


def _finalize_cost(results: list[ModelResult], price_overrides: dict[str, tuple[float, float]]) -> None:
    """Fill ``cost_usd`` on each result using the price table."""
    for r in results:
        price = _price_for(r.model, price_overrides)
        r.cost_usd = _estimate_cost(r.usage, price)


# --------------------------------------------------------------------------- #
# Output formatting.
# --------------------------------------------------------------------------- #

def _fmt_cost(usd: float) -> str:
    if usd <= 0:
        return "$0.00"
    if usd < 0.001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def _tier_label(index: int) -> str:
    return _TIER_LABELS[index] if index < len(_TIER_LABELS) else f"модель {index + 1}"


def _render_text(task: Task, results: list[ModelResult], models: list[str]) -> str:
    lines = [
        f"Задача: {task.title}",
        f"Условие: {task.text}",
        f"Эталон: {task.expected_answer or '—'}",
        "",
        "Сводка по моделям:",
        "",
        f"{'Модель' :<28} {'Тир' :<8} {'Время' :<9} {'Промпт' :<8} {'Ответ' :<8} "
        f"{'Всего' :<8} {'Стоимость' :<10} {'Качество'}",
        f"{'-'*28} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}",
    ]
    for i, r in enumerate(results):
        ok = "✅" if r.correct else "❌" if task.expected_answer else "—"
        lines.append(
            f"{r.model:<28} {_tier_label(i):<8} {r.elapsed_ms:<9.0f} "
            f"{r.prompt_tokens:<8} {r.completion_tokens:<8} {r.total_tokens:<8} "
            f"{_fmt_cost(r.cost_usd):<10} {ok}"
        )
    lines.append("")
    lines.append("Определения:")
    lines.append("  Тир         — условный уровень модели (слабая/средняя/сильная).")
    lines.append("  Время       — общее время ответа в миллисекундах.")
    lines.append("  Промпт/Ответ/Всего — токены из usage (если провайдер их отдаёт).")
    lines.append("  Стоимость   — расчётная цена в USD (0 — бесплатная/локальная).")
    lines.append("  Качество    — ✅/❌ по эталону (если задан), иначе — для ручной оценки.")
    lines.append("")
    lines.append("Полные ответы:")
    lines.append("")
    for i, r in enumerate(results):
        verdict = "✅" if r.correct else "❌"
        lines.append(f"--- {_tier_label(i)}: {r.model} {verdict} ---")
        lines.append("  " + r.response.replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)


def _render_markdown(
    task: Task,
    results: list[ModelResult],
    models: list[str],
    provider: str,
    base_url: str,
    runs: int,
) -> str:
    lines = [
        "# Сравнение моделей: слабая / средняя / сильная",
        "",
        f"**Задача:** {task.title}",
        "",
        f"**Условие:** {task.text}",
        "",
        f"**Эталонный ответ:** {task.expected_answer or '—'}",
        "",
        f"**Провайдер:** {provider}  ·  **Endpoint:** {base_url}  ·  **Прогонов:** {runs}",
        "",
        "## Сводная таблица",
        "",
        "| Тир | Модель | Время, мс | Промпт | Ответ | Всего токенов | Стоимость | Качество |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for i, r in enumerate(results):
        ok = "✅" if r.correct else "❌" if task.expected_answer else "—"
        lines.append(
            f"| {_tier_label(i)} | `{r.model}` | {r.elapsed_ms:.0f} | "
            f"{r.prompt_tokens} | {r.completion_tokens} | {r.total_tokens} | "
            f"{_fmt_cost(r.cost_usd)} | {ok} |"
        )
    lines.append("")
    lines.append(f"**Оценка качества:** {task.expected_answer and 'автоматическая по эталону' or 'вручную (эталона нет)'}")
    lines.append("")
    lines.append("## Полные ответы")
    lines.append("")
    for i, r in enumerate(results):
        verdict = "✅" if r.correct else "❌"
        lines.append(f"### {_tier_label(i)} — `{r.model}` {verdict}")
        lines.append("")
        lines.append(f"Время: {r.elapsed_ms:.0f} мс · Токены: {r.total_tokens} · Стоимость: {_fmt_cost(r.cost_usd)}")
        lines.append("")
        lines.append("```text")
        lines.append(r.response)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _save_json(task: Task, results: list[ModelResult], path: Path) -> None:
    payload = {
        "task": dataclasses.asdict(task),
        "models": [
            {
                "model": r.model,
                "elapsed_ms": round(r.elapsed_ms, 1),
                "first_byte_ms": round(r.first_byte_ms, 1) if r.first_byte_ms is not None else None,
                "usage": r.usage,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": round(r.cost_usd, 8),
                "correct": r.correct,
                "score_summary": r.score_summary,
                "response": r.response,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-models",
        description="Run the same request on weak/medium/strong models and compare "
        "quality, speed and cost.",
    )
    task_ids = [t.id for t in BUILTIN_TASKS]
    parser.add_argument(
        "--task-index",
        choices=task_ids,
        default=None,
        help=f"Built-in task to run: {', '.join(task_ids)}. Default: all.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Your own task text (overrides --task-index). Expected answer unknown.",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="Expected answer for --task, to enable automatic quality scoring.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model ids to compare. Default: {', '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the API base URL (else LLM_BASE_URL / .env).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the API key (else LLM_API_KEY / .env).",
    )
    parser.add_argument(
        "--price",
        action="append",
        default=None,
        metavar="MODEL=input,output",
        help="Override price for a model in USD per 1M tokens: e.g. "
        "'gpt-4o=2.50,10.00'. Repeatable. Unknown models cost 0.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gigachat"),
        default=None,
        help="Provider auth mode (else LLMConfig from env). 'gigachat' exchanges "
        "GIGACHAT_CLIENT_SECRET for an OAuth2 token before calling the API.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print request/response details (URL, model, payload, token usage) "
        "to stderr, same as the CLI's --details.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional directory or file to save the report. Use a .md or .json "
        "extension to choose format.",
    )
    return parser


def _parse_prices(overrides: list[str] | None) -> dict[str, tuple[float, float]]:
    """Parse ``--price`` values into a {model: (input, output)} map."""
    result = dict(_DEFAULT_PRICES)
    for item in overrides or []:
        model, _, rates = item.partition("=")
        model = model.strip()
        try:
            in_str, _, out_str = rates.replace(" ", "").partition(",")
            result[model] = (float(in_str), float(out_str))
        except ValueError:
            raise SystemExit(f"error: invalid --price '{item}' (expected MODEL=input,output)")
    return result


def _resolve_tasks(args: argparse.Namespace) -> list[Task]:
    if args.task:
        return [Task(id="custom", title="Своя задача", text=args.task, expected_answer=args.expected)]
    if args.task_index:
        return [t for t in BUILTIN_TASKS if t.id == args.task_index]
    return BUILTIN_TASKS


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _resolve_tasks(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prices = _parse_prices(args.price)

    base_config = LLMConfig.from_env().with_overrides(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    provider = args.provider or ("gigachat" if base_config.gigachat_client_secret else "openai")
    detail_printer = _DetailPrinter() if args.details else None

    def make_client(model: str, recorder: DetailListener | None = None) -> LLMClient:
        # Attach the user-facing --details printer alongside the measurement
        # recorder so both are active during a run.
        cfg = base_config.with_overrides(model=model)
        listeners: list[DetailListener] = []
        if detail_printer is not None:
            listeners.append(detail_printer)
        if recorder is not None:
            listeners.append(recorder)

        combined = _MultiListener(listeners) if len(listeners) > 1 else (listeners[0] if listeners else None)
        return _build_client(cfg, provider, detail_listener=combined)

    print(
        f"provider={provider} base_url={base_config.base_url} "
        f"models={models} tasks={[t.id for t in tasks]}\n"
    )

    output_dir = None
    if args.out:
        out_path = Path(args.out)
        output_dir = out_path.parent if out_path.suffix in (".md", ".json") else out_path

    for task in tasks:
        print("=" * 70)
        print(f"Задача: {task.title}")
        print(f"Условие: {task.text}")
        print("=" * 70)

        results: list[ModelResult] = []
        for i, model in enumerate(models):
            print(f"\n▶ {_tier_label(i)}: {model} ...")
            try:
                result = _run_model(make_client, model, task.text, task.expected_answer)
            except LLMError as exc:
                print(f"  ! ошибка: {exc}")
                continue
            results.append(result)
            ok = "✅" if result.correct else "❌"
            print(
                f"  готово за {result.elapsed_ms:.0f} мс · "
                f"токены {result.total_tokens} · {ok} · {result.score_summary}"
            )

        if not results:
            continue

        _finalize_cost(results, prices)

        print("\n" + _render_text(task, results, models))

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.out).stem if Path(args.out).suffix else f"{task.id}"
            ext = Path(args.out).suffix if Path(args.out).suffix else ".md"
            dest = output_dir / f"{stem}{ext}"
            if ext == ".json":
                _save_json(task, results, dest)
            else:
                dest.write_text(
                    _render_markdown(
                        task, results, models, provider, base_config.base_url, runs=1
                    ),
                    encoding="utf-8",
                )
            print(f"\nСохранено: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())