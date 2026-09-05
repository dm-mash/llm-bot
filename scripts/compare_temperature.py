#!/usr/bin/env python3
"""Compare sampling temperature settings on the same prompt.

Runs one prompt at ``temperature`` in {0.0, 0.7, 1.2}, repeating each setting
``--runs`` times, then reports per-temperature observations on:

  * accuracy   - how often / how correctly the answer matches the expected one
  * creativity - a rough textual signal (e.g. answer length, variety of words)
  * diversity  - how much answers differ between runs of the same temperature

It uses the project's service layer (``llm_bot.client.LLMClient``) so provider
and model are read from the environment / ``.env`` (see ``LLMConfig``). The
``temperature`` field is passed straight into the chat-completions payload, so
both OpenAI-compatible endpoints and GigaChat are supported automatically.

Examples:
    python scripts/compare_temperature.py
    python scripts/compare_temperature.py --task-index chickens
    python scripts/compare_temperature.py --task "Напиши слоган для кофейни" --runs 5
    python scripts/compare_temperature.py --provider gigachat
    python scripts/compare_temperature.py --out results/temperature.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/compare_temperature.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMClient, LLMError
from llm_bot.cli import _DetailPrinter
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import DetailListener
from llm_bot.gigachat import build_gigachat_client


# --------------------------------------------------------------------------- #
# Built-in tasks (same style as scripts/compare_methods.py).
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    title: str
    text: str
    expected_answer: str | None = None


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
        id="story",
        title="Короткая история",
        text=(
            "Напиши короткую историю (3–5 предложений) про потерявшегося "
            "робота, который однажды утром нашёл на крыше городскую голубку."
        ),
        expected_answer=None,
    ),
    Task(
        id="slogan",
        title="Слоган для кофейни",
        text="Придумай 3 оригинальных слогана для маленькой уютной кофейни.",
        expected_answer=None,
    ),
]


# Temperatures to compare, in order.
TEMPERATURES = [0.0, 0.7, 1.2]


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
# Scoring / analysis helpers.
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
    """Return (correct, summary). Robust to language/phrasing, like compare_methods."""
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
    """Approximate number of words in a text."""
    return len(_normalize(text).split())


def _unique_word_ratio(text: str) -> float:
    """Ratio of unique words to total words (rough vocabulary/variety signal)."""
    words = _normalize(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity over word sets; 1.0 = identical, 0.0 = disjoint."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa and not wb:
        return 1.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


@dataclasses.dataclass
class TempResult:
    temperature: float
    responses: list[str]
    correct_flags: list[bool]
    summaries: list[str]

    @property
    def accuracy(self) -> float:
        """Fraction of runs judged correct (only meaningful when an expected answer exists)."""
        if not self.correct_flags:
            return 0.0
        return sum(self.correct_flags) / len(self.correct_flags)

    @property
    def avg_words(self) -> float:
        if not self.responses:
            return 0.0
        return sum(_word_count(r) for r in self.responses) / len(self.responses)

    @property
    def avg_unique_ratio(self) -> float:
        if not self.responses:
            return 0.0
        return sum(_unique_word_ratio(r) for r in self.responses) / len(self.responses)

    @property
    def pair_jaccard(self) -> float:
        """Average Jaccard similarity across all pairs of runs (diversity, inverted)."""
        n = len(self.responses)
        if n < 2:
            return 1.0
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += _jaccard(self.responses[i], self.responses[j])
                pairs += 1
        return total / pairs

    @property
    def diversity(self) -> float:
        """1 - mean pairwise Jaccard; higher = more varied answers."""
        return 1.0 - self.pair_jaccard


def _run_temperature(
    make_client: Callable[[float], LLMClient],
    temperature: float,
    task_text: str,
    expected: str | None,
    runs: int,
) -> TempResult:
    responses: list[str] = []
    correct_flags: list[bool] = []
    summaries: list[str] = []
    for _ in range(runs):
        client = make_client(temperature)
        answer = client.send_prompt(task_text).strip()
        responses.append(answer)
        ok, summary = _score(answer, expected)
        correct_flags.append(ok)
        summaries.append(summary)
    return TempResult(
        temperature=temperature,
        responses=responses,
        correct_flags=correct_flags,
        summaries=summaries,
    )


# --------------------------------------------------------------------------- #
# Output formatting.
# --------------------------------------------------------------------------- #

def _render_text(task: Task, results: list[TempResult], runs: int) -> str:
    lines = [
        f"Задача: {task.title}",
        f"Условие: {task.text}",
        f"Эталон: {task.expected_answer or '—'}",
        f"Прогонов на температуру: {runs}",
        "",
        "Сводка по настройкам температуры:",
        "",
        f"{'Темп.' :<8} {'Точность' :<10} {'Ср. слов' :<10} {'Лексика' :<10} {'Разнообразие'}",
        f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}",
    ]
    for r in results:
        acc = (
            f"{r.accuracy:.0%}"
            if task.expected_answer is not None
            else "н/д (нет эталона)"
        )
        lines.append(
            f"{r.temperature:<8} {acc:<10} {r.avg_words:<10.1f} "
            f"{r.avg_unique_ratio:<10.2f} {r.diversity:<.2f}"
        )
    lines.append("")
    lines.append("Определения:")
    lines.append("  Точность    — доля ответов, совпавших с эталоном (если он задан).")
    lines.append("  Ср. слов    — средняя длина ответа в словах.")
    lines.append("  Лексика     — доля уникальных слов (грубый сигнал лексического богатства).")
    lines.append("  Разнообразие — 1 − средний Jaccard между прогонами (0=одинаковые, 1=разные).")
    lines.append("")
    lines.append("Полные ответы:")
    lines.append("")
    for r in results:
        lines.append(f"--- temperature = {r.temperature} ---")
        for i, resp in enumerate(r.responses):
            verdict = "✅" if r.correct_flags[i] else "❌"
            lines.append(f"  Прогон {i + 1} {verdict}:")
            lines.append("  " + resp.replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)


def _render_markdown(task: Task, results: list[TempResult], runs: int, provider: str, model: str) -> str:
    lines = [
        "# Сравнение температуры",
        "",
        f"**Задача:** {task.title}",
        "",
        f"**Условие:** {task.text}",
        "",
        f"**Эталонный ответ:** {task.expected_answer or '—'}",
        "",
        f"**Провайдер:** {provider}  ·  **Модель:** {model}  ·  **Прогонов на температуру:** {runs}",
        "",
        "## Сводная таблица",
        "",
        "| Темп. | Точность | Ср. слов | Лексика | Разнообразие |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        acc = (
            f"{r.accuracy:.0%}"
            if task.expected_answer is not None
            else "н/д (нет эталона)"
        )
        lines.append(
            f"| {r.temperature} | {acc} | {r.avg_words:.1f} | {r.avg_unique_ratio:.2f} | {r.diversity:.2f} |"
        )
    lines.append("")
    lines.append("## Полные ответы")
    lines.append("")
    for r in results:
        lines.append(f"### temperature = {r.temperature}")
        lines.append("")
        for i, resp in enumerate(r.responses):
            verdict = "✅" if r.correct_flags[i] else "❌"
            lines.append(f"**Прогон {i + 1} {verdict}**")
            lines.append("")
            lines.append("```text")
            lines.append(resp)
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def _save_json(task: Task, results: list[TempResult], path: Path) -> None:
    payload = {
        "task": dataclasses.asdict(task),
        "temperatures": [
            {
                "temperature": r.temperature,
                "accuracy": r.accuracy,
                "avg_words": r.avg_words,
                "avg_unique_ratio": r.avg_unique_ratio,
                "diversity": r.diversity,
                "pair_jaccard": r.pair_jaccard,
                "responses": [
                    {"run": i + 1, "correct": r.correct_flags[i], "text": resp}
                    for i, resp in enumerate(r.responses)
                ],
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
        prog="compare-temperature",
        description="Run one prompt at several temperatures and compare the results.",
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
        help="Your own task text (overrides --task-index). Expected answer is then "
        "unknown, so accuracy is manual.",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="Expected answer for --task, to enable automatic accuracy scoring.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many times to run each temperature (for diversity). Default: 3.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gigachat"),
        default=None,
        help="Provider auth mode (else LLMConfig from env). 'gigachat' exchanges "
        "GIGACHAT_CLIENT_SECRET for an OAuth2 token before calling the API.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model identifier (else LLM_MODEL / LLMConfig).",
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


def _resolve_tasks(args: argparse.Namespace) -> list[Task]:
    if args.task:
        return [Task(id="custom", title="Своя задача", text=args.task, expected_answer=args.expected)]
    if args.task_index:
        return [t for t in BUILTIN_TASKS if t.id == args.task_index]
    return BUILTIN_TASKS


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _resolve_tasks(args)

    base_config = LLMConfig.from_env().with_overrides(model=args.model)
    provider = args.provider or ("gigachat" if base_config.gigachat_client_secret else "openai")
    detail_listener = _DetailPrinter() if args.details else None

    def make_client(temperature: float) -> LLMClient:
        cfg = base_config.with_overrides(temperature=temperature)
        return _build_client(cfg, provider, detail_listener=detail_listener)

    print(
        f"provider={provider} model={base_config.model} "
        f"tasks={[t.id for t in tasks]} temps={TEMPERATURES} runs={args.runs}\n"
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

        results: list[TempResult] = []
        for temp in TEMPERATURES:
            print(f"\n▶ temperature = {temp} ...")
            try:
                result = _run_temperature(make_client, temp, task.text, task.expected_answer, args.runs)
            except LLMError as exc:
                print(f"  ! ошибка: {exc}")
                continue
            results.append(result)
            acc = f"{result.accuracy:.0%}" if task.expected_answer is not None else "н/д"
            print(f"  готово (точность: {acc}, разнообразие: {result.diversity:.2f})")

        if not results:
            continue

        print("\n" + _render_text(task, results, args.runs))

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.out).stem if Path(args.out).suffix else f"{task.id}"
            ext = Path(args.out).suffix if Path(args.out).suffix else ".md"
            dest = output_dir / f"{stem}{ext}"
            if ext == ".json":
                _save_json(task, results, dest)
            else:
                dest.write_text(
                    _render_markdown(task, results, args.runs, provider, base_config.model),
                    encoding="utf-8",
                )
            print(f"\nСохранено: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())