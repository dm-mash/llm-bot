#!/usr/bin/env python3
"""Compare 4 prompt strategies against the same task using the LLM API.

Runs a single (logical/mathematical) task through four different prompting
approaches and prints a comparison table plus per-method answers:

  1. direct        - raw task, no extra instructions
  2. step_by_step  - task + "solve step by step" instruction
  3. prompt_then_use - ask the model to draft a good prompt first, then solve
                       with that drafted prompt
  4. experts       - a panel of experts (analyst, engineer, critic), each gives
                     a solution; the critic reviews

It uses the project's service layer (``llm_bot.client.LLMClient``) so provider
and model are read from the environment / ``.env`` (see ``LLMConfig``). Both
OpenAI-compatible endpoints and GigaChat are supported automatically.

Examples:
    python scripts/compare_methods.py
    python scripts/compare_methods.py --task-index chickens
    python scripts/compare_methods.py --task "What is 7*6+9?"
    python scripts/compare_methods.py --provider gigachat
    python scripts/compare_methods.py --out results/comparison.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Callable

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/compare_methods.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMClient, LLMError
from llm_bot.cli import _DetailPrinter
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import DetailListener
from llm_bot.gigachat import build_gigachat_client


# --------------------------------------------------------------------------- #
# Built-in tasks. Each carries a canonical answer so we can score correctness.
# To use your own, pass --task (then expected_answer is ignored) or add an entry.
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
]


# --------------------------------------------------------------------------- #
# Prompt construction for the 4 methods.
# --------------------------------------------------------------------------- #

STEP_BY_STEP_INSTRUCTION = (
    "\n\nРешай пошагово. Сначала запиши ход рассуждений, затем финальный ответ."
)

# Each expert solves independently with its own specialization. The description
# is used as the system prompt for that role's call; it must NOT reference other
# experts' answers, because those are not passed to the model.
EXPERT_ROLES = [
    (
        "аналитик",
        "Ты — аналитик. Тщательно разбери задачу: выдели ключевые условия, "
        "разбей её на подзадачи и проверь каждое допущение. Найди несколько "
        "вариантов решения, взвесь их и дай обоснованный, проверяемый ответ. "
        "Отвечай строго по делу, без лишних пояснений.",
    ),
    (
        "инженер",
        "Ты — инженер. Подойди к задаче прагматично и количественно: составь "
        "необходимые уравнения или алгоритм, выполни расчёты и проверь их на "
        "непротиворечивость. Дай точный, однозначный ответ с явно показанными "
        "шагами вычислений. Отвечай строго по делу, без лишних пояснений.",
    ),
    (
        "критик",
        "Ты — критик. Решая задачу, активно ищи слабые места и ловушки: "
        "неверные допущения, пропущенные краевые случаи, ошибки в рассуждениях. "
        "Сформулируй решение максимально строго и надёжно, учтя найденные "
        "риски. Дай окончательный уверенный ответ. Отвечай строго по делу, "
        "без лишних пояснений.",
    ),
]


def _method_prompt(method: str, task_text: str) -> str:
    """Return the user prompt for a given method (except prompt_then_use/experts)."""
    if method == "direct":
        return task_text
    if method == "step_by_step":
        return task_text + STEP_BY_STEP_INSTRUCTION
    raise ValueError(f"Unexpected single-call method: {method}")


def _prompt_drafting_prompt(task_text: str) -> str:
    """First call of the prompt_then_use method: ask the model to draft a prompt."""
    return (
        "Ты — эксперт по составлению промптов. Составь оптимальный промпт, "
        "который поможет решить следующую задачу наиболее точно. Верни только "
        "сам текст промпта, без пояснений.\n\nЗадача:\n" + task_text
    )


# --------------------------------------------------------------------------- #
# Running one method.
# --------------------------------------------------------------------------- #

def _build_client(
    config: LLMConfig,
    provider: str,
    detail_listener: DetailListener | None = None,
) -> LLMClient:
    if provider == "gigachat":
        return build_gigachat_client(config, detail_listener=detail_listener)
    return LLMClient(config, detail_listener=detail_listener)


def _ask(
    make_client: Callable[[str | None], LLMClient],
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Send *prompt* via a client built by *make_client*.

    ``make_client`` builds a fresh :class:`LLMClient` for the current provider.
    When ``system_prompt`` is given, the client is configured to send that
    system message (used to give each expert its role).
    """
    client = make_client(system_prompt)
    return client.send_prompt(prompt)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _extract_numbers(text: str) -> set[str]:
    """Return the set of integer tokens (plain digits or with ','/'.' separators)
    appearing in *text*. Useful for scoring numeric answers regardless of language."""
    nums: set[str] = set()
    for token in re.findall(r"\d[\d\s.,]*", text):
        cleaned = re.sub(r"[^\d]", "", token)
        if cleaned:
            nums.add(cleaned)
    return nums


def _score(actual: str, expected: str | None) -> tuple[bool, str]:
    """Return (correct, summary) based on the expected answer when provided.

    Scoring is robust to language and phrasing:
      - If the expected answer contains numbers, we check that every number in
        it also appears in the actual answer (e.g. ``"Chickens: 16, Cows: 6"``
        matches the Russian canonical answer ``"6 коров, 16 кур"``).
      - Otherwise we fall back to a word-overlap check on normalized text.
    """
    if not expected:
        return True, "нет эталона — оцени вручную"

    expected_nums = _extract_numbers(expected)
    if expected_nums:
        actual_nums = _extract_numbers(actual)
        missing = expected_nums - actual_nums
        if not missing:
            return True, f"числа совпадают с эталоном ({expected})"
        return False, f"нет чисел {sorted(missing)} из эталона ({expected})"

    # No numbers in the expected answer -> fall back to word overlap.
    expected_words = _normalize(expected).split()
    hits = [w for w in expected_words if w in _normalize(actual)]
    ratio = len(hits) / max(len(expected_words), 1)
    if ratio >= 0.6:
        return True, f"содержит эталон ({expected})"
    return False, f"не совпадает с эталоном ({expected})"


@dataclasses.dataclass
class MethodResult:
    method: str
    prompts: list[str]
    responses: list[str]
    correct: bool
    summary: str
    labels: list[str] | None = None  # optional caption per response (e.g. expert role)


def _run_method(
    make_client: Callable[[str | None], LLMClient],
    method: str,
    task_text: str,
    expected: str | None,
) -> MethodResult:
    """Execute one method and collect its prompt(s) and response(s).

    ``make_client`` builds an :class:`LLMClient` for the current provider; it
    accepts an optional system prompt (used to give each expert its role).
    """
    if method == "prompt_then_use":
        # Call 1: draft a prompt. Call 2: solve using that drafted prompt.
        draft = _ask(make_client, _prompt_drafting_prompt(task_text)).strip()
        answer = _ask(make_client, draft)
        correct, summary = _score(answer, expected)
        return MethodResult(
            method=method,
            prompts=[_prompt_drafting_prompt(task_text), draft],
            responses=[draft, answer],
            correct=correct,
            summary=summary,
        )

    if method == "experts":
        # One independent call per expert role; the role description is sent as
        # the system prompt, and the user prompt is just the task text.
        prompts: list[str] = []
        responses: list[str] = []
        labels: list[str] = []
        for name, system_prompt in EXPERT_ROLES:
            prompts.append(task_text)
            responses.append(_ask(make_client, task_text, system_prompt=system_prompt))
            labels.append(f"{name}")
        # Use the last expert's reply (the critic) as the "final" answer for scoring.
        final = responses[-1]
        correct, summary = _score(final, expected)
        return MethodResult(
            method=method,
            prompts=prompts,
            responses=responses,
            correct=correct,
            summary=summary,
            labels=labels,
        )

    # direct / step_by_step : single call.
    prompt = _method_prompt(method, task_text)
    answer = _ask(make_client, prompt)
    correct, summary = _score(answer, expected)
    return MethodResult(
        method=method,
        prompts=[prompt],
        responses=[answer],
        correct=correct,
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Output formatting.
# --------------------------------------------------------------------------- #

_METHOD_LABELS = {
    "direct": "1. Прямой ответ",
    "step_by_step": "2. Решай пошагово",
    "prompt_then_use": "3. Сначала промпт → потом решение",
    "experts": "4. Группа экспертов",
}

_METHOD_ORDER = ["direct", "step_by_step", "prompt_then_use", "experts"]


def _response_label(r: MethodResult, index: int) -> str:
    """Return a caption for response *index* of method result *r* (e.g. an expert role)."""
    if r.labels and index < len(r.labels):
        return r.labels[index]
    return f"ответ {index + 1}"


def _render_text(task: Task, results: list[MethodResult]) -> str:
    """Render a human-readable comparison report (stdout)."""
    lines = [
        f"Задача: {task.title}",
        f"Условие: {task.text}",
        "",
        "Сравнение способов:",
        "",
        f"{'Способ':<32} {'Верно':<8} {'Комментарий'}",
        f"{'-' * 32} {'-' * 8} {'-' * 40}",
    ]
    for r in results:
        verdict = "да" if r.correct else "нет"
        lines.append(f"{_METHOD_LABELS[r.method]:<32} {verdict:<8} {r.summary}")
    lines.append("")
    lines.append("Подробные ответы:")
    lines.append("")
    for r in results:
        lines.append(f"{_METHOD_LABELS[r.method]}:")
        for i, resp in enumerate(r.responses):
            lines.append(f"  {_response_label(r, i)}:")
            lines.append("  " + resp.strip().replace("\n", "\n  "))
        lines.append("")
    return "\n".join(lines)


def _render_markdown(task: Task, results: list[MethodResult]) -> str:
    """Render a markdown report for --out <file>.md"""
    lines = [
        f"# Сравнение способов решения",
        "",
        f"**Задача:** {task.title}",
        "",
        f"**Условие:** {task.text}",
        "",
        f"**Эталонный ответ:** {task.expected_answer or '—'}",
        "",
        "## Таблица сравнения",
        "",
        "| Способ | Верно? | Комментарий |",
        "|---|---|---|",
    ]
    for r in results:
        verdict = "✅" if r.correct else "❌"
        lines.append(f"| {_METHOD_LABELS[r.method]} | {verdict} | {r.summary} |")
    lines.append("")
    lines.append("## Подробные ответы")
    lines.append("")
    for r in results:
        lines.append(f"### {_METHOD_LABELS[r.method]}")
        for i, resp in enumerate(r.responses):
            lines.append(f"**{_response_label(r, i)}:**")
            lines.append("")
            lines.append("```text")
            lines.append(resp.strip())
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def _save_json(task: Task, results: list[MethodResult], path: Path) -> None:
    payload = {
        "task": dataclasses.asdict(task),
        "results": [
            {
                "method": r.method,
                "label": _METHOD_LABELS[r.method],
                "prompts": r.prompts,
                "responses": r.responses,
                "labels": r.labels,
                "correct": r.correct,
                "summary": r.summary,
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
        prog="compare-methods",
        description=(
            "Run one task through 4 prompting strategies and compare the results."
        ),
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
        "unknown, so scoring is manual.",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="Expected answer for --task, to enable automatic scoring.",
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
        "--max-response-words",
        type=int,
        default=None,
        help="Target maximum length of each reply, in words. Added to the system "
        "prompt as a briefness instruction (else LLM_MAX_RESPONSE_WORDS). "
        "Leave unset for no limit.",
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
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset of methods to run, e.g. direct,experts.",
    )
    return parser


def _resolve_tasks(args: argparse.Namespace) -> list[Task]:
    if args.task:
        return [Task(id="custom", title="Своя задача", text=args.task, expected_answer=args.expected)]
    if args.task_index:
        return [t for t in BUILTIN_TASKS if t.id == args.task_index]
    return BUILTIN_TASKS


def _resolve_methods(args: argparse.Namespace) -> list[str]:
    if not args.methods:
        return list(_METHOD_ORDER)
    chosen = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in chosen if m not in _METHOD_ORDER]
    if invalid:
        raise SystemExit(f"error: unknown methods: {', '.join(invalid)} (allowed: {', '.join(_METHOD_ORDER)})")
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = _resolve_tasks(args)
    methods = _resolve_methods(args)

    config = LLMConfig.from_env()
    config = config.with_overrides(
        model=args.model,
        max_response_words=args.max_response_words,
    )
    provider = args.provider or ("gigachat" if config.gigachat_client_secret else "openai")
    detail_listener = _DetailPrinter() if args.details else None

    def make_client(system_prompt: str | None = None) -> LLMClient:
        """Build a client for the current provider; optionally override the system prompt."""
        cfg = (
            config.with_overrides(system_prompt=system_prompt)
            if system_prompt is not None
            else config
        )
        return _build_client(cfg, provider, detail_listener=detail_listener)

    print(f"provider={provider} model={config.model} tasks={[t.id for t in tasks]} methods={methods}\n")

    output_dir = None
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix in (".md", ".json"):
            output_dir = out_path.parent
        else:
            output_dir = out_path

    for task in tasks:
        print("=" * 70)
        print(f"Задача: {task.title}")
        print(f"Условие: {task.text}")
        print("=" * 70)
        results: list[MethodResult] = []
        for method in methods:
            print(f"\n▶ {_METHOD_LABELS[method]} ...")
            try:
                result = _run_method(make_client, method, task.text, task.expected_answer)
            except LLMError as exc:
                print(f"  ! ошибка: {exc}")
                continue
            results.append(result)
            verdict = "да" if result.correct else "нет"
            print(f"  готово (верно: {verdict}) — {result.summary}")

        if not results:
            continue

        print("\n" + _render_text(task, results))

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(args.out).stem if Path(args.out).suffix else f"{task.id}"
            ext = Path(args.out).suffix if Path(args.out).suffix else ".md"
            dest = output_dir / f"{stem}{ext}"
            if ext == ".json":
                _save_json(task, results, dest)
            else:
                dest.write_text(_render_markdown(task, results), encoding="utf-8")
            print(f"\nСохранено: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())