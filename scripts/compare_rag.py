#!/usr/bin/env python3
"""Compare the same questions answered with and without a local RAG context.

Deliverable for the «Первый RAG-запрос» task. For every question in the control
set it asks the model twice — once with nothing but the question, once with the
top chunks of a local document index as a grounding prefix — and then scores the
two answers against what the corpus actually says.

Scoring, all deterministic and offline-computable from the saved answers:

* **fact coverage** — share of ``expect`` strings found in the answer
  (case-insensitive substring). This is the primary metric: did the answer
  contain the fact, not did it read well.
* **retrieval hit** — did the expected source file reach the prompt at all?
  Separating this from the answer says *where* a failure happened: a miss here
  is a retrieval problem, a miss only in the answer is a reading problem.
* **citation** — did the answer cite the expected ``file:line``? The RAG block
  asks for this, so a grounded answer that cites nothing is a partial failure.
* **refusal** — for ``answerable: false``, did the model decline instead of
  inventing? Those two questions are the honest test of grounding: a model
  without RAG will happily answer "190 ₽" for a drink that isn't on the menu,
  and a number that exists in the repo but outside the corpus is exactly the
  case where confabulation is most tempting.

Two scores are therefore reported: ``fact_coverage`` over answerable questions
(does grounding get the facts in) and ``refusal`` over unanswerable ones (does
it keep its mouth shut), because a single average would let a good refusal rate
hide a bad answer or vice versa.

Examples:
    # both modes, both corpora, on a real model
    python scripts/compare_rag.py --questions rag_questions.yaml --model leanstral

    # one corpus, another model, more context per question
    python scripts/compare_rag.py --questions rag_questions.yaml --model groq-qwen \
        --corpus kb --top_k 6 --out results/rag_kb.md

    # no provider calls at all: score a saved run again (e.g. after editing
    # the scoring)
    python scripts/compare_rag.py --questions rag_questions.yaml \
        --from_json results/rag_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx
import yaml

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/compare_rag.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMClient, LLMError
from llm_bot.factory import build_client
from llm_bot.rag import Retriever
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.tokens import MESSAGE_OVERHEAD_TOKENS, estimate_tokens
from llm_bot.yaml_stores import YamlModelStore

MODES = ("rag", "no_rag")

#: Phrases that count as "I did not find it". Deliberately broad: the point is
#: to detect a confident fabrication, and a refusal rarely uses one fixed wording
#: ("информации об этом нет" and "нет информации" are the same statement). A bare
#: "нет" is NOT a match on its own — it appears inside perfectly confident
#: answers ("в кофейне нет Wi-Fi") and would turn a fabrication into a pass.
_REFUSAL_RE = re.compile(
    r"(не наш[её]л|не нахожу|не могу найти|не смог найти|не удалось найти"
    r"|информаци\w*[^.\n]{0,24}?нет\b|нет информации|не содержит"
    r"|не упоминается|не указано|не знаю|не в базе|не в документах"
    r"|отсутствует|не встречается|за рамками (?:этой |моей )?базы)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Control questions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Corpus:
    """One indexed body of documents the questions can be asked against."""

    name: str
    label: str
    index_path: Path
    input_dir: Path
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    """One control question with its expectations (see rag_questions.yaml)."""

    text: str
    corpus: str
    expect: tuple[str, ...]
    sources: tuple[str, ...]
    answerable: bool


@dataclass(frozen=True)
class Case:
    """A (question, mode) pair and everything observed while answering it."""

    index: int
    question: str
    corpus: str
    mode: str
    answerable: bool
    answer: str
    retrieved: tuple[str, ...] = ()
    context_tokens: int = 0
    prompt_tokens: int = 0
    elapsed: float = 0.0
    error: str = ""

    @property
    def retrieved_files(self) -> tuple[str, ...]:
        """Just the file part of the ``file:start-end`` citations of the hits."""
        return tuple(location.split(":", 1)[0] for location in self.retrieved)


def _spans(locations: Iterable[str]) -> set[tuple[str, int, int]]:
    """Parse ``file:start-end`` (or ``file``) into ``(file, start, end)``.

    A bare ``file`` yields ``(file, 0, 0)``, which overlaps nothing except
    another bare entry — deliberately, so an unknown line range is treated as
    "unknown", never as "matches everything".
    """
    out: set[tuple[str, int, int]] = set()
    for location in locations:
        match = re.fullmatch(r"(.+?):(\d+)-(\d+)", location.strip())
        if match:
            out.add((match.group(1), int(match.group(2)), int(match.group(3))))
        else:
            out.add((location.strip(), 0, 0))
    return out


def source_reached(sources: Sequence[str], retrieved: Sequence[str]) -> bool:
    """Did the retrieved set cover every expected source?

    A source given as ``file`` is satisfied by any chunk from that file. A
    source given as ``file:start-end`` demands line overlap with a retrieved
    chunk.

    The line-range form is the point: a file-level check reported question 1 as
    a retrieval success because three unrelated ``README.md`` chunks came back,
    while the chunk actually holding the answer was never in the prompt. That
    turned a retrieval failure into a "выдача: да" in the report, which is worse
    than a visible miss.
    """
    got = _spans(retrieved)
    if not got:
        return False
    for source in sources:
        want = _spans([source]).pop()
        name, start, end = want
        if start == 0 and end == 0:
            if not any(file == name for file, _, _ in got):
                return False
            continue
        if not any(
            file == name and start <= high and low <= end
            for file, low, high in got
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_questions(path: Path) -> tuple[dict[str, Corpus], list[Question]]:
    """Read the control set; fail loudly on a malformed file."""
    if not path.is_file():
        raise FileNotFoundError(f"вопросы не найдены: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидался словарь с corpora и questions")
    corpora_raw = raw.get("corpora") or {}
    if not isinstance(corpora_raw, dict) or not corpora_raw:
        raise ValueError(f"{path}: не задан раздел corpora")
    base = path.parent
    corpora: dict[str, Corpus] = {}
    for name, entry in corpora_raw.items():
        if not isinstance(entry, dict) or "index_dir" not in entry:
            raise ValueError(f"{path}: корпус {name!r} без index_dir")
        index_dir = Path(entry["index_dir"])
        if not index_dir.is_absolute():
            index_dir = base.parent / index_dir
        input_dir = Path(entry.get("input_dir", "."))
        if not input_dir.is_absolute():
            input_dir = base.parent / input_dir
        extensions = entry.get("extensions", [".md"])
        if isinstance(extensions, str):
            extensions = [extensions]
        corpora[name] = Corpus(
            name=name,
            label=str(entry.get("label") or name),
            index_path=index_dir / "index_structure.json",
            input_dir=input_dir,
            extensions=tuple(str(e) for e in extensions),
        )

    questions: list[Question] = []
    for position, entry in enumerate(raw.get("questions") or [], start=1):
        if not isinstance(entry, dict) or not entry.get("text"):
            raise ValueError(f"{path}: вопрос #{position} без поля text")
        corpus = str(entry.get("corpus") or "")
        if corpus not in corpora:
            raise ValueError(
                f"{path}: вопрос #{position} ссылается на неизвестный корпус "
                f"{corpus!r} (есть: {', '.join(corpora)})"
            )
        questions.append(
            Question(
                text=str(entry["text"]).strip(),
                corpus=corpus,
                expect=tuple(str(e) for e in entry.get("expect") or ()),
                sources=tuple(str(s) for s in entry.get("sources") or ()),
                # Default to answerable: a question is assumed to have an answer
                # unless it says otherwise, so forgetting the flag fails the
                # strict direction (must contain the fact) rather than hiding a
                # wrong answer in the refusal bucket.
                answerable=bool(entry.get("answerable", True)),
            )
        )
    if not questions:
        raise ValueError(f"{path}: нет ни одного вопроса")
    return corpora, questions


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    """Case-, whitespace- and markup-insensitive form for expectation matching.

    Markdown emphasis is stripped because the model reproduces the source
    formatting: a fact written as ``**до** приготовления`` in the document comes
    back as ``**до** приготовления`` in the answer, and without this the expected
    substring would not match across its own bold markers — scoring a correct
    answer as a miss.
    """
    plain = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", plain.strip().lower())


def fact_coverage(answer: str, expect: tuple[str, ...]) -> tuple[float, list[str], list[str]]:
    """Share of expected strings present in the answer.

    Returns ``(coverage, hit, missed)``. Matching is a case-insensitive
    substring test with collapsed whitespace, so a different phrasing of the
    same number still counts; a different value does not.
    """
    if not expect:
        return 1.0, [], []
    haystack = _norm(answer)
    hit = [item for item in expect if _norm(item) in haystack]
    missed = [item for item in expect if item not in hit]
    return len(hit) / len(expect), hit, missed


def source_files(sources: Sequence[str]) -> set[str]:
    """Just the file names of ``file`` / ``file:start-end`` sources."""
    return {source.split(":", 1)[0] for source in sources}


def cited_sources(answer: str) -> set[str]:
    """Files the answer cites, from any ``path/to/file.md:12-34`` reference."""
    found: set[str] = set()
    for match in re.finditer(r"[\w./-]+\.md:(\d+)-(\d+)", answer):
        found.add(match.group(0).rsplit(":", 1)[0])
    return found


def scored(case: Case, question: Question) -> dict[str, object]:
    """Everything measurable about one answered case."""
    coverage, hit, missed = fact_coverage(case.answer, question.expect)
    retrieval_hit = (
        source_reached(question.sources, case.retrieved)
        if question.sources
        else None
    )
    citation_hit = (
        bool(cited_sources(case.answer) & source_files(question.sources))
        if question.sources
        else None
    )
    refused = bool(_REFUSAL_RE.search(case.answer)) if not question.answerable else None
    if question.answerable:
        # The answer must carry every expected fact. The retrieval check applies
        # to the RAG mode only: in ``no_rag`` nothing is retrieved by
        # construction, so requiring a hit there would score the baseline as
        # broken instead of measuring what it actually is.
        passed = not case.error and coverage == 1.0
        if case.mode == "rag":
            passed = passed and retrieval_hit is not False
    else:
        # Nothing to confirm, so grounding means not inventing: no facts, no
        # claim of having found anything.
        passed = not case.error and bool(refused) and coverage == 1.0
    return {
        "fact_coverage": round(coverage, 3),
        "hit": hit,
        "missed": missed,
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "refused": refused,
        "cited_files": sorted(cited_sources(case.answer)),
        "retrieved_files": sorted(set(case.retrieved_files)),
        "passed": passed,
    }


def aggregate(cases: list[Case], questions: list[Question]) -> dict[str, object]:
    """Per-mode totals. Rates are per-mode means; ``overall_pass_rate`` mixes both."""
    by_index = {q_index: q for q_index, q in enumerate(questions, start=1)}
    summary: dict[str, dict[str, object]] = {}
    for mode in MODES:
        mode_cases = [(c, by_index[c.index]) for c in cases if c.mode == mode]
        if not mode_cases:
            continue
        answerable = [(c, q) for c, q in mode_cases if q.answerable]
        unanswerable = [(c, q) for c, q in mode_cases if not q.answerable]
        scores = {c.index: scored(c, q) for c, q in mode_cases}
        coverage = [
            float(scores[c.index]["fact_coverage"]) for c, _ in answerable
        ]
        refusals = [
            bool(scores[c.index]["refused"]) for c, _ in unanswerable
        ]
        summary[mode] = {
            "questions": len(mode_cases),
            "answered": sum(1 for c, _ in mode_cases if not c.error),
            "errors": [c.error for c, _ in mode_cases if c.error],
            # Quality: do the expected facts show up at all?
            "fact_coverage_mean": round(sum(coverage) / len(coverage), 3)
            if coverage
            else None,
            "fact_coverage_full": sum(1 for c in coverage if c == 1.0),
            "answerable": len(answerable),
            # Honesty: on questions the corpus cannot answer, does it stay silent?
            "refusal_rate": round(sum(refusals) / len(refusals), 3)
            if refusals
            else None,
            "unanswerable": len(unanswerable),
            # Where grounding should have shown up in the text.
            "retrieval_hit_rate": _rate(
                [
                    bool(scores[c.index]["retrieval_hit"])
                    for c, q in answerable
                    if q.sources
                ]
            ),
            "citation_rate": _rate(
                [
                    bool(scores[c.index]["citation_hit"])
                    for c, q in answerable
                    if q.sources
                ]
            ),
            "passed": sum(1 for s in scores.values() if s["passed"]),
            "pass_rate": round(
                sum(1 for s in scores.values() if s["passed"]) / len(mode_cases), 3
            ),
            "prompt_tokens_mean": round(
                sum(c.prompt_tokens for c, _ in mode_cases) / len(mode_cases), 1
            ),
            "elapsed_mean": round(
                sum(c.elapsed for c, _ in mode_cases) / len(mode_cases), 2
            ),
            "context_tokens_mean": round(
                sum(c.context_tokens for c, _ in mode_cases) / len(mode_cases), 1
            ),
        }
    return summary


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def build_agent_config(model_name: str) -> AgentConfig:
    """A minimal, corpus-agnostic agent config.

    Deliberately the same agent for both arms of the comparison: any persona
    difference would show up as a score difference and be mistaken for a RAG
    effect.

    The grounding rule is deliberately NOT repeated here. "Answer only from the
    context, cite ``file:lines``, refuse when the answer is absent" travels with
    the retrieved block (:mod:`llm_bot.rag`), identically for every agent — so
    restating it would only create a second copy to keep in sync, and a copy
    that could disagree with the one the bot actually sends.

    ``temperature`` is 0.1, not 0.0: the Mistral API rejects greedy sampling
    with ``top_p must be 1 when using greedy sampling`` and this project has no
    ``top_p`` knob, so a hard 0 would make every request an HTTP 400. Both arms
    use the same value, so the comparison stays fair.
    """
    return AgentConfig(
        name="researcher",
        model=model_name,
        default_system_prompt=(
            "Отвечай на том же языке, на котором написан вопрос."
        ),
        system_prompt="Ты отвечаешь на вопросы по документам. Отвечай кратко и по делу.",
        temperature=0.1,
    )


def run_case(
    client: LLMClient,
    question: Question,
    mode: str,
    index: int,
    retriever: Retriever | None,
) -> Case:
    """Answer one question in one mode; a provider failure is recorded, not raised."""
    prefix = ""
    retrieved: tuple[str, ...] = ()
    context_tokens = 0
    if mode == "rag":
        assert retriever is not None
        try:
            prefix, event = retriever.render(question.text)
            retrieved = event.retrieved
            context_tokens = event.context_tokens
        except Exception as exc:  # noqa: BLE001 - record, compare, do not crash
            return Case(
                index=index,
                question=question.text,
                corpus=question.corpus,
                mode=mode,
                answerable=question.answerable,
                answer="",
                retrieved=retrieved,
                error=f"retrieval: {exc}",
            )
    messages: list[dict[str, str]] = []
    if prefix:
        messages.append({"role": "system", "content": prefix})
    messages.append({"role": "user", "content": question.text})

    started = time.perf_counter()
    try:
        reply = client.chat(messages)
    except (LLMError, httpx.HTTPError) as exc:
        return Case(
            index=index,
            question=question.text,
            corpus=question.corpus,
            mode=mode,
            answerable=question.answerable,
            answer="",
            retrieved=retrieved,
            context_tokens=context_tokens,
            error=str(exc),
        )
    elapsed = time.perf_counter() - started
    prompt_tokens = sum(
        estimate_tokens(message["content"]) + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )
    return Case(
        index=index,
        question=question.text,
        corpus=question.corpus,
        mode=mode,
        answerable=question.answerable,
        answer=reply,
        retrieved=retrieved,
        context_tokens=context_tokens,
        prompt_tokens=prompt_tokens,
        elapsed=elapsed,
    )


def load_model(
    model_name: str, transport: httpx.BaseTransport | None
) -> tuple[LLMClient, AgentConfig]:
    """Build the client from ``data/models.yaml`` so runs use real credentials."""
    store = YamlModelStore()
    model: ModelConfig = store.get(model_name)
    agent = build_agent_config(model_name)
    return build_client(model, agent, transport=transport), agent


def run(
    corpora: dict[str, Corpus],
    questions: list[Question],
    client: LLMClient,
    *,
    corpus_filter: list[str] | None = None,
    top_k: int | None = None,
    max_context_tokens: int | None = None,
    on_case: Callable[[Case], None] | None = None,
) -> list[Case]:
    """Ask every question in both modes, one retriever per corpus (loaded once)."""
    retrievers: dict[str, Retriever | None] = {}
    cases: list[Case] = []
    for index, question in enumerate(questions, start=1):
        if corpus_filter and question.corpus not in corpus_filter:
            continue
        if question.corpus not in retrievers:
            corpus = corpora[question.corpus]
            kwargs: dict[str, object] = {}
            if top_k is not None:
                kwargs["top_k"] = top_k
            if max_context_tokens is not None:
                kwargs["max_context_tokens"] = max_context_tokens
            retrievers[question.corpus] = Retriever(corpus.index_path, **kwargs)
        for mode in MODES:
            case = run_case(
                client,
                question,
                mode,
                index,
                retrievers[question.corpus] if mode == "rag" else None,
            )
            cases.append(case)
            if on_case is not None:
                on_case(case)
    return cases


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _tick(value: bool | None) -> str:
    if value is None:
        return "—"
    return "да" if value else "нет"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".lstrip("0")


def render_report(
    corpora: dict[str, Corpus],
    questions: list[Question],
    cases: list[Case],
    summary: dict[str, object],
    *,
    model_name: str,
    top_k: int | None,
    max_context_tokens: int | None,
    temperature: float | None = None,
) -> str:
    by_index = {q_index: q for q_index, q in enumerate(questions, start=1)}
    sampling = (
        f"temperature={temperature:g}"
        if temperature is not None
        else "детерминированная оценка по сохранённым ответам"
    )
    lines: list[str] = [
        "# RAG против модели без RAG",
        "",
        "Каждый контрольный вопрос задан дважды: с найденным контекстом "
        "(`rag`) и без него (`no_rag`). Оценки детерминированные — подстрочный "
        "поиск ожиданий в сохранённых ответах, поэтому повторный прогон по JSON "
        "даёт те же числа.",
        "",
        "## Настройки",
        "",
        f"- Модель: `{model_name}`",
        f"- Режим: оба, {sampling}",
    ]
    if top_k is not None:
        lines.append(f"- Чанков на вопрос (top_k): {top_k}")
    if max_context_tokens is not None:
        lines.append(f"- Бюджет контекста, токенов: {max_context_tokens}")
    for corpus in corpora.values():
        lines.append(
            f"- Корпус `{corpus.name}`: индекс `{corpus.index_path}`, "
            f"документы `{corpus.input_dir}` — {corpus.label}"
        )

    lines += [
        "",
        "## Итоги",
        "",
        "| Режим | Вопросов | Покрытие фактов | Полных | Отказов | "
        "Попадание в выдачу | Цитации | PASS | prompt-токенов | с/вопрос |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        row = summary.get(mode)
        if not row:
            continue
        lines.append(
            f"| {'с RAG' if mode == 'rag' else 'без RAG'} "
            f"| {row['questions']} "
            f"| {_pct(row['fact_coverage_mean'])} "
            f"| {row['fact_coverage_full']}/{row['answerable']} "
            f"| {_pct(row['refusal_rate'])} "
            f"| {_pct(row['retrieval_hit_rate'])} "
            f"| {_pct(row['citation_rate'])} "
            f"| {row['passed']}/{row['questions']} ({_pct(row['pass_rate'])}) "
            f"| {row['prompt_tokens_mean']} "
            f"| {row['elapsed_mean']} |"
        )

    lines += [
        "",
        "«Покрытие фактов» — доля ожидаемых строк, найденных в ответе, по "
        "вопросам с ответом. «Отказы» — доля честных «не нашёл» по вопросам, "
        "которых в базе нет. Они считаются раздельно: усреднять их в одно "
        "число нельзя, сильный отказ легко спрятал бы слабые ответы.",
        "",
        "## По вопросам",
        "",
        "| # | Вопрос | Режим | Покрытие | Выдача | Цитация | Отказ | PASS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in sorted(cases, key=lambda c: (c.index, c.mode)):
        question = by_index[case.index]
        score = scored(case, question)
        files = ", ".join(case.retrieved[:2]) or "—"
        lines.append(
            f"| {case.index} "
            f"| {question.text[:60]}{'…' if len(question.text) > 60 else ''} "
            f"| {'rag' if case.mode == 'rag' else 'no_rag'} "
            f"| {_pct(float(score['fact_coverage']))} "
            f"| {_tick(score['retrieval_hit'] if isinstance(score['retrieval_hit'], bool) else None)} ({files}) "
            f"| {_tick(score['citation_hit'] if isinstance(score['citation_hit'], bool) else None)} "
            f"| {_tick(score['refused'] if isinstance(score['refused'], bool) else None)} "
            f"| {_tick(bool(score['passed']))} |"
        )

    lines += ["", "## Ошибки", ""]
    failed = [c for c in cases if c.error]
    if failed:
        lines.append(f"Не удалось получить ответ: {len(failed)} из {len(cases)}.")
        for case in failed:
            lines.append(f"- #{case.index} [{case.mode}] {case.error}")
    else:
        lines.append("Все запросы прошли, ошибок провайдера нет.")

    lines += [
        "",
        "## Как читать",
        "",
        "- «Выдача» показывает, дошёл ли до промпта чанк с нужным ответом, а не "
        "просто файл: сверяется пересечение строк с `sources` из YAML. Промах "
        "здесь — проблема поиска (чанк не нашёлся или упал за бюджетом), а не "
        "модели. Раньше сверка шла по имени файла, и вопрос, чей ответ в промпт "
        "не попал, проходил как успешный.",
        "- «Цитация» — упомянул ли ответ источник в формате `файл:строки`, "
        "который требовал блок контекста.",
        "- У неответимых вопросов ожиданий нет: PASS означает отказ, а любое "
        "конкретное число в ответе — конфабуляцию.",
        "- Полные ответы и события поиска — в `*.json` рядом с этим отчётом.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сравнить ответы модели с RAG и без него.",
    )
    parser.add_argument(
        "--questions",
        default="rag_questions.example.yaml",
        help="YAML с контрольными вопросами (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ключ модели из data/models.yaml (default: RAG_COMPARE_MODEL из "
        "окружения, иначе leanstral).",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpora",
        help="Ограничить прогон одним корпусом (можно указать несколько раз).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Сколько чанков доставать на вопрос (default: как в Retriever).",
    )
    parser.add_argument(
        "--rag-max-tokens",
        type=int,
        default=None,
        help="Бюджет блока контекста в токенах.",
    )
    parser.add_argument(
        "--out",
        default="results/rag_comparison.md",
        help="Markdown-отчёт (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="JSON с ответами (default: тот же путь с расширением .json).",
    )
    parser.add_argument(
        "--from_json",
        default=None,
        help="Пересчитать метрики по сохранённому JSON, не обращаясь к API.",
    )
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="Печатать полные ответы в stdout.",
    )
    return parser


def save_json(
    path: Path,
    corpora: dict[str, Corpus],
    questions: list[Question],
    cases: list[Case],
    summary: dict[str, object],
    **meta: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **meta,
                "corpora": {
                    name: {
                        "label": c.label,
                        "index_path": str(c.index_path),
                        "input_dir": str(c.input_dir),
                    }
                    for name, c in corpora.items()
                },
                "questions": [
                    {
                        "index": index,
                        "text": q.text,
                        "corpus": q.corpus,
                        "answerable": q.answerable,
                        "expect": list(q.expect),
                        "sources": list(q.sources),
                    }
                    for index, q in enumerate(questions, start=1)
                ],
                "cases": [
                    {
                        **vars(case),
                        "score": scored(
                            case,
                            next(
                                q
                                for i, q in enumerate(questions, start=1)
                                if i == case.index
                            ),
                        ),
                    }
                    for case in cases
                ],
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_saved_cases(path: Path) -> tuple[list[Question], list[Case]]:
    """Rebuild questions and cases from a saved run, for offline re-scoring."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        Question(
            text=str(entry["text"]),
            corpus=str(entry.get("corpus", "")),
            expect=tuple(entry.get("expect") or ()),
            sources=tuple(entry.get("sources") or ()),
            answerable=bool(entry.get("answerable", True)),
        )
        for entry in payload["questions"]
    ]
    cases = [
        Case(
            index=int(entry["index"]),
            question=str(entry.get("question", "")),
            corpus=str(entry.get("corpus", "")),
            mode=str(entry["mode"]),
            answerable=bool(entry.get("answerable", True)),
            answer=str(entry.get("answer", "")),
            retrieved=tuple(entry.get("retrieved") or ()),
            context_tokens=int(entry.get("context_tokens", 0)),
            prompt_tokens=int(entry.get("prompt_tokens", 0)),
            elapsed=float(entry.get("elapsed", 0.0)),
            error=str(entry.get("error", "")),
        )
        for entry in payload["cases"]
    ]
    return questions, cases


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions_path = Path(args.questions)

    if args.from_json:
        saved_questions, cases = load_saved_cases(Path(args.from_json))
        # Grade with the YAML on disk when it is there: ``sources`` moved from
        # "any chunk of this file" to "a chunk overlapping these lines", and
        # rescoring against the copies frozen inside the JSON silently reported
        # the old verdict.
        questions = saved_questions
        if questions_path.is_file():
            corpora, fresh = load_questions(questions_path)
            if len(fresh) != len(saved_questions):
                print(
                    f"warn: в {questions_path} вопросов {len(fresh)}, а в "
                    f"{args.from_json} — {len(saved_questions)}; считаю по JSON",
                    file=sys.stderr,
                )
            else:
                questions = fresh
        else:
            print(
                f"warn: {questions_path} не найден — считаю по вопросам из "
                f"{args.from_json}",
                file=sys.stderr,
            )
        corpora = {
            "repo": Corpus("repo", "Документы репозитория", Path(), Path(), (".md",)),
            "kb": Corpus("kb", "База знаний кофейни «Зёрна»", Path(), Path(), (".md",)),
        }
        summary = aggregate(cases, questions)
        report = render_report(
            corpora,
            questions,
            cases,
            summary,
            model_name="(из сохранённого прогона)",
            top_k=None,
            max_context_tokens=None,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"\nОтчёт: {out}")
        return 0

    try:
        corpora, questions = load_questions(questions_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.corpora:
        unknown = [name for name in args.corpora if name not in corpora]
        if unknown:
            print(
                f"error: неизвестный корпус: {', '.join(unknown)} "
                f"(в YAML есть: {', '.join(corpora)})",
                file=sys.stderr,
            )
            return 2
        wanted = {name: corpora[name] for name in args.corpora}
    else:
        wanted = corpora
    # Checked before the model: a missing index is a configuration error the
    # user has to fix, and reporting it as a 0% score would bury it under a
    # table of numbers that looks like a real result.
    missing = [c.index_path for c in wanted.values() if not c.index_path.is_file()]
    if missing:
        for path in missing:
            print(
                f"error: индекс не найден: {path}\n"
                f"  соберите его: python scripts/index_documents.py "
                f"--input_dir <docs> --index_dir <dir> --strategy structure",
                file=sys.stderr,
            )
        return 2

    model_name = args.model or os.environ.get("RAG_COMPARE_MODEL") or "leanstral"
    try:
        client, agent = load_model(model_name, None)
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"error: не удалось собрать модель {model_name!r}: {exc} "
            f"(проверьте data/models.yaml и .env)",
            file=sys.stderr,
        )
        return 2

    print(f"Модель: {model_name} | вопросов: {len(questions)} | режимов: 2")
    started = time.perf_counter()
    cases = run(
        corpora,
        questions,
        client,
        corpus_filter=args.corpora,
        top_k=args.top_k,
        max_context_tokens=args.rag_max_tokens,
        on_case=lambda case: print(
            f"  #{case.index} [{case.mode:>6}] "
            f"{'ошибка: ' + case.error if case.error else 'ок'}"
        ),
    )
    summary = aggregate(cases, questions)

    out = Path(args.out)
    json_path = Path(args.json_path) if args.json_path else out.with_suffix(".json")
    save_json(
        json_path,
        corpora,
        questions,
        cases,
        summary,
        model=model_name,
        top_k=args.top_k,
        rag_max_tokens=args.rag_max_tokens,
        elapsed=round(time.perf_counter() - started, 2),
    )
    report = render_report(
        corpora,
        questions,
        cases,
        summary,
        model_name=model_name,
        top_k=args.top_k,
        max_context_tokens=args.rag_max_tokens,
        temperature=agent.temperature,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print()
    print(report)
    if args.show_answers:
        print("## Полные ответы\n")
        for case in sorted(cases, key=lambda c: (c.index, c.mode)):
            print(f"### #{case.index} [{case.mode}] {case.question}\n")
            print(case.answer.strip() or "(пусто)")
            print()
    print(f"Отчёт: {out}\nJSON:  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())