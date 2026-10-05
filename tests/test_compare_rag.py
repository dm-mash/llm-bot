"""Tests for scripts/compare_rag.py: loading, scoring, and the two-mode run.

All offline: the LLM is an ``httpx.MockTransport`` and the retrieval is the
day-21 ``FakeEmbedder``, so no model is downloaded and no network is touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "compare_rag", ROOT / "scripts" / "compare_rag.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses look up ``cls.__module__`` in
    # sys.modules while building the class, so an unregistered module breaks.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp_rag = _load_script()

from llm_bot import rag as ragmod  # noqa: E402
from tests.test_index_documents import FakeEmbedder  # noqa: E402
from tests.test_rag import make_index  # noqa: E402


QUESTIONS_YAML = """
corpora:
  kb:
    index_dir: {index_dir}
    input_dir: .
    extensions: .md
    label: База про тесты
questions:
  - text: Часы работы кофейни
    corpus: kb
    answerable: true
    expect: ["9:00"]
    sources: ["kb/hours.md"]
  - text: Сколько стоит латте на кокосовом молоке
    corpus: kb
    answerable: false
    expect: []
    sources: []
"""


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    index_dir = tmp_path / "kb_emb"
    make_index(
        index_dir,
        [
            ("kb/hours.md", "Часы работы: с 9:00 до 21:00.", "Часы работы"),
            ("kb/menu.md", "Крем: 190 ₽.", "Напитки"),
        ],
    )
    payload = QUESTIONS_YAML.format(index_dir=index_dir)
    path = tmp_path / "questions.yaml"
    path.write_text(payload, encoding="utf-8")
    return path


def _scripted_client(replies: dict[str, str], captured: list[dict]) -> httpx.Client:
    """A client whose answers depend on whether a RAG block was sent."""
    state = {"turn": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        captured.append(payload)
        state["turn"] += 1
        text = replies.get(str(state["turn"]), "заглушка")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": text}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return handler  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_load_questions_reads_corpora_and_questions(questions_file: Path) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    assert set(corpora) == {"kb"}
    assert corpora["kb"].index_path.name == "index_structure.json"
    assert corpora["kb"].label == "База про тесты"
    assert len(questions) == 2
    assert questions[0].expect == ("9:00",)
    assert questions[0].answerable is True
    assert questions[1].answerable is False


def test_a_question_defaults_to_answerable() -> None:
    """A forgotten ``answerable`` flag must fail strict, not land in the
    lenient refusal bucket and quietly improve the score."""
    entry = cmp_rag.Question(
        text="q", corpus="kb", expect=("190 ₽",), sources=(), answerable=True
    )
    assert entry.answerable is True


def test_load_questions_rejects_a_broken_file(tmp_path: Path) -> None:
    cases = {
        "no_corpora": "questions: []\n",
        "unknown_corpus": (
            "corpora:\n  kb: {index_dir: d}\nquestions:\n  - {text: q, corpus: zz}\n"
        ),
        "no_text": "corpora:\n  kb: {index_dir: d}\nquestions:\n  - {corpus: kb}\n",
        "empty": "corpora:\n  kb: {index_dir: d}\nquestions: []\n",
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            cmp_rag.load_questions(path)
    with pytest.raises(FileNotFoundError):
        cmp_rag.load_questions(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_fact_coverage_counts_expected_strings() -> None:
    coverage, hit, missed = cmp_rag.fact_coverage(
        "Мы с 9:00 до 21:00, вход свободный", ("9:00", "21:00")
    )
    assert coverage == 1.0
    assert hit == ["9:00", "21:00"]
    assert missed == []


def test_fact_coverage_is_case_and_whitespace_insensitive() -> None:
    coverage, _, _ = cmp_rag.fact_coverage("С   9:00", ("с 9:00",))
    assert coverage == 1.0


def test_fact_coverage_reports_what_was_missed() -> None:
    coverage, hit, missed = cmp_rag.fact_coverage(
        "Цена 190 рублей", ("190 ₽", "300 мл")
    )
    assert coverage == 0.0
    assert hit == []
    assert missed == ["190 ₽", "300 мл"]


def test_fact_coverage_without_expectations_is_full() -> None:
    assert cmp_rag.fact_coverage("что угодно", ())[0] == 1.0


def test_cited_sources_reads_file_line_ranges() -> None:
    found = cmp_rag.cited_sources("по [kb/hours.md:11-15], см. kb/menu.md:8-17")
    assert found == {"kb/hours.md", "kb/menu.md"}


def test_cited_sources_accepts_cyrillic_and_spaces_in_a_pdf_name() -> None:
    """The scanned names hold Cyrillic, spaces and dots.

    A pattern limited to ``[\\w./-]`` or to ``.md`` returned nothing for every
    one of them, so the citation score read 0.000 while retrieval was healthy —
    a broken metric that looks exactly like a broken answer.
    """
    answer = (
        "Занятие на 20 минут "
        "[ID-250-248-463-709 Отчёт Б.pdf:12-38]"
    )
    assert cmp_rag.cited_sources(answer) == {
        "ID-250-248-463-709 Отчёт Б.pdf"
    }


def test_cited_sources_score_a_cyrillic_name_against_a_source() -> None:
    """The parsed name must intersect the expected source, not just look like one."""
    sources = ["ID-250-905-682-540 Занятие_60мин.pdf:5-12"]
    answer = "(ID-250-905-682-540 Занятие_60мин.pdf:5-12)"
    assert cmp_rag.cited_sources(answer) & cmp_rag.source_files(sources)


def test_cited_sources_ignores_a_page_suffix_after_the_range() -> None:
    """``file.pdf:12-38 — page 1`` must not swallow the page label."""
    answer = "[ID-250-801-540-383 Занятие.pdf:1-9 — page 1]"
    assert cmp_rag.cited_sources(answer) == {
        "ID-250-801-540-383 Занятие.pdf"
    }


def test_cited_sources_ignores_prose_and_prices() -> None:
    assert cmp_rag.cited_sources("Стоит 190 руб, точной цифры нет") == set()
    assert cmp_rag.cited_sources("в файле data/docs/ нет ничего") == set()


def test_cited_sources_reads_other_indexed_extensions() -> None:
    found = cmp_rag.cited_sources(
        "[llm_bot/rag.py:12-34] и scripts/index_documents.py:300-320"
    )
    assert found == {"llm_bot/rag.py", "scripts/index_documents.py"}


def test_refusal_detection() -> None:
    assert ragmod.REFUSAL_RE.search("Я не нашёл это в базе")
    assert ragmod.REFUSAL_RE.search("Информации об этом нет")
    assert not ragmod.REFUSAL_RE.search("Стоит 190 ₽")


def _question(**kwargs) -> object:
    base = {
        "text": "q",
        "corpus": "kb",
        "expect": ("190 ₽",),
        "sources": ("kb/menu.md",),
        "answerable": True,
    }
    base.update(kwargs)
    return cmp_rag.Question(**base)


def test_scoring_separates_retrieval_from_reading() -> None:
    question = _question()
    # The expected source reached the prompt but the answer still lacks the fact:
    # the reading failed, not the search.
    case = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=True,
        answer="Кофе стоит 180 ₽",
        retrieved=("kb/menu.md:1-5",),
    )
    score = cmp_rag.scored(case, question)
    assert score["retrieval_hit"] is True
    assert score["fact_coverage"] == 0.0
    assert score["passed"] is False

    # Conversely: the model knew the fact but the chunk never arrived.
    case2 = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="no_rag", answerable=True,
        answer="Это стоит 190 ₽",
        retrieved=(),
    )
    score2 = cmp_rag.scored(case2, question)
    assert score2["retrieval_hit"] is False
    assert score2["fact_coverage"] == 1.0


def test_scoring_requires_a_citation_when_a_source_is_known() -> None:
    question = _question()
    grounded = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=True,
        answer="Стоит 190 ₽ [kb/menu.md:1-5]",
        retrieved=("kb/menu.md:1-5",),
    )
    assert cmp_rag.scored(grounded, question)["citation_hit"] is True
    ungrounded = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=True,
        answer="Стоит 190 ₽",
        retrieved=("kb/menu.md:1-5",),
    )
    assert cmp_rag.scored(ungrounded, question)["citation_hit"] is False


def test_an_unanswerable_question_passes_only_on_refusal() -> None:
    question = _question(answerable=False, expect=(), sources=())
    refusal = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=False,
        answer="Я не нашёл такого напитка в базе",
    )
    assert cmp_rag.scored(refusal, question)["passed"] is True
    fabrication = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="no_rag", answerable=False,
        answer="Кокосовый латте стоит 280 ₽",
    )
    assert cmp_rag.scored(fabrication, question)["passed"] is False


def test_an_error_never_passes() -> None:
    question = _question()
    failed = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=True,
        answer="Стоит 190 ₽", error="429", retrieved=("kb/menu.md:1-5",),
    )
    assert cmp_rag.scored(failed, question)["passed"] is False


def test_aggregate_keeps_quality_and_honesty_apart() -> None:
    questions = [
        _question(),
        _question(answerable=False, expect=(), sources=()),
    ]
    # The grounded arm answers in the format the protocol asks for and carries
    # the source list the code appends, so it is a genuinely backed answer and
    # earns the 1.0 that a bare line-range citation used to earn by default.
    rag = [
        cmp_rag.Case(index=1, question="q", corpus="kb", mode="rag",
                     answerable=True,
                     answer='190 ₽ [1] «Крем: 190 ₽»\n\n'
                            + ragmod.SOURCES_HEADING
                            + "\n[1] kb/menu.md:1-5 — Напитки",
                     retrieved=("kb/menu.md:1-5",),
                     chunk_texts=("Крем: 190 ₽, объём 300 мл.",),
                     prompt_tokens=100),
        cmp_rag.Case(index=2, question="q", corpus="kb", mode="rag",
                     answerable=False, answer="не нашёл", prompt_tokens=120),
    ]
    plain = [
        cmp_rag.Case(index=1, question="q", corpus="kb", mode="no_rag",
                     answerable=True, answer="190 ₽", prompt_tokens=20),
        cmp_rag.Case(index=2, question="q", corpus="kb", mode="no_rag",
                     answerable=False, answer="280 ₽", prompt_tokens=20),
    ]
    summary = cmp_rag.aggregate(rag + plain, questions)
    assert summary["rag"]["fact_coverage_mean"] == 1.0
    assert summary["rag"]["refusal_rate"] == 1.0
    assert summary["rag"]["pass_rate"] == 1.0
    assert summary["no_rag"]["fact_coverage_mean"] == 1.0
    # The ungrounded mode got the fact right but invented the second answer.
    assert summary["no_rag"]["refusal_rate"] == 0.0
    assert summary["no_rag"]["pass_rate"] == 0.5
    # RAG costs tokens: that is the price of grounding.
    assert summary["rag"]["prompt_tokens_mean"] > summary["no_rag"]["prompt_tokens_mean"]


# --------------------------------------------------------------------------- #
# Running both modes
# --------------------------------------------------------------------------- #


def _client(replies: dict[str, str], captured: list[dict]):
    handler = _scripted_client(replies, captured)
    from llm_bot.config import LLMConfig

    from llm_bot.client import LLMClient

    config = LLMConfig(
        base_url="https://example.test/v1", api_key="k", model="m",
        temperature=0.0, max_tokens=64,
        default_system_prompt="Отвечай на том же языке, на котором написан вопрос.",
    )
    return LLMClient(config, transport=httpx.MockTransport(handler))


def test_run_asks_every_question_in_both_modes(
    questions_file: Path, tmp_path: Path
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    index_path = corpora["kb"].index_path

    # Force the fake embedder so nothing is downloaded.
    original = cmp_rag.Retriever
    cmp_rag.Retriever = lambda path, **kw: original(
        path, embedder=FakeEmbedder("fake-mini"), **kw
    )
    captured: list[dict] = []
    client = _client(
        {
            # Answers in the format the protocol now asks for: a chunk number and
            # the phrase copied out of that chunk. ``finalize_answer`` runs on
            # these exactly as it runs on a live reply, so the run exercises the
            # source list and both audits rather than bypassing them.
            "1": 'Мы с 9:00 до 21:00 [1] «Часы работы: с 9:00 до 21:00»',
            "2": "С 9:00 до 21:00",
            "3": "не нашёл",
            "4": "250 ₽",
        },
        captured,
    )
    try:
        cases = cmp_rag.run(corpora, questions, client)
    finally:
        cmp_rag.Retriever = original

    assert len(cases) == 4
    assert [c.mode for c in cases] == ["rag", "no_rag", "rag", "no_rag"]
    summary = cmp_rag.aggregate(cases, questions)
    assert summary["rag"]["pass_rate"] == 1.0
    # The baseline knows the opening hours (fact coverage 1.0) but invents a
    # price for the drink that is not on the menu — exactly the asymmetry the
    # control set is built to expose.
    assert summary["no_rag"]["fact_coverage_mean"] == 1.0
    assert summary["no_rag"]["refusal_rate"] == 0.0
    assert summary["no_rag"]["pass_rate"] == 0.5

    # The RAG request carries a grounding prefix; the plain one does not.
    rag_payloads = [p for p in captured if any(
        m["role"] == "system" and "локальной базы" in m["content"] for m in p["messages"]
    )]
    plain_payloads = [p for p in captured if not any(
        m["role"] == "system" and "локальной базы" in m["content"] for m in p["messages"]
    )]
    assert len(rag_payloads) == 2
    assert len(plain_payloads) == 2
    # The user's question reaches the model unchanged in both modes.
    for payload in captured:
        assert payload["messages"][-1]["role"] == "user"


def test_run_records_a_retrieval_failure_instead_of_crashing(
    questions_file: Path,
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)

    class Broken:
        def render(self, question):
            raise RuntimeError("index unreadable")

    captured: list[dict] = []
    client = _client({"1": "ок"}, captured)
    case = cmp_rag.run_case(client, questions[0], "rag", 1, Broken())
    assert case.error.startswith("retrieval:")
    assert case.answer == ""


def test_run_can_filter_by_corpus(questions_file: Path) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    captured: list[dict] = []
    client = _client({}, captured)
    cases = cmp_rag.run(corpora, questions, client, corpus_filter=["kb"])
    assert cases  # kb is the only corpus, so nothing is filtered out
    assert all(c.corpus == "kb" for c in cases)
    cases = cmp_rag.run(corpora, questions, client, corpus_filter=["repo"])
    assert cases == []


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_report_mentions_both_modes_and_the_honesty_axis(
    questions_file: Path,
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="rag",
                     answerable=True, answer="С 9:00 [kb/hours.md:11-15]",
                     retrieved=("kb/hours.md:11-15",), prompt_tokens=90,
                     context_tokens=40),
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="no_rag",
                     answerable=True, answer="С 9:00", prompt_tokens=10),
        cmp_rag.Case(index=2, question=questions[1].text, corpus="kb", mode="rag",
                     answerable=False, answer="не нашёл", prompt_tokens=70),
        cmp_rag.Case(index=2, question=questions[1].text, corpus="kb", mode="no_rag",
                     answerable=False, answer="250 ₽", prompt_tokens=10),
    ]
    summary = cmp_rag.aggregate(cases, questions)
    report = cmp_rag.render_report(
        corpora, questions, cases, summary,
        model_name="fake", top_k=4, max_context_tokens=None,
    )
    assert "RAG против модели без RAG" in report
    assert "с RAG" in report and "без RAG" in report
    # The two axes stay separate in the table.
    assert "Покрытие фактов" in report and "Отказов" in report
    assert "fake" in report
    # Per-question rows for all four cases.
    assert report.count("| rag |") + report.count("| no_rag |") == 4
    # An error row is reported rather than hidden.
    assert "Ошибки" in report


def test_report_flags_errors(questions_file: Path) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="rag",
                     answerable=True, answer="", error="429 rate limited"),
    ]
    summary = cmp_rag.aggregate(cases, questions[:1])
    report = cmp_rag.render_report(
        corpora, questions[:1], cases, summary, model_name="fake",
        top_k=None, max_context_tokens=None,
    )
    assert "429 rate limited" in report
    assert "Не удалось получить ответ" in report


def test_every_declared_source_span_actually_contains_its_facts() -> None:
    """The control set must not drift away from the documents it grades.

    Each answerable question declares ``file:start-end`` and the substrings that
    have to appear there. If an edit shifts the lines, the span goes stale and
    the question silently starts failing for the wrong reason — which is exactly
    the confusion the retrieval fix is meant to remove.
    """
    config = yaml.safe_load(
        (ROOT / "rag_questions.example.yaml").read_text(encoding="utf-8")
    )
    roots = {
        name: ROOT / spec["input_dir"]
        for name, spec in config["corpora"].items()
    }
    checked = 0
    for index, question in enumerate(config["questions"], start=1):
        if not question.get("expect") or not question.get("sources"):
            continue
        for source in question["sources"]:
            name, _, span = source.partition(":")
            start, _, end = span.partition("-")
            path = roots[question["corpus"]] / name
            assert path.is_file(), f"#{index}: нет файла {path}"
            lines = path.read_text(encoding="utf-8").splitlines()
            text = "\n".join(lines[int(start) - 1 : int(end)])
            plain = text.lower().replace("*", "")
            for fact in question["expect"]:
                assert fact.lower().replace("*", "") in plain, (
                    f"#{index}: факт {fact!r} не лежит в {source}"
                )
            checked += 1
    assert checked == 8


def test_citation_check_ignores_the_line_range_in_sources() -> None:
    """``sources`` carries lines now; a citation names only the file."""
    assert cmp_rag.source_files(["README.md:1304-1310", "kb/menu.md"]) == {
        "README.md", "kb/menu.md"
    }
    question = cmp_rag.Question(
        text="q", corpus="repo", expect=("128",),
        sources=("README.md:1304-1310",), answerable=True,
    )
    case = cmp_rag.Case(
        index=1, question="q", corpus="repo", mode="rag", answerable=True,
        answer="окно 128 токенов [README.md:1304-1310]",
        retrieved=("README.md:1304-1310",),
    )
    score = cmp_rag.scored(case, question)
    assert score["citation_hit"] is True
    assert score["cited_files"] == ["README.md"]
    # A citation to some other file is still not a hit.
    other = cmp_rag.Case(
        index=1, question="q", corpus="repo", mode="rag", answerable=True,
        answer="окно 128 токенов [kb/menu.md:1-9]",
        retrieved=("README.md:1304-1310",),
    )
    assert cmp_rag.scored(other, question)["citation_hit"] is False


def test_retrieval_hit_demands_the_chunk_not_just_the_file() -> None:
    """Three unrelated chunks of the right file must not count as a hit.

    Question 1 scored "выдача: да" because ``sources`` was checked at file level:
    the retriever returned three unrelated ``README.md`` chunks while the chunk
    holding the answer was never in the prompt. A metric that flatters a
    retrieval failure is worse than a visible miss.
    """
    right_file = ("README.md:949-951", "README.md:995-1000", "tasks/day22.md:22-22")
    assert cmp_rag.source_reached(["README.md"], right_file) is True
    assert cmp_rag.source_reached(["README.md:1304-1310"], right_file) is False
    assert cmp_rag.source_reached(["README.md:1304-1310"],
                                  ["README.md:1304-1310"]) is True
    # Overlap, not equality: a neighbouring chunk that shares the span counts.
    assert cmp_rag.source_reached(["README.md:1304-1310"],
                                  ["README.md:1300-1306"]) is True
    assert cmp_rag.source_reached(["README.md:1304-1310"],
                                  ["README.md:1305-1311"]) is True
    assert cmp_rag.source_reached(["README.md:1304-1310"],
                                  ["README.md:1311-1318"]) is False
    assert cmp_rag.source_reached(["README.md"], []) is False
    # Same file, different document entirely.
    assert cmp_rag.source_reached(["plans/compression.md:69-77"],
                                  ["README.md:69-77"]) is False


def test_a_wrong_chunk_fails_the_question_even_when_the_facts_are_right() -> None:
    """Coverage and retrieval are separate verdicts; both must hold to pass."""
    question = cmp_rag.Question(
        text="q", corpus="repo", expect=("paraphrase-multilingual",),
        sources=("README.md:1304-1310",), answerable=True,
    )
    # A grounded answer as the runtime now produces it: the source list the code
    # appends, and a phrase quoted from the chunk the citation points at. A
    # hand-written answer with neither cannot pass, and that is the point — this
    # fixture used to pass on a bare line-range citation, which is the number
    # day24 exists to correct.
    grounded = (
        "paraphrase-multilingual-MiniLM-L12-v2 [1] «paraphrase-multilingual-"
        "MiniLM-L12-v2»\n\n" + ragmod.SOURCES_HEADING
        + "\n[1] README.md:1304-1310 — Модели"
    )
    with_answer = cmp_rag.Case(
        index=1, question="q", corpus="repo", mode="rag", answerable=True,
        answer=grounded,
        retrieved=("README.md:1304-1310",),
        chunk_texts=("Модели: paraphrase-multilingual-MiniLM-L12-v2.",),
    )
    without_answer = cmp_rag.Case(
        index=1, question="q", corpus="repo", mode="rag", answerable=True,
        answer=grounded,
        retrieved=("README.md:949-951",),
        chunk_texts=("Установка: pip install llm-bot.",),
    )
    assert cmp_rag.scored(with_answer, question)["retrieval_hit"] is True
    assert cmp_rag.scored(with_answer, question)["passed"] is True

    score = cmp_rag.scored(without_answer, question)
    assert score["fact_coverage"] == 1.0   # the fact is there
    assert score["retrieval_hit"] is False  # but not from the retrieved chunk
    assert score["passed"] is False


def test_markdown_emphasis_does_not_break_expectation_matching() -> None:
    """A model that quotes the source formatting must not be scored as a miss.

    The document says ``**до** приготовления`` and the model reproduces the bold
    markers, so the expected substring ``до приготовления`` is split by ``**``.
    That is a formatting difference, not a wrong fact.
    """
    coverage, hit, missed = cmp_rag.fact_coverage(
        "Бариста предупреждает **до** приготовления, а не после.", ("до приготовления",)
    )
    assert coverage == 1.0
    assert hit == ["до приготовления"]
    assert missed == []
    # ...and the strip must not make a genuinely absent fact present.
    assert cmp_rag.fact_coverage("после приготовления", ("до приготовления",))[0] == 0.0


def test_report_states_the_temperature_actually_used(questions_file: Path) -> None:
    """A hardcoded "temperature=0.0" would misreport a run that used 0.1."""
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [cmp_rag.Case(index=1, question=questions[0].text, corpus="kb",
                          mode="rag", answerable=True, answer="С 9:00")]
    report = cmp_rag.render_report(
        corpora, questions[:1], cases, cmp_rag.aggregate(cases, questions[:1]),
        model_name="leanstral", top_k=None, max_context_tokens=None,
        temperature=0.1,
    )
    assert "temperature=0.1" in report
    assert "temperature=0.0" not in report


def test_saved_json_can_be_rescored_offline(tmp_path: Path, questions_file: Path) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="rag",
                     answerable=True, answer="С 9:00 [kb/hours.md:11-15]",
                     retrieved=("kb/hours.md:11-15",)),
        cmp_rag.Case(index=2, question=questions[1].text, corpus="kb", mode="rag",
                     answerable=False, answer="не нашёл"),
    ]
    summary = cmp_rag.aggregate(cases, questions)
    path = tmp_path / "run.json"
    cmp_rag.save_json(path, corpora, questions, cases, summary, model="fake")
    assert path.is_file()

    reloaded_questions, reloaded_cases = cmp_rag.load_saved_cases(path)
    assert len(reloaded_questions) == len(questions)
    assert len(reloaded_cases) == len(cases)
    # Re-scoring a saved run reproduces the numbers: no API, same verdict.
    again = cmp_rag.aggregate(reloaded_cases, reloaded_questions)
    assert again["rag"]["pass_rate"] == summary["rag"]["pass_rate"]


def test_saved_json_contains_the_answers_and_scores(
    tmp_path: Path, questions_file: Path
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="rag",
                     answerable=True, answer="С 9:00 [kb/hours.md:11-15]",
                     retrieved=("kb/hours.md:11-15",)),
    ]
    path = tmp_path / "run.json"
    cmp_rag.save_json(path, corpora, questions, cases,
                      cmp_rag.aggregate(cases, questions[:1]), model="fake")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"] == "fake"
    assert payload["cases"][0]["answer"] == "С 9:00 [kb/hours.md:11-15]"
    assert payload["cases"][0]["retrieved"] == ["kb/hours.md:11-15"]
    assert payload["cases"][0]["score"]["citation_hit"] is True
    assert payload["questions"][0]["expect"] == ["9:00"]


def test_main_rescores_from_json_without_any_client(
    tmp_path: Path, questions_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    cases = [
        cmp_rag.Case(index=1, question=questions[0].text, corpus="kb", mode="rag",
                     answerable=True, answer="С 9:00 [kb/hours.md:11-15]",
                     retrieved=("kb/hours.md:11-15,")),
        cmp_rag.Case(index=2, question=questions[1].text, corpus="kb", mode="rag",
                     answerable=False, answer="не нашёл"),
    ]
    saved = tmp_path / "run.json"
    cmp_rag.save_json(saved, corpora, questions, cases,
                      cmp_rag.aggregate(cases, questions), model="fake")
    out = tmp_path / "rescored.md"

    def explode(*args, **kwargs):
        raise AssertionError("--from_json must not touch the provider")

    monkeypatch.setattr(cmp_rag, "load_model", explode)
    code = cmp_rag.main([
        "--questions", str(questions_file), "--from_json", str(saved),
        "--out", str(out),
    ])
    assert code == 0
    assert "RAG против модели без RAG" in out.read_text(encoding="utf-8")


def test_rescoring_uses_the_edited_yaml_not_the_copy_inside_the_json(
    tmp_path: Path, questions_file: Path
) -> None:
    """Editing ``sources`` must change the verdict on the next rescore.

    The JSON stores the questions it ran with. Grading with those copies means a
    corrected control set looks unchanged: the old file-level ``sources`` were
    restored silently and question 1 kept reporting a retrieval hit for three
    unrelated chunks.
    """
    corpora, questions = cmp_rag.load_questions(questions_file)
    loose = cmp_rag.Question(
        text=questions[0].text, corpus="kb", expect=questions[0].expect,
        sources=("kb/menu.md",), answerable=True,
    )
    strict = cmp_rag.Question(
        text=questions[0].text, corpus="kb", expect=questions[0].expect,
        sources=("kb/menu.md:40-44",), answerable=True,
    )
    retrieved = ("kb/menu.md:1-3", "kb/other.md:9-11")
    cases = [
        cmp_rag.Case(index=1, question=loose.text, corpus="kb", mode="rag",
                     answerable=True, answer="С 9:00 до 21:00",
                     retrieved=retrieved)
    ]
    saved = tmp_path / "run.json"
    cmp_rag.save_json(saved, corpora, [loose], cases,
                      cmp_rag.aggregate(cases, [loose]), model="fake")

    relaxed_yaml = questions_file.read_text(encoding="utf-8").replace(
        "sources: [\"kb/hours.md\"]", 'sources: ["kb/menu.md:40-44"]'
    )
    questions_file.write_text(relaxed_yaml, encoding="utf-8")
    try:
        reloaded_questions, reloaded_cases = cmp_rag.load_saved_cases(saved)
        assert cmp_rag.scored(reloaded_cases[0], reloaded_questions[0])[
            "retrieval_hit"
        ] is True
        fresh = cmp_rag.load_questions(questions_file)[1]
        assert cmp_rag.scored(reloaded_cases[0], fresh[0])["retrieval_hit"] is False
        assert cmp_rag.scored(cases[0], strict)["retrieval_hit"] is False
    finally:
        questions_file.write_text(relaxed_yaml.replace(
            'sources: ["kb/menu.md:40-44"]', 'sources: ["kb/hours.md"]'
        ), encoding="utf-8")


def test_main_reports_a_missing_index_without_calling_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A missing index is a configuration error, not a 0% score."""

    def explode(*args, **kwargs):
        raise AssertionError("the model must not be built without an index")

    monkeypatch.setattr(cmp_rag, "load_model", explode)
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        QUESTIONS_YAML.format(index_dir=tmp_path / "does_not_exist"),
        encoding="utf-8",
    )
    code = cmp_rag.main([
        "--questions", str(broken), "--model", "fake",
        "--out", str(tmp_path / "x.md"),
    ])
    assert code == 2
    assert "индекс не найден" in capsys.readouterr().err
    assert not (tmp_path / "x.md").exists()


def test_main_rejects_an_unknown_corpus_filter(
    tmp_path: Path, questions_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("the model must not be built for an unknown corpus")

    monkeypatch.setattr(cmp_rag, "load_model", explode)
    code = cmp_rag.main([
        "--questions", str(questions_file), "--corpus", "nope",
        "--out", str(tmp_path / "x.md"),
    ])
    assert code == 2

# --------------------------------------------------------------------------- #
# The third arm: rag_rerank
# --------------------------------------------------------------------------- #


class SpyReranker:
    """Records the shortlist size it is handed; reorders nothing."""

    name = "fake-cross"

    def __init__(self) -> None:
        self.shortlists: list[int] = []

    def rerank(self, question, hits, top_k):
        self.shortlists.append(len(hits))
        return list(hits)[:top_k]


def test_modes_for_adds_the_third_arm_only_on_request() -> None:
    """The day-22 two-arm run must stay byte-identical by default."""
    assert cmp_rag.modes_for(False) == cmp_rag.MODES == ("rag", "no_rag")
    assert cmp_rag.modes_for(True) == ("rag", "rag_rerank", "no_rag")


def test_the_rerank_arm_runs_next_to_the_plain_one(
    questions_file: Path, tmp_path: Path
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    reranker = SpyReranker()
    original = cmp_rag.Retriever
    cmp_rag.Retriever = lambda path, **kw: original(
        path, embedder=FakeEmbedder("fake-mini"), **kw
    )
    client = _client({"1": "С 9:00 до 21:00", "2": "С 9:00 до 21:00",
                      "3": "С 9:00 до 21:00", "4": "не нашёл",
                      "5": "С 9:00 до 21:00", "6": "не нашёл"}, [])
    try:
        cases = cmp_rag.run(
            corpora, questions, client, reranker=reranker,
            candidate_k=3, top_k=1,
            modes=cmp_rag.modes_for(True),
        )
    finally:
        cmp_rag.Retriever = original

    assert [c.mode for c in cases] == ["rag", "rag_rerank", "no_rag"] * 2
    summary = cmp_rag.aggregate(cases, questions)
    assert set(summary) == {"rag", "rag_rerank", "no_rag"}
    # Both retrieval arms were judged against the same `sources` rule.
    assert summary["rag_rerank"]["retrieval_hit_rate"] is not None
    assert reranker.shortlists, "the reranker was never called"


def test_the_rerank_arm_needs_a_reranker_rather_than_silently_skipping(
    questions_file: Path,
) -> None:
    corpora, questions = cmp_rag.load_questions(questions_file)
    original = cmp_rag.Retriever
    cmp_rag.Retriever = lambda path, **kw: original(
        path, embedder=FakeEmbedder("fake-mini"), **kw
    )
    try:
        with pytest.raises(ValueError, match="требует reranker"):
            cmp_rag.run(
                corpora, questions, _client({}, []),
                modes=("rag_rerank",),
            )
    finally:
        cmp_rag.Retriever = original


def test_the_two_retrieval_arms_do_not_share_a_retriever(
    questions_file: Path,
) -> None:
    """One shared retriever would leak the reranker's state into the `rag` arm."""
    corpora, questions = cmp_rag.load_questions(questions_file)
    original = cmp_rag.Retriever
    built: list[dict] = []

    def spy(path, **kw):
        built.append(kw)
        return original(path, embedder=FakeEmbedder("fake-mini"), **kw)

    cmp_rag.Retriever = spy
    client = _client({}, [])
    try:
        cmp_rag.run(
            corpora, questions, client, reranker=SpyReranker(),
            top_k=2, candidate_k=7, modes=cmp_rag.modes_for(True),
        )
    finally:
        cmp_rag.Retriever = original

    with_reranker = [kw for kw in built if "reranker" in kw]
    without = [kw for kw in built if "reranker" not in kw]
    assert len(with_reranker) == 1 and len(without) == 1
    assert with_reranker[0]["candidate_k"] == 7
    assert with_reranker[0]["top_k"] == 2
    assert "candidate_k" not in without[0]


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #


def _sweep_cases() -> tuple[list[cmp_rag.Case], list[cmp_rag.Question]]:
    """Three cases shaped so a threshold visibly costs something.

    Question 2's own chunk scores *below* the distractor that outranks it, and
    the trap scores below both. That is the real shape of the problem: cutting on
    score removes the trap last and the answer second.
    """
    questions = [
        cmp_rag.Question("часы", "kb", ("21:00",), ("kb/hours.md:11-15",), True),
        cmp_rag.Question("меню", "kb", ("190",), ("kb/menu.md:1-5",), True),
        cmp_rag.Question("скидка", "kb", (), (), False),
    ]
    cases = [
        cmp_rag.Case(
            index=1, question="часы", corpus="kb", mode=cmp_rag.RERANK_MODE,
            answerable=True, answer="",
            scored=(("kb/hours.md:11-15", 5.0), ("kb/menu.md:1-5", -1.0)),
        ),
        cmp_rag.Case(
            index=2, question="меню", corpus="kb", mode=cmp_rag.RERANK_MODE,
            answerable=True, answer="",
            scored=(("kb/hours.md:11-15", 5.0), ("kb/menu.md:1-5", -3.0)),
        ),
        cmp_rag.Case(
            index=3, question="скидка", corpus="kb", mode=cmp_rag.RERANK_MODE,
            answerable=False, answer="",
            scored=(("kb/hours.md:11-15", 1.5),),
        ),
    ]
    return cases, questions


def test_the_sweep_replays_the_saved_shortlist_without_any_model() -> None:
    cases, questions = _sweep_cases()
    rows = cmp_rag.threshold_sweep(cases, questions, top_k=2)
    assert [row["threshold"] for row in rows] == list(cmp_rag.SWEEP_THRESHOLDS)
    # No filter: every answerable question keeps its chunk.
    assert rows[0]["hits"] == 2 and rows[0]["checked"] == 2
    assert rows[0]["rate"] == 1.0
    assert rows[0]["traps_armed"] == 1  # the trap got something either way

    by_threshold = {row["threshold"]: row for row in rows}
    # -2.0 drops the menu chunk but not the hours one: one answer is already lost.
    assert by_threshold[-2.0]["hits"] == 1
    # 2.0 finally disarms the trap, and by then half the answers are gone.
    # This trade-off is the whole argument for leaving the threshold off.
    assert by_threshold[2.0]["traps_armed"] == 0
    assert by_threshold[2.0]["hits"] == 1


def test_the_sweep_counts_the_chunks_a_threshold_would_send() -> None:
    cases, questions = _sweep_cases()
    rows = {row["threshold"]: row for row in
            cmp_rag.threshold_sweep(cases, questions, top_k=2)}
    assert rows[None]["chunks_mean"] == pytest.approx(5 / 3, abs=0.01)
    # At 2.0 the trap is empty and the menu question keeps only its wrong chunk:
    # two chunks left, one of them still useless.
    assert rows[2.0]["chunks_mean"] == pytest.approx(2 / 3, abs=0.01)


def test_the_sweep_needs_the_rerank_arm_and_ignores_others() -> None:
    cases, questions = _sweep_cases()
    cases.append(
        cmp_rag.Case(index=1, question="часы", corpus="kb", mode="rag",
                     answerable=True, answer="", scored=(("kb/hours.md:11-15", 5.0),))
    )
    rows = cmp_rag.threshold_sweep(cases, questions, top_k=2)
    assert rows[0]["checked"] == 2  # the `rag` case was not counted


def test_the_report_shows_the_sweep_and_the_reranker_settings() -> None:
    cases, questions = _sweep_cases()
    corpora = {"kb": cmp_rag.Corpus("kb", "Кофейня", Path(), Path(), (".md",))}
    summary = cmp_rag.aggregate(cases, questions)
    report = cmp_rag.render_report(
        corpora, questions, cases, summary,
        model_name="fake", top_k=2, max_context_tokens=None,
        rerank_model="fake-cross", candidate_k=20, min_rerank_score=None,
        sweep=cmp_rag.threshold_sweep(cases, questions, top_k=2),
    )
    assert "Порог отсечения" in report
    assert "fake-cross" in report
    assert "| выключен | 2/2 | 1/1 |" in report
    # A threshold being off is stated, so its absence is not read as an oversight.
    assert "Порог отсечения: выключен" in report
    assert "с RAG + реранкер" in report


def test_the_report_names_the_threshold_when_one_was_used() -> None:
    cases, questions = _sweep_cases()
    corpora = {"kb": cmp_rag.Corpus("kb", "Кофейня", Path(), Path(), (".md",))}
    report = cmp_rag.render_report(
        corpora, questions, cases, cmp_rag.aggregate(cases, questions),
        model_name="fake", top_k=2, max_context_tokens=None,
        rerank_model="fake-cross", candidate_k=20, min_rerank_score=0.5,
    )
    assert "Порог отсечения: 0.5" in report


def test_a_saved_run_keeps_its_rerank_diagnostics(tmp_path: Path) -> None:
    cases, questions = _sweep_cases()
    path = tmp_path / "run.json"
    cmp_rag.save_json(
        path, {}, questions, cases, cmp_rag.aggregate(cases, questions),
        model="fake", top_k=2, rerank_model="fake-cross",
        rerank_candidates=20, rerank_min_score=None,
    )
    _, reloaded = cmp_rag.load_saved_cases(path)
    assert reloaded[0].scored == (("kb/hours.md:11-15", 5.0), ("kb/menu.md:1-5", -1.0))
    assert reloaded[0].mode == cmp_rag.RERANK_MODE
    meta = cmp_rag.load_saved_meta(path)
    assert meta["top_k"] == 2 and meta["rerank_model"] == "fake-cross"


@pytest.mark.parametrize(
    "phrase",
    [
        "минимальный срок действия не указан",
        "срок не указан в документе",
        "не указано",
        "не указана",
        "не указаны",
    ],
)
def test_a_refusal_is_recognised_in_every_gender(phrase: str) -> None:
    """The masculine form was missing and scored a correct refusal as a failure.

    Found on a PDF corpus: the model answered «срок не указан» — a
    refusal — and the ``не указано``-only pattern counted it as a confabulation,
    understating the ``rag_rerank`` refusal rate by one question in six.
    """
    assert ragmod.REFUSAL_RE.search(phrase), phrase


@pytest.mark.parametrize(
    "phrase",
    [
        "в кофейне нет Wi-Fi",
        "в стоимость входит один сеанс",
        "сеансы проводятся по выходным дням",
    ],
)
def test_a_confident_answer_is_not_mistaken_for_a_refusal(phrase: str) -> None:
    """The widening above must not start swallowing real answers."""
    assert not ragmod.REFUSAL_RE.search(phrase), phrase


# --------------------------------------------------------------------------- #
# The PDF control set
# --------------------------------------------------------------------------- #

PDF_INDEX = ROOT / "data/certs_emb/index_structure.json"
PDF_QUESTIONS = ROOT / "data/certs_eval/questions.yaml"


@pytest.mark.skipif(
    not (PDF_INDEX.is_file() and (ROOT / "data/certs").is_dir()),
    reason="нужен локальный корпус PDF: data/ в .gitignore",
)
def test_every_pdf_span_contains_its_facts() -> None:
    """The same anti-drift guard the day-22 set has, for the PDF corpus.

    The corpus and its questions both live under git-ignored ``data/``, so
    silent drift is the normal failure mode here:
    bump pypdf and every line number shifts, and the benchmark starts reporting
    a retrieval problem when the cause was the extraction.

    The text is re-extracted from the PDFs with the *indexer's own* function
    rather than reassembled from the stored chunks. A chunk's ``start_line`` is
    the start of its section, not of its text — headers and page markers sit in
    between — so a chunk-based reconstruction would report offsets that were
    never the ones the citations were written against.
    """
    from scripts.index_documents import extract_pdf_text, split_lines

    config = yaml.safe_load(PDF_QUESTIONS.read_text(encoding="utf-8"))
    cache: dict[str, list[str]] = {}

    def lines_of(source: str) -> list[str]:
        if source not in cache:
            text, _ = extract_pdf_text(ROOT / "data/certs" / source, max_pages=1)
            cache[source] = split_lines(text)
        return cache[source]

    answerable = 0
    for index_no, question in enumerate(config["questions"], start=1):
        if not question.get("sources"):
            assert not question.get("expect"), (
                f"#{index_no}: неответимый вопрос не должен иметь ожиданий"
            )
            continue
        answerable += 1
        # Facts are checked against the *union* of the declared spans: a question
        # may deliberately spread them ("where" in one chunk, "how old" in
        # another). Demanding each fact inside each span would reject exactly the
        # questions this corpus is built around.
        parts: list[str] = []
        for source in question["sources"]:
            name, _, span = source.partition(":")
            start, _, end = span.partition("-")
            document = lines_of(name)
            assert 1 <= int(start) <= int(end) <= len(document), (
                f"#{index_no}: {source} выходит за пределы документа "
                f"({len(document)} строк)"
            )
            parts.append("\n".join(document[int(start) - 1 : int(end)]))
        plain = "\n".join(parts).lower().replace("*", "")
        for fact in question["expect"]:
            assert fact.lower().replace("*", "") in plain, (
                f"#{index_no}: факт {fact!r} не лежит в {question['sources']}"
            )
    assert answerable == 34, f"ожидалось 34 ответимых вопроса, а их {answerable}"


# --------------------------------------------------------------------------- #
# The three checks
# --------------------------------------------------------------------------- #


def _grounded_case(**overrides) -> "cmp_rag.Case":
    """An answer that passes all three checks: source list, citation, quote."""
    fields = dict(
        index=1,
        question="q",
        corpus="kb",
        mode="rag",
        answerable=True,
        answer='190 ₽ [1] «Крем: 190 ₽»\n\n'
               + ragmod.SOURCES_HEADING
               + "\n[1] kb/menu.md:1-5 — Напитки",
        retrieved=("kb/menu.md:1-5",),
        chunk_texts=("Крем: 190 ₽, объём 300 мл.",),
    )
    fields.update(overrides)
    return cmp_rag.Case(**fields)


def _price_question() -> "cmp_rag.Question":
    return cmp_rag.Question(
        text="q", corpus="kb", expect=("190",),
        sources=("kb/menu.md:1-5",), answerable=True,
    )


def test_a_correct_answer_with_a_source_a_citation_and_a_quote_passes() -> None:
    """The three things day24 asks of every answer, in one scored case."""
    score = cmp_rag.scored(_grounded_case(), _price_question())

    assert score["sources_listed"] is True
    assert score["citation_ok"] is True
    assert score["quote_ok"] is True
    assert score["grounding"] == ragmod.Grounding.GROUNDED.value
    assert score["passed"] is True


def test_a_right_answer_cited_from_the_right_chunk_fails_without_a_quote() -> None:
    """The gap this task exists to close.

    This answer is not wrong in any way a fact check can see: every expected fact
    is present and the citation names a line range the block really contained. It
    failed anyway, because nothing checked the wording. Scoring it as a pass is
    what produced a 37/40 that meant very little.
    """
    score = cmp_rag.scored(
        _grounded_case(answer="190 ₽ [1]\n\n" + ragmod.SOURCES_HEADING
                                + "\n[1] kb/menu.md:1-5 — Напитки"),
        _price_question(),
    )

    assert score["fact_coverage"] == 1.0
    assert score["citation_ok"] is True
    assert score["quote_ok"] is False
    assert score["passed"] is False


def test_a_paraphrased_quote_fails_the_question() -> None:
    """Plausible wording, wrong words. Only a substring test can tell, which is
    why the check is a substring test and not a second model."""
    score = cmp_rag.scored(
        _grounded_case(answer='Крем стоит 190 рублей [1] «Крем стоит 190 рублей»'
                                "\n\n" + ragmod.SOURCES_HEADING
                                + "\n[1] kb/menu.md:1-5 — Напитки"),
        _price_question(),
    )

    assert score["paraphrased_quotes"], "the paraphrase must be reported"
    assert score["grounding"] == ragmod.Grounding.UNGROUNDED.value
    assert score["passed"] is False


def test_an_answer_with_no_source_list_fails_the_question() -> None:
    """The list is appended by code, so this only happens when nothing was
    retrieved — and a question with no block cannot be answered from one."""
    score = cmp_rag.scored(
        _grounded_case(answer='190 ₽ [1] «Крем: 190 ₽»'),
        _price_question(),
    )

    assert score["sources_listed"] is False
    assert score["passed"] is False


def test_a_refusal_is_not_required_to_list_sources() -> None:
    """An unanswerable question has no sources to list, so the three checks are
    skipped for it — the answer already says it found nothing. Requiring a source
    list here would score honesty as a failure."""
    question = cmp_rag.Question(
        text="q", corpus="kb", expect=(), sources=(), answerable=False,
    )
    case = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="rag", answerable=False,
        answer="не нашёл ответ в этом контексте", prompt_tokens=10,
    )

    score = cmp_rag.scored(case, question)
    assert score["sources_listed"] is False
    assert score["passed"] is True


def test_the_checks_are_recomputed_from_saved_chunks_not_trusted_from_json() -> None:
    """A run saved before this feature must still be checkable.

    ``--from_json`` exists to re-score a saved run after an edit to the scoring.
    If the verdict were read out of the JSON, the one case that matters most —
    a run judged by the old rules — could never be judged by the new ones.
    """
    case = _grounded_case(grounding=None, citations=None, quotes=None)
    score = cmp_rag.scored(case, _price_question())

    assert score["grounding"] == ragmod.Grounding.GROUNDED.value
    assert score["passed"] is True


def test_the_baseline_arm_is_not_demanded_to_cite_anything() -> None:
    """``no_rag`` has no block by construction, so the three checks cannot apply.

    Requiring them would score the baseline as broken instead of measuring what it
    is, which is the whole reason it is in the table.
    """
    question = _price_question()
    case = cmp_rag.Case(
        index=1, question="q", corpus="kb", mode="no_rag", answerable=True,
        answer="190 ₽", prompt_tokens=20,
    )

    score = cmp_rag.scored(case, question)
    assert score["fact_coverage"] == 1.0
    assert score["passed"] is True
