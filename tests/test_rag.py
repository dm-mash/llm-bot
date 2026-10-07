"""Tests for the RAG layer: prompt block, token budget, and session wiring."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from llm_bot import rag as ragmod
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.rag import (
    DEFAULT_MAX_CONTEXT_RATIO,
    NO_ANSWER_TEXT,
    UNSUPPORTED_QUOTE_TEXT,
    Grounding,
    RagEvent,
    RagHit,
    Retriever,
    audit_quotes,
    finalize_answer,
    is_refusal,
    rag_budget_tokens,
    render_sources_footer,
    sources_from_text,
)
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.tokens import estimate_tokens

# Reuse the day-21 deterministic embedder: same idea (zlib.crc32 instead of
# ``hash()``), no torch, no network.
from tests.test_index_documents import FakeEmbedder


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def make_index(
    tmp_path: Path,
    chunks: list[tuple[str, str, str]],
    *,
    strategy: str = "structure",
    name: str = "fake-mini",
) -> Path:
    """Write a minimal but valid index: ``(source, text, section)`` per chunk."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    embedder = FakeEmbedder(name)
    payload = {
        "version": "1.0",
        "chunking_strategy": strategy,
        "embedding_model": name,
        "dimension": embedder.dim,
        "chunks": [
            {
                "chunk_id": f"{source}:{position}",
                "document_id": source,
                "chunk_position": position,
                "text": text,
                "token_count": estimate_tokens(text),
                "embedding": embedder.encode([text])[0],
                "metadata": {
                    "source": source,
                    "start_line": position * 10 + 1,
                    "end_line": position * 10 + 5,
                    "section": section,
                    "title": section,
                },
            }
            for position, (source, text, section) in enumerate(chunks)
        ],
    }
    path = tmp_path / f"index_{strategy}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


#: Question texts are chosen so they literally share words with the chunk they
#: must retrieve: FakeEmbedder is a bag of word hashes, so an overlapping word
#: is what makes a ranking assertion deterministic (a paraphrased question
#: would tie at zero and the winner would be arbitrary).
QUESTIONS = {
    "hours": "часы работы",
    "menu": "крем объём",
    "limits": "8000 TPM",
}


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    return make_index(
        tmp_path,
        [
            ("kb/menu.md", "Крем: 190 ₽, объём 300 мл.", "Напитки"),
            ("kb/hours.md", "Часы работы: с 8:00 до 22:00.", "Часы работы"),
            ("docs/limits.md", "Groq free 8000 TPM returns 429.", "Лимиты"),
        ],
    )


def retriever(index_path: Path, **kwargs) -> Retriever:
    kwargs.setdefault("embedder", FakeEmbedder("fake-mini"))
    return Retriever(index_path, **kwargs)


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name: str) -> ModelConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name: str) -> AgentConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты полезный помощник.",
        temperature=0.5,
        max_tokens=64,
    )


def _make_session(tmp_path: Path, transport, *, invariants: bool = False, **kwargs):
    model_cfg = ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )
    return make_session(
        "s1",
        "assistant",
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(_agent_config()),
        session_store=JsonSessionStore(str(tmp_path / "sessions")),
        transport=transport,
        invariants=invariants,
        **kwargs,
    )


def _capturing_transport(
    reply: str | Callable[[dict], str] = "reply-text",
) -> tuple[httpx.MockTransport, list[dict]]:
    """A transport that records every request payload it is handed."""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        payloads.append(payload)
        content = reply(payload) if callable(reply) else reply
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": content}}
                ]
            },
        )

    return httpx.MockTransport(handler), payloads


def _systems(payload: dict) -> list[str]:
    return [m["content"] for m in payload["messages"] if m["role"] == "system"]


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_budget_is_the_smaller_of_the_cap_and_the_window_share() -> None:
    assert rag_budget_tokens(None, None) is None
    assert rag_budget_tokens(0, 8192) is None
    assert rag_budget_tokens(None, 8192) == int(8192 * DEFAULT_MAX_CONTEXT_RATIO)
    # An explicit cap wins when it is smaller than the window share...
    assert rag_budget_tokens(100, 8192) == 100
    # ... and the window share wins when it is smaller.
    assert rag_budget_tokens(8000, 1000) == 250


def test_fit_words_cuts_on_a_word_boundary() -> None:
    assert ragmod._fit_words("слово", 100) == "слово"
    assert ragmod._fit_words("любой текст", 0) == ""
    cut = ragmod._fit_words("одно слово и ещё немного слов", 20)
    assert len(cut) <= 21
    assert cut.endswith("…")
    # The last visible word is whole, never sliced in half.
    body = cut[:-1].rstrip()
    assert body.split(" ")[-1] in "одно слово и ещё немного слов".split(" ")


def test_fit_words_returns_nothing_when_no_word_fits() -> None:
    """A half-word would read as a real token from the document — refuse it."""
    assert ragmod._fit_words("одно", 2) == ""
    assert ragmod._fit_words("длинноенеразбиваемо", 5) == ""


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def test_retrieve_returns_hits_with_traceable_locations(index_path: Path) -> None:
    hits, event = retriever(index_path, top_k=1).retrieve(QUESTIONS["hours"])
    assert [hit.source for hit in hits] == ["kb/hours.md"]
    assert hits[0].location == "kb/hours.md:11-15"
    assert hits[0].section == "Часы работы"
    assert hits[0].text.startswith("Часы работы")
    assert isinstance(event, RagEvent)
    assert event.candidates == 1
    assert event.retrieved == ("kb/hours.md:11-15",)
    assert event.elapsed >= 0.0


def test_the_best_matching_chunk_ranks_first(index_path: Path) -> None:
    instance = retriever(index_path, top_k=3)
    for key, expected in (
        ("hours", "kb/hours.md"),
        ("menu", "kb/menu.md"),
        ("limits", "docs/limits.md"),
    ):
        hits, _ = instance.retrieve(QUESTIONS[key])
        assert hits[0].source == expected, key


def test_retrieve_respects_top_k_and_orders_by_score(index_path: Path) -> None:
    hits, event = retriever(index_path, top_k=2).retrieve(QUESTIONS["hours"])
    assert len(hits) == 2
    assert event.candidates == 2
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_a_missing_or_foreign_index_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build it first"):
        Retriever(tmp_path / "nope.json", embedder=FakeEmbedder())
    foreign = tmp_path / "index_structure.json"
    foreign.write_text('{"chunks": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an index"):
        Retriever(foreign, embedder=FakeEmbedder())


def test_an_empty_index_renders_nothing(tmp_path: Path) -> None:
    block, event = retriever(make_index(tmp_path, [])).render("любой вопрос")
    assert block == ""
    assert event.candidates == 0 and event.context_tokens == 0


def test_embedder_is_loaded_lazily_and_from_the_index_itself(
    index_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing heavy happens until a question is asked (no torch at startup)."""
    instance = Retriever(index_path)  # no embedder injected
    assert instance._embedder is None
    assert instance.embedding_model == "fake-mini"
    assert instance.chunking_strategy == "structure"

    asked: list[str] = []

    def fake_embedder(name: str) -> FakeEmbedder:
        asked.append(name)
        return FakeEmbedder(name)

    monkeypatch.setattr(ragmod, "Embedder", fake_embedder)
    instance.retrieve(QUESTIONS["hours"])
    # The model name comes out of the index, never from configuration: the
    # stored vectors were produced by *that* model and are not comparable with
    # any other. (Same rule as the indexer's --reuse path.)
    assert asked == ["fake-mini"]
    assert instance.embedder.name == "fake-mini"


# --------------------------------------------------------------------------- #
# Block rendering
# --------------------------------------------------------------------------- #


def test_block_states_the_protocol_and_cites_sources(index_path: Path) -> None:
    block, event = retriever(index_path, top_k=1).render(QUESTIONS["hours"])
    assert "Отвечай ТОЛЬКО по этому контексту" in block
    assert "не нашёл ответ" in block
    assert "дословную фразу" in block
    assert "[1] kb/hours.md:11-15 — Часы работы" in block
    assert "с 8:00 до 22:00" in block
    assert event.dropped == 0
    assert event.context_tokens == estimate_tokens(block) + 4


def test_the_grounding_rule_exists_in_exactly_one_place() -> None:
    """The protocol must not be restated in an agent config.

    It travels with the block, so every agent is grounded identically and the
    rule cannot drift out of sync with what the bot actually sends. A second
    copy in an agent's ``system_prompt`` is exactly what this test forbids: it
    already happened once and had to be fixed by hand in three places.

    Read from the repository root, so it also covers ``agents.example.yaml``.
    """
    root = Path(__file__).resolve().parent.parent
    patterns = ("Отвечай ТОЛЬКО по этому контексту", "строго по нему")
    # ``llm_bot/rag.py`` is where the rule belongs, so it is the allow-list
    # rather than the target: the check is that nobody else states it.
    sources = [root / "data" / "agents.yaml", root / "agents.example.yaml"]
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in sources
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if any(pattern in line for pattern in patterns)
        # Comments are allowed to *name* the rule while explaining it is not
        # duplicated; only an actual instruction is a second copy.
        and not line.lstrip().startswith("#")
    ]
    assert offenders == [], (
        "правило цитирования продублировано вне блока контекста: " + ", ".join(offenders)
    )


def test_block_drops_the_worst_hits_to_fit_the_budget(index_path: Path) -> None:
    # The header alone is ~55 tokens, so a 100-token budget leaves room for one
    # chunk but not for three.
    block, event = retriever(index_path, top_k=3, max_context_tokens=100).render(
        QUESTIONS["hours"]
    )
    assert event.candidates == 3
    assert event.dropped >= 1
    assert len(event.retrieved) == 3 - event.dropped
    assert estimate_tokens(block) + 4 <= 100
    # Whatever stayed in the block is cited by location, so a reader can trace
    # every claim back to a file.
    for location in event.retrieved:
        assert location in block


def test_citations_in_an_answer_name_the_files_to_keep_in_the_dialog(
    index_path: Path,
) -> None:
    """A follow-up with no product in it («а он долго действует?») used to
    answer from whichever sibling document the reranker liked, so the file the
    dialog is actually about has to be able to claim a slot."""
    hits, _ = retriever(
        index_path, top_k=2, reranker=FakeReranker(), candidate_k=3
    ).retrieve(QUESTIONS["limits"], prefer_sources=["kb/menu.md"])
    assert [hit.source for hit in hits] == ["docs/limits.md", "kb/menu.md"]


def test_a_preferred_document_may_arrive_as_a_bare_name(index_path: Path) -> None:
    """The index stores bare file names while the session can hold a path."""
    hits, _ = retriever(
        index_path, top_k=2, reranker=FakeReranker(), candidate_k=3
    ).retrieve(QUESTIONS["limits"], prefer_sources=["data/kb/menu.md"])
    assert "kb/menu.md" in [hit.source for hit in hits]


def test_reserving_slots_never_grows_the_block(index_path: Path) -> None:
    for top_k in (2, 3, 4):
        hits, _ = retriever(
            index_path, top_k=top_k, reranker=FakeReranker(), candidate_k=3
        ).retrieve(
            QUESTIONS["limits"],
            prefer_sources=["kb/menu.md", "kb/hours.md", "docs/limits.md"],
        )
        assert len(hits) <= top_k


def test_the_top_hit_survives_reservation(index_path: Path) -> None:
    """The best hit already earned its slot; a reserved one displaces a weaker."""
    hits, _ = retriever(
        index_path, top_k=3, reranker=FakeReranker(), candidate_k=3
    ).retrieve(QUESTIONS["limits"], prefer_sources=["kb/menu.md"])
    assert hits[0].source == "docs/limits.md"


def test_citations_are_read_out_of_an_answer() -> None:
    text = (
        "Занятие, 20 минут [data/docs/Отчёт А.pdf:1-27] "
        "и [Отчёт Б.pdf:19-27], но не [kb/menu.md]"
    )
    assert ragmod.sources_from_text(text, limit=3) == [
        "data/docs/Отчёт А.pdf",
        "Отчёт Б.pdf",
        "kb/menu.md",
    ]


def test_text_without_citations_names_no_documents() -> None:
    assert ragmod.sources_from_text("Занятие, длительность 20 минут.") == []


def test_a_rewritten_document_id_is_not_a_source(
    index_path: Path,
) -> None:
    """The model mostly complies with ``[file:lines]`` but rewrites what it
    copies: 12 of 15 citations to one document came back with a changed id and
    nothing noticed."""
    text = "Отчёт Б [ID-250-250-463-709 Отчёт Б.pdf:1-10]."
    cleaned, audit = ragmod.audit_citations(
        text, ["ID-250-248-463-709 Отчёт Б.pdf:1-10"]
    )
    assert cleaned == f"Отчёт Б {ragmod.UNSUPPORTED_CITATION}."
    assert audit.dropped == ("ID-250-250-463-709 Отчёт Б.pdf",)
    assert not audit.clean


def test_a_line_range_outside_the_block_is_not_a_source() -> None:
    """Right file, lines the block never contained — still not evidence."""
    cleaned, audit = ragmod.audit_citations(
        "20 минут [kb/hours.md:19-27].",
        ["kb/hours.md:11-15"],
    )
    assert cleaned == f"20 минут {ragmod.UNSUPPORTED_CITATION}."
    assert audit.dropped == ("kb/hours.md",)


def test_a_citation_the_block_backs_is_left_alone() -> None:
    text = "20 минут [kb/hours.md:11-15] и [kb/hours.md]."
    cleaned, audit = ragmod.audit_citations(
        text, ["kb/hours.md:11-15", "docs/limits.md:6-8"]
    )
    assert cleaned == text
    assert audit.clean
    assert len(audit.kept) == 2


def test_citation_lines_may_rest_inside_a_wider_hit() -> None:
    """A narrower range inside the retrieved span is still backed by it."""
    cleaned, _ = ragmod.audit_citations(
        "20 минут [kb/hours.md:12-14].", ["kb/hours.md:11-15"]
    )
    assert cleaned == "20 минут [kb/hours.md:12-14]."


def test_a_file_named_only_as_a_path_still_matches() -> None:
    cleaned, _ = ragmod.audit_citations(
        "20 минут [data/kb/hours.md:11-15].", ["kb/hours.md:11-15"]
    )
    assert cleaned == "20 минут [data/kb/hours.md:11-15]."


def test_the_marker_itself_is_not_treated_as_a_new_citation() -> None:
    """The model copies the marker back, which must not pile up in the audit."""
    text = f"Уже было: {ragmod.UNSUPPORTED_CITATION}"
    cleaned, audit = ragmod.audit_citations(text, ["kb/hours.md:11-15"])
    assert cleaned == text
    assert audit.dropped == ()
    # Nothing real is cited here, and saying so is the point.
    assert audit.uncited is True


def test_an_answer_is_left_alone_when_no_evidence_was_sent() -> None:
    """Nothing was retrieved, so there is nothing to check against — the model
    may still be citing a tool result or its own knowledge."""
    text = "Ответ [kb/hours.md:11-15]."
    assert ragmod.audit_citations(text, ()) == (text, ragmod.CitationAudit())


def test_the_session_marks_a_fabricated_citation_in_the_reply(
    tmp_path: Path, index_path: Path
) -> None:
    transport, _ = _capturing_transport("20 минут [kb/menu.md:1-3] и цена.")
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )
    session.chat(QUESTIONS["hours"])

    assert ragmod.UNSUPPORTED_CITATION in session.history[-1]["content"]
    assert session.last_citation_audit is not None
    assert session.last_citation_audit.dropped == ("kb/menu.md",)


def test_the_session_leaves_invariant_ids_alone(
    tmp_path: Path, index_path: Path
) -> None:
    """The prompt renders every invariant as ``- [STACK-1] (kind) ...`` and the
    model works its checklist into the reply. Those brackets name rules, not
    documents: auditing them replaced five valid ids with the unsupported
    marker and logged five warnings on an otherwise correct answer."""
    transport, _ = _capturing_transport(
        "Проверяю запрос: [kb/hours.md] ок [STACK-1] и [RULE-LANG]."
    )
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1),
        invariants=True,
    )
    session.chat(QUESTIONS["hours"])

    content = session.history[-1]["content"]
    assert "[STACK-1]" in content and "[RULE-LANG]" in content
    assert ragmod.UNSUPPORTED_CITATION not in content
    audit = session.last_citation_audit
    assert audit is not None
    assert audit.kept == ("kb/hours.md",)
    assert audit.dropped == ()


def test_citations_can_be_switched_off_without_losing_the_grounding_rules(
    index_path: Path,
) -> None:
    """``--rag-no-cite`` hides sources from the user for a step where they are
    noise. The grounding and the no-blending rules have to stay: turning off the
    citation format must not turn off "answer only from this context"."""
    on, _ = retriever(index_path, top_k=1).render(QUESTIONS["hours"])
    off, _ = retriever(index_path, top_k=1, cite=False).render(QUESTIONS["hours"])

    # Asserts the demand and its worked example, not the exact wording: the
    # wording was rewritten once to raise the quote rate, and a test that froze
    # the sentence would have turned that measurement into an obstacle.
    assert ragmod._RAG_PROTOCOL in on
    assert "укажи номер" in on
    assert "«срок — 6 месяцев»" in on
    assert ragmod._RAG_PROTOCOL_NO_CITATIONS in off
    assert "укажи этот номер" not in off
    # Removing the demand for the handle was measured to be insufficient: the
    # model drops the brackets and then narrates the source in prose. The
    # no-cite clause has to forbid naming it in words as well.
    assert "ни имена файлов" in off
    assert "его номер" in off
    # A quote with no number is an unbacked claim wearing quotation marks, so it
    # goes with the handle.
    assert "ни цитаты в кавычках" in off
    for block in (on, off):
        assert ragmod._RAG_NO_BLENDING in block
        assert "ничего не додумывай" in block


def test_no_cite_strips_the_citations_from_the_reply(
    tmp_path: Path, index_path: Path
) -> None:
    """--rag-no-cite has to leave no citation in the reply, not just no marker.

    The first version of this flag only dropped the cite instruction from the
    prompt, and that was not enough: the block renders every hit as
    ``Файл: <name>:<lines>`` and the model copies the shape whether or not it
    was asked to. A demo run came back with the answer on one line and
    ``[Отчёт Б.pdf:11-19]`` on the next. A flag
    named "no cite" that prints a source is worse than no flag at all.
    """
    transport, _ = _capturing_transport("Ссылаюсь на [kb/hours.md:1-2] и [выдумка.md].")
    session = _make_session(
        tmp_path,
        transport,
        retriever=retriever(index_path, top_k=1, cite=False),
        rag_cite=False,
    )
    session.chat(QUESTIONS["hours"])

    reply = session.history[-1]["content"]
    assert "kb/hours.md" not in reply
    assert "выдумка.md" not in reply
    assert "[" not in reply and "]" not in reply
    assert session.last_citation_audit is None


def test_no_cite_keeps_the_punctuation_the_brackets_left_behind(
    tmp_path: Path, index_path: Path
) -> None:
    transport, _ = _capturing_transport("С 8:00 до 22:00 [kb/hours.md:1-2].")
    session = _make_session(
        tmp_path,
        transport,
        retriever=retriever(index_path, top_k=1, cite=False),
        rag_cite=False,
    )
    session.chat(QUESTIONS["hours"])

    reply = session.history[-1]["content"]
    assert reply.endswith("22:00.")
    assert " ." not in reply


def test_the_block_names_the_near_duplicate_hazard(index_path: Path) -> None:
    """Two documents shared 15 of 18 indexed lines, and the model answered a
    question about one of them from the other's chunk. The protocol has to name
    that failure, or it comes straight back."""
    block, _ = retriever(index_path, top_k=1).render(QUESTIONS["hours"])
    assert ragmod._RAG_PROTOCOL in block
    assert ragmod._RAG_NO_BLENDING in block
    assert "не взаимозаменяемы" in block


def test_a_budget_that_fits_one_hit_keeps_the_best(index_path: Path) -> None:
    # Derived from the real header instead of a literal: the budget rules must
    # keep working when the protocol above the chunks changes length.
    full, _ = retriever(index_path, top_k=1, max_context_tokens=100_000).render(
        QUESTIONS["hours"]
    )
    budget = estimate_tokens(full) + 8
    block, event = retriever(index_path, top_k=1, max_context_tokens=budget).render(
        QUESTIONS["hours"]
    )
    assert event.dropped == 0
    assert "kb/hours.md" in block
    assert "с 8:00 до 22:00" in block
    assert estimate_tokens(block) + 4 <= budget


def test_a_tiny_budget_truncates_the_top_hit_instead_of_cutting_mid_word(
    index_path: Path,
) -> None:
    # Enough for the protocol plus part of the top hit, so it must be cut short.
    probe = retriever(index_path, top_k=1, max_context_tokens=100_000)
    head_text = probe._head()
    whole, _ = probe.render(QUESTIONS["hours"])
    body_text = whole[len(head_text) :].strip()
    source_line = body_text.splitlines()[0]
    head_tokens = estimate_tokens(head_text)
    # ``_fit`` can only truncate once the remaining room is positive, and only
    # stops keeping whole hits above this size: the window between the two is
    # where truncation happens at all.
    lowest = (len(head_text) + len(source_line) + 2) // 4 + 1
    highest = ragmod.MESSAGE_OVERHEAD_TOKENS + head_tokens + estimate_tokens(body_text)
    assert lowest < highest, "protocol leaves no room for part of a single hit"
    budget = lowest + (highest - lowest) // 2
    block, event = retriever(index_path, top_k=3, max_context_tokens=budget).render(
        QUESTIONS["hours"]
    )
    assert event.dropped == 2
    assert "…" in block
    assert estimate_tokens(block) + 4 <= budget
    # The last line of the block is the top hit, cut to whole words only.
    body = block.rsplit("\n", 1)[-1]
    assert body.endswith("…")
    assert "Часы работы: с 8:00 до 22:00".startswith(body[:-1].strip())


def test_a_budget_too_small_for_any_word_sends_no_context(index_path: Path) -> None:
    """An empty block is honest; a header with no evidence is not."""
    block, event = retriever(index_path, top_k=3, max_context_tokens=40).render(
        QUESTIONS["hours"]
    )
    assert block == ""
    assert event.dropped == 3
    assert event.retrieved == ()
    assert event.context_tokens == 0


def test_no_budget_keeps_every_hit(index_path: Path) -> None:
    block, event = Retriever(
        index_path,
        embedder=FakeEmbedder("fake-mini"),
        top_k=3,
        max_context_tokens=None,
        context_window=None,
    ).render(QUESTIONS["hours"])
    assert event.dropped == 0
    assert "kb/menu.md" in block and "docs/limits.md" in block


def test_context_window_is_used_when_no_cap_is_given(index_path: Path) -> None:
    small = Retriever(
        index_path, embedder=FakeEmbedder("fake-mini"), top_k=3, context_window=200
    )
    assert small.budget_tokens == int(200 * DEFAULT_MAX_CONTEXT_RATIO)
    _, event = small.render(QUESTIONS["hours"])
    assert event.dropped >= 1


def test_hits_carry_the_metadata_the_citation_format_needs(index_path: Path) -> None:
    """The prompt cites ``[file:start-end]``; that must come from real metadata."""
    hits, _ = retriever(index_path).retrieve("меню")
    for hit in hits:
        assert isinstance(hit, RagHit)
        assert hit.source and hit.start_line > 0
        assert hit.end_line >= hit.start_line
        assert hit.text.strip()
        assert hit.location == f"{hit.source}:{hit.start_line}-{hit.end_line}"


# --------------------------------------------------------------------------- #
# Session wiring
# --------------------------------------------------------------------------- #


def test_session_injects_the_block_and_keeps_it_out_of_history(
    tmp_path: Path, index_path: Path
) -> None:
    transport, payloads = _capturing_transport("Мы с 8:00 до 22:00.")
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )

    assert session.chat(QUESTIONS["hours"]).startswith("Мы с 8:00 до 22:00.")

    blocks = [b for b in _systems(payloads[0]) if "Контекст из локальной базы" in b]
    assert len(blocks) == 1
    assert "[1] kb/hours.md:11-15 — Часы работы" in blocks[0]
    assert session.rag_events[-1].question == QUESTIONS["hours"]
    assert "kb/hours.md" in session.last_rag_event.retrieved[0]
    # The block's handles must address the hits that were actually sent, or a
    # [1] in the reply can point at evidence the model never saw.
    sent = session.last_rag_event.sources
    assert [hit.location for hit in sent] == list(session.last_rag_event.retrieved)
    # This answer named no chunk, so it gets no source list: the footer states
    # what the reply leaned on, and an uncited reply leaned on nothing. It must
    # also stay out of the context, where the model could read its own answer
    # back and start copying the footer's shape. The chunks *were* sent here, so
    # the reply carries the "went unused" note rather than the "none found" one.
    reply = session.chat("ещё раз")
    assert ragmod.SOURCES_HEADING not in reply
    assert ragmod.UNUSED_SOURCES_NOTE in reply
    assert ragmod.NO_SOURCES_NOTE not in reply
    assert not any("Источники:" in b for b in _systems(payloads[1]))


def test_session_keeps_the_dialog_on_the_document_it_already_cited(
    tmp_path: Path, index_path: Path
) -> None:
    """The second turn names no product, so only the citation the model itself
    produced can say which file the customer is asking about."""
    replies = iter(["Часы [kb/hours.md:11-15]", "Лимиты [docs/limits.md:6-8]"])
    transport, _ = _capturing_transport(lambda _payload: next(replies))

    class _Recording:
        def __init__(self) -> None:
            self.seen: list[tuple[str, list[str]]] = []

        def render(self, question, *, prefer_sources=()):
            self.seen.append((question, list(prefer_sources)))
            return "", RagEvent(question=question)

    spy = _Recording()
    session = _make_session(tmp_path, transport, retriever=spy)
    session.chat(QUESTIONS["hours"])
    session.chat(QUESTIONS["limits"])

    assert spy.seen[0] == (QUESTIONS["hours"], [])
    assert spy.seen[1] == (QUESTIONS["limits"], ["kb/hours.md"])


def test_the_rag_block_is_a_prefix_not_history(
    tmp_path: Path, index_path: Path
) -> None:
    """The context is rebuilt per question and never persisted with the dialog."""
    transport, payloads = _capturing_transport()
    store_dir = tmp_path / "sessions"
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )
    session.chat(QUESTIONS["hours"])
    session.chat(QUESTIONS["limits"])

    saved = (store_dir / "s1.json").read_text(encoding="utf-8")
    # Nothing about the retrieved documents leaks into the durable dialog: on
    # the next turn the model must be grounded again, not remember evidence.
    assert "Контекст из локальной базы" not in saved
    assert "Контекст из локальной базы" not in json.dumps(
        session.history, ensure_ascii=False
    )
    assert len(session.rag_events) == 2
    # Each turn carried its own context, and the second one differs.
    first = _systems(payloads[0])
    second = _systems(payloads[1])
    assert any("kb/hours.md" in b for b in first if "Контекст" in b)
    assert any("docs/limits.md" in b for b in second if "Контекст" in b)


def test_session_without_rag_sends_no_block(tmp_path: Path) -> None:
    transport, payloads = _capturing_transport("ок")
    session = _make_session(tmp_path, transport)
    assert session.chat("привет") == "ок"
    assert not any("Контекст из локальной базы" in b for b in _systems(payloads[0]))
    assert session.rag_events == []


def test_rag_top_k_and_token_cap_reach_the_retriever(
    tmp_path: Path, index_path: Path
) -> None:
    transport, _ = _capturing_transport()
    session = _make_session(
        tmp_path,
        transport,
        rag_index=str(index_path),
        rag_top_k=1,
        rag_max_context_tokens=200,
    )
    built = session._retriever
    assert isinstance(built, Retriever)
    assert built.top_k == 1
    assert built.budget_tokens == 200
    assert built.embedding_model == "fake-mini"


def test_the_factory_wires_the_index_path_and_top_k(
    tmp_path: Path, index_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rag_index`` alone is enough: the session ends up with a real retriever."""
    monkeypatch.setattr(ragmod, "Embedder", lambda name: FakeEmbedder(name))
    transport, payloads = _capturing_transport("ок")
    session = _make_session(
        tmp_path, transport, rag_index=str(index_path), rag_top_k=3
    )
    assert isinstance(session._retriever, Retriever)
    assert session._retriever.top_k == 3
    session.chat(QUESTIONS["hours"])
    assert any("Контекст из локальной базы" in b for b in _systems(payloads[0]))


def test_a_failing_retriever_degrades_to_a_plain_answer(
    tmp_path: Path, index_path: Path
) -> None:
    """Retrieval is never a reason to fail the turn: the bot answers anyway."""

    class BrokenRetriever:
        top_k = 3

        def render(self, question: str):
            raise RuntimeError("index unreadable")

    transport, payloads = _capturing_transport("ответ без RAG")
    session = _make_session(
        tmp_path, transport, retriever=BrokenRetriever()
    )
    assert session.chat("вопрос") == "ответ без RAG"
    assert not any("Контекст из локальной базы" in b for b in _systems(payloads[0]))
    assert session.rag_events == []


def test_an_empty_index_still_answers(tmp_path: Path) -> None:
    transport, payloads = _capturing_transport("ничего не нашёл")
    session = _make_session(
        tmp_path, transport, retriever=retriever(make_index(tmp_path, []))
    )
    assert session.chat("вопрос") == f"ничего не нашёл\n\n{ragmod.NO_SOURCES_NOTE}"
    assert not any("Контекст из локальной базы" in b for b in _systems(payloads[0]))
    assert session.rag_events[-1].candidates == 0


def test_rag_block_is_placed_after_tools_and_before_the_role_prompt(
    tmp_path: Path, index_path: Path
) -> None:
    """Order matters: what it can call, then the evidence, then its role."""

    class ToolsOnly:
        """Enough of MCPRouter for the prefix order (no tool is ever called)."""

        max_rounds = 1

        def render_tools_block(self) -> str:
            return "ДОСТУПНЫЕ ИНСТРУМЕНТЫ: пример"

        def parse_directives(self, reply: str) -> list:
            return []

        def call_tool(self, directive):
            raise AssertionError("the reply must not call a tool here")

    transport, payloads = _capturing_transport("ок")
    session = _make_session(
        tmp_path,
        transport,
        retriever=retriever(index_path, top_k=1),
        mcp_router=ToolsOnly(),
    )
    session.chat(QUESTIONS["hours"])

    systems = _systems(payloads[0])
    tools = next(i for i, b in enumerate(systems) if "ИНСТРУМЕНТЫ" in b)
    grounded = next(
        i for i, b in enumerate(systems) if "Контекст из локальной базы" in b
    )
    role = next(i for i, b in enumerate(systems) if "полезный помощник" in b)
    assert tools < grounded < role


def test_a_broken_index_is_reported_before_the_first_question(tmp_path: Path) -> None:
    """Startup fails loudly rather than silently answering ungrounded."""
    transport, _ = _capturing_transport()
    with pytest.raises(FileNotFoundError, match="build it first"):
        _make_session(tmp_path, transport, rag_index=str(tmp_path / "nope.json"))


def test_the_cli_only_turns_rag_on_with_the_flag() -> None:
    from llm_bot.cli import build_parser

    parser = build_parser()
    off = parser.parse_args(["--agent", "assistant", "привет"])
    assert off.rag is False
    on = parser.parse_args(
        ["--agent", "assistant", "--rag", "--rag-top-k", "2", "привет"]
    )
    assert on.rag is True
    assert on.rag_top_k == 2
    assert on.rag_index.endswith("index_structure.json")
    assert on.rag_max_tokens is None

# --------------------------------------------------------------------------- #
# Re-ranking
# --------------------------------------------------------------------------- #


class FakeReranker:
    """Ranks by shared words, so it can disagree with ``FakeEmbedder``."""

    name = "fake-cross"

    def __init__(self, prefer: str = "") -> None:
        self.prefer = prefer
        self.calls: list[tuple[str, int]] = []

    def rerank(self, question, hits, top_k):
        self.calls.append((question, len(hits)))
        words = set(question.lower().split())

        def key(item):
            score, record = item
            text = str(record.get("text", "")).lower()
            bonus = 1 if self.prefer and self.prefer in text else 0
            return (-(bonus + len(words & set(text.split()))), -score)

        return [(score, record) for score, record in sorted(hits, key=key)[:top_k]]


def test_a_reranker_reorders_the_same_index(index_path: Path) -> None:
    """The shortlist gets wider, the prompt stays ``top_k``."""
    instance = retriever(
        index_path, top_k=1, reranker=FakeReranker(prefer="8000"), candidate_k=3
    )
    hits, event = instance.retrieve(QUESTIONS["limits"])
    assert [hit.source for hit in hits] == ["docs/limits.md"]
    assert event.reranked == 3  # looked at the whole shortlist
    assert event.candidates == 1  # but sent only one chunk
    assert instance._reranker.calls == [(QUESTIONS["limits"], 3)]


def test_without_a_reranker_the_dense_order_is_untouched(index_path: Path) -> None:
    """``candidate_k`` must not silently widen the shortlist with no reranker."""
    instance = retriever(index_path, top_k=2, candidate_k=99)
    hits, event = instance.retrieve(QUESTIONS["hours"])
    assert event.reranked == 0
    assert event.scored == ()
    assert [hit.source for hit in hits] == ["kb/hours.md", "kb/menu.md"]


def test_the_candidate_k_is_never_narrower_than_top_k(index_path: Path) -> None:
    instance = retriever(
        index_path, top_k=3, reranker=FakeReranker(), candidate_k=1
    )
    assert instance.candidate_k == 3


def test_the_event_carries_the_whole_shortlist_for_offline_analysis(
    index_path: Path,
) -> None:
    """``scored`` is what the threshold sweep replays instead of the model."""
    instance = retriever(index_path, top_k=1, reranker=FakeReranker(), candidate_k=3)
    _, event = instance.retrieve(QUESTIONS["hours"])
    assert len(event.scored) == 3
    locations = [location for location, _ in event.scored]
    assert all(":" in location for location in locations)
    scores = [score for _, score in event.scored]
    assert scores == sorted(scores, reverse=True)


def test_a_threshold_can_empty_the_context(index_path: Path) -> None:
    """The honest outcome for a question the corpus cannot answer."""
    instance = retriever(
        index_path,
        top_k=3,
        reranker=FakeReranker(),
        candidate_k=3,
        min_rerank_score=99.0,
    )
    block, event = instance.render(QUESTIONS["hours"])
    assert block == ""
    assert event.candidates == 0
    assert event.filtered == 3
    assert event.context_tokens == 0


def test_no_threshold_means_nothing_is_filtered(index_path: Path) -> None:
    instance = retriever(index_path, top_k=3, reranker=FakeReranker(), candidate_k=3)
    _, event = instance.retrieve(QUESTIONS["hours"])
    assert event.filtered == 0


def test_the_default_candidate_k_is_the_measured_one(index_path: Path) -> None:
    """A reranker with no explicit ``candidate_k`` must still get 20 candidates.

    Regression guard: ``DEFAULT_RERANK_CANDIDATES`` existed but was never read,
    so the shortlist silently collapsed to ``top_k`` (4) and the reranker only
    re-ordered what dense search had already picked — the arm then reported the
    identical retrieval rate to plain ``rag``, which is what made the bug
    visible.
    """
    from llm_bot.rerank import DEFAULT_RERANK_CANDIDATES

    instance = retriever(index_path, top_k=4, reranker=FakeReranker())
    assert instance.candidate_k == DEFAULT_RERANK_CANDIDATES == 20


def test_an_explicit_candidate_k_still_wins(index_path: Path) -> None:
    instance = retriever(
        index_path, top_k=4, reranker=FakeReranker(), candidate_k=50
    )
    assert instance.candidate_k == 50


def test_no_reranker_means_no_widened_shortlist(index_path: Path) -> None:
    instance = retriever(index_path, top_k=4)
    assert instance.candidate_k == 4


def test_the_cli_wires_the_reranker_flags() -> None:
    from llm_bot.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["--agent", "assistant", "--rag", "--rag-rerank",
         "--rag-rerank-candidates", "12", "--rag-rerank-min-score", "-1.5",
         "привет"]
    )
    assert args.rag_rerank is True
    assert args.rag_rerank_candidates == 12
    assert args.rag_rerank_min_score == -1.5
    # Off unless asked for.
    assert parser.parse_args(["--agent", "assistant", "--rag", "привет"]).rag_rerank is False


def test_reranker_flags_without_rag_are_an_error(tmp_path, capsys) -> None:
    """Otherwise the bot looks like it searched and found nothing."""
    from llm_bot.cli import main as cli_main

    code = cli_main(["--agent", "assistant", "--rag-rerank", "привет"])
    assert code == 2
    assert "require --rag" in capsys.readouterr().err


def test_the_factory_leaves_the_reranker_out_by_default(tmp_path: Path) -> None:
    """The plain RAG path must not load a second model."""
    index_path = make_index(tmp_path, [("kb/menu.md", "Крем: 190 \u20bd.", "Напитки")])
    session = _make_session(
        tmp_path, _capturing_transport()[0], rag_index=str(index_path)
    )
    assert session._retriever.reranker is None
    assert session._retriever.candidate_k == session._retriever.top_k


def test_the_factory_injects_a_reranker_when_asked(tmp_path: Path) -> None:
    index_path = make_index(tmp_path, [("kb/menu.md", "Крем: 190 \u20bd.", "Напитки")])
    # Patched where the factory looks it up, not where it is defined.
    sentinel = FakeReranker()
    from llm_bot import factory as factorymod

    original = factorymod.Reranker
    factorymod.Reranker = lambda *a, **kw: sentinel
    try:
        session = _make_session(
            tmp_path, _capturing_transport()[0],
            rag_index=str(index_path), rag_rerank=True,
        )
    finally:
        factorymod.Reranker = original
    assert session._retriever.reranker is sentinel
    assert session._retriever.candidate_k == 20


def test_two_preferred_documents_both_keep_their_slot() -> None:
    """Reserving one document must not cost the other its slot.

    Every reservation used to write to the last element, so a second preferred
    document silently overwrote the slot the first one had just taken.
    """
    def hit(source: str, score: float) -> tuple[float, dict]:
        return score, {"metadata": {"source": source}}

    raw = [hit("a.md", 1.0), hit("b.md", 0.9), hit("c.md", 0.8), hit("d.md", 0.7)]
    pool = [*raw, hit("wanted.md", 0.1), hit("also.md", 0.2)]
    chosen = ragmod._reserve_slots(raw, pool, ["wanted.md", "also.md"], 4)
    assert [source_of(item) for item in chosen] == [
        "a.md", "b.md", "also.md", "wanted.md"
    ]
    assert len(chosen) == 4


def source_of(item: tuple[float, dict]) -> str:
    return str(item[1]["metadata"]["source"])


def test_an_answer_with_no_citation_at_all_is_flagged(tmp_path: Path) -> None:
    text, audit = ragmod.audit_citations(
        "Длительность — 20 минут.", ["kb/limits.md:11-19"]
    )
    assert text == "Длительность — 20 минут."
    assert audit.uncited is True
    assert audit.dropped == ()


def test_a_turn_without_a_block_is_not_called_uncited() -> None:
    _, audit = ragmod.audit_citations("Привет!", [])
    assert audit.uncited is False


def test_a_citation_may_name_a_document_by_its_id() -> None:
    """The model writes ``[ID-250-439-931-337]`` and drops the title."""
    text, audit = ragmod.audit_citations(
        "Рост до 190 см [ID-250-439-931-337].",
        ["ID-250-439-931-337 Отчёт А.pdf:19-27"],
    )
    assert audit.dropped == ()
    assert audit.kept == ("ID-250-439-931-337",)


def test_a_bare_number_is_not_a_source_when_two_documents_share_it() -> None:
    _, audit = ragmod.audit_citations(
        "Рост до 190 см [ID-250-439-931-337].",
        [
            "ID-250-439-931-337 Отчёт А.pdf:1-18",
            "ID-250-439-931-337 Отчёт Б.pdf:1-18",
        ],
    )
    assert audit.dropped == ("ID-250-439-931-337",)


def test_a_wrong_number_is_still_rejected() -> None:
    _, audit = ragmod.audit_citations(
        "Длительность 20 минут [ID-250-250-463-709].",
        ["ID-250-248-463-709 Отчёт Б.pdf:11-19"],
    )
    assert audit.dropped == ("ID-250-250-463-709",)


def test_a_narrower_range_inside_the_chunk_is_kept() -> None:
    """The block carries lines 1-15 and the model cites the two it actually read.

    Observed live on a duration question: the answer came back as
    ``...pdf:8-10`` while the chunk sent was ``...pdf:1-15``. Requiring the
    ranges to be equal instead of nested would turn that into a false
    "unsupported source" and start dropping honest citations.
    """
    text, audit = ragmod.audit_citations(
        "Длительность 1 день [ID-250-863-857-532 Услуга.pdf:8-10].",
        ["ID-250-863-857-532 Услуга.pdf:1-15"],
    )
    assert audit.dropped == ()
    assert "8-10" in text


def test_a_citation_naming_the_service_instead_of_the_file_is_dropped() -> None:
    """The live run quoted the service title, which names no retrievable source.

    ``[Услуга «Вечер»]`` reads like a citation and lands in the
    reply as if it were verified, so the audit has to catch it — otherwise the
    demo answer carries a confident, unsourced label.
    """
    text, audit = ragmod.audit_citations(
        "Услуга проходит в клубе [Услуга «Вечер»].",
        ["ID-250-863-857-532 Услуга.pdf:1-15"],
    )
    assert audit.dropped == ("Услуга «Вечер»",)
    assert ragmod.UNSUPPORTED_CITATION in text


def test_strip_citations_removes_both_kinds_of_citation() -> None:
    """A reply with no sources must not keep the one the audit would have caught.

    Leaving the unbacked citation behind is what a caller asking for a clean
    answer does not want: ``[Услуга «Вечер»]`` names no file and
    still reads as a source.
    """
    text = ragmod.strip_citations(
        "Длительность 1 день [ID-250-863-857-532 Услуга.pdf:8-10] "
        "на [Услуга «Вечер»]."
    )
    assert text == "Длительность 1 день на."


def test_strip_citations_keeps_invariant_ids() -> None:
    """An invariant names a rule, not a document, so it survives.

    Only the bracket goes; the words around it are the model's own prose and are
    left alone rather than second-guessed.
    """
    text = ragmod.strip_citations(
        "Соблюдаю [STACK-1] и [ARCH-1], источник [kb/hours.md:1-2].",
        ignore=["STACK-1", "ARCH-1"],
    )
    assert text == "Соблюдаю [STACK-1] и [ARCH-1], источник."


def test_strip_citations_tidies_the_spacing_it_leaves() -> None:
    assert ragmod.strip_citations("С 8:00 до 22:00 [kb/hours.md:1-2].") == (
        "С 8:00 до 22:00."
    )


# --------------------------------------------------------------------------- #
# Verbatim quotes
# --------------------------------------------------------------------------- #


def test_a_quote_the_chunk_does_not_contain_is_marked(index_path: Path) -> None:
    """A citation says which chunk; only the quote says which words.

    Measured, this is the pair the citation check alone misses: on a corpus of
    near-duplicate documents the model cited the right file with the right lines
    and still stated the price wrong, because nothing looked at the wording. So
    the protocol now asks for the phrase, and a phrase that is not in the chunk
    it is attributed to is counted rather than passed on as verified.

    The words stay in the reply. Replacing them with a placeholder used to leave
    «[цитата не подтверждена]» standing in the middle of a sentence, which reads
    as a broken bot and destroys the sentence it was reporting on. finalize_answer
    turns the count into one line beside the source list.
    """
    hit = retriever(index_path, top_k=1).retrieve(QUESTIONS["hours"])[0][0]
    assert "с 8:00 до 22:00" in hit.text

    reply, audit = audit_quotes(
        'Часы [1]: «Часы работы: с 8:00 до 23:00».',
        [hit],
    )

    assert "Часы работы: с 8:00 до 23:00" in reply
    assert audit.dropped == ("Часы работы: с 8:00 до 23:00",)


def test_a_quote_that_is_verbatim_in_the_chunk_survives(index_path: Path) -> None:
    """The check is a substring test, not a paraphrase test, and must behave
    like one: quoting correctly is what earns the answer, so the only thing
    standing between a good reply and the marker must be the model's own care."""
    hit = retriever(index_path, top_k=1).retrieve(QUESTIONS["hours"])[0][0]
    reply, audit = audit_quotes('Часы [1]: «с 8:00 до 22:00».', [hit])

    assert audit.kept == ("с 8:00 до 22:00",)
    assert UNSUPPORTED_QUOTE_TEXT not in reply


def test_a_quoted_name_from_the_question_is_not_a_claim_about_a_chunk(
    index_path: Path,
) -> None:
    """«Тростниковый крем» is the user's own wording, not evidence.

    Only quotes the model attributes to a citation are checked. Otherwise the
    audit would fail every answer that names the product the customer asked
    about, and a check that cries wolf gets switched off.
    """
    hit = retriever(index_path, top_k=1).retrieve(QUESTIONS["hours"])[0][0]
    reply, audit = audit_quotes('Напиток «Тростниковый крем» — это [1].', [hit])

    assert reply == 'Напиток «Тростниковый крем» — это [1].'
    assert not audit.dropped
    assert not audit.kept


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def test_an_answer_with_a_confirmed_quote_is_grounded(
    tmp_path: Path, index_path: Path
) -> None:
    """The three things day24 asks of every answer, and this turn has all of
    them: a source, a citation, and a quote that is really in that chunk."""
    transport, _ = _capturing_transport(
        "Часы [1]: с 8:00 до 22:00 — «Часы работы: с 8:00 до 22:00»."
    )
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )
    session.chat(QUESTIONS["hours"])

    verdict = session.last_grounding
    assert verdict is not None
    assert verdict.status is Grounding.GROUNDED
    assert verdict.quotes == ("Часы работы: с 8:00 до 22:00",)
    assert verdict.clean


def test_an_answer_without_a_quote_is_ungrounded_even_with_a_good_citation(
    tmp_path: Path, index_path: Path
) -> None:
    """A valid [1] and no quote means the claim itself was never checked.

    This is the state 11 of 20 answers on a saved run were in, and it is exactly
    what a citation-only rule counts as success.
    """
    transport, _ = _capturing_transport("Часы [1]: с 8:00 до 22:00.")
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )
    reply = session.chat(QUESTIONS["hours"])

    assert session.last_grounding.status is Grounding.UNGROUNDED
    # The source list is code-built, so it is there even with no quote.
    assert ragmod.SOURCES_HEADING in reply


def test_a_refusal_is_not_punished_for_carrying_no_citation(
    tmp_path: Path, index_path: Path
) -> None:
    """Declining is the correct behaviour when the block has no answer.

    Judged before the audits, so a refusal is never reported as ungrounded for
    the thing it is right about.
    """
    transport, _ = _capturing_transport(
        "Не нашёл ответ в этом контексте. Уточните, пожалуйста, что нужно."
    )
    session = _make_session(
        tmp_path, transport, retriever=retriever(index_path, top_k=1)
    )
    session.chat("где душ")

    assert session.last_grounding.status is Grounding.REFUSED


def test_strict_mode_withdraws_an_answer_nothing_backs(
    tmp_path: Path, index_path: Path
) -> None:
    """--rag-strict replaces the claim, rather than shipping it with a marker.

    A price or a limit stated wrongly is the failure this whole mechanism is
    for, so the flag's job is to make the reply say nothing rather than to make
    it say something with a warning attached.
    """
    transport, _ = _capturing_transport("Часы [1]: с 8:00 до 23:00.")
    session = _make_session(
        tmp_path,
        transport,
        retriever=retriever(index_path, top_k=1),
        rag_strict=True,
    )
    reply = session.chat(QUESTIONS["hours"])

    assert reply == NO_ANSWER_TEXT
    assert "23:00" not in reply
    # The footer went with the claim: sources for a withdrawn answer would read
    # as support for it.
    assert "Источники:" not in reply
    assert session.last_grounding.status is Grounding.UNGROUNDED


def test_strict_mode_leaves_a_grounded_answer_alone(
    tmp_path: Path, index_path: Path
) -> None:
    """Strict must not be a blanket refusal. The whole point of judging on the
    quote rather than on a score is that a correct answer is distinguishable."""
    transport, _ = _capturing_transport(
        "Часы [1]: с 8:00 до 22:00 — «Часы работы: с 8:00 до 22:00»."
    )
    session = _make_session(
        tmp_path,
        transport,
        retriever=retriever(index_path, top_k=1),
        rag_strict=True,
    )
    reply = session.chat(QUESTIONS["hours"])

    assert reply.startswith("Часы [1]: с 8:00 до 22:00")
    assert ragmod.SOURCES_HEADING in reply


def test_a_quote_in_backticks_is_checked_too() -> None:
    """Measured, not assumed: on a run over technical documents the model quoted
    in backticks, not «». An audit blind to that reports "the model does not
    quote" about answers that quote constantly — and every fact in such a corpus
    is a code span or a number anyway.
    """
    hit = RagHit(
        "kb/limits.md", 1, 5, "Лимиты", "Лимит `max_summary_ratio` = 0.3.", 1.0, "c1"
    )
    reply, audit = audit_quotes("По умолчанию `max_summary_ratio` = 0.3 [1].", [hit])

    assert audit.kept == ("max_summary_ratio",)
    assert UNSUPPORTED_QUOTE_TEXT not in reply


def test_markdown_emphasis_inside_a_quote_is_not_a_paraphrase() -> None:
    """Reproducing the source's own `**` is quoting it exactly.

    The scorer strips emphasis for the same reason; an audit that failed correct
    answers would be switched off, and a switched-off audit protects nothing.
    """
    hit = RagHit(
        "kb/limits.md", 1, 5, "Лимиты", "Правило **жёсткое**: 30% лимита.", 1.0, "c1"
    )
    reply, audit = audit_quotes("Правило «**жёсткое**: 30% лимита» [1].", [hit])

    assert audit.kept, audit
    assert UNSUPPORTED_QUOTE_TEXT not in reply


def test_a_quote_is_attributed_to_a_citation_that_follows_it() -> None:
    """Measured both ways: the model wrote `[1] «фраза»` about as often as
    `«фраза» [1]`. Reading the marker only backwards silently skipped half."""
    hit = RagHit(
        "kb/hours.md", 11, 15, "Часы", "Часы работы: с 8:00 до 22:00.", 1.0, "c1"
    )

    assert audit_quotes("«с 8:00 до 22:00» [1].", [hit])[1].kept
    assert audit_quotes("[1]. Часы: «с 8:00 до 22:00».", [hit])[1].kept


def test_a_nested_quote_is_not_emitted_twice() -> None:
    """A «…`code`…» span matches both the guillemets and the backticks.

    Checking the inner span as well would splice the same words into the reply a
    second time, which is a corruption the user reads, not a metric that moves.
    """
    hit = RagHit("kb/limits.md", 1, 5, "Лимиты", "Лимит `ratio` = 0.3.", 1.0, "c1")
    reply, audit = audit_quotes("«Лимит `ratio` = 0.3.» [1]", [hit])

    assert reply.count("ratio") == 1
    assert audit.kept == ("Лимит `ratio` = 0.3.",)


def test_an_uncheckable_quote_is_counted_rather_than_silently_skipped() -> None:
    """A quote with a marker a sentence away is not the model's claim about a
    chunk — it is usually the customer's own wording. Leaving it alone is right;
    leaving it *unmentioned* would let a check that examines a third of its input
    report as one that examines all of it."""
    hit = RagHit("kb/hours.md", 11, 15, "Часы", "Часы работы: с 8:00 до 22:00.", 1.0, "c1")
    reply, audit = audit_quotes("Напиток «Тростниковый крем» стоит [1]", [hit])

    assert audit.unchecked == ("Тростниковый крем",)
    assert not audit.dropped
    assert reply == "Напиток «Тростниковый крем» стоит [1]"


def test_the_footer_names_the_chunk_id_not_only_the_line_range() -> None:
    """A line range moves every time the file is re-indexed; the id does not."""
    footer = render_sources_footer(
        [RagHit("kb/hours.md", 11, 15, "Часы работы", "Часы", 1.0, "structure-003-001")],
        (1,),
    )

    assert "[1] kb/hours.md:11-15 — Часы работы" in footer
    assert "chunk structure-003-001" in footer
    assert ragmod.SOURCES_HEADING in footer
    # One source gets a singular heading. A lone «[2]» under «Источники» reads as
    # an off-by-one rather than as the chunk number the answer used.
    assert "Источник:" in footer


def test_the_footer_lists_only_the_chunks_the_answer_cited() -> None:
    """Retrieval returns a shortlist; the answer leans on part of it.

    Measured on a real session: a four-chunk block, an answer citing two of them,
    and a footer naming all four as sources. The extra lines are the ones a reader
    cannot tell apart from real support, and they claim provenance the answer never
    had — the reply asserts two facts and points at two documents.
    """
    hits = [
        RagHit(f"kb/doc{n}.md", 1, 5, f"Раздел {n}", f"Факт {n}.", 0.9, f"c{n}")
        for n in (1, 2, 3, 4)
    ]
    event = RagEvent(
        question="q",
        retrieved=tuple(hit.location for hit in hits),
        candidates=4,
        sources=tuple(hits),
    )

    final = finalize_answer("Высота 800 м [1]. Разница — 250 м [2].", event)
    footer = final.text.split(ragmod.SOURCES_HEADING, 1)[1]

    assert "[1] kb/doc1.md:1-5" in footer
    assert "[2] kb/doc2.md:1-5" in footer
    assert "doc3.md" not in footer
    assert "doc4.md" not in footer


def test_the_footer_keeps_the_numbers_the_answer_used() -> None:
    """Renumbering the list would break the only correspondence the reader has:
    a ``[3]`` in the text has to mean ``[3]`` in the list, even when the answer
    never mentioned [1]."""
    hits = [
        RagHit(f"kb/doc{n}.md", 1, 5, "", f"Факт {n}.", 0.9, f"c{n}")
        for n in (1, 2, 3)
    ]
    event = RagEvent(
        question="q",
        retrieved=tuple(hit.location for hit in hits),
        candidates=3,
        sources=tuple(hits),
    )

    footer = finalize_answer("Факт [3].", event).text.split(ragmod.SOURCES_HEADING, 1)[1]

    assert "[3] kb/doc3.md:1-5" in footer
    assert "[1]" not in footer and "[2]" not in footer


def test_a_refusal_carries_no_source_list() -> None:
    """«I did not find it» followed by a source list says the opposite.

    The list was built from everything that was sent, so a refusal listed the
    documents it had failed to find the answer in — presented as though they
    backed a claim that was never made.
    """
    hit = RagHit("kb/hours.md", 11, 15, "Часы работы", "Часы работы: с 8:00 до 22:00.", 1.0, "c1")
    event = RagEvent(
        question="где душ", retrieved=(hit.location,), candidates=1, sources=(hit,)
    )

    final = finalize_answer("Не нашёл ответ в этом контексте. Уточните, пожалуйста.", event)

    assert ragmod.SOURCES_HEADING not in final.text
    assert final.grounding.status is Grounding.REFUSED


def test_a_citation_to_a_chunk_that_was_never_sent_is_rejected() -> None:
    """Retrieval returns a shortlist; the budget may drop from it before sending.

    Checking against the shortlist rather than what was sent means a hit the
    model never saw passes as evidence — and a fact it could only have known from
    somewhere else is exactly what the audit is for.
    """
    dropped = RagHit("kb/dropped.md", 1, 5, "Прочее", "Выдуманная высота 999 м.", 9.9, "d1")
    sent = RagHit("kb/sent.md", 19, 27, "Прыжок", "Прыжок с высоты 800 м.", 0.6, "c1")
    event = RagEvent(
        question="с какой высоты",
        retrieved=(dropped.location, sent.location),
        candidates=2,
        sources=(sent,),
    )

    final = finalize_answer("С высоты 999 м [kb/dropped.md:1-5].", event)

    assert final.citations.dropped == ("kb/dropped.md",)
    assert ragmod.UNSUPPORTED_CITATION_TEXT in final.text
    assert final.grounding.status is Grounding.UNGROUNDED


def test_no_chunks_or_no_citations_means_no_footer() -> None:
    """An empty source list under a heading would promise evidence and show none."""
    assert render_sources_footer([]) == ""
    hit = RagHit("kb/hours.md", 11, 15, "Часы", "Часы", 1.0, "c1")
    assert render_sources_footer([hit]) == ""
    assert render_sources_footer([hit], ()) == ""


def test_finalize_answer_is_the_one_place_a_reply_is_checked() -> None:
    """The benchmark calls this too. Two implementations is how a report ends up
    scoring the bot's idea of a backed answer instead of the bot's."""
    event = RagEvent(
        question="часы работы",
        retrieved=("kb/hours.md:11-15",),
        candidates=1,
        context_tokens=10,
        sources=(
            RagHit(
                "kb/hours.md", 11, 15, "Часы", "Часы работы: с 8:00 до 22:00.",
                1.0,
                "c1",
            ),
        ),
    )

    final = finalize_answer("Часы [1]: «с 8:00 до 22:00».", event)

    assert final.citations is not None
    assert final.citations.kept == ("kb/hours.md:11-15",)
    assert final.quotes is not None and final.quotes.kept == ("с 8:00 до 22:00",)
    assert final.grounding.status is Grounding.GROUNDED
    assert ragmod.SOURCES_HEADING in final.text


def test_finalize_answer_reports_an_unjudged_reply_as_unchecked() -> None:
    """With citations off there is nothing to verify, and an unjudged answer must
    not be mistaken for a checked one."""
    event = RagEvent(
        question="q",
        retrieved=("kb/hours.md:11-15",),
        candidates=1,
        sources=(RagHit("kb/hours.md", 11, 15, "Часы", "Часы", 1.0, "c1"),),
    )

    final = finalize_answer("Часы [kb/hours.md:11-15].", event, cite=False)

    assert final.citations is None
    assert final.quotes is None
    assert final.grounding.checked is False
    assert final.grounding.status is Grounding.UNGROUNDED
    assert "kb/hours.md" not in final.text


def test_the_verdict_says_which_check_failed_when_nothing_is_wrong_with_it() -> None:
    """The answer from a real session: right sources, right citation, no quote.

    It was reported as ungrounded with no reason attached, because every reason
    list came back empty — a citation was confirmed, so nothing was unsupported,
    and nothing was paraphrased, so nothing was bad. The failure was real
    (no phrase to check the wording against) and the diagnostic was silent about
    it, which is the one combination that cannot be acted on.
    """
    hit = RagHit("kb/jump.pdf", 19, 27, "page 1", "Прыжок с высоты 800 метров.", 1.0, "c1")
    event = RagEvent(
        question="с какой высоты",
        retrieved=(hit.location,),
        candidates=1,
        sources=(hit,),
    )

    verdict = finalize_answer("С высоты 800 м [1].", event).grounding

    assert verdict.status is Grounding.UNGROUNDED
    assert verdict.supported == (hit.location,)
    assert not verdict.unsupported and not verdict.bad_quotes
    # The reason has to be nameable, or the state is indistinguishable from a
    # bug in the checker.
    assert verdict.unquoted is True
    assert verdict.uncited is False


def test_a_quoted_phrase_that_contradicts_the_chunk_passes_the_quote_check() -> None:
    """A limit this task cannot close, pinned so it is not forgotten.

    The model states 900 m and quotes the chunk's 800 m correctly. The quote is
    verbatim, so every deterministic check passes — and the answer is still wrong.
    Catching it needs a second model to judge meaning, which is exactly what this
    mechanism avoids. Recorded as a known ceiling rather than left as a surprise.
    """
    hit = RagHit("kb/jump.pdf", 19, 27, "page 1", "Прыжок с высоты 800 метров.", 1.0, "c1")
    event = RagEvent(
        question="с какой высоты", retrieved=(hit.location,), candidates=1, sources=(hit,)
    )

    verdict = finalize_answer("С высоты 900 м [1] «Прыжок с высоты 800 метров».", event).grounding

    assert verdict.status is Grounding.GROUNDED


def test_a_handle_resolves_to_its_document_when_following_the_subject() -> None:
    """A follow-up turn is kept on the document the dialog is already about.

    A chunk handle names a position in a block the customer never saw, so
    ``[2]`` is not a document. The name lives in the source list the code
    appended, and reading the reply without it put the literal string ``"2"``
    into the next turn's retrieval bias — a filename that does not exist.
    """
    hits = [
        RagHit("kb/dive.md", 1, 15, "page 1", "Урок дайвинга проходит во Владивостоке.", 0.9, "c1"),
        RagHit("kb/jump.pdf", 19, 27, "page 1", "Прыжок с высоты 800 м.", 0.8, "c2"),
    ]
    event = RagEvent(
        question="где проходит урок дайвинга",
        retrieved=tuple(hit.location for hit in hits),
        candidates=2,
        sources=tuple(hits),
    )

    reply = finalize_answer(
        "Урок дайвинга проходит во Владивостоке [1].", event
    ).text

    assert "[1]" in reply and "Источник" in reply
    assert sources_from_text(reply, limit=2) == ["kb/dive.md"]


def test_the_source_list_is_not_mistaken_for_the_model_citing_documents() -> None:
    """The list is our output, and it repeats handles rather than documents.

    Read as citations it contributed a bare handle to every follow-up turn's
    retrieval bias, on top of whatever the model actually cited — so a reply that
    cited nothing yields no subject at all rather than one made of our own output.
    """
    hits = [RagHit("kb/dive.md", 1, 15, "page 1", "Урок дайвинга.", 0.9, "c1")]
    footer = render_sources_footer(hits, (1,))

    assert sources_from_text("Ответ без единой ссылки." + footer, limit=2) == []


def test_a_handle_with_nothing_to_resolve_it_is_dropped_not_passed_on() -> None:
    """Returning ``"2"`` would bias the next turn towards a missing filename,
    which is worse than admitting this turn named no document."""
    assert sources_from_text("Ответ [2] без списка источников.", limit=2) == []


def test_a_quote_between_two_citations_is_resolved_by_which_chunk_holds_it() -> None:
    """The model writes ``[1] «фраза» [2]`` for a phrase taken from the second.

    A positional rule marks those wrong: the phrase is verbatim in one chunk and
    absent from the other, so guessing by proximity fails half of them — on a real
    session it turned «Владивосток. Дайвинг-клуб» into «цитата не подтверждена»
    when it sat in the block right there. The candidates are tried in order and
    the one that actually contains the phrase decides, which keeps the property
    that matters: absent from all of them, it is still refused.
    """
    first = RagHit("kb/dive.md", 16, 29, "page 1", "Количество участников: 2.", 0.9, "c1")
    second = RagHit("kb/dive.md", 1, 15, "page 1", "Место проведения: Владивосток.", 0.8, "c2")
    event = RagEvent(
        question="где проходит",
        retrieved=(first.location, second.location),
        candidates=2,
        sources=(first, second),
    )

    good = finalize_answer('Проходит во Владивостоке [1] «Место проведения: Владивосток» [2].', event)
    assert good.quotes.kept == ("Место проведения: Владивосток",)
    assert good.grounding.status is Grounding.GROUNDED

    bad = finalize_answer('Проходит во Владивостоке [1] «Владивосток. Выдумка» [2].', event)
    assert bad.quotes.dropped == ("Владивосток. Выдумка",)
    assert bad.grounding.status is Grounding.UNGROUNDED


# --------------------------------------------------------------------------- #
# "Sources always": an answer with no sources says which of the two gaps it is
# --------------------------------------------------------------------------- #


def test_a_note_never_reads_as_the_start_of_a_source_list() -> None:
    """The stem every parser looks for must not be a word a note opens with.

    Otherwise ``scripts/batch_ask.py`` splits the answer at the note and renders
    the note itself as a source line — a parser fix per call site instead of one
    wording decision.
    """
    for note in (ragmod.NO_SOURCES_NOTE, ragmod.UNUSED_SOURCES_NOTE):
        assert ragmod.SOURCES_HEADING not in note


def _sent(hit: RagHit) -> RagEvent:
    """One retrieval that sent one chunk."""
    return RagEvent(
        question="вопрос",
        retrieved=(hit.location,),
        candidates=1,
        sources=(hit,),
    )


def test_no_retrieval_at_all_carries_no_note() -> None:
    """Nothing was searched, so claiming documents were not found would be false."""
    final = ragmod.finalize_answer("обычный ответ", None)

    assert final.text == "обычный ответ"
    assert ragmod.sources_note_reason(final.text) is None


def test_nothing_retrieved_says_the_documents_were_not_found() -> None:
    final = ragmod.finalize_answer(
        "мой ответ без опоры",
        RagEvent(question="вопрос", retrieved=(), candidates=0),
    )

    assert ragmod.sources_note_reason(final.text) == ragmod.NOTE_NOT_FOUND
    assert final.text.startswith("мой ответ без опоры")


def test_sent_chunks_left_uncited_say_they_went_unused() -> None:
    """The other gap, and the one worth keeping visible: the model had the
    evidence in front of it and did not lean on any of it."""
    event = _sent(RagHit("kb/a.md", 1, 3, "Часы", "Часы работы.", 1.0, "c1"))
    final = ragmod.finalize_answer("Часы работы.", event)

    assert ragmod.sources_note_reason(final.text) == ragmod.NOTE_UNUSED


def test_a_refusal_still_declares_which_gap_it_was() -> None:
    """A refusal says "I did not find it" but not whether documents were in front
    of it. The reader needs that, and it is what keeps "sources always" true when
    the correct answer is to decline.

    The note claims no support, so it cannot contradict the refusal the way a
    source list would — that is why the exemption this replaces was wrong.
    """
    event = _sent(RagHit("kb/a.md", 1, 3, "Часы", "Часы работы.", 1.0, "c1"))
    final = ragmod.finalize_answer(ragmod.NO_ANSWER_TEXT, event)

    assert ragmod.sources_note_reason(final.text) == ragmod.NOTE_UNUSED
    assert "Источник:" not in final.text


def test_a_cited_answer_carries_no_note() -> None:
    event = _sent(RagHit("kb/a.md", 1, 3, "Часы", "Часы работы.", 1.0, "c1"))
    final = ragmod.finalize_answer("Часы работы [1] «Часы работы».", event)

    assert ragmod.sources_note_reason(final.text) is None
    assert "Источник" in final.text


def test_the_bots_own_refusal_reads_as_a_refusal() -> None:
    """The contract NO_ANSWER_TEXT's docstring states, made testable.

    Strict mode replaces every unbacked answer with this sentence. If the detector
    missed it, the benchmark would score correct behaviour as a confident
    ungrounded answer — the refusal rate would be understated and the mode would
    look like a regression it is not.
    """
    assert is_refusal(NO_ANSWER_TEXT)


def test_notes_are_told_apart_by_reason() -> None:
    assert ragmod.sources_note_reason(ragmod.NO_SOURCES_NOTE) == ragmod.NOTE_NOT_FOUND
    assert ragmod.sources_note_reason(ragmod.UNUSED_SOURCES_NOTE) == ragmod.NOTE_UNUSED


# --------------------------------------------------------------------------- #
# The model writing its own source list
# --------------------------------------------------------------------------- #


IMITATED = (
    "Да, страховка включена в стоимость прыжка. [1]\n\n"
    "Для прыжка в тандеме: страховка 500 руб. отдельно. [4]\n\n"
    "Источники (фрагменты из ответа):\n"
    "[1] Прыжок с парашютом.pdf:1-9 — page 1 · chunk structure-006-000\n"
    "[4] Прыжок в тандеме.pdf:1-10 — page 1 · chunk structure-005-000\n"
    "Источники (фрагменты из ответа):\n"
    "[1] Релаксация в фитобочке с обертыванием.pdf:1-9 — page 1 · chunk structure-007-000\n"
    "[4] Прыжок с парашютом.pdf:28-35 — page 1 · chunk structure-006-004"
)


def test_a_source_list_the_model_invented_is_removed() -> None:
    """Two source lists in one answer, and the reader cannot tell which to
    believe.

    Real run, question about insurance: the model wrote its own footer in the
    shape the protocol shows it, and the code appended the true one after it. The
    invented list named «Релаксация в фитобочке» — a document about a float
    session that had nothing to do with the answer. Both lists looked right, and
    the invented one was audited as if it were the model's citations.
    """
    hits = (
        RagHit("kb/Прыжок с парашютом.pdf", 1, 9, "page 1",
               "Страховка включена.", 1.0, "c1"),
        RagHit("kb/Релаксация в фитобочке.pdf", 1, 9, "page 1",
               "Полотенце включено.", 1.0, "c2"),
        RagHit("kb/Дайвинг.pdf", 1, 9, "page 1",
               "Инструктор включён.", 1.0, "c3"),
        RagHit("kb/Прыжок в тандеме.pdf", 1, 10, "page 1",
               "Страховка 500 руб.", 1.0, "c4"),
    )
    final = ragmod.finalize_answer(
        IMITATED,
        RagEvent(
            question="а есть страховка",
            retrieved=tuple(h.location for h in hits),
            candidates=len(hits),
            sources=hits,
        ),
    )

    assert "Релаксация в фитобочке" not in final.text
    assert "Источники" not in final.text.split(ragmod.SOURCES_HEADING)[0]
    # Exactly one list, and it is the one this code printed.
    assert final.text.count(ragmod.SOURCES_HEADING) == 1
    assert "Прыжок с парашютом.pdf" in final.text
    assert "Прыжок в тандеме.pdf" in final.text


def test_stripping_leaves_an_answer_with_no_list_alone() -> None:
    clean = "Страховка включена в стоимость. [1]\n\nВсё."
    assert ragmod.strip_imitated_footer(clean) == clean


def test_a_word_about_sources_in_the_prose_is_not_a_list() -> None:
    """The heading pattern must not eat prose: «см. раздел Источники: ...» is a
    sentence about documents, not a list of them."""
    text = "См. раздел Источники: там перечислены цены.\nКонец."
    assert ragmod.strip_imitated_footer(text) == text


# --------------------------------------------------------------------------- #
# The model repeats itself
# --------------------------------------------------------------------------- #


def _four_chunks() -> RagEvent:
    hits = tuple(
        RagHit(f"kb/d{i}.md", 1, 5, "стр", f"текст {i}", 1.0, f"c{i}")
        for i in range(4)
    )
    return RagEvent(
        question="вопрос",
        retrieved=tuple(h.location for h in hits),
        candidates=4,
        sources=hits,
    )


LOOPED = "[1] «карамельный сироп».\n\n" * 30


def test_a_reply_that_repeats_one_line_is_refused() -> None:
    """The model fell into a loop and the audit turned it into 31 copies of a
    marker. What made it unrecoverable is that this text goes into the history:
    the next turn reads a page of identical lines as the most recent thing in
    context, which is exactly the pattern the model is already in. Refusing keeps
    a short honest sentence in the history instead."""
    final = ragmod.finalize_answer(LOOPED, _four_chunks())

    assert "карамельный" not in final.text
    assert final.text.startswith(ragmod.NO_ANSWER_TEXT)
    assert ragmod.sources_note_reason(final.text) == ragmod.NOTE_UNUSED
    assert final.grounding.checked is True


def test_the_loop_guard_does_not_fire_on_a_real_answer() -> None:
    """It has to be tight. A list of similar items, a table and a reply that
    repeats a caveat are all normal answers, and refusing them would be worse
    than the loop it guards against."""
    table = "\n".join(f"| {i} | напиток | 100 ₽ |" for i in range(1, 21))
    caveat = "Важно: это верно на март.\n\nВажно: это верно на март."
    listed = "\n".join(f"- карамельный сироп, {i} ₽" for i in range(1, 21))

    assert ragmod.looks_looped(table) is False
    assert ragmod.looks_looped(caveat) is False
    assert ragmod.looks_looped(listed) is False
    assert ragmod.looks_looped(LOOPED) is True


def test_a_short_repetition_is_not_treated_as_a_loop() -> None:
    """Ten copies of a line is a bad habit; thirty is a loop. The threshold is
    set where it only catches the case that actually ruined a run."""
    assert ragmod.looks_looped("[1] «фраза».\n" * 10) is False
    assert ragmod.looks_looped("[1] «фраза».\n" * 12) is True


# --------------------------------------------------------------------------- #
# A repeated problem is one problem, not six
# --------------------------------------------------------------------------- #


def test_six_identical_diagnostics_read_as_one_line_with_a_count() -> None:
    """The model repeats a sentence, so it repeats its citation and its quote,
    and the audit then reported «цитата не подтверждена» six times over.

    Every line that printed the raw list repeated the same non-information, and
    the count of lines said nothing except how long the loop was. What a reader
    needs is which problems happened and how many times.
    """
    assert ragmod.summarise(["цитата"] * 6) == ["цитата (6×)"]
    assert ragmod.summarise(["цитата", "цитата"]) == ["цитата (2×)"]
    assert ragmod.summarise([]) == []
    assert ragmod.summarise(["а", "б", "а"]) == ["а (2×)", "б"]


def test_adjacent_markers_are_collapsed_into_one_with_a_count() -> None:
    """When the model repeats a fragment inside one line, the audit leaves a
    string of identical markers side by side. The marker is there so an
    unverified claim cannot read as verified, and a run of them does the
    opposite — the eye skips straight past it. The count stays, because how many
    claims went unsupported is part of what the answer has to disclose."""
    wall = ("[источник не подтверждён] " * 6).strip()

    out = ragmod.collapse_repeated_markers(wall)

    assert out.count("[источник не подтверждён]") == 1
    assert "×6" in out


def test_markers_on_separate_claims_are_never_merged() -> None:
    """Each bullet is its own claim with its own citation, so each keeps its own
    marker. Merging them would say one claim was unverified when four were —
    which is the number the reader actually needs."""
    listed = (
        "- Полёты по выходным дням [2] [источник не подтверждён]\n"
        "- Перенос даты из-за погоды [2] [источник не подтверждён]\n"
        "- Не рекомендуется при слабом вестибулярном аппарате [2] "
        "[источник не подтверждён]\n"
        "- Сертификат действует 6 месяцев [2] [источник не подтверждён]"
    )

    out = ragmod.collapse_repeated_markers(listed)

    assert out.count("[источник не подтверждён]") == 4
    assert "×" not in out


def test_a_single_marker_and_different_ones_are_left_alone() -> None:
    once = "Факт [1] [источник не подтверждён]. Ещё [2] [цитата не подтверждена]."
    assert ragmod.collapse_repeated_markers(once) == once


def test_a_quote_marker_copied_from_history_is_not_read_as_a_citation() -> None:
    """Real log line, real cause.

    «RAG citations not in the retrieved block: цитата не подтверждена» — six
    times, with the text of a *quote* marker in the list of dropped *citations*.
    The answer had never named such a document. The model had quoted back a
    marker from the previous turn, and the citation audit treated the marker as
    a citation to an unknown document, then relabelled it: the text ended up
    saying «источник не подтверждён» about a quote that had already been
    rejected. Two different problems reported as one, pointing at the wrong one.
    """
    handles = ("kb/a.pdf:1-8",)
    reply = "Воздух несёт вас вверх [1] [цитата не подтверждена]."

    text, audit = ragmod.audit_citations(reply, handles, handles=handles)

    assert audit.dropped == ()
    assert audit.kept == ("kb/a.pdf:1-8",)
    assert text == reply, "маркер должен остаться маркером, а не сменить смысл"


def test_both_of_the_audit_markers_are_recognised_when_copied_back() -> None:
    for marker in (ragmod.UNSUPPORTED_CITATION, ragmod.UNSUPPORTED_QUOTE):
        text, audit = ragmod.audit_citations(
            f"Прошлое утверждение [1] {marker}.", ("kb/a.pdf:1-8",),
            handles=("kb/a.pdf:1-8",),
        )
        assert audit.dropped == (), marker
        assert text == f"Прошлое утверждение [1] {marker}."


def test_wording_attached_to_a_rejected_citation_is_marked_too() -> None:
    """The citation was marked but the wording and its quote were not, so the
    reader saw invented text carrying an invented quote with one bracket beside
    it.

    Real question about a paraglider: the answer explained how a paraglider works
    (no engine, control lines, running down a hill), claimed flights happen at
    weekends, and gave a six-month validity period. The document it cited says
    none of that — only that there is one participant and where the field is. The
    quote check had nothing to work with, because the citation it belonged to had
    already been rejected, and so it passed it by default.
    """
    hit = RagHit("kb/Полёт на паралёте.pdf", 1, 8, "page 1",
                 "Поздравляем! Количество участников: 1.", 1.0, "c")
    event = RagEvent(question="а на параплате это что?",
                     retrieved=(hit.location,), candidates=1, sources=(hit,))
    reply = (
        "Это когда вы летите с парашютом, который сам не имеет мотора [2] "
        "«ветер несёт вас в воздухе». Полёты по выходным дням [2] «выходные». "
        "Сертификат действует 6 месяцев [2] «6 месяцев»."
    )

    final = ragmod.finalize_answer(reply, event)

    # The wording is the model's own and stays readable. What changes is that the
    # answer is no longer allowed to pass as checked: the count is stated, and the
    # citations that failed are marked, so a reader is told rather than misled.
    assert "6 месяцев" in final.text
    assert ragmod.UNSUPPORTED_CITATION in final.text
    assert "Формулировки проверены частично" in final.text
    # Every claim in this answer was an unattributed quote, so once they are all
    # rejected there is nothing but markers left — and leading debris is stripped.
    # What the reader gets is the honest sentence, not a page of placeholders.
    assert ragmod.UNUSED_SOURCES_NOTE in final.text
    assert final.grounding.status is ragmod.Grounding.UNGROUNDED


def test_a_quote_beside_a_valid_citation_is_not_swept_up_by_the_new_rule() -> None:
    """The regression this rule could cause.

    Marking uncited quotes is only safe because it is gated on a citation having
    failed. An answer whose citations all check out must come through untouched,
    or every ordinary reply gets flagged.
    """
    hit = RagHit("kb/a.pdf", 1, 5, "стр", "Количество участников: 1.", 1.0, "c")
    event = RagEvent(question="сколько участников", retrieved=(hit.location,),
                     candidates=1, sources=(hit,))
    reply = 'Количество участников: 1 [1] «Количество участников: 1».'

    final = ragmod.finalize_answer(reply, event)

    assert final.grounding.status is ragmod.Grounding.GROUNDED
    assert ragmod.UNSUPPORTED_QUOTE not in final.text
    assert ragmod.UNSUPPORTED_CITATION not in final.text


def test_a_copied_hit_line_without_a_heading_is_removed() -> None:
    """Real case, from a manual session.

    The reply carried ``[4] Чудеса на виражах.pdf:17-24`` as its own source line,
    and the footer the code printed said the same handle was ``:1-10``. Two line
    ranges for one source, side by side, with the invented one reading as
    authoritative — because the model copies the per-hit header it was shown
    above every chunk. The heading-based strip did not catch it: there was no
    heading.
    """
    reply = (
        "Старт с аэродрома Новонежино [4].\n\n"
        "[4] Чудеса на виражах.pdf:17-24 — page 1 · chunk structure-011-002"
    )
    hits = tuple(
        RagHit(f"kb/Док {i}.pdf", 1, 10, "page 1", f"текст {i}", 1.0, f"c{i}")
        for i in range(1, 4)
    )
    hit = RagHit("kb/Чудеса на виражах.pdf", 1, 10, "page 1",
                 "Место проведения: аэродром Новонежино", 1.0, "c4")
    sources = hits + (hit,)
    event = RagEvent(question="а откуда старт",
                     retrieved=tuple(h.location for h in sources),
                     candidates=len(sources), sources=sources)

    final = ragmod.finalize_answer(reply, event)

    assert "17-24" not in final.text
    assert final.text.count(":1-10") == 1, "осталась ровно одна, настоящая строка"


def test_prose_with_a_file_name_or_a_line_range_is_not_stripped() -> None:
    """Only the block's own header shape: ``[N] file:lines — section``. Prose that
    happens to mention a document, or a citation followed by numbers, is an
    answer and must survive."""
    prose = "См. также Чудеса на виражах.pdf — там детали. Правило [4] в строках 17-24."
    assert ragmod.strip_imitated_footer(prose) == prose


def test_a_document_title_in_quotes_is_a_name_not_a_claim() -> None:
    """Real reply, two near-identical certificates in one corpus:

    «Уточните, какой именно документ вас интересует — [цитата не подтверждена]
    или [цитата не подтверждена]?»

    The model was naming the two documents so the user could pick, and both names
    were replaced by the audit marker. Read as a defect rather than as a question,
    and the question itself was destroyed — which is the only thing the sentence
    was for.
    """
    hits = (
        RagHit("kb/Чудеса на виражах.pdf", 1, 10, "page 1", "Поздравляем!", 1.0, "c1"),
        RagHit("kb/Полёт на паралёте.pdf", 1, 8, "page 1", "Поздравляем!", 1.0, "c2"),
    )
    event = RagEvent(question="на самолете?",
                     retrieved=tuple(h.location for h in hits),
                     candidates=2, sources=hits)
    reply = (
        "Уточните, какой именно документ вас интересует — «Чудеса на виражах» [1] "
        "или «Полёт на паралёте» [2]?"
    )

    out = ragmod.finalize_answer(reply, event).text

    assert "«Чудеса на виражах»" in out
    assert "«Полёт на паралёте»" in out
    assert ragmod.UNSUPPORTED_QUOTE not in out


def test_a_quote_that_is_not_a_title_is_still_marked() -> None:
    """The exemption is for names only. Wording the chunk does not contain stays
    marked, including in the same answer as a document title."""
    hits = (
        RagHit("kb/Чудеса на виражах.pdf", 1, 10, "page 1", "Поздравляем!", 1.0, "c1"),
    )
    event = RagEvent(question="вопрос", retrieved=(hits[0].location,),
                     candidates=1, sources=hits)
    reply = "Документ «Чудеса на виражах» [1] говорит, что «ветер несёт вверх» [1]."

    out = ragmod.finalize_answer(reply, event).text

    # A title is not a claim and is left alone; wording the chunk does not contain
    # is the model's own and also stays readable — but it is counted, so the reply
    # cannot be mistaken for a verified one.
    assert "«Чудеса на виражах»" in out
    assert "«ветер несёт вверх»" in out
    assert "Формулировки проверены частично" in out


def test_naming_a_document_does_not_count_as_evidence() -> None:
    """The exemption that stopped «Чудеса на виражах» being replaced by a marker
    opened a hole: the title went into ``kept``, and ``kept_quotes`` is what the
    grounding verdict requires. A reply that cited one wrong chunk, proved
    nothing, and only named a document then came back GROUNDED — so the CLI
    printed no warning at all, and the wrong answer looked clean.

    Real case: «Полёт на самолёте как наблюдатель возможен [4]» where [4] pointed at
    a parachuting document, with no quote anywhere, and no ``[rag]`` line.
    """
    hits = (
        RagHit("kb/Чудеса на виражах.pdf", 1, 24, "page 1",
               "участник не пилотирует самолёт", 1.0, "c4"),
        RagHit("kb/Прыжок с парашютом.pdf", 1, 9, "page 1",
               "свободный полёт", 1.0, "c2"),
    )
    event = RagEvent(question="ну на самолете всё-таки",
                     retrieved=tuple(h.location for h in hits),
                     candidates=2, sources=hits)
    reply = ("Полёт на самолёте как наблюдатель возможен [2]. "
             "Уточните, какой документ — «Чудеса на виражах» или другой?")

    final = ragmod.finalize_answer(reply, event)

    assert "«Чудеса на виражах»" in final.text
    assert ragmod.UNSUPPORTED_QUOTE not in final.text
    # And it is still an unproven answer.
    assert final.quotes.kept == ()
    assert final.grounding.status is ragmod.Grounding.UNGROUNDED


# --------------------------------------------------------------------------- #
# Handles as ranges and lists
# --------------------------------------------------------------------------- #


HANDLES = ("a.pdf:1-2", "b.pdf:1-2", "c.pdf:1-2", "d.pdf:1-2")


def test_a_range_of_handles_is_read_as_several_chunks() -> None:
    """Real log line: «citations not in the retrieved block: 1-3, 1,3».

    The model cited ``[1-3]`` and ``[1,3]`` meaning chunks 1 and 3. Both were
    parsed as a *single* document whose name is the string ``1-3`` — a document
    that does not exist — so a correct attribution came back unconfirmed, and the
    chunks it named were missing from the source list too.
    """
    for text, wanted in (
        ("Ограничения [1-3]", (1, 2, 3)),      # a range covers everything between
        ("Оба [1,3]", (1, 3)),                 # a list names exactly these
        ("С пробелом [1, 3]", (1, 3)),
    ):
        out, audit = ragmod.audit_citations(text, HANDLES, handles=HANDLES)
        assert audit.dropped == (), text
        assert audit.used_handles == wanted, text
        assert out == text, text


def test_a_range_backwards_reads_the_same() -> None:
    out, audit = ragmod.audit_citations("Всё [3-1]", HANDLES, handles=HANDLES)
    assert audit.dropped == ()
    assert audit.used_handles == (1, 2, 3)


def test_a_range_naming_a_chunk_that_was_not_sent_is_still_rejected() -> None:
    """The new form must not become a way through: a range is a claim about
    several chunks, and if one of them was never sent the bracket is wrong."""
    out, audit = ragmod.audit_citations("Всё [1-5]", HANDLES, handles=HANDLES)

    assert audit.dropped == ("5",)
    assert audit.used_handles == (1, 2, 3, 4)
    assert ragmod.UNSUPPORTED_CITATION in out


def test_a_line_range_is_not_mistaken_for_a_handle_range() -> None:
    """``file.pdf:17-24`` is one document with a range of lines, not handles 17
    through 24. Getting this wrong would reject every correct file citation."""
    locations = ("Чудеса на виражах.pdf:1-24",)
    out, audit = ragmod.audit_citations(
        "Фрагмент [Чудеса на виражах.pdf:17-24]", locations, handles=locations
    )
    assert audit.dropped == ()
    assert out == "Фрагмент [Чудеса на виражах.pdf:17-24]"


def test_an_answer_does_not_open_with_marker_debris() -> None:
    """Real reply:

    ``[1] [цитата не подтверждена] [1]`` then a bare fragment, then the answer.

    The model started mid-thought with citation brackets; the audits replaced the
    quote inside them and left a line with no words in it, above the answer. It
    reads as a broken interface, and it is the first thing in the reply.
    """
    debris = (
        "[1] [цитата не подтверждена] [1]\n\n"
        "[6] «Ореховый — 50 ₽ добавки» [6]\n\n"
        "Проблема: ореховый сироп."
    )

    assert ragmod._strip_leading_debris(debris).startswith("[6] «Ореховый")
    assert ragmod._strip_leading_debris(debris).startswith(
        "[6] «Ореховый")


def test_debris_stripping_leaves_prose_where_it_is() -> None:
    """Only leading lines, and only lines with no words at all. A line with a
    single word is content, and debris in the middle of a reply is not the defect
    being fixed here — eating it would be a guess."""
    assert ragmod._strip_leading_debris("[1] Проблема тут.\n[2]") == \
        "[1] Проблема тут.\n[2]"
    assert ragmod._strip_leading_debris(
        "Первая строка.\n\n[1] [1]\n\nВторая."
    ) == "Первая строка.\n\n[1] [1]\n\nВторая."


def test_evidence_front_loaded_above_the_answer_is_removed() -> None:
    """Real reply, first turn of the coffee dialogue:

    ``[1] «Ореховый сироп содержит фундук и миндаль…» [1]`` and
    ``[6] «Ореховый — 50 ₽ добавки» [6]``, and only then the prose.

    The model checks its evidence and prints it before it starts writing. Every
    word in those lines is already in the source list the code appends, so they
    repeat what the reader is shown anyway — and above the answer they make a
    correct reply look like it failed to start.
    """
    reply = (
        "[1] «Ореховый сироп содержит фундук и миндаль» [1]\n\n"
        "[6] «Ореховый — 50 ₽ добавки» [6]\n\n"
        "Проблема в сиропе. Он содержит орехи [1].\n\n"
        "Что делать:\n- Не предлагать [1]"
    )

    out = ragmod._drop_standalone_evidence(reply)

    assert out.startswith("Проблема в сиропе.")
    assert "50 ₽ добавки» [6]" not in out
    # The citation inside the prose stays; it is the answer's own evidence.
    assert "содержит орехи [1]" in out


def test_an_answer_made_only_of_evidence_lines_is_left_alone() -> None:
    """Stripping it would leave nothing to show, so the lines stay and the
    grounding note says what went wrong instead."""
    only = "[1] «текст» [1]\n[2] «ещё» [2]"
    assert ragmod._drop_standalone_evidence(only) == only


def test_an_ordinary_reply_is_untouched() -> None:
    reply = "Проблема в сиропе. [1] «фраза»"
    assert ragmod._drop_standalone_evidence(reply) == reply


def test_finalize_answer_can_be_applied_twice() -> None:
    """The note was printed twice, stacked.

    A turn that reuses evidence from earlier sends its reply back through this
    function, and the note appended the first time is still in the text — so the
    reader got two identical lines. Reported from a run whose whole answer was a
    refusal: «Документы не использованы — ответ не подтверждён.» immediately
    followed by the same sentence.
    """
    hits = (RagHit("kb/a.pdf", 1, 5, "стр", "текст", 1.0, "c"),)
    event = RagEvent(question="дд", retrieved=(hits[0].location,),
                     candidates=1, sources=hits)
    reply = "Я не нашёл ответа на «дд» в базе знаний кофейни."

    once = ragmod.finalize_answer(reply, event).text
    twice = ragmod.finalize_answer(once, event).text

    assert once.count(ragmod.UNUSED_SOURCES_NOTE) == 1
    assert twice.count(ragmod.UNUSED_SOURCES_NOTE) == 1
