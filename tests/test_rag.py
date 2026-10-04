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
    RagEvent,
    RagHit,
    Retriever,
    rag_budget_tokens,
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
    assert "[kb/hours.md:11-15] — Часы работы" in block
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

    assert ragmod._RAG_PROTOCOL in on and "укажи источник" in on
    assert ragmod._RAG_PROTOCOL_NO_CITATIONS in off
    assert "укажи источник" not in off
    # Removing the demand for [file:lines] was measured to be insufficient: the
    # model drops the brackets and then narrates the source in prose. The
    # no-cite clause has to forbid naming it in words as well.
    assert "ни словами" in off
    assert "его номер" in off
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

    assert session.chat(QUESTIONS["hours"]) == "Мы с 8:00 до 22:00."

    blocks = [b for b in _systems(payloads[0]) if "Контекст из локальной базы" in b]
    assert len(blocks) == 1
    assert "[kb/hours.md:11-15]" in blocks[0]
    assert session.rag_events[-1].question == QUESTIONS["hours"]
    assert "kb/hours.md" in session.last_rag_event.retrieved[0]


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
    assert session.chat("вопрос") == "ничего не нашёл"
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
