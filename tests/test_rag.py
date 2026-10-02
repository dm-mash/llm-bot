"""Tests for the RAG layer: prompt block, token budget, and session wiring."""

from __future__ import annotations

import json
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


def _make_session(tmp_path: Path, transport, **kwargs):
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
        invariants=False,
        **kwargs,
    )


def _capturing_transport(reply: str = "reply-text") -> tuple[httpx.MockTransport, list[dict]]:
    """A transport that records every request payload it is handed."""
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": reply}}
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


def test_a_budget_that_fits_one_hit_keeps_the_best(index_path: Path) -> None:
    block, event = retriever(index_path, top_k=1, max_context_tokens=100).render(
        QUESTIONS["hours"]
    )
    assert event.dropped == 0
    assert "kb/hours.md" in block
    assert "с 8:00 до 22:00" in block
    assert estimate_tokens(block) + 4 <= 100


def test_a_tiny_budget_truncates_the_top_hit_instead_of_cutting_mid_word(
    index_path: Path,
) -> None:
    block, event = retriever(index_path, top_k=3, max_context_tokens=70).render(
        QUESTIONS["hours"]
    )
    assert event.dropped == 2
    assert "…" in block
    assert estimate_tokens(block) + 4 <= 70
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