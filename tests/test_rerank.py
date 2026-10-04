"""Tests for the second retrieval stage: ``llm_bot.rerank``.

Offline throughout. ``sentence_transformers`` is never imported: the tests drive
:class:`FakeReranker`, which implements the same two methods the retriever calls
(``score`` and ``rerank``). A real cross-encoder would make the suite need torch
and a ~130 MB download, and it would assert on a model's opinion rather than on
this project's logic.
"""

from __future__ import annotations

import pytest

from llm_bot.rerank import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MODEL,
    Reranker,
    apply_threshold,
    window_tokens,
    passage,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class FakeCrossEncoder:
    """Scores ``(query, passage)`` by the number of shared words.

    Similar in spirit to ``FakeEmbedder`` but deliberately independent of it, so
    a test can build a shortlist where the two disagree: the chunk with the
    *lower* cosine score shares words with the question and must still win. A
    reranker that just echoed the first stage could not pass that.
    """

    max_seq_length = 512

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.pairs.extend(pairs)
        return [
            float(
                len(set(query.lower().split()) & set(text.lower().split()))
            )
            for query, text in pairs
        ]


@pytest.fixture
def fake_reranker() -> Reranker:
    """A :class:`Reranker` with its model swapped out, constructor bypassed."""
    instance = Reranker.__new__(Reranker)
    instance._model = FakeCrossEncoder()
    instance.name = "fake-cross"
    instance.batch_size = 4
    instance.max_tokens = 512
    return instance


def chunk(source: str, text: str, section: str = "", lines=(1, 10)) -> dict:
    return {
        "chunk_id": f"{source}:{lines[0]}",
        "text": text,
        "metadata": {
            "source": source,
            "start_line": lines[0],
            "end_line": lines[1],
            "section": section,
        },
    }


# --------------------------------------------------------------------------- #
# passage()
# --------------------------------------------------------------------------- #


def test_passage_leads_with_the_source_and_section() -> None:
    """The header is worth 3 questions out of 34 on a near-duplicate corpus."""
    text = passage(chunk("a.pdf", "Возраст: 6 лет.", "Условия"))
    assert text == "a.pdf — Условия\nВозраст: 6 лет."


def test_passage_falls_back_to_the_title_when_there_is_no_section() -> None:
    record = chunk("a.md", "тело", section="")
    record["metadata"]["title"] = "Заголовок"
    assert passage(record) == "a.md — Заголовок\nтело"


def test_passage_uses_no_header_when_there_is_nothing_to_name() -> None:
    record = {"text": "тело", "metadata": {}}
    assert passage(record) == "тело"


def test_passage_caps_the_header_so_the_body_is_not_truncated() -> None:
    """A pathological filename must not eat the 512-token model window."""
    text = passage(chunk("x" * 500, "важный факт", "раздел"))
    assert len(text.splitlines()[0]) <= 200
    assert "важный факт" in text


# --------------------------------------------------------------------------- #
# Reranker.rerank()
# --------------------------------------------------------------------------- #


def test_rerank_reorders_and_reports_cross_encoder_scores(fake_reranker) -> None:
    hits = [
        (0.9, chunk("a.md", "общее")),
        (0.8, chunk("b.md", "часы работы")),
    ]
    ranked = fake_reranker.rerank("часы работы", hits, top_k=2)
    # The chunk sharing a word with the question beats the higher cosine score.
    assert ranked[0][1]["metadata"]["source"] == "b.md"
    assert ranked[1][1]["metadata"]["source"] == "a.md"
    # Scores are cross-encoder logits now, so they replace the cosine values and
    # are sorted descending.
    assert ranked[0][0] > ranked[1][0]


def test_rerank_respects_top_k(fake_reranker) -> None:
    hits = [(float(i), chunk(f"{i}.md", "текст")) for i in range(10)]
    assert len(fake_reranker.rerank("текст", hits, top_k=3)) == 3


def test_rerank_passes_the_source_header_to_the_model(fake_reranker) -> None:
    fake_reranker.rerank(
        "часы работы", [(1.0, chunk("kb/hours.md", "часы работы", "Часы"))], top_k=1
    )
    assert fake_reranker._model.pairs == [
        ("часы работы", "kb/hours.md — Часы\nчасы работы")
    ]


def test_rerank_of_nothing_is_nothing(fake_reranker) -> None:
    assert fake_reranker.rerank("вопрос", [], top_k=4) == []


def test_rerank_clamps_top_k_above_the_shortlist(fake_reranker) -> None:
    hits = [(1.0, chunk("a.md", "текст"))]
    assert len(fake_reranker.rerank("текст", hits, top_k=50)) == 1


def test_ties_keep_the_dense_order_so_benchmarks_stay_deterministic(
    fake_reranker,
) -> None:
    """Identical passages must not reshuffle between runs."""
    hits = [(0.9, chunk("a.md", "текст")), (0.8, chunk("b.md", "текст"))]
    ranked = fake_reranker.rerank("текст", hits, top_k=2)
    assert [hit[1]["metadata"]["source"] for hit in ranked] == ["a.md", "b.md"]


# --------------------------------------------------------------------------- #
# score() / apply_threshold()
# --------------------------------------------------------------------------- #


def test_score_of_no_pairs_is_empty(fake_reranker) -> None:
    assert fake_reranker.score([]) == []


def test_threshold_is_off_by_default(fake_reranker) -> None:
    hits = [(-5.0, chunk("a.md", "текст")), (-9.0, chunk("b.md", "текст"))]
    assert apply_threshold(hits, None) == hits


def test_threshold_drops_the_weak_hits(fake_reranker) -> None:
    hits = [(5.0, chunk("a.md", "текст")), (-5.0, chunk("b.md", "текст"))]
    assert apply_threshold(hits, 0.0) == [hits[0]]


def test_threshold_may_drop_everything(fake_reranker) -> None:
    """Empty context on an unanswerable question is the point of the filter."""
    hits = [(-5.0, chunk("a.md", "текст"))]
    assert apply_threshold(hits, 0.0) == []


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_defaults_are_the_measured_ones() -> None:
    """Both are corpus measurements, so a silent change here is a silent change
    to every number in the report."""
    assert DEFAULT_RERANK_MODEL == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # mMARCO is multilingual; an English-only reranker scores ~0 on Cyrillic.
    assert "mmarco" in DEFAULT_RERANK_MODEL
    assert DEFAULT_RERANK_CANDIDATES == 20


def test_the_window_prefers_the_undeprecated_attribute() -> None:
    """``max_length`` warns in sentence-transformers 6 and may go stale."""

    class Modern:
        max_seq_length = 384
        max_length = 512  # deprecated alias, deliberately wrong

    class Legacy:
        max_length = 320

    class Bare:
        pass

    assert window_tokens(Modern()) == 384
    assert window_tokens(Legacy()) == 320
    assert window_tokens(Bare()) == 512