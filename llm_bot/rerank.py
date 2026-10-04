"""Second retrieval stage: re-score the dense shortlist with a cross-encoder.

Dense search compares one vector per chunk, so a chunk that is on-topic for the
whole document can outrank the chunk that states the fact. On a 13-file
near-duplicate corpus that put the right chunk at rank 11 for a question
about one service's height and weight limits — present, close, and still
outside ``--rag-top-k``. This module fixes that shape of miss: take a wide
shortlist, let a model read each (question, chunk) pair properly, and keep the
best few.

Two defaults are set by measurement on that corpus, not by taste:

``DEFAULT_RERANK_CANDIDATES = 20``
    Recall@20 was 34/34 against 22/34 at rank 4, so a 20-chunk shortlist loses
    no answerable question before the re-scoring starts.

the ``source — section`` header in :func:`passage`
    Feeding the filename to the cross-encoder moved rank-4 recall from 29/34 to
    32/34. The corpus is full of near-duplicate documents whose bodies overlap
    heavily, and the name is what tells them apart — the body text alone does
    not.

There is deliberately **no score threshold**. The two score distributions
    overlap: the lowest-scoring correct chunk sits at -4.014 while an
    unanswerable question about a product that is genuinely absent from the
    corpus peaks at +0.671, above ten correct answers, because the document set
    really is about that product. Cutting at 0 buys 3 refusals and costs 9
    answers. Filtering stays optional (``--rerank-min-score``) and off unless
    asked for.

``sentence_transformers`` is imported inside :meth:`Reranker.__init__`, as in
:mod:`llm_bot.retrieval`, so ``llm_bot`` stays importable without torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Multilingual (mMARCO) and small (12 layers, 384 hidden). A Russian-only or
#: English-only cross-encoder is not an option here: the questions and the
#: documents are Russian, and the English mmarco-* models of the same size score
#: near zero on Cyrillic.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

#: How many chunks dense retrieval hands to the cross-encoder. Wider than
#: ``top_k`` on purpose: this is the recall ceiling for the whole stage.
DEFAULT_RERANK_CANDIDATES = 20


def passage(chunk: dict[str, Any]) -> str:
    """The text the cross-encoder sees: source, section, then the body.

    The header is worth three questions out of 34 at rank 4 — see the module
    docstring. It is capped so one pathological filename cannot eat the model's
    512-token window and truncate the actual evidence.
    """
    meta = chunk.get("metadata", {}) or {}
    source = str(meta.get("source", "")).strip()
    section = str(meta.get("section") or meta.get("title") or "").strip()
    head = f"{source} — {section}".strip(" —")[:200]
    text = str(chunk.get("text", "")).strip()
    return f"{head}\n{text}" if head else text


def window_tokens(model: Any) -> int:
    """The cross-encoder's sequence window.

    ``max_length`` was renamed to ``max_seq_length`` in sentence-transformers 6
    and the old attribute still works but warns. Kept as a function so the
    preference is testable without downloading a model.
    """
    return int(
        getattr(model, "max_seq_length", None)
        or getattr(model, "max_length", 512)
        or 512
    )


class Reranker:
    """Thin wrapper around a sentence-transformers cross-encoder.

    Scores are raw logits on an unbounded scale: they rank well and mean
    nothing on their own, which is exactly why no threshold is applied by
    default. Scores replace the cosine values in the returned hits — after
    re-ranking, that is the number that decided the order.
    """

    def __init__(
        self,
        name: str = DEFAULT_RERANK_MODEL,
        local_only: bool = False,
        *,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        try:
            self._model = CrossEncoder(name, local_files_only=local_only)
        except Exception:  # noqa: BLE001 - offline run without a warm HF cache
            if not local_only:
                raise
            print(f"[warn] {name} is not cached, falling back to a download")
            self._model = CrossEncoder(name)
        self.name = name
        self.batch_size = batch_size
        self.max_tokens = window_tokens(self._model)

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[float]:
        """Score ``(question, passage)`` pairs; higher means more relevant."""
        if not pairs:
            return []
        values = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in values]

    def rerank(
        self,
        question: str,
        hits: Sequence[tuple[float, dict[str, Any]]],
        top_k: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Re-order a dense shortlist and keep the best ``top_k``.

        ``hits`` are ``(cosine, chunk)`` pairs as returned by
        :func:`llm_bot.retrieval.query_index`; the returned scores are
        cross-encoder scores, not cosine. Order is stable for ties, so a
        deterministic benchmark stays deterministic.
        """
        if not hits:
            return []
        scores = self.score([(question, passage(chunk)) for _, chunk in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        top_k = max(1, min(top_k, len(hits)))
        return [(scores[i], hits[i][1]) for i in order[:top_k]]


def apply_threshold(
    hits: Sequence[tuple[float, dict[str, Any]]], min_score: float | None
) -> list[tuple[float, dict[str, Any]]]:
    """Drop hits scoring below ``min_score``; ``None`` disables the filter.

    Allowed to return nothing: on a question the corpus cannot answer, an empty
    context block is the honest outcome — the model then has nothing to ground
    on and is expected to refuse. That is the only way a threshold can help, and
    a near-duplicate corpus says it costs far more than it buys.
    """
    if min_score is None:
        return list(hits)
    return [hit for hit in hits if hit[0] >= min_score]