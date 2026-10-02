"""Retrieval-augmented generation over the local document index.

The index itself is built by ``scripts/index_documents.py`` and read/searched
through :mod:`llm_bot.retrieval`; this module is the part the agent uses at
request time. A :class:`Retriever` turns one user question into a small block of
the most similar chunks, rendered as a prefix system message that states the
protocol the model must follow:

* answer **only** from the context;
* cite ``file:line`` for every fact, so a claim can be traced to its chunk;
* say plainly that the answer was not found, instead of guessing — the whole
  point of grounding is that a wrong confident answer is worse than a refusal.

The block is a *prefix*, never history: it is rebuilt for every question and is
not persisted with the dialog, so a session does not accumulate stale context
(the same rule the project's ARCH-2 invariant states for tool results).

Retrieval is a pure prefix-building step, so it cannot decide anything by
itself — the model always sees the user's question unchanged. Nothing here can
make the bot act on retrieved text as if it were an instruction (that risk is
covered by the invariant layer, which audits the final reply).

Cost control: the embedder is loaded on the first query, never in
``__init__``, so building a retriever (and failing fast on a missing index)
stays free. The block is capped in tokens like the rolling summary is
(``tokens.py``: ``ceil(chars / 4)``), and when it does not fit, the *worst*
hits are dropped rather than truncating the best one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from .retrieval import Embedder, load_index, query_index
from .tokens import MESSAGE_OVERHEAD_TOKENS, estimate_tokens

#: Section marker + protocol, rendered above the chunks.
_RAG_HEADER = "Контекст из локальной базы знаний:"
_RAG_PROTOCOL = (
    "Отвечай ТОЛЬКО по этому контексту. Для каждого факта укажи источник "
    "в формате [файл:строки]. Если в контексте нет ответа — прямо скажи, "
    "что не нашёл ответ, и ничего не додумывай."
)

#: Share of the context window the block may occupy when no explicit cap is
#: given. Small on purpose: grounding is worth less than the dialog itself, and
#: a long block of low-scoring chunks tends to dilute the question.
DEFAULT_MAX_CONTEXT_RATIO = 0.25

#: Chars per token in the local estimator (``llm_bot.tokens`` divides by 4).
CHARS_PER_TOKEN = 4

#: How many chunks to retrieve per question. The day-21 benchmark puts a
#: single good chunk at rank 1 for roughly 60% of queries, and recall@3 is only
#: ~15 points above recall@1 — so a small number beats a big one, both for
#: prompt budget and for distraction.
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class RagHit:
    """One retrieved chunk, reduced to what the prompt and diagnostics need."""

    source: str
    start_line: int
    end_line: int
    section: str
    text: str
    score: float
    chunk_id: str = ""

    @property
    def location(self) -> str:
        return f"{self.source}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class RagEvent:
    """Diagnostics of one retrieval (mirrors ``MemoryEvent`` / ``MCPEvent``)."""

    question: str
    retrieved: tuple[str, ...]
    candidates: int
    dropped: int = 0
    context_tokens: int = 0
    elapsed: float = 0.0


def rag_budget_tokens(
    max_context_tokens: int | None,
    context_window: int | None,
    ratio: float = DEFAULT_MAX_CONTEXT_RATIO,
) -> int | None:
    """Return the maximum block size in tokens, or ``None`` for no cap.

    The smaller of the explicit cap and the model's context window times
    ``ratio``. When neither is known, returns ``None`` (no cap) — same contract
    as :func:`llm_bot.agent.summary_budget_chars`.
    """
    budget = max_context_tokens
    if context_window is not None:
        by_window = int(context_window * ratio)
        budget = by_window if budget is None else min(budget, by_window)
    if budget is None or budget <= 0:
        return None
    return budget


def _fit_words(text: str, limit: int) -> str:
    """Return ``text`` cut to at most ``limit`` chars, never mid-word.

    A half-word in retrieved evidence reads as if it were a real token from the
    document, which is exactly the kind of small lie grounding must not tell —
    so when not even one whole word fits, the result is ``""`` and the caller
    sends no context at all. Halving the limit converges quickly, so callers
    that need a strictly-fitting block can retry with a smaller one.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    space = text.rfind(" ", 0, limit)
    if space <= 0:
        return ""
    return text[:space].rstrip() + "…"


def _hit_source_line(hit: RagHit) -> str:
    section = f" — {hit.section}" if hit.section else ""
    return f"[{hit.location}]{section}"


class Retriever:
    """Question → top-k chunks of a local index → prompt block.

    The index is read (and validated) eagerly so a missing or foreign file
    fails at construction; the embedding model is loaded lazily on the first
    query, because loading it costs seconds and megabytes and a run that asks
    no questions should not pay for it.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        embedder: Embedder | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_context_tokens: int | None = None,
        context_window: int | None = None,
        max_context_ratio: float = DEFAULT_MAX_CONTEXT_RATIO,
    ) -> None:
        self.index_path = Path(index_path)
        self._index = load_index(self.index_path)
        self._embedder = embedder
        self.top_k = max(1, int(top_k))
        self.max_context_tokens = max_context_tokens
        self.budget_tokens = rag_budget_tokens(
            max_context_tokens, context_window, max_context_ratio
        )
        self.chunking_strategy = str(self._index.get("chunking_strategy", ""))
        self.embedding_model = str(self._index.get("embedding_model", ""))

    # -- retrieval ---------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        """The embedding model, loaded on first use from the index's own name.

        The name comes out of the index rather than from configuration: the
        stored vectors were produced by *that* model, and vectors from two
        models are not comparable. This mirrors the ``--reuse`` path of the
        indexing script, which rejects a contradicting ``--model``.
        """
        if self._embedder is None:
            from .retrieval import DEFAULT_EMBEDDING_MODEL

            self._embedder = Embedder(self.embedding_model or DEFAULT_EMBEDDING_MODEL)
        return self._embedder

    def retrieve(self, question: str) -> tuple[list[RagHit], RagEvent]:
        """Return the top-k chunks for ``question`` plus a diagnostics event."""
        started = time.perf_counter()
        answers = query_index(
            self._index, self.embedder, [question], top_k=self.top_k
        )
        hits: list[RagHit] = []
        for score, record in answers[0] if answers else []:
            meta = record.get("metadata", {}) or {}
            hits.append(
                RagHit(
                    source=str(meta.get("source", "?")),
                    start_line=int(meta.get("start_line", 0) or 0),
                    end_line=int(meta.get("end_line", 0) or 0),
                    section=str(meta.get("section") or meta.get("title") or ""),
                    text=str(record.get("text", "")),
                    score=float(score),
                    chunk_id=str(record.get("chunk_id", "")),
                )
            )
        event = RagEvent(
            question=question,
            retrieved=tuple(hit.location for hit in hits),
            candidates=len(hits),
            elapsed=time.perf_counter() - started,
        )
        return hits, event

    # -- rendering ---------------------------------------------------------

    def render(self, question: str) -> tuple[str, RagEvent]:
        """Build the prompt block for ``question`` ("" when nothing was found).

        Hits are kept in score order until the token budget runs out; the ones
        that do not fit are dropped (recorded in the event) rather than
        truncating the best hit, since a half-chunk is worse than one chunk
        less. When not even the best hit fits on its own, its text is truncated
        to the remaining space so the model still sees the strongest evidence.
        """
        hits, event = self.retrieve(question)
        if not hits:
            return "", event
        block, dropped = self._fit(hits)
        kept = hits[: len(hits) - dropped] if block else []
        return block, replace(
            event,
            retrieved=tuple(hit.location for hit in kept),
            dropped=dropped,
            context_tokens=(
                MESSAGE_OVERHEAD_TOKENS + estimate_tokens(block) if block else 0
            ),
        )

    def _fit(self, hits: list[RagHit]) -> tuple[str, int]:
        """Render as many top hits as the budget allows; return (block, dropped)."""
        budget = self.budget_tokens
        if budget is None:
            return self._block(hits), 0

        kept: list[str] = []
        dropped = 0
        for position, hit in enumerate(hits):
            candidate = [*kept, self._hit_text(hit)]
            if self._size(self._block_from_texts(candidate)) <= budget:
                kept.append(self._hit_text(hit))
            else:
                dropped = len(hits) - position
                break
        if kept:
            return self._block_from_texts(kept), dropped
        # Not even the best hit fits whole: show it truncated to the space that
        # is left. The char-per-token estimate is approximate, so shorten in a
        # loop until the rendered block really fits — a block that overshoots the
        # budget is exactly the kind of silent overflow the cap exists to stop.
        source = _hit_source_line(hits[0])
        full = hits[0].text.strip()
        room = budget * CHARS_PER_TOKEN - len(self._head()) - len(source) - 2
        while room > 0:
            body = _fit_words(full, room)
            if not body:
                break
            block = self._block_from_texts([f"{source}\n{body}"])
            if self._size(block) <= budget:
                return block, len(hits) - 1
            # The char-per-token estimate overshot; halve and retry so the loop
            # always terminates.
            room //= 2
        # Not even one whole word survives: there is no honest context to send.
        # An empty block is better than one that looks like evidence but is not.
        return "", len(hits)

    # -- block formatting --------------------------------------------------

    def _head(self) -> str:
        return f"{_RAG_HEADER}\n{_RAG_PROTOCOL}\n"

    def _hit_text(self, hit: RagHit) -> str:
        return f"{_hit_source_line(hit)}\n{hit.text.strip()}"

    def _block(self, hits: list[RagHit]) -> str:
        return self._block_from_texts([self._hit_text(hit) for hit in hits])

    def _block_from_texts(self, texts: list[str]) -> str:
        lines = [self._head()]
        for text in texts:
            lines.append("")
            lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _size(block: str) -> int:
        return MESSAGE_OVERHEAD_TOKENS + estimate_tokens(block)