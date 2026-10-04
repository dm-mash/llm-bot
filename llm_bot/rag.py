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

Two stages, either or both. Dense search alone returns ``top_k`` chunks by
cosine; passing a :class:`~llm_bot.rerank.Reranker` makes the retriever ask
dense for ``candidate_k`` chunks instead, let a cross-encoder read each
(question, chunk) pair, and keep the best ``top_k`` of those. The shortlist stays
wide on purpose — the extra chunks never reach the prompt, so this costs
re-ranking time and no tokens.

Cost control: the embedder is loaded on the first query, never in
``__init__``, so building a retriever (and failing fast on a missing index)
stays free. The block is capped in tokens like the rolling summary is
(``tokens.py``: ``ceil(chars / 4)``), and when it does not fit, the *worst*
hits are dropped rather than truncating the best one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from collections.abc import Sequence
from typing import Any

from .rerank import DEFAULT_RERANK_CANDIDATES, Reranker, apply_threshold
from .retrieval import Embedder, load_index, query_index
from .tokens import MESSAGE_OVERHEAD_TOKENS, estimate_tokens

#: Section marker + protocol, rendered above the chunks.
_RAG_HEADER = "Контекст из локальной базы знаний:"
_RAG_PROTOCOL = (
    "Отвечай ТОЛЬКО по этому контексту. Для каждого факта укажи источник "
    "в формате [файл:строки]. Если в контексте нет ответа — прямо скажи, "
    "что не нашёл ответ, и ничего не додумывай."
)

#: Same rule with the citation clause removed, for when sources are not shown.
#: The grounding and the "say you did not find it" half stay: only the demand
#: to tag every fact with ``[file:lines]`` goes away.
#:
#: Merely dropping that demand is not enough. The model answers "20 минут
#: [file.pdf:19]", loses the brackets, and then narrates the same source in
#: prose: "Эта информация указана в документе «Отчёт Б» на странице 11-19". So
#: the clause has to forbid the source in words too. Prose cannot be stripped
#: afterwards without mangling the answer, which makes this line the only thing
#: standing between the flag and a reply that still reads like it has sources.
_RAG_PROTOCOL_NO_CITATIONS = (
    "Отвечай ТОЛЬКО по этому контексту. Если в контексте нет ответа — прямо скажи, "
    "что не нашёл ответ, и ничего не додумывай. "
    "Не указывай источники: ни в скобках, ни словами. Не называй файл, документ, "
    "его номер, страницу или диапазон строк — ответь только по существу."
)

#: Second rule, added after a near-duplicate corpus exposed a specific failure:
#: two documents for the same offering shared 15 of their 18 indexed lines
#: and differed in three (the title line, one word of the service description,
#: and a number). The model answered a follow-up about the first document from
#: the second one's chunk and then cited the first — a confidently wrong answer
#: with a citation that
#: contradicted it. Ranking was not at fault: both chunks were in the block.
#: Naming the hazard explicitly is what moved the answer.
_RAG_NO_BLENDING = (
    "ВНИМАНИЕ: в базе могут быть несколько РАЗНЫХ документов с похожим названием, "
    "и они не взаимозаменяемы. Каждый факт принадлежит ровно одному документу: бери его "
    "только из блока с тем же именем файла и перескажи точную формулировку оттуда. "
    "Не переноси свойства одного документа на другой. Если клиент не указал, о каком "
    "из них идёт речь, покажи различие явно или уточни, какой он имеет в виду."
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
#:
#: Corrected on a near-duplicate corpus, where that rule cost 10 questions: the
#: day-21 set has no near-duplicate documents, so a flat cosine score had little
#: to confuse. There, dense alone reaches 22/34 at this k while the same shortlist
#: re-ranked reaches 32/34 — the small-k argument holds for the *prompt*, and the
#: shortlist the re-ranker reads is separately sized by ``candidate_k``.
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
    #: How many chunks the re-ranker looked at. Equals ``candidates`` in the
    #: dense-only mode, and is what makes the recall ceiling of the second stage
    #: visible in a log rather than a guess.
    reranked: int = 0
    #: Hits the optional score filter removed. Zero unless ``min_score`` is set.
    filtered: int = 0
    #: Every shortlist entry with its cross-encoder score, best first, before
    #: the top-k cut. This is what a threshold sweep is computed from, so the
    #: sweep in the benchmark report stays reproducible offline.
    scored: tuple[tuple[str, float], ...] = field(default=())


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


#: ``[file.md]`` or ``[file.md:12-18]`` — the citation shapes the protocol asks
#: for. Only square brackets: the corpus has file names with parentheses and
#: colons is common in them, so this stays deliberately narrow. The line range
#: is captured separately because it has to be checked, not just displayed.
_CITATION_RE = re.compile(
    r"\[([^\[\]\n]{1,200}?)(?::(\d+)(?:-(\d+))?)?\]"
)

#: What an unsupported citation turns into. Marked rather than deleted on
#: purpose: a wrong source in brackets reads as a verified one, so quietly
#: removing it would hide the very thing this audit exists to surface.
#: The text inside the brackets, kept separate so the audit can recognise a
#: marker the model copied back instead of counting it as a fresh citation.
UNSUPPORTED_CITATION_TEXT = "источник не подтверждён"
UNSUPPORTED_CITATION = f"[{UNSUPPORTED_CITATION_TEXT}]"


@dataclass
class CitationAudit:
    """Which citations in an answer the retrieved block can actually back."""

    kept: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: The block was sent and the answer cited nothing at all. Nothing to
    #: replace, so the text is left alone, but a factual turn with no source is
    #: exactly the case the protocol exists for and it must not pass silently.
    uncited: bool = False

    @property
    def clean(self) -> bool:
        return not self.dropped


def _parse_location(location: str) -> tuple[str, int, int] | None:
    """``file.md:12-18`` -> (name, 12, 18); ``None`` when there is no range."""
    name, separator, span = location.rpartition(":")
    if not separator or "-" not in span:
        return None
    start, _, end = span.partition("-")
    if not (start.isdigit() and end.isdigit()):
        return None
    return name, int(start), int(end)


def audit_citations(
    text: str,
    locations: Sequence[str],
    *,
    also_backed: Sequence[str] = (),
    ignore: Sequence[str] = (),
) -> tuple[str, CitationAudit]:
    """Replace citations in ``text`` that the block at ``locations`` cannot back.

    The protocol asks for ``[file:lines]``, and the model mostly complies — but
    it rewrites what it copies. On a corpus of near-duplicate documents 12 of 15
    citations to one file came back with a single digit changed from the name on
    disk: the digits moved and nothing noticed. So a citation is checked against
    what was actually sent, both the file name and the line range, and anything
    else is replaced with :data:`UNSUPPORTED_CITATION`.

    ``ignore`` holds brackets that are *not* source citations and must survive
    untouched. The prompt renders every invariant as ``- [STACK-1] (kind) ...``,
    and when the model works its invariant checklist into the reply it echoes
    those ids verbatim. They name rules, not documents, so leaving them out of
    ``kept``/``dropped`` keeps the reply readable and keeps the audit honest:
    before this, a perfectly correct answer logged five "citations not in the
    block" warnings and replaced five valid ids with the unsupported marker.
    """
    known: list[tuple[str, int, int]] = []
    for location in (*locations, *also_backed):
        parsed = _parse_location(location)
        if parsed is not None:
            known.append(parsed)
    # A citation may name a document by its id alone, as in ``[ID-123-456]``.
    # That is the same source, not a different one, so it is accepted when the
    # id belongs to exactly one document in play: if two of them carried the same
    # id, "the id" would not identify anything and the citation has to stay
    # unsupported.
    numbers: dict[tuple[str, ...], set[str]] = {}
    for name, _, _ in known:
        identifier = _digit_id(name)
        if identifier:
            numbers.setdefault(identifier, set()).add(name)

    def names_match(cited: str, name: str) -> bool:
        if _same_source(name, cited):
            return True
        identifier = _digit_id(cited)
        return bool(identifier) and len(numbers.get(identifier, ())) == 1
    if not known:
        return text, CitationAudit()

    kept: list[str] = []
    dropped: list[str] = []
    skipped = {item.strip() for item in ignore}

    def replace(match: re.Match[str]) -> str:
        cited = match.group(1).strip()
        if cited in skipped:
            return match.group(0)
        if cited == UNSUPPORTED_CITATION_TEXT:
            # A previous turn already said this; re-checking it would pile up
            # duplicates in the audit and in the warning log.
            return match.group(0)
        start = match.group(2)
        end = match.group(3) or start
        supported = False
        for name, first, last in known:
            if not names_match(cited, name):
                continue
            if start is None or (int(start) >= first and int(end) <= last):
                supported = True
                break
        if supported:
            kept.append(cited)
            return match.group(0)
        dropped.append(cited)
        return UNSUPPORTED_CITATION

    cleaned = _CITATION_RE.sub(replace, text)
    return cleaned, CitationAudit(
        kept=tuple(kept),
        dropped=tuple(dropped),
        uncited=not kept and not dropped,
    )


def strip_citations(text: str, *, ignore: Sequence[str] = ()) -> str:
    """Remove every source citation from a reply that was asked not to carry any.

    Dropping the cite instruction from the prompt is not enough on its own: the
    block renders each hit as ``Файл: <name>:<lines>`` and the model copies that
    shape into the reply even when nothing asked it to. Measured on a corpus of
    near-duplicate documents — a run with the instruction removed still came back
    with ``[Отчёт Б.pdf:11-19]`` on the line after the answer. A flag that
    promises no sources and prints one is worse than no flag, so the brackets go.

    Both kinds of citation go: the ones the block backs and the ones it does not.
    Leaving the unbacked ones is what a caller asking for a clean answer does
    not want — ``[Услуга «Вечер»]`` names no file and reads as a verified source.
    ``ignore`` keeps the ids that are not sources, for the same reason the audit
    keeps them: an invariant echoed from the protocol names a rule, not a
    document.
    """
    skipped = {item.strip() for item in ignore}

    def drop(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1).strip() in skipped else ""

    cleaned = _CITATION_RE.sub(drop, text)
    # Removing brackets leaves the punctuation they were attached to stranded,
    # so "20 минут [file:19]." comes back as "20 минут ."
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sources_from_text(text: str, *, limit: int = 2) -> list[str]:
    """File names cited in ``text``, most recent first, as bare names.

    Used to keep a follow-up question on the document the dialog is already
    about. Order is left as found — the caller wants the first documents that
    were cited, not a frequency ranking, because a bot repeats the same file.
    """
    names: list[str] = []
    for match in _CITATION_RE.finditer(text):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _hit_source_line(hit: RagHit) -> str:
    section = f" — {hit.section}" if hit.section else ""
    return f"[{hit.location}]{section}"


def _digit_id(name: str) -> tuple[str, ...]:
    """The numeric id in a name: ``ID-250-439-931-337 report.pdf`` ->
    ``("250", "439", "931", "337")``. Empty when there is nothing numeric."""
    groups = re.findall(r"\d+", name)
    return tuple(groups) if len(groups) >= 3 else ()


def _same_source(left: str, right: str) -> bool:
    """Do a cited name and an index source point at the same file?

    An index stores either a bare file name (scanned PDFs) or a path
    relative to the corpus (``kb/hours.md``), and the citation in an answer is
    whatever the model echoed back — so an exact match is not enough on its own.
    """
    return left == right or PurePath(left).name == PurePath(right).name


def _reserve_slots(
    raw: list[tuple[float, dict[str, Any]]],
    pool: list[tuple[float, dict[str, Any]]],
    prefer_sources: Sequence[str],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    """Swap the weakest hits for one hit per preferred source that is missing.

    ``top_k`` is left alone on purpose: a reserved slot costs a slot, and
    growing the block instead would quietly push the token budget (and with it
    the hits the user does need) around. Order is preserved for everything that
    was already in, so score order still drives the ranking.
    """
    def source_of(item: tuple[float, dict[str, Any]]) -> str:
        meta = item[1].get("metadata", {}) or {}
        return str(meta.get("source", "?"))

    wanted = [source for source in dict.fromkeys(prefer_sources) if source]
    if not wanted or len(raw) < 2:
        return raw
    chosen = list(raw)
    present = {source_of(item) for item in chosen}
    # At most half the slots: a reserved hit displaces a scored one, and the
    # top hit has already earned its place. Displacing from the tail keeps that.
    limit = max(1, top_k // 2)
    reserved = 0
    for source in wanted:
        if reserved >= limit:
            break
        if source in present:
            continue
        for item in pool:
            if _same_source(source_of(item), source):
                # Each reservation takes the next slot from the tail, moving
                # left: writing to ``chosen[-1]`` every time would let a second
                # preferred document overwrite the slot the first one just took
                # and the reservation would silently do half its job.
                position = len(chosen) - 1 - reserved
                present.discard(source_of(chosen[position]))
                chosen[position] = item
                present.add(source)
                reserved += 1
                break
    return chosen


def _location_of(chunk: dict[str, Any]) -> str:
    """``file:start-end`` for a raw index chunk, matching :attr:`RagHit.location`."""
    meta = chunk.get("metadata", {}) or {}
    return (
        f"{meta.get('source', '?')}:{int(meta.get('start_line', 0) or 0)}"
        f"-{int(meta.get('end_line', 0) or 0)}"
    )


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
        reranker: Reranker | None = None,
        candidate_k: int | None = None,
        min_rerank_score: float | None = None,
        cite: bool = True,
    ) -> None:
        self.index_path = Path(index_path)
        self.cite = cite
        self._index = load_index(self.index_path)
        self._embedder = embedder
        self.top_k = max(1, int(top_k))
        self.max_context_tokens = max_context_tokens
        self.budget_tokens = rag_budget_tokens(
            max_context_tokens, context_window, max_context_ratio
        )
        self.chunking_strategy = str(self._index.get("chunking_strategy", ""))
        self.embedding_model = str(self._index.get("embedding_model", ""))
        # Injected, never constructed here: loading a second model costs
        # seconds and megabytes, and a run that never asks a question should
        # not pay for it. Without a reranker ``candidate_k`` would be a
        # silently ignored flag, so it collapses back to ``top_k`` instead.
        self._reranker = reranker
        default_candidates = (
            DEFAULT_RERANK_CANDIDATES if reranker is not None else self.top_k
        )
        self.candidate_k = max(
            self.top_k,
            int(candidate_k if candidate_k is not None else default_candidates),
        )
        self.min_rerank_score = min_rerank_score

    # -- retrieval ---------------------------------------------------------

    @property
    def reranker(self) -> Reranker | None:
        """The second stage, when one was injected."""
        return self._reranker

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

    def retrieve(
        self, question: str, *, prefer_sources: Sequence[str] = ()
    ) -> tuple[list[RagHit], RagEvent]:
        """Return the top-k chunks for ``question`` plus a diagnostics event.

        Dense first, re-ranking second (if a reranker was injected), then the
        optional score filter, and only then the top-k cut — so a threshold
        lowers the *effective* top-k instead of hiding chunks the filter kept.

        ``prefer_sources`` reserves one slot per document that the conversation
        is already about. A follow-up like «а он долго действует?» names no
        product at all, and every document in the corpus repeats the same
        "в течение 6 месяцев" boilerplate — so on that question the customer's
        own document fell out of the block entirely and the answer came from an
        unrelated business whose document does state a flat validity period.
        Preferring the documents already cited keeps the dialog on its subject
        without teaching the model anything new.
        """
        started = time.perf_counter()
        deep_k = self.candidate_k if self._reranker is not None else self.top_k
        answers = query_index(self._index, self.embedder, [question], top_k=deep_k)
        raw = answers[0] if answers else []
        reranked = len(raw) if self._reranker is not None else 0

        scored: tuple[tuple[str, float], ...] = ()
        if self._reranker is not None and raw:
            reranked = len(raw)
            raw = self._reranker.rerank(question, raw, self.candidate_k)
            scored = tuple(
                (_location_of(chunk), float(score)) for score, chunk in raw
            )

        before_filter = len(raw)
        kept = apply_threshold(raw, self.min_rerank_score)
        filtered = before_filter - len(kept)
        raw = kept[: self.top_k]
        if prefer_sources:
            raw = _reserve_slots(raw, kept, prefer_sources, self.top_k)

        hits: list[RagHit] = []
        for score, record in raw:
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
            reranked=reranked,
            filtered=filtered,
            scored=scored,
        )
        return hits, event

    # -- rendering ---------------------------------------------------------

    def render(
        self, question: str, *, prefer_sources: Sequence[str] = ()
    ) -> tuple[str, RagEvent]:
        """Build the prompt block for ``question`` ("" when nothing was found).

        Hits are kept in score order until the token budget runs out; the ones
        that do not fit are dropped (recorded in the event) rather than
        truncating the best hit, since a half-chunk is worse than one chunk
        less. When not even the best hit fits on its own, its text is truncated
        to the remaining space so the model still sees the strongest evidence.

        ``prefer_sources`` passes through to :meth:`retrieve`.
        """
        hits, event = self.retrieve(question, prefer_sources=prefer_sources)
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
        for position, text in enumerate(self._texts(hits)):
            if self._size(self._block_from_texts([*kept, text])) <= budget:
                kept.append(text)
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
        protocol = _RAG_PROTOCOL if self.cite else _RAG_PROTOCOL_NO_CITATIONS
        return f"{_RAG_HEADER}\n{protocol}\n{_RAG_NO_BLENDING}\n"

    def _hit_text(self, hit: RagHit) -> str:
        return f"{_hit_source_line(hit)}\n{hit.text.strip()}"

    def _texts(self, hits: Sequence[RagHit]) -> list[str]:
        """Each hit as ``header`` + body.

        The single place a hit becomes block text: the budgeted path, the
        unbudgeted path and the truncation fallback all go through it, so what
        one of them sends the others cannot fall out of.

        This used to insert a card of «what this document says that its twin
        does not» above the body, computed in the indexer. It measured worse:
        over the same 10-turn dialog x 3 runs, 15/30 with the card and 16/30
        without. The card only appears when the retrieved chunk happens to
        contain its lines — absent for the chunk that says «сделать это
        самостоятельно», the one case it was built for — and it puts the
        sibling's file name directly above the text, which is what the
        no-blending rule is about. Kept out of the block, deliberately.
        """
        return [self._hit_text(hit) for hit in hits]

    def _block(self, hits: list[RagHit]) -> str:
        return self._block_from_texts(self._texts(hits))

    def _block_from_texts(self, texts: list[str]) -> str:
        lines = [self._head()]
        for text in texts:
            lines.append("")
            lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _size(block: str) -> int:
        return MESSAGE_OVERHEAD_TOKENS + estimate_tokens(block)