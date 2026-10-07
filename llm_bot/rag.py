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
from enum import Enum
from pathlib import Path, PurePath
from collections.abc import Sequence
from typing import Any

from .rerank import DEFAULT_RERANK_CANDIDATES, Reranker, apply_threshold
from .retrieval import Embedder, load_index, query_index
from .tokens import MESSAGE_OVERHEAD_TOKENS, estimate_tokens

#: Section marker + protocol, rendered above the chunks.
_RAG_HEADER = "Контекст из локальной базы знаний:"
_RAG_PROTOCOL = (
    "Отвечай ТОЛЬКО по этому контексту. Каждый фрагмент в контексте помечен "
    "номером в квадратных скобках. После каждого факта приведи дословную фразу "
    "из этого фрагмента в кавычках, а затем укажи номер — например: срок — "
    "6 месяцев [1] «срок — 6 месяцев». Не переписывай имя файла. Если в "
    "контексте нет ответа — прямо скажи, что не нашёл ответ, уточни, что именно "
    "нужно, и ничего не додумывай."
)
# The worked example in that rule is not decoration. Asked to "quote a phrase"
# in the abstract, the model produced one verbatim quote in 15 of 32 answers;
# shown the shape, in 24 of 32. It matters that the example is concrete rather
# than a placeholder: `«фраза из фрагмента» [1]` scored 17 of 24, and an earlier
# placeholder of this kind is what produced `[файл: kofeinya_zerna.md:14]` in a
# saved run — a citation naming nothing that could be retrieved. The example is
# neutral on purpose, and was checked for being copied verbatim into answers
# (24 answers, 0 occurrences).

#: Same rule with the citation clause removed, for when sources are not shown.
#: The grounding and the "say you did not find it" half stay: only the demand
#: to tag every fact with its chunk number goes away.
#:
#: Merely dropping that demand is not enough. The model answers "20 минут
#: [file.pdf:19]", loses the brackets, and then narrates the same source in
#: prose: "Эта информация указана в документе «Отчёт Б» на странице 11-19". So
#: the clause has to forbid the source in words too. Prose cannot be stripped
#: afterwards without mangling the answer, which makes this line the only thing
#: standing between the flag and a reply that still reads like it has sources.
#: The quoted fragment goes with it: a quote with no number is an unbacked claim
#: wearing quotation marks.
_RAG_PROTOCOL_NO_CITATIONS = (
    "Отвечай ТОЛЬКО по этому контексту. Если в контексте нет ответа — прямо скажи, "
    "что не нашёл ответ, уточни, что именно нужно, и ничего не додумывай. "
    "Не указывай источники: ни номера фрагментов, ни имена файлов, ни цитаты в "
    "кавычках. Не называй документ, его номер, страницу или диапазон строк — "
    "ответь только по существу."
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

#: Replacement for a quoted fragment the pointed-at chunk does not contain. Kept
#: as a marker rather than deleted, for the reason the citation marker is kept: a
#: confident sentence with its quote silently removed reads as a fact the sources
#: back.
UNSUPPORTED_QUOTE_TEXT = "цитата не подтверждена"
UNSUPPORTED_QUOTE = f"[{UNSUPPORTED_QUOTE_TEXT}]"

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
    #: The hits that were actually sent, in block order, with everything a
    #: source needs to be shown: name, section, ``chunk_id`` and body text.
    #: ``retrieved`` above is only the locations, which is enough to validate a
    #: citation but not to print a source line or to check a quote against the
    #: chunk it claims to come from. Narrowed by :meth:`Retriever.render` to the
    #: hits the budget kept, so a handle can never point at evidence that was not
    #: sent.
    sources: tuple[RagHit, ...] = field(default=())


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
def _cited_handles(cited: str) -> list[int] | None:
    """Handles a citation names, or ``None`` when it names a document instead.

    The model cites ``[3]`` but also ``[1-3]`` and ``[1,3]``, meaning several
    chunks. Those were being read as one document whose name is the string
    ``1-3``, so a correct attribution came back as «источник не подтверждён» —
    with the chunks it named missing from the footer as well. A line range
    (``файл.pdf:17-24``) is not a handle list: the hyphen there belongs to the
    colon, which is why only text with no colon is considered.
    """
    text = cited.strip()
    if ":" in text:
        return None
    if not re.fullmatch(r"[0-9,\s]+(?:[-–—][0-9]+)?", text):
        return None
    numbers: list[int] = []
    for part in re.split(r"[,]", text):
        part = part.strip()
        if not part:
            continue
        bounds = re.split(r"[-–—]", part)
        try:
            values = [int(b) for b in bounds if b.strip()]
        except ValueError:
            return None
        if not values:
            return None
        if len(values) == 1:
            numbers.append(values[0])
        elif values[0] <= values[1]:
            numbers.extend(range(values[0], values[1] + 1))
        else:
            numbers.extend(range(values[1], values[0] + 1))
    return numbers or None


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


def summarise(items: Sequence[str]) -> list[str]:
    """Collapse repeats: one line per distinct problem, with a count.

    A model that repeats a sentence also repeats its citation and its quote, so
    the raw audit reported «цитата не подтверждена» six times over and every
    report line that printed them repeated the same non-information. What a
    reader needs is which problems happened and how often — one line each is
    enough to act on.
    """
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    out = []
    for item, n in counts.items():
        out.append(item if n == 1 else f"{item} ({n}×)")
    return out


_REPEAT_MARKER_RE = re.compile(
    r"(?P<marker>"
    + re.escape(UNSUPPORTED_QUOTE)
    + r"|"
    + re.escape(UNSUPPORTED_CITATION)
    + r")(?:\s+(?P=marker))*"
)


def collapse_repeated_markers(text: str) -> str:
    """``[x] [x] [x]`` becomes ``[x] ×3``.

    The markers exist so an unverified claim cannot read as verified, and a wall
    of identical ones does the opposite: it pushes the reader's eye past the
    marker entirely. The count is kept, because how many claims were unsupported
    is part of what the answer has to disclose.
    """
    def run(match: re.Match[str]) -> str:
        n = len(re.findall(re.escape(match.group("marker")), match.group(0)))
        return match.group("marker") if n == 1 else f"{match.group('marker')} ×{n}"

    return _REPEAT_MARKER_RE.sub(run, text)


@dataclass
class CitationAudit:
    """Which citations in an answer the retrieved block can actually back."""

    kept: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: Handles the answer actually leaned on, in the order they first appear.
    #: Retrieval sent more than the answer used — on a four-chunk block the model
    #: cited two — and a source list that repeats the whole shortlist claims
    #: provenance the answer never had.
    used_handles: tuple[int, ...] = ()
    #: The block was sent and the answer cited nothing at all. Nothing to
    #: replace, so the text is left alone, but a factual turn with no source is
    #: exactly the case the protocol exists for and it must not pass silently.
    uncited: bool = False

    @property
    def clean(self) -> bool:
        return not self.dropped


@dataclass
class QuoteAudit:
    """Which quoted fragments the chunk they are attributed to really contains.

    Same shape as :class:`CitationAudit` on purpose: the two checks answer the
    same question about different evidence — a citation says *which* chunk, a
    quote says *which words* — so the verdict, the CLI line and the benchmark all
    read them the same way.
    """

    kept: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: Quotes that match the chunk exactly once spacing is ignored — the model
    #: corrected a typo in the source. Not evidence in the strict sense, so not
    #: in ``kept``; not a fabrication either, so not in ``dropped``.
    approx: tuple[str, ...] = ()
    #: Quoted fragments that could not be checked because no citation claimed
    #: them. Reported rather than dropped silently: a check that quietly skips
    #: half its input is indistinguishable from a check that passes everything,
    #: and the size of this number is what says which one it is.
    unchecked: tuple[str, ...] = ()
    #: No quote in the answer was confirmed verbatim. Set on every path, including
    #: the ones where the answer contains no quotes at all: "nothing to check" and
    #: "nothing checked" are the same risk here, and a flag that was only set on
    #: the one path that had quotes reported an unquoted answer as fine.
    uncited: bool = True

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
    handles: Sequence[str] = (),
) -> tuple[str, CitationAudit]:
    """Replace citations in ``text`` that the block at ``locations`` cannot back.

    The protocol asks for the chunk's number, and the model mostly complies — but
    it rewrites what it copies. On a corpus of near-duplicate documents 12 of 15
    citations to one file came back with a single digit changed from the name on
    disk: the digits moved and nothing noticed. So a citation is checked against
    what was actually sent, both the file name and the line range, and anything
    else is replaced with :data:`UNSUPPORTED_CITATION`.

    ``handles`` is the block's locations in printed order, so ``handles[n-1]`` is
    what ``[n]`` names. A number is resolved to the hit it points at and judged by
    that hit's real name and range, which is the point of printing a number: the
    model never has to reproduce a file name to be credited, and a name it still
    tries to reproduce anyway is checked the same way as before.

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
    by_handle: dict[str, str] = {}
    #: First handle per file, for a citation that named a file instead of a number.
    handle_of_name: dict[str, int] = {}
    for number, location in enumerate(handles, 1):
        if _parse_location(location) is not None:
            by_handle[str(number)] = location
            handle_of_name.setdefault(location.split(":", 1)[0], number)
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
    used: list[int] = []
    skipped = {item.strip() for item in ignore}

    def replace(match: re.Match[str]) -> str:
        cited = match.group(1).strip()
        if cited in skipped:
            return match.group(0)
        if cited in (UNSUPPORTED_CITATION_TEXT, UNSUPPORTED_QUOTE_TEXT):
            # One of this audit's own markers, copied back out of the history.
            # It is not a fresh claim, so it must not be counted as one: a
            # previous turn already reported it, and re-reporting would both
            # duplicate the warning and — because the two markers mean different
            # things — relabel a rejected quote as a rejected citation. That
            # showed up as «RAG citations not in the retrieved block: цитата не
            # подтверждена», which accuses the answer of citing a document it
            # never named.
            return match.group(0)
        start = match.group(2)
        end = match.group(3) or start
        pointed = by_handle.get(cited)
        if pointed is not None:
            # A handle stands for the whole hit it was printed above, so it is
            # supported as long as that hit was really sent — there is no range
            # to re-derive and no name to get wrong. Logged as the full location
            # so ``kept`` reads the same whichever form the model chose.
            kept.append(pointed)
            if int(cited) not in used:
                used.append(int(cited))
            return match.group(0)
        numbers = _cited_handles(cited)
        if numbers is not None:
            # A range or a list. Credit every handle that really was sent, so
            # the chunks the model meant appear in the footer; the bracket is
            # marked only if some of the numbers named a chunk that was not.
            missing = [n for n in numbers if str(n) not in by_handle]
            for n in numbers:
                location = by_handle.get(str(n))
                if location is not None:
                    kept.append(location)
                    if n not in used:
                        used.append(n)
            if not missing:
                return match.group(0)
            dropped.extend(str(n) for n in missing)
            return UNSUPPORTED_CITATION
        matched: str | None = None
        for name, first, last in known:
            if not names_match(cited, name):
                continue
            if start is None or (int(start) >= first and int(end) <= last):
                matched = name
                break
        if matched is not None:
            # A citation that named a file still points at one of the printed
            # chunks, so it belongs in the source list under that chunk's number.
            kept.append(cited)
            number = handle_of_name.get(matched)
            if number is not None and number not in used:
                used.append(number)
            return match.group(0)
        dropped.append(cited)
        return UNSUPPORTED_CITATION

    cleaned = _CITATION_RE.sub(replace, text)
    return cleaned, CitationAudit(
        kept=tuple(kept),
        dropped=tuple(dropped),
        used_handles=tuple(used),
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


#: Stem of the source list, exported so checks and parsers look for it instead of
#: repeating the wording. A test asserting on a hardcoded string breaks when
#: someone improves the wording; one that greps for this constant keeps measuring
#: the thing it was written for. It is a stem, not a full heading, because the
#: list has two forms and detection must hold for both — see
#: :func:`render_sources_footer`.
SOURCES_HEADING = "Источник"

#: Two distinct reasons why an answer can carry no sources, worded differently on
#: purpose. Both are appended by code, so the model cannot omit them and cannot
#: produce one where it does not belong.
#:
#: The first is the index's verdict: retrieval was asked and sent nothing. The
#: second is the model's: the chunks were in front of it and none was cited. One
#: is a gap in the data, the other is a gap in the answer, and they send you to
#: different places — folding them into a single "no sources" would hide exactly
#: the second one, which is the signal that the model stopped using evidence.
#:
#: They say «документы», not «источники», on purpose. :data:`SOURCES_HEADING` is
#: the stem every parser looks for to find where the source list begins, and a
#: note opening with the same word would be read as the start of that list — by
#: ``scripts/batch_ask.py`` it would become a source line made of the note.
NO_SOURCES_NOTE = "Документы не найдены — ответ не подтверждён."
UNUSED_SOURCES_NOTE = "Документы не использованы — ответ не подтверждён."


def unverified_note(dropped: int, unchecked: int) -> str:
    """One line, once, instead of a placeholder in every sentence.

    Says how much of the answer could not be checked against the chunk it names,
    which is the fact worth knowing. It goes next to the source list rather than
    inside the prose, because inside the prose it lands mid-sentence and the
    sentence stops meaning anything.
    """
    parts = []
    if dropped:
        parts.append(
            f"{dropped} " + _plural(dropped, "цитата не совпала", "цитаты не совпали",
                                    "цитат не совпало")
        )
    if unchecked:
        parts.append(
            f"{unchecked} " + _plural(unchecked, "фраза", "фразы", "фраз")
            + " без ссылки на источник"
        )
    if not parts:
        return ""
    return "Формулировки проверены частично: " + "; ".join(parts) + \
        " — сверьте их с источниками ниже."


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many

NOTE_NOT_FOUND = "not_found"
NOTE_UNUSED = "unused"

_SOURCES_NOTES = {
    NOTE_NOT_FOUND: NO_SOURCES_NOTE,
    NOTE_UNUSED: UNUSED_SOURCES_NOTE,
}
_NOTE_BY_TEXT = {note: reason for reason, note in _SOURCES_NOTES.items()}


def sources_note_reason(text: str) -> str | None:
    """Which "no sources" note *text* carries, or ``None`` if it carries neither.

    Reports classify with this instead of matching wording of their own, so
    rewording a note leaves the counts intact instead of quietly zeroing them.
    """
    for note, reason in _NOTE_BY_TEXT.items():
        if note in text:
            return reason
    return None


def render_sources_footer(
    sources: Sequence[RagHit], used: Sequence[int] = ()
) -> str:
    """Return the source list for the chunks the answer leaned on.

    Built by code rather than asked from the model. The measurement is why: the
    block asks for a chunk number, yet on a saved run only 9 of 20 answers carried
    any citation at all, and the ones that did were the ones where the model
    happened to copy the shape faithfully. A file name with spaces and
    non-ASCII letters is a bad thing to make correctness depend on — the run also
    produced ``[файл: report.pdf:14]``, the protocol's own placeholder with the
    name glued on, which names no retrievable chunk. Nothing here depends on the
    model cooperating, so "every answer carries its sources" stops being a request
    and becomes a property of the reply.

    Only ``used`` is listed, not everything that was sent. Retrieval returns a
    shortlist and the answer leans on part of it: a four-chunk block where the
    model cited one chunk does not have four sources, it has one. Printing the
    rest claims provenance the answer never had, and the extra lines are exactly
    the ones a reader cannot tell apart from real support.

    The number is the chunk's own handle, never renumbered — a ``[3]`` in the
    text and ``[3]`` here must be the same chunk, since that correspondence is
    the only way a reader can check anything. The cost is that a one-source
    answer reads "Источник [2]" while having nothing else, which looks like an
    off-by-one rather than a reference. So the wording carries the reference
    instead of relying on the list position: singular for one source, and an
    explicit note that the numbers are the ones used in the answer for several.
    Neither form renumbers anything.

    An answer that cited nothing gets no list. A refusal has no evidence, and a
    source list under «I did not find it» says the opposite.
    """
    if not sources or not used:
        return ""
    chosen = [
        (number, sources[number - 1])
        for number in sorted({n for n in used if 1 <= n <= len(sources)})
    ]
    lines = ["", _source_line(chosen)]
    return "\n".join(lines)


def _source_line(chosen: Sequence[tuple[int, RagHit]]) -> str:
    """One rendered entry per used chunk, under the right heading."""
    rendered = []
    for number, hit in chosen:
        parts = [f"[{number}] {hit.location}"]
        if hit.section:
            parts.append(f"— {hit.section}")
        if hit.chunk_id:
            parts.append(f"· chunk {hit.chunk_id}")
        rendered.append(" ".join(parts))
    # Only the heading differs between one source and several; every entry keeps
    # the same shape so the handle reads as a reference in both. The singular
    # exists because a lone «[2]» under a plural heading looks like an off-by-one
    # rather than the chunk number the answer actually used.
    if len(rendered) == 1:
        return f"{SOURCES_HEADING}:\n" + rendered[0]
    return f"{SOURCES_HEADING}и (фрагменты из ответа):\n" + "\n".join(rendered)


#: Quote forms a verbatim phrase may take. «» is what the prompt asks for; the
#: straight pair is accepted because models switch to it unprompted, and the
#: backtick pair because a measured run showed the model quoting code spans in a
#: corpus where every fact *is* a code span or a number — a check blind to that
#: reports "the model does not quote" about answers that quote constantly.
#: Minimum length is per-form: an identifier is one character shorter than prose.
_QUOTE_FORMS: tuple[tuple[int, str, str], ...] = (
    (3, "«", "»"),
    (3, '"', '"'),
    (2, "`", "`"),
)


def _norm_quote(value: str) -> str:
    """Fold a quote and a chunk body to a form that can be compared.

    Case, surrounding punctuation, non-breaking spaces and hyphenation are the
    only differences a faithful paraphrase of a quote can have, so they are
    normalised away. Everything else is left alone: a quote that only matches
    after heavy normalisation is not evidence, it is a coincidence of numbers.
    """
    value = value.replace("\u00a0", " ").replace("\u2011", "-").replace("\u2013", "-")
    value = value.replace("\u2014", "-").strip().strip(".,;:!?")
    # Markdown emphasis is formatting, not wording. A model reproducing the
    # source's own ``**`` and ``_`` has quoted it exactly, and scoring that as a
    # paraphrase would teach the auditor to fail correct answers.
    value = re.sub(r"[*_`]+", "", value)
    return re.sub(r"\s+", " ", value).casefold()


def _tight(value: str) -> str:
    """``_norm_quote`` with every space removed.

    Needed because a source can be missing a space where the model puts one. The
    coffee corpus says «фундук иминдаль»; a model quoting it as «фундук и
    миндаль» has quoted it faithfully and corrected a typo, and
    :func:`_norm_quote` cannot see that — collapsing runs of whitespace does not
    add a space that is not there.

    This only forgives spacing. Every other character still has to be present in
    the same order, so a paraphrase cannot pass by accident; the cost is that a
    real rewrite which happens to differ only in spacing is called approximate,
    which is what it is.
    """
    return _norm_quote(value).replace(" ", "")


def _quotes_in(text: str) -> list[tuple[str, int, int]]:
    """Every quoted fragment as ``(body, start, end)`` spans of ``text``."""
    found: list[tuple[str, int, int]] = []
    for minimum, opener, closer in _QUOTE_FORMS:
        pattern = (
            rf"{re.escape(opener)}"
            rf"([^{re.escape(closer)}\n]{{{minimum},200}})"
            rf"{re.escape(closer)}"
        )
        for match in re.finditer(pattern, text):
            found.append((match.group(1), match.start(), match.end()))
    return sorted(found, key=lambda item: item[1])


#: A citation that *follows* a quote only tags it when almost nothing sits
#: between them — only spaces and punctuation, no words. "«фраза» [1]" and
#: "«фраза», [1]" tag the quote; "Продукт «Название» стоит [1]" does not, because
#: there the marker belongs to the sentence and the quoted word is the
#: customer's own. Distancing the two is what keeps a name the customer typed
#: from being scored as a claim the chunk has to contain.
_QUOTE_CITATION_MAX_GAP = 12


def _owners(
    text: str, markers: list[tuple[int, str]], start: int, end: int
) -> tuple[str, ...]:
    """The citations this quote span could belong to, best guess first.

    More than one when the quote sits between two markers, which is what the
    model does: ``[1] «фраза» [2]`` for a phrase drawn from the second chunk. A
    single positional rule fails half of those — the phrase is verbatim in one
    chunk and absent from the other, so choosing by position alone marks a
    correct quote as unbacked. The candidates are returned in preference order
    and the caller settles it by looking in them, which keeps the property that
    matters: a phrase in none of them is still refused.
    """
    before = [name for position, name in markers if position < start]
    found: list[str] = []
    if before:
        found.append(before[-1])
    # A marker may also tag the quote by following it, but only immediately: a
    # marker a sentence later belongs to that sentence.
    for position, name in markers:
        if position <= end:
            continue
        gap = text[end:position]
        if len(gap) > _QUOTE_CITATION_MAX_GAP:
            break
        if any(character.isalpha() for character in gap):
            break
        found.append(name)
        break
    return tuple(found)


def audit_quotes(
    text: str,
    sources: Sequence[RagHit],
    *,
    ignore: Sequence[str] = (),
    mark_unchecked: bool = False,
) -> tuple[str, QuoteAudit]:
    """Replace quoted fragments that the pointed-at chunk does not contain.

    A citation says *which* chunk a claim came from; it says nothing about the
    wording. The model can cite a correct chunk and still state the fact wrongly,
    and the line-range check passes either way. So the protocol also asks for the
    phrase itself, and this checks that the phrase is really in the chunk — which
    is the one claim in an answer that can be settled without a model.

    Only quotes a citation owns are considered, and a citation may sit on either
    side: measured, the model wrote ``[1] «фраза»`` about as often as
    ``«фраза» [1]``, so reading the marker only backwards silently skipped half
    of them. A quoted product name in the user's own question («Тростниковый
    крем») is not evidence and not a failure; a quoted fragment the model
    attributes to a chunk is a claim about that chunk, and if the words are not
    there it has to be marked.

    ``ignore`` works as in :func:`audit_citations`: brackets that are not sources
    have no chunk to be checked against, and the quotes under them are left alone.

    ``mark_unchecked`` covers the gap that let fabricated wording through. A
    quote nobody cited is normally left alone — it may be the user's own words.
    But when some citation in the same answer *was* rejected, the uncited quotes
    are no longer ambiguous: the model did attach them to a source, that source
    did not survive, and leaving the phrase unmarked presents it as established.
    Real case: a question about a paraglider came back with an explanation of how
    a paraglider works, a claim about weekend flights and a six-month validity
    period — none of it in the document the citation pointed at. The citation was
    marked; the wording and its quote were not, so the reader saw invented text
    carrying an invented quote and only a bracket beside it.
    """
    if not sources:
        return text, QuoteAudit()
    by_handle = {str(number): hit for number, hit in enumerate(sources, 1)}
    skipped = {item.strip() for item in ignore}

    # Names of the documents actually in play. A model that has to disambiguate
    # two similar certificates writes their titles — «Чудеса на виражах» or
    # «Полёт на паралёте» — and that is a name, not a claim about wording.
    # Auditing it produced «Уточните, какой именно документ вас интересует —
    # [цитата не подтверждена] или [цитата не подтверждена]?», which reads as
    # breakage rather than as a document name.
    document_names = {
        _norm_quote(hit.location.split(":", 1)[0])
        for hit in sources
        if hit.location
    }
    document_names.discard("")
    for hit in sources:
        stem = Path(hit.location.split(":", 1)[0]).stem
        if stem:
            document_names.add(_norm_quote(stem))

    def is_document_name(body: str) -> bool:
        needle = _norm_quote(body)
        if not needle:
            return False
        return any(
            needle == name or needle in name or name in needle
            for name in document_names
            if name
        )

    # Citation spans first, so each quote can be attributed to the nearest one
    # on either side of it.
    markers = [
        (match.start(), match.group(1).strip())
        for match in _CITATION_RE.finditer(text)
        if match.group(1).strip() not in skipped
    ]
    if not markers:
        return text, QuoteAudit()

    kept: list[str] = []
    dropped: list[str] = []
    unchecked: list[str] = []
    approx: list[str] = []
    out: list[str] = []
    cursor = 0
    for body, start, end in _quotes_in(text):
        if start < cursor:
            # Already inside a quote that was checked as a whole. Checking the
            # inner span as well would emit the text twice, once from each.
            continue
        candidates = [
            hit for hit in (by_handle.get(name) for name in _owners(text, markers, start, end))
            if hit is not None
        ]
        if not candidates:
            # No citation claims this quote. Left exactly as written — it may be
            # the customer's own wording — unless a citation in this answer was
            # already rejected, in which case the model did mean to attribute it
            # to a source and the attribution failed.
            unchecked.append(body)
            # Deliberately nothing inserted here. An inline placeholder in the
            # middle of a sentence replaced the model's own words and left
            # «но [цитата не подтверждена] не значит [цитата не подтверждена]» —
            # the sentence stopped carrying its meaning to make a point no reader
            # could act on. The count goes next to the source list instead, where
            # it says something and does not cut prose in half.
            continue
        needle = _norm_quote(body)
        target = next(
            (hit for hit in candidates if needle in _norm_quote(hit.text)), None
        )
        out.append(text[cursor:start])
        if target is not None:
            kept.append(body)
            out.append(text[start:end])
        elif any(
            _tight(body) in _tight(hit.text) for hit in candidates
        ):
            # Same characters, different spacing. The model quoted the chunk and
            # silently fixed its typing; calling that a paraphrase would mark a
            # faithful quotation as fabricated. Kept, and reported apart from the
            # quotes that match exactly, so the two are never confused.
            approx.append(body)
            out.append(text[start:end])
        else:
            if is_document_name(body):
                # The document's own title, which the chunk does not contain
                # because the chunk *is* that document. Naming it is not a claim
                # about wording, so it is not marked — and it is deliberately
                # NOT counted as evidence either. Counting it made a reply that
                # only named a document pass as grounded: ``kept_quotes`` was
                # satisfied by the title, so an answer citing one wrong chunk and
                # proving nothing printed no warning at all.
                unchecked.append(body)
                out.append(text[start:end])
            else:
                dropped.append(body)
                out.append(text[start:end])
        cursor = end
    if not out:
        return text, QuoteAudit(
            kept=tuple(kept),
            dropped=tuple(dropped),
            approx=tuple(approx),
            unchecked=tuple(unchecked),
            uncited=not kept,
        )
    out.append(text[cursor:])
    return "".join(out), QuoteAudit(
        kept=tuple(kept),
        dropped=tuple(dropped),
        unchecked=tuple(unchecked),
        uncited=not kept,
    )


def _handles_to_files(footer: str) -> dict[str, str]:
    """``handle -> file name`` from the source list this module wrote.

    Parsed rather than pattern-matched against the citation regex, because the
    list puts the handle in brackets and the document *outside* them —
    ``[2] kb/hours.md:11-15`` — which is the point of the format.
    """
    mapping: dict[str, str] = {}
    for line in footer.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        handle, _, rest = stripped[1:].partition("]")
        rest = rest.strip().lstrip(":").strip()
        location = re.split(r"\s+[—·]", rest, maxsplit=1)[0].strip()
        if handle.strip().isdigit() and location:
            mapping[handle.strip()] = location.split(":", 1)[0]
    return mapping


def sources_from_text(text: str, *, limit: int = 2) -> list[str]:
    """Documents cited in ``text``, most recent first, as bare file names.

    Used to keep a follow-up question on the document the dialog is already
    about. Order is left as found — the caller wants the first documents that
    were cited, not a frequency ranking, because a bot repeats the same file.

    Two things about the modern reply shape are handled here. A chunk handle
    names a position in a block the reader never saw, so ``[2]`` is not a
    document and must never be passed on as one — the file name comes from the
    source list the code appended, which prints the location for exactly that
    handle. And the list itself is skipped: it repeats handles rather than
    documents, so reading it as citations put the literal string ``"2"`` into
    the next turn's retrieval bias.
    """
    body, _, footer = text.partition(SOURCES_HEADING)
    printed = _handles_to_files(footer)
    names: list[str] = []
    for match in _CITATION_RE.finditer(body):
        cited = match.group(1).strip()
        if cited.isdigit():
            # A handle with no entry in the source list names nothing we can
            # retrieve. Passing "2" on as a document biases the next turn's
            # search towards a filename that does not exist.
            if cited not in printed:
                continue
            name = printed[cited]
        else:
            name = cited
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _hit_source_line(hit: RagHit, handle: int = 0) -> str:
    """Header line for one hit: ``[1] file:11-15 — section``.

    The number is what the model is asked to cite. A file name with spaces and
    non-ASCII letters is a bad thing to ask a model to retype: measured on a
    near-duplicate corpus the model produced ``[файл: report.pdf:14]`` — the
    protocol's own placeholder, copied literally, with the name tacked on — and
    the audit could not match it. A number has no such failure mode, and the full
    location stays in the block for the reader and for the audit.
    """
    section = f" — {hit.section}" if hit.section else ""
    if handle:
        return f"[{handle}] {hit.location}{section}"
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
                    chunk_id=str(
                        meta.get("chunk_id")
                        or record.get("chunk_id")
                        or record.get("id")
                        or ""
                    ),
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
            sources=tuple(hits),
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
            sources=tuple(kept),
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
        source = _hit_source_line(hits[0], 1)
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

    def _hit_text(self, hit: RagHit, handle: int) -> str:
        return f"{_hit_source_line(hit, handle)}\n{hit.text.strip()}"

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
        return [self._hit_text(hit, handle) for handle, hit in enumerate(hits, 1)]

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

# --------------------------------------------------------------------------- #
# Refusals and grounding
# --------------------------------------------------------------------------- #

#: The reply declined to answer from the block. Lived here since the benchmark
#: kept its own copy, which meant the CLI and the report could disagree about
#: whether one and the same answer was a refusal — and a disagreement about that
#: is exactly what makes a grounding number meaningless.
REFUSAL_RE = re.compile(
    r"(не наш[её]л|не нахожу|не могу найти|не смог найти|не удалось найти"
    r"|не могу ответить"
    r"|информаци\w*[^.\n]{0,24}?нет\b|нет информации|не содержит"
    r"|не упоминается|не указан\w*|не знаю|не в базе|не в документах"
    r"|отсутствует|не встречается|за рамками (?:этой |моей )?базы)",
    re.IGNORECASE,
)

#: What the bot says instead of a claim it cannot back. Fixed wording on
#: purpose: it is the one sentence that must be recognisable by the same
#: :func:`is_refusal` the benchmark uses, and free wording cannot be.
#:
#: The pattern below is what keeps that promise, so it is stated as a test: a
#: refusal wording the detector misses is scored as a confident answer, which
#: makes strict mode look worse than it is and hides the behaviour it exists for.
NO_ANSWER_TEXT = "Не могу ответить по этому контексту. Уточните, пожалуйста, вопрос."


def is_refusal(text: str) -> bool:
    """``True`` when ``text`` declines rather than answers.

    Deliberately the same predicate the benchmark scores with. A second
    implementation is how a report ends up claiming a refusal rate the runtime
    would never produce.
    """
    return bool(REFUSAL_RE.search(text))


class Grounding(str, Enum):
    """What a reply did with the block it was given."""

    #: Facts, each with a chunk number, and every quoted phrase found verbatim in
    #: the chunk it was attributed to. The only verdict that counts as an answer.
    GROUNDED = "grounded"
    #: Declined. Not a failure — the point of the protocol is that a block
    #: without the answer produces this.
    REFUSED = "refused"
    #: Answered anyway, with nothing checkable behind it.
    UNGROUNDED = "ungrounded"

    def __bool__(self) -> bool:
        """So ``if session.last_grounding:`` reads as "there was a verdict"."""
        return True


@dataclass(frozen=True)
class GroundingVerdict:
    """The verdict plus the evidence behind it, for the CLI and the report."""

    status: Grounding
    #: Chunks cited and confirmed against the block.
    supported: tuple[str, ...] = ()
    #: Chunks the block cannot back.
    unsupported: tuple[str, ...] = ()
    #: Quotes found verbatim in the chunk they were attributed to.
    quotes: tuple[str, ...] = ()
    #: Quotes the cited chunk does not contain.
    bad_quotes: tuple[str, ...] = ()
    #: The block was sent, the model answered, and it cited nothing. Recorded
    #: because it is the common case: 11 of 20 answers on a saved run.
    uncited: bool = False
    #: It cited a chunk and quoted nothing from it. A separate flag from
    #: :attr:`uncited` because it is the more confusing one: the answer carries
    #: sources, names them correctly, reads as if it were checked, and the claim
    #: in it has not been compared against a single word of the evidence. With
    #: only :attr:`uncited` in the verdict, the diagnostic for this state had
    #: nothing to print and said nothing at all.
    unquoted: bool = False
    #: ``False`` when there was nothing to check — citations were switched off,
    #: so the answer is neither backed nor caught out. Kept separate from
    #: :attr:`clean` so an unjudged answer is never counted as a good one.
    checked: bool = True

    @property
    def clean(self) -> bool:
        return self.checked and not self.unsupported and not self.bad_quotes


def judge_grounding(
    reply: str,
    event: RagEvent | None,
    *,
    citations: CitationAudit | None = None,
    quotes: QuoteAudit | None = None,
) -> GroundingVerdict:
    """Decide whether ``reply`` is backed by the block it was sent.

    Based on what can be checked without a model, on purpose. A reranker score
    cannot be used for this: measured on a corpus where the trap question's
    highest-scoring chunk is wrong, its score is ``+0.7`` while a correct chunk
    for a different question sits at ``-4.0``. The two populations overlap, so
    any threshold either drops good answers or lets bad ones through. Whether the
    model actually quoted the chunk it cites is checkable exactly, and that is
    the strongest signal available without asking another model to grade the
    first one.

    Refusal is checked before anything else, because a reply that declined is
    correct by definition and must not be punished for carrying no citation.
    """
    if event is None or not event.sources:
        # No block was sent, so nothing can be grounded in one. Not a refusal —
        # the caller only reaches this with RAG on and retrieval empty, which
        # :meth:`Retriever.render` already treats as an answerable-with-nothing.
        return GroundingVerdict(Grounding.UNGROUNDED)
    if is_refusal(reply):
        return GroundingVerdict(Grounding.REFUSED)

    supported = tuple(citations.kept) if citations else ()
    unsupported = tuple(citations.dropped) if citations else ()
    kept_quotes = tuple(quotes.kept) if quotes else ()
    bad_quotes = tuple(quotes.dropped) if quotes else ()
    uncited = bool(citations.uncited) if citations else not supported
    unquoted = bool(quotes.uncited) if quotes else not kept_quotes

    return GroundingVerdict(
        # A refusal-shaped answer aside, a claim counts only if it points at a
        # chunk that was really sent *and* backs the claim with words from it.
        status=(
            Grounding.GROUNDED
            if supported and kept_quotes and not unsupported and not bad_quotes
            else Grounding.UNGROUNDED
        ),
        supported=supported,
        unsupported=unsupported,
        quotes=kept_quotes,
        bad_quotes=bad_quotes,
        uncited=uncited,
        unquoted=unquoted,
    )


@dataclass(frozen=True)
class FinalAnswer:
    """A reply after the retrieved block has had its say about it.

    ``citations`` and ``quotes`` are ``None`` when nothing was checked — the
    caller asked for no sources, so there is no evidence to verify. That is not
    the same as an empty audit, which means the answer was checked and carried
    neither a citation nor a quote, and the report must not confuse the two.
    """

    text: str
    citations: CitationAudit | None
    quotes: QuoteAudit | None
    grounding: GroundingVerdict


#: The model imitates the source list when the protocol shows its shape, and a
#: second list is the result. Both are plausible-looking, and one of them is
#: invented: a real run answered a question about insurance with a footer naming
#: "Релаксация в фитобочке", which retrieval had returned for some other
#: reason. A reader cannot tell which list to believe, and the audit was reading
#: the invented one.
_IMITATED_FOOTER_RE = re.compile(
    r"^[ \t]*" + re.escape(SOURCES_HEADING) + r"и?[ \t]*(?:\([^\n]*\))?[ \t]*:[ \t]*\n"
    r"(?:[ \t]*(?:\[[^\]]*\][^\n]*|[-–—*][^\n]*|chunk [^\n]*)(?:\n|\Z))+",
    re.IGNORECASE | re.MULTILINE,
)

#: A bare copy of the block's own per-hit header, with no list around it. The
#: block shows the model ``[4] file.pdf:17-24 — page 1 · chunk structure-011-002``
#: above every chunk, and the model sometimes reprints one of those lines as if it
#: were the source list. Real case: the reply carried
#: ``[4] Чудеса на виражах.pdf:17-24`` while the real footer said the same handle
#: was ``:1-10`` — two line ranges for one source, side by side, and the invented
#: one read as authoritative.
_COPIED_HIT_LINE_RE = re.compile(
    r"^\[[^\]]{1,8}\][ \t]+[^\n:]{1,160}?:\d+(?:-\d+)?[ \t]*(?:—|–|-|·).+"
    r"(?:\n|\Z)",
    re.MULTILINE,
)


def strip_imitated_footer(text: str) -> str:
    """Remove a source list the model wrote itself.

    Repeated until nothing changes, because a model that invents a list often
    writes two or three back to back and each one starts a line the previous
    match ended on.

    Everything from the heading down goes, because a half-trimmed list leaves
    the reader worse off than none — an unpaired ``[4]`` above and a list without
    it below. The list this code appends afterwards is the true one.
    """
    for _ in range(4):
        cleaned = _IMITATED_FOOTER_RE.sub("", text)
        cleaned = _COPIED_HIT_LINE_RE.sub("", cleaned)
        # The removed list leaves the blank lines that surrounded it behind.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).rstrip()
        if cleaned == text.rstrip():
            return cleaned
        text = cleaned
    return text


#: A reply this degenerate is not a short answer, it is the model repeating one
#: fragment until it runs out. Measured on a 14-turn coffee dialog: 31 identical
#: markers per turn from turn 3 onward, and every turn after it identical too.
#:
#: This matters beyond the one bad turn. The audit replaces the repeated
#: fragment with a marker, and that text goes into the history the model reads
#: next — so a highly repetitive line becomes the most recent thing the model
#: saw, and the pattern it is already in. Measured: 402 such markers in one
#: report. Refusing is the honest response, and it is also the one that stops the
#: contamination.
_LOOP_MIN_LINES = 12
_LOOP_MIN_RATIO = 0.8


def looks_looped(text: str) -> bool:
    """True when a reply is the same line over and over.

    Lines, not sentences: the audit collapses a repeated *sentence* into a
    repeated marker, so counting sentences after the fact would miss the case
    that matters. The threshold is deliberately high — a list of similar items
    or a table is not a loop, and a run that genuinely repeats a caveat twice
    should not be refused for it.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < _LOOP_MIN_LINES:
        return False
    unique = len(set(lines))
    return (len(lines) - unique) / len(lines) >= _LOOP_MIN_RATIO


def _is_debris_line(line: str) -> bool:
    """True when a line is made of markers and punctuation, with no words.

    Checked by taking the line apart rather than by matching a shape: markers sit
    next to each other separated by brackets, and a pattern for the whole line has
    to get every separator right to notice there is nothing underneath them.
    """
    rest = line
    for marker in (UNSUPPORTED_QUOTE, UNSUPPORTED_CITATION,
                   UNSUPPORTED_QUOTE_TEXT, UNSUPPORTED_CITATION_TEXT):
        rest = rest.replace(marker, " ")
    rest = re.sub(r"\[[^\]]{0,8}\]", " ", rest)
    return not re.sub(r"[\s\W_]+", "", rest, flags=re.UNICODE)


def _strip_leading_debris(text: str) -> str:
    """Drop opening lines that carry nothing but markers.

    A reply that starts mid-thought can open with citation brackets the model was
    still writing when it changed its mind. After the audits replace the quotes
    inside them, what is left is ``[1] [цитата не подтверждена] [1]`` — a line
    with no words at all, sitting above the actual answer.

    Only leading lines are touched. A line containing a single word is prose and
    stays, because that judgement is not available here and guessing it would eat
    content the model did write.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or _is_debris_line(lines[i])):
        i += 1
    return "\n".join(lines[i:]).lstrip("\n")


def _is_evidence_only_line(line: str) -> bool:
    """True when a line is nothing but citations, quotes and punctuation."""
    rest = line
    for quote in ("«[^»]*»", '"[^"]*"'):
        rest = re.sub(quote, " ", rest)
    rest = re.sub(r"\[[^\]]{0,200}\]", " ", rest)
    rest = re.sub(r":[0-9]+(?:-[0-9]+)?", " ", rest)
    return not re.sub(r"[\s\W_]+", "", rest, flags=re.UNICODE)


def _drop_standalone_evidence(text: str) -> str:
    """Remove lines that hold only citations and quoted fragments.

    The model sometimes front-loads the evidence it checked and then writes the
    answer — two lines of ``[1] «…» [1]`` above the prose. Every word in them is
    already printed in the source list this code appends, so they add nothing for
    the reader and make the reply look like it failed to start.

    Guarded on something being left: a reply that is *only* evidence lines still
    has to say something, so in that case the lines stay.
    """
    lines = text.splitlines()
    kept = [l for l in lines if not _is_evidence_only_line(l)]
    if not "".join(kept).strip():
        return text
    return "\n".join(kept)


def finalize_answer(
    reply: str,
    event: RagEvent | None,
    *,
    cite: bool = True,
    invariant_ids: Sequence[str] = (),
) -> FinalAnswer:
    """Check a reply against the block it was sent, and return it ready to show.

    The one place this happens. :class:`~llm_bot.agent.Session` calls it before a
    reply is remembered or persisted, and ``scripts/compare_rag.py`` calls it on
    the answers it collects. Both used to be able to hold their own idea of what a
    backed answer looks like, which is how a report ends up scoring the bot's
    behaviour instead of the bot's.

    Order matters and is the whole point: citations first, because a quote is
    attributed to a chunk by the marker in front of it; then quotes, against the
    hits those markers resolved to; then the source list, appended rather than
    asked for so it cannot be forgotten.

    An answer that ends with neither a source list nor a note is a defect. Before
    this, "cited nothing" and "was told nothing" produced byte-identical text, so
    a run could quietly degrade and the report would still look healthy; now one
    of the two notes always appears, and which one says where to look.
    """
    if event is None:
        # RAG was not in play for this turn. Saying "sources not found" here
        # would report a document search that never ran.
        return FinalAnswer(
            reply, None, None, GroundingVerdict(Grounding.UNGROUNDED, checked=False)
        )

    if not event.sources:
        # Retrieval ran and nothing survived to be sent, so the model had no
        # evidence at all. The answer is unchecked rather than wrong — mark it,
        # or it reads as a plain reply with nothing to doubt it by.
        return FinalAnswer(
            f"{reply}\n\n{NO_SOURCES_NOTE}",
            None,
            None,
            GroundingVerdict(Grounding.UNGROUNDED, checked=False),
        )

    if looks_looped(reply):
        # Refuse rather than show thirty copies of one line. Saying so also
        # keeps the history clean: the next turn reads a short, honest sentence
        # instead of a page of repeated markers, which is what drove the loop in
        # the first place.
        return FinalAnswer(
            f"{NO_ANSWER_TEXT}\n\n{UNUSED_SOURCES_NOTE}",
            None,
            None,
            GroundingVerdict(Grounding.UNGROUNDED, checked=True),
        )

    if not cite:
        # Nothing to verify: without a citation and a quote there is no evidence
        # to check, so the verdict says so rather than blaming the answer. No note
        # either — sources were switched off on purpose, which is not a failure.
        return FinalAnswer(
            strip_citations(reply, ignore=invariant_ids),
            None,
            None,
            GroundingVerdict(Grounding.UNGROUNDED, checked=False),
        )

    handles = [hit.location for hit in event.sources]
    # Before anything is checked. An invented source list has to go first, or it
    # gets audited as if it were the model's citations — which is how a footer
    # naming an unrelated document could pass for evidence.
    reply = strip_imitated_footer(reply)
    text, citations = audit_citations(
        reply,
        # The chunks that were sent, not the ones retrieval returned. A hit can
        # be retrieved and then dropped for the token budget, and a citation to a
        # chunk the model never saw is exactly the failure this audit exists for.
        handles,
        ignore=invariant_ids,
        handles=handles,
    )
    text, quotes = audit_quotes(
        text,
        event.sources,
        ignore=invariant_ids,
        # A rejected citation somewhere in the answer means the wording attached
        # to it is unattributable, not merely uncited.
        mark_unchecked=bool(citations.dropped),
    )
    used = citations.used_handles if citations else ()
    note = unverified_note(len(quotes.dropped), len(quotes.unchecked))
    if note:
        text += f"\n\n{note}"
    text += render_sources_footer(event.sources, used)
    # Chunks were sent and none of them was cited. A refusal gets the note too:
    # it declines in its own words, but the reader still has to know whether the
    # bot had documents in front of it and left them alone — that is the
    # difference between "found nothing" and "refused to answer". The note makes
    # no claim of support, so unlike a source list it cannot contradict a
    # refusal the way «Источник [1]» under «не нашёл» would.
    if not used and UNUSED_SOURCES_NOTE not in text:
        # Idempotent. A turn that reuses evidence from earlier passes its reply
        # back through here, and the note this function appended last time is
        # still in the text — appending again printed it twice, stacked:
        # «Документы не использованы — ответ не подтверждён.» on the line above
        # the identical line.
        text += f"\n\n{UNUSED_SOURCES_NOTE}"
    return FinalAnswer(
        _strip_leading_debris(
            _drop_standalone_evidence(collapse_repeated_markers(text))
        ),
        citations,
        quotes,
        judge_grounding(reply, event, citations=citations, quotes=quotes),
    )
