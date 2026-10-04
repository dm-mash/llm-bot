#!/usr/bin/env python3
"""Local document indexing pipeline: chunking -> embeddings -> JSON index.

Deliverable for the «Индексация документов» task: it reads a corpus (Markdown,
source code, plain text, PDF), builds a local index with embeddings and
per-chunk metadata, and compares two chunking strategies on it:

* **fixed_size** — every chunk is a sliding window of ``--chunk_size`` tokens
  (or characters, ``--unit chars``) with ``--chunk_overlap`` tokens of overlap;
* **structure** — chunks follow the document's own structure: heading paths for
  Markdown, ``def``/``class`` blocks for Python, pages for PDF, blank-line
  blocks for everything else.

Every chunk carries the metadata the task asks for — ``source``, ``title``,
``section``, ``chunk_id`` — plus ``document_id``, ``chunk_position``,
``start_line``/``end_line``, ``char_count``, ``token_count`` and
``content_hash``, so any hit is traceable back to ``file:line``.

The two indices are compared twice:

* **structural statistics** — chunk counts, size distribution, share of tiny
  and oversized chunks, duplicate rate, corpus coverage, index size;
* **a retrieval benchmark** — queries whose gold answer is a known line span of
  a document. By default the queries are the section headings themselves
  (and ``def``/``class`` names for code); ``--eval`` adds hand-written domain
  queries. Scoring: recall@1/3/5 and MRR@10.

Gold spans are computed from the *raw* documents, so both strategies are
measured against the same ground truth no matter how they were chunked — a
fixed-size chunk can be a hit for a heading query as long as it overlaps the
section's lines.

Storage is plain JSON (``--index_dir``) — no vector database needed. The
markdown report goes to ``--out``.

Examples:
    # ~60 pages of project docs: two strategies + markdown report
    python scripts/index_documents.py --input_dir . --extensions .md

    # docs + source code, smaller windows
    python scripts/index_documents.py --input_dir . --extensions .md,.py \
        --chunk_size 180 --chunk_overlap 30

    # hand-written retrieval queries instead of heading-as-query
    python scripts/index_documents.py --input_dir . --extensions .md \
        --eval index_queries.example.yaml

    # only one strategy, character-based windows, no markdown report
    python scripts/index_documents.py --input_dir docs --extensions .md \
        --strategy structure --unit chars --out ""
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/index_documents.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Search lives in the package, not here: the bot needs it at runtime and both
# sides must score hits identically, so the script re-exports it instead of
# keeping a second copy. ``Embedder`` in particular is still patched by name in
# the tests, which this import keeps working.
from llm_bot.retrieval import (
    INDEX_VERSION,
    Embedder,
    ModelTokenizer,
    load_index,
    query_index,
    render_answers,
    search,
)
from llm_bot.retrieval import DEFAULT_EMBEDDING_MODEL

# Chosen by measuring, not by reputation — see ``llm_bot/retrieval.py`` for the
# full rationale and the numbers behind the choice.
DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL

# The indices hold one 384-float vector per chunk (~3 KB of JSON), so they go to
# a git-ignored directory by default; the report is small and belongs in results/.
DEFAULT_INDEX_DIR = "data/emb"
DEFAULT_REPORT = "results/index_documents_report.md"

# Directories that are never part of a document corpus: VCS, virtualenvs,
# caches and this project's own runtime/generated data.
DEFAULT_EXCLUDES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".idea", ".vscode", "dist", "build", "data", "results",
    # The second RAG corpus (the demo knowledge base) is indexed on its own;
    # keeping it out of the repository corpus stops it from polluting the
    # day-21 benchmark, whose numbers must stay reproducible.
    "knowledge_base",
})

PDF_EXT = ".pdf"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
PY_BLOCK_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<kind>def|class|async def)\s+(?P<name>\w+)")
#: Page separator between extracted PDF pages. ``splitlines()`` treats ``\f`` as
#: a line boundary and would silently drop it, so the text is split on ``\n``
#: instead — see :func:`split_lines`.
PAGE_BREAK = "\f"


def split_lines(text: str) -> list[str]:
    """Split on newlines only, keeping ``\\f`` inside the lines.

    ``str.splitlines`` breaks on ``\\f`` (and on \\x1c-\\x1e, \\x85, \\u2028,
    \\u2029), which erases the page marker before any page-aware parsing can see
    it: every PDF then arrives as one unlabelled section and ``page_of_line``
    always returns "". Only ``\\n`` is a line separator for this corpus.
    """
    return text.split("\n")

#: One «token» for the pure-Python fallback tokenizer: a run of non-space chars.
WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Section:
    """A structural unit of a document with its 1-based inclusive line span.

    ``path`` is the heading path for Markdown (``"A > B"``), the top-level
    symbol name for Python, the page number for PDF, and an empty string for
    plain text.
    """

    path: str
    title: str
    start_line: int
    end_line: int

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclasses.dataclass(frozen=True)
class Document:
    """A source file loaded into memory as a list of lines (1-based access)."""

    path: Path
    rel_path: str
    ext: str
    title: str
    lines: tuple[str, ...]
    sections: tuple[Section, ...] = ()
    page_starts: tuple[int, ...] = ()
    # Line ranges a chunker must not cut at a blank line (fenced code). Filled by
    # the parsers that have such a notion; prose documents legitimately have none.
    verbatim: tuple[tuple[int, int], ...] = ()
    #: PDF pages left out by ``--pdf_max_pages``. Non-zero means the index holds
    #: a prefix of the file, which the report must disclose.
    pages_dropped: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line) + 1 for line in self.lines)


@dataclasses.dataclass
class Chunk:
    """A retrievable unit: text plus the metadata that makes it traceable.

    ``char_start``/``char_end`` are byte-exact offsets into the document, so a
    hit can be checked by span overlap; ``start_line``/``end_line`` are the
    line range a human reads.
    """

    chunk_id: str
    text: str
    strategy: str
    document_id: int
    source: str
    title: str
    section: str
    section_title: str
    chunk_position: int
    start_line: int
    end_line: int
    char_start: int = 0
    char_end: int = 0
    char_count: int = 0
    token_count: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.char_end:
            self.char_end = self.char_start + len(self.text)
        if not self.content_hash:
            self.content_hash = hash_text(self.text)

    def metadata(self) -> dict[str, object]:
        """The metadata block stored next to the embedding."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "section_title": self.section_title,
            "document_id": self.document_id,
            "chunk_position": self.chunk_position,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "chunking_strategy": self.strategy,
        }


@dataclasses.dataclass(frozen=True)
class EvalQuery:
    """A benchmark query: the gold answer is a span of one document."""

    text: str
    source: str
    section: str
    start_line: int
    end_line: int
    char_start: int = 0
    char_end: int = 0
    kind: str = "heading"

    @property
    def gold_id(self) -> str:
        return f"{self.source}:{self.start_line}-{self.end_line}"


@dataclasses.dataclass
class StrategyResult:
    """Chunks, embeddings and both flavours of metrics for one strategy."""

    strategy: str
    chunks: list[Chunk]
    embeddings: list[list[float]]
    index: dict[str, object]
    stats: dict[str, float]
    metrics: dict[str, float]
    encode_seconds: float

    def index_path(self, index_dir: Path) -> Path:
        return index_dir / f"index_{self.strategy}.json"


def hash_text(text: str) -> str:
    """Short, stable content hash used for duplicate detection."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def normalize_extensions(*raw: str | Iterable[str]) -> tuple[str, ...]:
    """``.md``, ``".md,.PY"``, ``[".md"]`` or a mix -> ``(".md", ".py")``."""
    exts: list[str] = []
    for item in raw:
        items = [item] if isinstance(item, str) else list(item)
        for entry in items:
            for part in str(entry).split(","):
                part = part.strip().lower()
                if not part:
                    continue
                exts.append(part if part.startswith(".") else f".{part}")
    return tuple(dict.fromkeys(exts))


def iter_files(
    input_dir: Path,
    extensions: Sequence[str],
    excludes: frozenset[str] = DEFAULT_EXCLUDES,
) -> Iterator[Path]:
    """Yield corpus files under ``input_dir``.

    Sub-directories are pruned by name (``DEFAULT_EXCLUDES`` plus dotted
    names), but the root itself never is: ``--input_dir results`` still indexes
    the project's ``results/``.
    """
    root = input_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in excludes and not d.startswith(".")
        )
        current = Path(dirpath)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = current / name
            if path.suffix.lower() not in extensions:
                continue
            if path.is_file():
                yield path


def extract_pdf_text(path: Path, max_pages: int = 0) -> tuple[str, int]:
    """Extract PDF text, keeping ``\\f`` page breaks for page-aware chunking.

    ``max_pages`` keeps only the leading pages (0 = all) and returns the number
    of pages actually dropped, so the report can say plainly that the index is
    not the whole file. Silently indexing a prefix is the kind of quiet gap that
    makes a later "the fact is not in the corpus" verdict wrong: the answer sat
    on page 3 of a 4-page PDF and the corpus claim was never checked.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF support needs pypdf: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    kept = reader.pages if max_pages <= 0 else reader.pages[:max_pages]
    pages = [(page.extract_text() or "").rstrip() for page in kept]
    # The marker gets a line of its own: page text is rstripped, so a bare join
    # would weld the last line of page N to the first line of page N+1 and hand
    # the chunker a section boundary in the middle of a line.
    return f"\n{PAGE_BREAK}\n".join(pages), len(reader.pages) - len(kept)


def document_title(rel_path: str, ext: str, lines: Sequence[str]) -> str:
    """The document title: first H1 for Markdown, stem for everything else."""
    if ext == ".md":
        for line in lines:
            match = HEADING_RE.match(line)
            if match and len(match.group(1)) == 1:
                return match.group(2).strip() or Path(rel_path).stem
            if line.strip():
                break
    return Path(rel_path).stem


def load_document(
    path: Path,
    root: Path,
    ext: str,
    pdf_max_pages: int = 0,
) -> Document | None:
    """Read one file into a :class:`Document`; ``None`` if it is unusable."""
    pages_dropped = 0
    try:
        if ext == PDF_EXT:
            raw, pages_dropped = extract_pdf_text(path, pdf_max_pages)
        else:
            raw = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, RuntimeError):
        return None
    lines = tuple(split_lines(raw))
    if not any(line.strip() for line in lines):
        return None
    rel_path = path.resolve().relative_to(root.resolve()).as_posix()
    title = document_title(rel_path, ext, lines)
    if ext == ".md":
        sections = parse_headings(lines)
    elif ext == ".py":
        sections = parse_code_blocks(lines)
    else:
        sections = parse_generic_blocks(lines)
    page_starts = tuple(start for start, _ in page_spans(lines))
    return Document(
        verbatim=tuple(fenced_ranges(lines)),
        path=path,
        rel_path=rel_path,
        ext=ext,
        title=title,
        lines=lines,
        sections=sections,
        page_starts=page_starts,
        pages_dropped=pages_dropped,
    )


def collect_documents(
    input_dir: Path,
    extensions: Sequence[str],
    excludes: frozenset[str] = DEFAULT_EXCLUDES,
    max_docs: int = 0,
    pdf_max_pages: int = 0,
) -> tuple[list[Document], list[str]]:
    """Load every corpus file. Returns ``(documents, notes)`` with skip reasons.

    ``notes`` groups repeated skip reasons (one empty ``__init__.py`` per
    package is not worth a report line each), and PDF files are skipped with a
    hint when the optional ``pypdf`` dependency is missing.

    ``pdf_max_pages`` is surfaced as a note of its own: a truncated corpus has to
    declare itself, otherwise "the fact is not in the index" and "the fact was
    never indexed" look identical in the report.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    documents: list[Document] = []
    skipped: Counter[str] = Counter()
    dropped_pages = 0
    for path in iter_files(input_dir, extensions, excludes):
        ext = path.suffix.lower()
        doc = load_document(path, input_dir, ext, pdf_max_pages)
        if doc is None:
            reason = "no text layer (install pypdf for PDF)" if ext == PDF_EXT else (
                "empty or unreadable"
            )
            skipped[f"{path.suffix or path.name}: {reason}"] += 1
            continue
        documents.append(doc)
        dropped_pages += doc.pages_dropped
        if max_docs and len(documents) >= max_docs:
            break
    if dropped_pages:
        skipped[f"PDF: {dropped_pages} страниц не попали в индекс "
                f"(--pdf_max_pages {pdf_max_pages})"] += 1
    notes = [
        f"{count} × {reason}" if count > 1 else reason
        for reason, count in sorted(skipped.items())
    ]
    return documents, notes


# --------------------------------------------------------------------------- #
# Structure parsing
# --------------------------------------------------------------------------- #


def parse_headings(lines: Sequence[str]) -> tuple[Section, ...]:
    """Markdown sections by heading, with a fence-aware scan.

    Fenced code blocks are skipped, so ``# comment`` lines inside ``````` ``` ````
    examples never become sections. Each section ends right before the next
    heading of any level; the heading line itself belongs to the section (the
    chunk text keeps it, which makes the chunk self-describing).
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    starts: list[tuple[int, int, str, str]] = []  # (level, line, path, title)
    fenced = lines_in_ranges(fenced_ranges(lines))
    for number, line in enumerate(lines, 1):
        if number in fenced:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        level, title = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        starts.append((level, number, " > ".join(t for _, t in stack), title))
    if not starts:
        return (Section("", "", 1, len(lines)),)
    if starts[0][1] > 1:
        sections.append(Section("", "", 1, starts[0][1] - 1))
    for index, (_, number, path, title) in enumerate(starts):
        end = starts[index + 1][1] - 1 if index + 1 < len(starts) else len(lines)
        sections.append(Section(path, title, number, end))
    return tuple(sections)


def parse_code_blocks(lines: Sequence[str]) -> tuple[Section, ...]:
    """Python top-level ``def``/``class`` blocks, each with its leading comments.

    The preamble before the first block becomes section ``"<module>"``; a
    block's span starts at its decorators/comment block so docstrings and
    comments stay with the symbol they document.
    """
    sections: list[Section] = []
    starts: list[tuple[int, str, str]] = []  # (line, path, title)
    fenced = lines_in_ranges(fenced_ranges(lines))
    for number, line in enumerate(lines, 1):
        if number in fenced or not line[:1].isalpha():
            continue
        match = PY_BLOCK_RE.match(line)
        if not match or match.group("indent"):
            continue
        kind = "class" if match.group("kind") == "class" else "def"
        starts.append((number, match.group("name"), f"{kind} {match.group('name')}"))
    if not starts:
        return (Section("<module>", "<module>", 1, len(lines)),)
    if starts[0][0] > 1:
        sections.append(Section("<module>", "<module>", 1, starts[0][0] - 1))
    for index, (number, name, title) in enumerate(starts):
        start = number
        for back in range(number - 2, -1, -1):
            candidate = lines[back].strip()
            if not candidate:
                break
            if candidate.startswith(("#", "@")):
                start = back + 1
                continue
            break
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        sections.append(Section(name, title, start, end))
    return tuple(sections)


def page_spans(lines: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """1-based inclusive line spans of the ``\\f``-separated pages.

    The marker normally sits on a line of its own (see
    :func:`extract_pdf_text`) and belongs to no page, so it is skipped. A marker
    found inside a line instead closes the page at that line: the text after it
    is the next page's, and dropping the tail would lose content.
    """
    spans: list[tuple[int, int]] = []
    start = 1
    for number, line in enumerate(lines, 1):
        if PAGE_BREAK not in line:
            continue
        # ``str.strip()`` eats ``\f`` too (it is whitespace), so the line is
        # emptied first and only ``\f`` is left over.
        if line.replace(PAGE_BREAK, "").strip() == "":
            if number - 1 >= start:
                spans.append((start, number - 1))
            start = number + 1
        else:
            spans.append((start, number))
            start = number
    if start <= len(lines):
        spans.append((start, len(lines)))
    return tuple(spans)


def parse_generic_blocks(lines: Sequence[str]) -> tuple[Section, ...]:
    """Plain text / YAML: pages (``\\f``) if present, otherwise the whole file."""
    spans = page_spans(lines)
    whole = (1, len(lines))
    if not spans or spans == (whole,):
        return (Section("", "", 1, len(lines)),)
    return tuple(
        Section(f"page {index}", f"page {index}", start, end)
        for index, (start, end) in enumerate(spans, 1)
    )


def fenced_ranges(lines: Sequence[str]) -> list[tuple[int, int]]:
    """1-based inclusive line ranges of ``` / ~~~ fenced blocks, delimiters included.

    One state machine for every parser that needs to know "is this line code, not
    structure?": a heading inside a bash block is not a heading, a ``def`` inside
    a markdown example is not a symbol. The ranges are also what keeps the
    chunker from splitting a fenced example at a blank line inside it — a blank
    line is a paragraph boundary everywhere except inside verbatim text.
    """
    ranges: list[tuple[int, int]] = []
    fence: str | None = None
    opened = 0
    for number, line in enumerate(lines, 1):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence, opened = marker[0] * 3, number
            elif marker.startswith(fence):
                ranges.append((opened, number))
                fence = None
    if fence is not None:  # unterminated fence: to the end of the document
        ranges.append((opened, len(lines)))
    return ranges


def lines_in_ranges(
    ranges: Sequence[tuple[int, int]], start: int = 1, end: int | None = None
) -> set[int]:
    """The line numbers covered by ``ranges`` (optionally clipped to a span)."""
    last = end if end is not None else max((hi for _, hi in ranges), default=0)
    return {
        number
        for low, high in ranges
        for number in range(max(low, start), min(high, last) + 1)
    }


def blank_line_runs(
    lines: Sequence[str],
    start: int,
    end: int,
    protected: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Non-blank 1-based inclusive line runs inside ``[start, end]``.

    A blank line inside a ``protected`` range is not a boundary: splitting a
    fenced code block at one of its own blank lines would produce chunks that
    are not valid code and not findable by the words inside them.
    """
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    verbatim = lines_in_ranges(protected, start, end) if protected else set()
    for number in range(start, end + 1):
        if number - 1 < len(lines) and lines[number - 1].strip():
            if run_start is None:
                run_start = number
        elif run_start is not None and number not in verbatim:
            runs.append((run_start, number - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def page_of_line(doc: Document, line: int) -> str:
    """PDF page label for a 1-based line, if the document has page breaks.

    ``bisect_right`` on the first line of each page maps every line of that page
    to its own number, including the last one, because the next page's first
    line is strictly greater.
    """
    if not doc.page_starts:
        return ""
    index = bisect.bisect_right(doc.page_starts, line)
    return f"page {index}" if index else ""


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


class CharTokenizer:
    """Whitespace tokenizer returning character spans.

    Used when no embedding model is available, and by the tests. Tokens are
    word-shaped, so windows never cut a word in half.
    """

    name = "whitespace"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in WORD_RE.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.spans(text))


# --------------------------------------------------------------------------- #
# Chunkers
# --------------------------------------------------------------------------- #


def slice_lines(doc: Document, start: int, end: int) -> str:
    """Text of the inclusive 1-based line span, without surrounding blank lines."""
    return "\n".join(doc.lines[start - 1:end]).strip()


def line_starts(doc: Document) -> list[int]:
    """Character offset of every line start, for offset -> line lookups."""
    starts = [0]
    for index, line in enumerate(doc.lines):
        starts.append(starts[index] + len(line) + 1)
    return starts


def line_of_offset(starts: Sequence[int], offset: int) -> int:
    """1-based line number containing ``offset``."""
    return max(1, min(len(starts), bisect.bisect_right(starts, offset)))


def char_span(
    text: str, starts: Sequence[int], start_line: int, end_line: int
) -> tuple[int, int]:
    """Character span of the inclusive 1-based line range, whitespace-trimmed.

    The result is exactly the span of ``slice_lines(...).strip()``, so a gold
    span and the span of the chunk that answers it are computed the same way and
    can be compared with a strict containment check. Leaving the trailing
    newline (or the blank lines at the end of a section) inside the gold span
    makes every chunk look like it is missing the tail of the answer.
    """
    first = starts[start_line - 1]
    last = min(starts[min(end_line, len(starts) - 1)], len(text))
    while first < last and text[first].isspace():
        first += 1
    while last > first and text[last - 1].isspace():
        last -= 1
    return first, last


def snap_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a span outwards to whitespace so a window never cuts a word."""
    if start > 0:
        while start > 0 and not text[start - 1].isspace():
            start -= 1
    if end < len(text):
        while end < len(text) and not text[end].isspace():
            end += 1
    return start, end


class FixedSizeChunker:
    """Strategy 1: fixed-size sliding windows over tokens (or characters).

    Windows are cut on token offsets *in context* — the tokenizer is run once
    over the whole document — and then grown outwards to whitespace so no word
    is cut in half. Growing can push a window past its budget (the neighbouring
    word is added in full), so the window is trimmed back token by token and
    finally pulled to the previous word boundary. The result is a chunk of at
    most ``size`` tokens that still starts and ends on whitespace.

    The chunk is the character span itself, not whole lines: a line-aligned
    variant would silently embed far more than ``size`` tokens on any document
    with long lines, and would defeat the point of a fixed-size strategy.
    """

    name = "fixed_size"

    def __init__(
        self,
        size: int = 200,
        overlap: int = 0,
        unit: str = "tokens",
        tokenizer: CharTokenizer | ModelTokenizer | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("chunk size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.unit = unit
        self.tokenizer = tokenizer or CharTokenizer()

    @property
    def params(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "chunk_size": self.size,
            "chunk_overlap": self.overlap,
        }

    def _token_windows(self, text: str) -> Iterator[tuple[int, int]]:
        """Character spans of the token windows, each within the budget."""
        spans = self.tokenizer.spans(text)
        if not spans:
            return
        starts = [span[0] for span in spans]
        step = self.size - self.overlap
        for index in range(0, len(spans), step):
            window = spans[index:index + self.size]
            if not window:
                return
            start = window[0][0]
            end = window[-1][1]
            start, end = snap_span(text, start, end)
            low = bisect.bisect_left(starts, start)
            high = bisect.bisect_right(starts, end - 1, low)
            grown = high - low > self.size
            while high - low > self.size and high > low:
                high -= 1
                end = spans[high - 1][1]
            if high - low > self.size:
                return  # a single token longer than the whole window
            if grown:
                # Trimming cut a token in half; step back to the word boundary.
                # Only after a trim: pulling back an untouched window would
                # silently drop the end of the document.
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            yield start, end
            if index + self.size >= len(spans):
                return

    def _char_windows(self, text: str) -> Iterator[tuple[int, int]]:
        """Character spans of the character windows, each within the budget."""
        step = self.size - self.overlap
        for start in range(0, len(text), step):
            end = min(len(text), start + self.size)
            start, end = snap_span(text, start, end)
            if end - start > self.size:
                space = text.rfind(" ", start, end)
                if space > start + self.size - 1:
                    end = space
                end = min(end, start + self.size)
            if end > start:
                yield start, end

    def chunk(self, doc: Document, document_id: int) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        windows = (
            self._char_windows(text)
            if self.unit == "chars"
            else self._token_windows(text)
        )
        line_index = line_starts(doc)
        chunks: list[Chunk] = []
        position = 0
        for start, end in windows:
            raw = text[start:end]
            body = raw.strip()
            if not body:
                continue
            # Offsets must match the stripped text, otherwise they point into
            # the whitespace that was trimmed off.
            offset = start + (len(raw) - len(raw.lstrip()))
            chunks.append(Chunk(
                chunk_id=f"fixed_size-{document_id:03d}-{position:03d}",
                text=body,
                strategy=self.name,
                document_id=document_id,
                source=doc.rel_path,
                title=doc.title,
                section="",
                section_title="",
                chunk_position=position,
                start_line=line_of_offset(line_index, offset),
                end_line=line_of_offset(line_index, offset + len(body) - 1),
                char_start=offset,
                char_end=offset + len(body),
            ))
            position += 1
        return chunks


#: «Количество человек: 1», «Длительность: 20 минут», «Место проведения: зал» —
#: a labelled value, the shape a service form or an invoice puts its facts in.
#: A capital letter starts the label so that prose with a colon mid-sentence
#: does not qualify.
_FIELD_LINE = re.compile(r"^\s*[А-ЯЁA-Z][^:.\n]{2,38}:\s*\S")


class StructureChunker:
    """Strategy 2: chunks follow headings / symbols / pages / blank lines.

    The size limit is expressed in **tokens**, not characters, because the limit
    that actually matters is the embedding model's window: a 1200-character
    Russian chunk is ~320 tokens, and MiniLM silently keeps only the first 256.
    Sizing in characters needs a chars-per-token guess that is off by 2x on a
    code-heavy corpus; counting tokens costs one cheap pass over the text.

    A section that fits is one chunk. An oversized section is packed on paragraph
    (blank-line) boundaries and, if a window still does not fit, bisected on line
    boundaries — the first window always starts at the heading line, so a chunk
    never loses the title that makes it self-describing for retrieval.
    Pieces smaller than ``min_chars`` are merged with their neighbour *inside the
    same section only*: merging across sections would destroy the ``section``
    metadata, which is the whole point of this strategy.
    """

    name = "structure"

    def __init__(
        self,
        max_tokens: int = 254,
        min_chars: int = 200,
        max_chars: int = 0,
        tokenizer: CharTokenizer | ModelTokenizer | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if min_chars < 0:
            raise ValueError("min_chars must not be negative")
        if max_chars < 0:
            raise ValueError("max_chars must not be negative")
        self.max_tokens = max_tokens
        self.min_chars = min_chars
        self.max_chars = max_chars  # optional secondary cap, 0 = off
        self.tokenizer = tokenizer or CharTokenizer()

    @property
    def params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "min_chars": self.min_chars,
        }
        if self.max_chars:
            params["max_chars"] = self.max_chars
        return params

    def chunk(self, doc: Document, document_id: int) -> list[Chunk]:
        # One cheap tokenization pass per line, reused by the packing heuristic.
        line_tokens = [self.tokenizer.count(line) for line in doc.lines]
        starts = line_starts(doc)
        pieces: list[tuple[str, str, int, int]] = []  # path, title, start, end
        # A document whose structure was never parsed is still indexed: without
        # this fallback it would silently disappear from the index.
        sections = doc.sections or (Section("", "", 1, len(doc.lines)),)
        for section in sections:
            section_path = (
                section.path
                or page_of_line(doc, section.start_line)
                or doc.title
            )
            section_title = section.title or section_path
            if self._fits(doc, section.start_line, section.end_line):
                pieces.append(
                    (section_path, section_title, section.start_line, section.end_line)
                )
                continue
            pieces.extend(
                self._split_section(
                    doc, line_tokens, section, section_path, section_title
                )
            )
        pieces = self._absorb_heading_only(doc, pieces)
        pieces = self._absorb_field_lines(doc, pieces)
        chunks: list[Chunk] = []
        position = 0
        for section_path, section_title, start, end in pieces:
            body = slice_lines(doc, start, end)
            if not body:
                continue
            # char_span trims exactly like slice_lines(...).strip(), so the
            # stored offsets index the chunk text itself.
            first, last = char_span(doc.text, starts, start, end)
            chunks.append(Chunk(
                chunk_id=f"structure-{document_id:03d}-{position:03d}",
                text=body,
                strategy=self.name,
                document_id=document_id,
                source=doc.rel_path,
                title=doc.title,
                section=section_path,
                section_title=section_title,
                chunk_position=position,
                start_line=start,
                end_line=end,
                char_start=first,
                char_end=last,
            ))
            position += 1
        return chunks

    def _absorb_field_lines(
        self, doc: Document, pieces: list[tuple[str, str, int, int]]
    ) -> list[tuple[str, str, int, int]]:
        """Extend the opening chunk so it also carries the «label: value» lines.

        A vague question — «а ещё есть занятия какие-то?» — matches the chunk
        with the product title, and on these documents that chunk is the
        marketing intro: no headcount, no duration, no venue. Those sit two lines
        further down and were split into the *next* chunk, so the block held four
        product titles and not one fact, and the model filled the gap from its
        neighbours. That is how one service came back with another one's
        duration and venue — both plausible, both from the corpus, and both
        attached to the wrong product.

        Extending the line range keeps the chunk a verbatim slice of the document
        (``char_span`` stays exact), and only when the result still fits the
        budget. It stops at the next section so ``section`` metadata stays true.
        """
        if not pieces:
            return pieces
        section_path, section_title, start, end = pieces[0]
        last = max(
            (number for number, line in enumerate(doc.lines, 1)
             if _FIELD_LINE.match(line)),
            default=0,
        )
        # The field lines may sit in the next *window* of the same section; stop
        # only where a different section begins, so ``section`` stays truthful.
        run_end = end
        for path, _window_title, _start, window_end in pieces[1:]:
            if path != section_path:
                break
            run_end = max(run_end, window_end)
        last = min(last, run_end)
        if last <= end or not (start <= last):
            return pieces
        if not self._fits(doc, start, last):
            return pieces
        return [(section_path, section_title, start, last), *pieces[1:]]

    def _fits(self, doc: Document, start: int, end: int) -> bool:
        """Exact check: does the assembled line range respect both budgets?"""
        text = slice_lines(doc, start, end)
        if self.tokenizer.count(text) > self.max_tokens:
            return False
        return not self.max_chars or len(text) <= self.max_chars

    def _split_section(
        self,
        doc: Document,
        line_tokens: Sequence[int],
        section: Section,
        section_path: str,
        section_title: str,
    ) -> list[tuple[str, str, int, int]]:
        """Pack an oversized section into windows and make each one fit.

        The packing uses the cheap sum of per-line counts — good enough to
        decide where a window ends — and every window is then verified with an
        exact count of the assembled text, because a sum of per-line counts
        systematically underestimates (the tokenizer spends tokens on the
        newlines it joins lines with).
        """
        windows: list[tuple[int, int]] = []
        start = section.start_line
        used = 0
        for para_start, para_end in blank_line_runs(
            doc.lines, section.start_line, section.end_line, doc.verbatim
        ):
            block = sum(line_tokens[para_start - 1:para_end])
            if used and used + block > self.max_tokens:
                windows.append((start, para_start - 1))
                start, used = para_start, 0
            used += block
        windows.append((start, section.end_line))
        out: list[tuple[str, str, int, int]] = []
        for window_start, window_end in windows:
            for part in self._enforce_budget(doc, window_start, window_end):
                out.append((section_path, section_title, part[0], part[1]))
        return self._merge_tiny(out, doc)

    def _enforce_budget(self, doc: Document, start: int, end: int) -> list[tuple[int, int]]:
        """Split a window on line boundaries until every part fits.

        A single line that does not fit (minified code, a long URL) is kept
        whole: cutting mid-line would corrupt the chunk text for a truncation
        the model would do anyway.
        """
        if self._fits(doc, start, end) or start >= end:
            return [(start, end)]
        middle = (start + end) // 2
        return (
            self._enforce_budget(doc, start, middle)
            + self._enforce_budget(doc, middle + 1, end)
        )

    def _small(self, doc: Document, start: int, end: int) -> bool:
        return len(slice_lines(doc, start, end)) < self.min_chars

    def _heading_only(self, doc: Document, start: int, end: int) -> bool:
        """True when the piece is Markdown headings and blank lines, nothing else.

        A heading with no prose of its own cannot answer anything, yet it embeds
        as a near-perfect match for its own title ("Часы работы" vs "Часы работы
        кофейни"), so it outranks the section that does hold the answer. This is
        not a size problem and ``min_chars`` cannot catch it — a long title is
        still a title.
        """
        for line in split_lines(slice_lines(doc, start, end)):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return False
        return True

    def _merge_tiny(
        self, pieces: list[tuple[str, str, int, int]], doc: Document
    ) -> list[tuple[str, str, int, int]]:
        """Merge undersized pieces with an adjacent piece of the same section.

        Both directions, because a piece can be too small on either side. A
        heading with no body of its own is first in its section and can only be
        merged forward — leaving it alone produces a chunk that is a title and
        nothing else, which retrieves on the title but answers nothing.

        Only if the merge still fits the budget: a short chunk costs a few tokens
        of noise, an over-budget chunk costs a silently truncated tail.
        """
        merged: list[tuple[str, str, int, int]] = []
        for path, title, start, end in pieces:
            if merged:
                p_path, _, p_start, p_end = merged[-1]
                if p_path == path and self._small(doc, start, end) and self._fits(
                    doc, p_start, end
                ):
                    merged[-1] = (p_path, title, p_start, end)
                    continue
            merged.append((path, title, start, end))
        out: list[tuple[str, str, int, int]] = []
        position = 0
        while position < len(merged):
            path, title, start, end = merged[position]
            following = merged[position + 1] if position + 1 < len(merged) else None
            if (
                following is not None
                and following[0] == path
                and self._small(doc, start, end)
                and self._fits(doc, start, following[3])
            ):
                out.append((path, title, start, following[3]))
                position += 2
                continue
            out.append(merged[position])
            position += 1
        return out

    def _absorb_heading_only(
        self, doc: Document, pieces: list[tuple[str, str, int, int]]
    ) -> list[tuple[str, str, int, int]]:
        """Fold a heading-only piece into the piece that follows it.

        Runs across the whole document, not per section, because the interesting
        case is precisely a heading whose body is empty and whose content lives
        in its *children* — a different section path. Merging stays limited to
        heading-only pieces: a piece with real text is never pulled across a
        section boundary, that would mix unrelated sections into one chunk. The
        following piece's path wins, being the more specific of the two.
        """
        out: list[tuple[str, str, int, int]] = []
        position = 0
        while position < len(pieces):
            path, title, start, end = pieces[position]
            following = pieces[position + 1] if position + 1 < len(pieces) else None
            if (
                following is not None
                and self._heading_only(doc, start, end)
                and self._fits(doc, start, following[3])
            ):
                out.append((following[0], following[1], start, following[3]))
                position += 2
                continue
            out.append(pieces[position])
            position += 1
        return out


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #

def build_index(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    strategy: str,
    corpus: dict[str, object],
    params: dict[str, object],
    embedder_name: str,
    dim: int,
) -> dict[str, object]:
    """Assemble the JSON-serializable index for one chunking strategy."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
        )
    return {
        "version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunking_strategy": strategy,
        "embedding_model": embedder_name,
        "embedding_dim": dim,
        "chunking_params": params,
        "corpus": corpus,
        "chunks": [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": list(embedding),
                "metadata": chunk.metadata(),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ],
    }


def save_index(index: dict[str, object], path: Path) -> Path:
    """Write an index as compact JSON (embeddings dominate the file size)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def corpus_summary(
    documents: Sequence[Document],
    tokenizer: CharTokenizer | ModelTokenizer,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Corpus-level numbers, including the «pages» estimate the task asks for."""
    chars = sum(doc.char_count for doc in documents)
    summary: dict[str, object] = {
        "documents": len(documents),
        "chars": chars,
        "lines": sum(len(doc.lines) for doc in documents),
        "tokens": sum(tokenizer.count(doc.text) for doc in documents),
        "pages_estimate": round(chars / 3000, 1),
        "extensions": sorted({doc.ext for doc in documents}),
        "sources": [doc.rel_path for doc in documents],
    }
    if extra:
        summary.update(extra)
    return summary


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


def is_hit(chunk: Chunk, query: EvalQuery) -> bool:
    """A retrieved chunk answers a query when it covers the gold span.

    Character spans are used when both sides have them (exact overlap, so a
    fixed-size window that only grazes the section's last line is not counted);
    otherwise the coarser line range is compared.
    """
    if chunk.source != query.source:
        return False
    if chunk.char_end and query.char_end:
        return chunk.char_start < query.char_end and chunk.char_end > query.char_start
    return (
        chunk.start_line <= query.end_line and chunk.end_line >= query.start_line
    )


def build_queries(
    documents: Sequence[Document],
    min_section_chars: int = 200,
    tokenizer: CharTokenizer | ModelTokenizer | None = None,
) -> list[EvalQuery]:
    """Auto-generated benchmark: every section heading is a query.

    The gold answer is the section's span, computed from the raw document —
    independent of how either strategy was chunked, which is what makes the
    two strategies comparable. Sections that carry too little text (a table of
    contents line, a one-line section) are skipped as queries.
    """
    counter = tokenizer or CharTokenizer()
    queries: list[EvalQuery] = []
    for doc in documents:
        starts = line_starts(doc)
        for section in doc.sections:
            label = section.title or section.path
            if not label or len(label.split()) < 2:
                continue
            body = slice_lines(doc, section.start_line, section.end_line)
            if len(body) < min_section_chars or counter.count(body) < 10:
                continue
            first, last = char_span(
                doc.text, starts, section.start_line, section.end_line
            )
            queries.append(EvalQuery(
                text=label,
                source=doc.rel_path,
                section=section.path,
                start_line=section.start_line,
                end_line=section.end_line,
                char_start=first,
                char_end=last,
                kind="heading" if doc.ext == ".md" else "structure",
            ))
    return queries


def load_queries(
    path: Path, documents: Sequence[Document]
) -> list[EvalQuery]:
    """Read hand-written queries from YAML or JSON.

    Schema::

        queries:
          - text: "как подключить GigaChat"
            source: README.md
            section: "Configuration > Example: use GigaChat (Sber)"   # optional
            kind: manual

    Without ``section`` the gold span is the whole document.
    """
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        payload = yaml.safe_load(raw) or {}
    else:
        payload = json.loads(raw)
    by_source = {doc.rel_path: doc for doc in documents}
    queries: list[EvalQuery] = []
    for item in payload.get("queries", []):
        source = str(item.get("source", ""))
        if source not in by_source:
            raise ValueError(f"eval query {item.get('text')!r}: unknown source {source!r}")
        doc = by_source[source]
        section = str(item.get("section", "") or "")
        span = (1, len(doc.lines))
        if section:
            match = _find_section(doc, section)
            if match is None:
                leaf = section.split(" > ")[-1].strip()
                close = [
                    other.path
                    for other in doc.sections
                    if leaf[:12] and leaf[:12] in other.path
                ][:3]
                hint = f"; похожие разделы: {close}" if close else ""
                raise ValueError(
                    f"eval query {item.get('text')!r}: section {section!r} not found "
                    f"in {source}{hint}"
                )
            span = (match.start_line, match.end_line)
        first, last = char_span(doc.text, line_starts(doc), span[0], span[1])
        queries.append(EvalQuery(
            text=str(item["text"]),
            source=source,
            section=section,
            start_line=span[0],
            end_line=span[1],
            char_start=first,
            char_end=last,
            kind=str(item.get("kind", "manual")),
        ))
    return queries


def _find_section(doc: Document, path: str) -> Section | None:
    """Resolve a section path: exact, suffix, then unique leaf-title match.

    The leaf fallback exists for hand-written ``--eval`` files: naming
    ``"Invariants (hard constraints)"`` is what a human writes, not the full
    ``"A > B > C"`` ancestry. Every stage requires a *unique* match, so a name
    shared by two sections resolves to nothing and the caller reports it
    instead of scoring against an arbitrary section.
    """
    leaf = path.split(" > ")[-1].strip()
    stages = (
        [section for section in doc.sections if section.path == path],
        [section for section in doc.sections if section.path.endswith(path)],
        [
            section
            for section in doc.sections
            if leaf in (section.title, section.path.split(" > ")[-1])
        ],
    )
    for matches in stages:
        if matches:
            return matches[0] if len(matches) == 1 else None
    return None


def evaluate(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    queries: Sequence[EvalQuery],
    query_vectors: Sequence[Sequence[float]],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Recall@k and MRR@10 of one strategy over the benchmark queries."""
    if not queries:
        return {}
    depth = max(max(ks), 10)
    hits = {k: 0 for k in ks}
    reciprocal_rank = 0.0
    misses: list[str] = []
    for query, vector in zip(queries, query_vectors):
        ranked = search(embeddings, vector, top_k=depth)
        rank = 0
        for position, (index, _score) in enumerate(ranked, 1):
            if is_hit(chunks[index], query):
                rank = position
                break
        if rank:
            reciprocal_rank += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1
        else:
            misses.append(query.gold_id)
    total = len(queries)
    metrics: dict[str, float] = {
        "queries": float(total),
        "mrr@10": round(reciprocal_rank / total, 4),
    }
    for k in ks:
        metrics[f"recall@{k}"] = round(hits[k] / total, 4)
    if misses:
        metrics["misses"] = ", ".join(misses[:5])  # type: ignore[assignment]
    return metrics


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def chunk_stats(
    chunks: Sequence[Chunk],
    corpus_chars: int,
    params: dict[str, object],
    index_bytes: int = 0,
    tokenizer: CharTokenizer | ModelTokenizer | None = None,
    window_tokens: int = 0,
) -> dict[str, float]:
    """Size distribution, duplication and coverage for one strategy.

    ``over_window_share`` is the share of chunks longer than the embedding
    model's context window: their tail is silently dropped by the model, which
    is the main way an index quietly loses information.
    ``over_budget_share`` is the same check against the chunker's own budget —
    it should be ~0 and is what proves the strategy respects its own limit.

    It is split in two, because the two causes need different reactions.
    ``single_line_overflow_share`` is chunks that are one whole line: the
    chunker refuses to cut a line in half (see ``StructureChunker``), so those
    are a deliberate, inspectable exception. ``over_budget_multiline_share`` is
    the strict invariant — a chunk spanning several lines yet over budget would
    mean the splitting logic is broken, and it must be 0.0 on every run.
    """
    if not chunks:
        return {"chunks": 0, "chars": 0}
    counter = tokenizer or CharTokenizer()
    sizes = [chunk.char_count for chunk in chunks]
    tokens = [counter.count(chunk.text) for chunk in chunks]
    hashes = {chunk.content_hash for chunk in chunks}
    total = sum(sizes)
    max_chars = int(params.get("max_chars") or 0)
    min_chars = int(params.get("min_chars") or 0)
    budget = int(params.get("max_tokens") or params.get("chunk_size") or 0)
    over = [count > budget for count in tokens] if budget else [False] * len(tokens)
    single_line_over = [
        flag and chunk.start_line == chunk.end_line
        for flag, chunk in zip(over, chunks)
    ]
    return {
        "chunks": float(len(chunks)),
        "chars": float(total),
        "chars_avg": round(statistics.fmean(sizes), 1),
        "chars_median": round(statistics.median(sizes), 1),
        "chars_min": float(min(sizes)),
        "chars_max": float(max(sizes)),
        "chars_std": round(statistics.pstdev(sizes), 1),
        "tokens_avg": round(statistics.fmean(tokens), 1),
        "tokens_max": float(max(tokens)),
        "sections": float(len({(c.source, c.section) for c in chunks})),
        "duplicate_share": round(1 - len(hashes) / len(chunks), 4),
        "coverage_share": round(min(total / corpus_chars, 1.0) if corpus_chars else 0.0, 4),
        "tiny_share": round(
            sum(1 for size in sizes if size < min_chars) / len(sizes), 4
        ) if min_chars else 0.0,
        "over_chars_share": round(
            sum(1 for size in sizes if size > max_chars) / len(sizes), 4
        ) if max_chars else 0.0,
        "over_budget_share": round(sum(over) / len(tokens), 4) if budget else 0.0,
        "single_line_overflow_share": (
            round(sum(single_line_over) / len(tokens), 4) if budget else 0.0
        ),
        "over_budget_multiline_share": (
            round(
                sum(1 for flag, line in zip(over, single_line_over)
                    if flag and not line) / len(tokens),
                4,
            )
            if budget else 0.0
        ),
        "over_window_share": round(
            sum(1 for count in tokens if count > window_tokens) / len(tokens), 4
        ) if window_tokens else 0.0,
        "index_kb": round(index_bytes / 1024, 1),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_markdown(
    corpus: dict[str, object],
    results: Sequence[StrategyResult],
    queries: Sequence[EvalQuery],
    notes: Sequence[str],
    command: str,
    index_dir: str = DEFAULT_INDEX_DIR,
) -> str:
    """Markdown report: corpus, metadata sample, statistics, benchmark."""
    lines = [
        "# Индексация документов: локальный индекс с эмбеддингами",
        "",
        f"- Дата: {datetime.now().isoformat(timespec='seconds')}",
        f"- Команда: `{command}`",
        f"- Модель эмбеддингов: `{corpus.get('embedding_model', DEFAULT_MODEL)}`",
        f"- Корпус: **{corpus.get('documents', 0)} документов**, "
        f"{corpus.get('chars', 0)} символов ≈ {corpus.get('pages_estimate', 0)} "
        f"страниц (3000 символов/страница)",
        f"- Расширения: {', '.join(corpus.get('extensions', []))}",
        "- Параметры chunking: "
        + "; ".join(
            f"`{result.strategy}` = {json.dumps(result.index['chunking_params'])}"
            for result in results
        ),
        "",
    ]

    if notes:
        lines += [
            "### Пропущенные файлы",
            "",
            *(f"- {note}" for note in notes),
            "",
        ]

    sample = next(
        (
            chunk
            for result in results
            for chunk in result.chunks
            if chunk.section
        ),
        None,
    ) or next(
        (chunk for result in results for chunk in result.chunks), None
    )
    if sample is not None:
        lines += [
            "## Метаданные чанка",
            "",
            "Каждый чанк хранит источник, заголовок документа, путь секции и "
            "диапазон строк — любой ответ индекса можно проверить по `file:line`.",
            "",
            "```json",
            json.dumps(
                {
                    "id": sample.chunk_id,
                    "metadata": sample.metadata(),
                    "text": sample.text[:220] + ("…" if len(sample.text) > 220 else ""),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]

    lines += [
        "## Сравнение стратегий chunking",
        "",
    ]
    fixed = next((r for r in results if r.strategy == "fixed_size"), None)
    structure = next((r for r in results if r.strategy == "structure"), None)
    stat_keys = [
        ("chunks", "чанков", "num"),
        ("chars_avg", "символов на чанк (среднее)", "num"),
        ("chars_median", "символов на чанк (медиана)", "num"),
        ("chars_min", "мин. размер, символов", "num"),
        ("chars_max", "макс. размер, символов", "num"),
        ("chars_std", "разброс (σ), символов", "num"),
        ("tokens_avg", "токенов на чанк (среднее)", "num"),
        ("tokens_max", "макс. чанк, токенов", "num"),
        ("over_budget_share", "доля чанков длиннее бюджета стратегии", "pct"),
        ("single_line_overflow_share", "…из них одиночная строка (не режется)", "pct"),
        ("over_budget_multiline_share", "…из них многострочных (должно быть 0)", "pct"),
        ("over_window_share", "доля чанков длиннее окна модели", "pct"),
        ("tiny_share", "доля чанков < min_chars", "pct"),
        ("duplicate_share", "доля дубликатов", "pct"),
        ("coverage_share", "покрытие корпуса", "pct"),
        ("encode_seconds", "время эмбеддингов, с", "num"),
        ("index_kb", "размер индекса, КБ", "num"),
    ]
    lines.append(
        "| Метрика | " + " | ".join(r.strategy for r in results) + " |"
    )
    lines.append("|---|" + "---:|" * len(results))
    for key, label, kind in stat_keys:
        cells = " | ".join(
            _fmt_metric(result.stats.get(key), kind) for result in results
        )
        lines.append(f"| {label} | {cells} |")

    lines += [
        "",
        "### Retrieval-бенчмарк",
        "",
    ]
    if queries:
        lines += [
            f"Запросов: **{len(queries)}** "
            f"(эталон — точный диапазон символов раздела; hit — чанк "
            f"пересекается с ним; "
            f"{sum(1 for q in queries if q.kind == 'manual')} вручную, "
            f"{sum(1 for q in queries if q.kind != 'manual')} из заголовков).",
            "",
            "| Метрика | " + " | ".join(r.strategy for r in results) + " |",
            "|---|" + "---:|" * len(results),
        ]
        for key, label in (
            ("recall@1", "recall@1"),
            ("recall@3", "recall@3"),
            ("recall@5", "recall@5"),
            ("mrr@10", "MRR@10"),
        ):
            lines.append(
                f"| {label} | "
                + " | ".join(_fmt_metric(r.metrics.get(key), "pct") for r in results)
                + " |"
            )
        lines.append("")
        lines += _conclusion(fixed, structure, queries)
    else:
        lines += [
            "Запросов нет — бенчмарк пропущен. Добавьте `--eval "
            "index_queries.example.yaml` или увеличьте корпус: запросы "
            "генерируются из заголовков секций.",
            "",
        ]

    lines += [
        "## Как читать индекс",
        "",
        "```python",
        "import json, math",
        "",
        f"index = json.load(open('{index_dir}/index_structure.json', encoding='utf-8'))",
        "query = index['chunks'][0]   # в проде здесь вектор настоящего вопроса",
        "ranked = sorted(",
        "    ((math.fsum(a * b for a, b in zip(row['embedding'], query['embedding'])),",
        "      row['metadata']) for row in index['chunks']),",
        "    key=lambda pair: pair[0],   # ключ обязателен: при равных score",
        "    reverse=True,               # сравниваются dict-ы, и сортировка падает",
        ")",
        "for score, meta in ranked[:3]:",
        "    print(f\"{score:.3f} {meta['source']}:{meta['start_line']}-\"",
        "          f\"{meta['end_line']}  {meta['section']}\")",
        "```",
        "",
        "Векторы нормализованы, поэтому скалярное произведение — это косинусная",
        "близость.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fmt_metric(value: object, kind: str = "num") -> str:
    """Format a metric for the report: ``pct`` turns a 0..1 share into a percent."""
    if value is None:
        return "—"
    if kind == "pct" and isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _conclusion(
    fixed: StrategyResult | None,
    structure: StrategyResult | None,
    queries: Sequence[EvalQuery],
) -> list[str]:
    """Plain-language verdict derived from the measured numbers."""
    if fixed is None or structure is None or not queries:
        return []
    lines = ["### Вывод", ""]
    for key in ("recall@1", "recall@5"):
        left = float(fixed.metrics.get(key, 0.0))
        right = float(structure.metrics.get(key, 0.0))
        if abs(right - left) < 0.005:
            lines.append(f"- {key}: стратегии сравнимы ({_fmt_metric(left, 'pct')}).")
        else:
            winner = "structure" if right > left else "fixed_size"
            lines.append(
                f"- {key}: лучше **{winner}** "
                f"({_fmt_metric(max(left, right), 'pct')} против "
                f"{_fmt_metric(min(left, right), 'pct')})."
            )
    big = float(fixed.stats.get("chars_median", 0.0))
    small = float(structure.stats.get("chars_median", 0.0))
    if abs(big - small) <= 0.1 * max(big, small):
        lines.append(
            f"- Медианный размер чанка почти одинаков ({big:g} против {small:g} "
            "символов): разница не в размере, а в границах — у structure чанк "
            "совпадает с разделом документа."
        )
    else:
        lines.append(
            f"- Медианный чанк: {big:g} символов у fixed_size против {small:g} "
            "у structure — "
            + (
                "структурный чанк ближе к смысловому блоку, поэтому запрос про "
                "заголовок попадает в него целиком."
                if small < big else
                "fixed_size ближе к смысловому блоку на этом корпусе."
            )
        )
    tiny_fixed = float(fixed.stats.get("tiny_share", 0.0))
    tiny_structure = float(structure.stats.get("tiny_share", 0.0))
    if tiny_structure > tiny_fixed + 0.05:
        lines.append(
            f"- Плата за структуру: {_fmt_metric(tiny_structure, 'pct')} чанков "
            f"короче min_chars против {_fmt_metric(tiny_fixed, 'pct')} — это "
            "короткие разделы целиком: склеивать их можно только с соседним "
            "разделом, а это уже ломает метаданные `section`. Поднимать "
            "`--structure_min_chars` вверх бесполезно, опускать — вниз."
        )
    lines.append(
        f"- Цена: structure даёт "
        f"{int(float(structure.stats.get('chunks', 0)))} чанков "
        f"({float(structure.stats.get('index_kb', 0)):g} КБ) против "
        f"{int(float(fixed.stats.get('chunks', 0)))} "
        f"({float(fixed.stats.get('index_kb', 0)):g} КБ) у fixed_size."
    )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def drop_duplicates(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Remove exact-duplicate texts (same content hash), keeping the first."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        unique.append(chunk)
    return unique


def structure_max_tokens(
    window_tokens: int, requested: int | None
) -> int:
    """Token budget for a structure chunk: the model window minus the margin.

    Anything above the window is not embedded at all — the model truncates the
    tail — so the default simply fills the window instead of guessing a
    character size and hoping it fits.
    """
    if requested is not None:
        return requested
    return max(32, window_tokens - 2)  # [CLS] / [SEP]


def _warn_about_overflow(
    strategy: str,
    chunks: Sequence[Chunk],
    stats: dict[str, object],
    budget: int,
    window_tokens: int,
) -> None:
    """Name every chunk the embedding model will see only partly.

    Two very different problems hide behind "over budget": a chunk that spans
    several lines means the splitting logic failed, and a chunk that is a single
    line means the source has a line longer than the model window — the chunker
    keeps it whole on purpose, but its tail is still dropped by the model, so
    the words in it become unsearchable. Printing the locations turns a silent
    0.3% into something a reader can check.
    """
    if stats.get("over_budget_multiline_share"):
        print(f"[warn] {strategy}: многострочный чанк длиннее бюджета "
              f"{budget} — это ошибка разбиения, StrategyResult.stats её покажет")
    limit = budget or window_tokens
    over = [
        chunk for chunk in chunks
        if chunk.start_line == chunk.end_line and chunk.token_count > limit
    ]
    if over:
        spots = ", ".join(
            f"{chunk.source}:{chunk.start_line} ({chunk.token_count} токенов)"
            for chunk in over[:5]
        )
        tail = f" и ещё {len(over) - 5}" if len(over) > 5 else ""
        print(f"[warn] {strategy}: {len(over)} чанков — одиночная строка длиннее "
              f"{limit} токенов, хвост не попадёт в эмбеддинг: {spots}{tail}")


def run_strategy(
    strategy: str,
    documents: Sequence[Document],
    chunker: FixedSizeChunker | StructureChunker,
    embedder: Embedder,
    corpus: dict[str, object],
    batch_size: int,
    dedup: bool,
) -> StrategyResult:
    """Chunk -> embed -> index -> statistics for one strategy."""
    chunks: list[Chunk] = []
    for document_id, document in enumerate(documents):
        chunks.extend(chunker.chunk(document, document_id))
    if dedup:
        chunks = drop_duplicates(chunks)
    if not chunks:
        raise RuntimeError(f"strategy {strategy} produced no chunks")
    # Token counts are filled here rather than in the chunkers, so both
    # strategies store the same tokenizer's numbers in their metadata.
    counter = embedder.tokenizer
    for chunk in chunks:
        chunk.token_count = counter.count(chunk.text)
    started = time.perf_counter()
    embeddings = embedder.encode([chunk.text for chunk in chunks], batch_size)
    elapsed = time.perf_counter() - started
    index = build_index(
        chunks=chunks,
        embeddings=embeddings,
        strategy=strategy,
        corpus=corpus,
        params=chunker.params,
        embedder_name=embedder.name,
        dim=embedder.dim,
    )
    stats = chunk_stats(
        chunks,
        corpus_chars=int(corpus.get("chars", 0)),  # type: ignore[arg-type]
        params=chunker.params,
        tokenizer=embedder.tokenizer,
        window_tokens=embedder.max_tokens,
    )
    stats["encode_seconds"] = round(elapsed, 2)
    _warn_about_overflow(
        strategy,
        chunks,
        stats,
        budget=int(chunker.params.get("max_tokens") or 0)
        or int(chunker.params.get("chunk_size") or 0),
        window_tokens=embedder.max_tokens,
    )
    return StrategyResult(
        strategy=strategy,
        chunks=chunks,
        embeddings=embeddings,
        index=index,
        stats=stats,
        metrics={},
        encode_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=Path, default=None,
                        help="Directory with the corpus to index "
                             "(required unless --reuse).")
    parser.add_argument("--extensions", type=str, default=".md,.py,.txt,.yaml",
                        help="Extensions to index (default: %(default)s).")
    parser.add_argument("--index_dir", type=Path,
                        default=Path(DEFAULT_INDEX_DIR),
                        help="Where the JSON indices are written "
                             "(default: %(default)s; git-ignored, because the "
                             "embeddings are megabytes).")
    parser.add_argument("--out", type=str,
                        default=DEFAULT_REPORT,
                        help="Markdown report path; \"\" disables it "
                             "(default: %(default)s).")
    parser.add_argument("--chunk_size", type=int, default=120,
                        help="Fixed-size window in tokens, must fit the model "
                             "window (default: %(default)s).")
    parser.add_argument("--chunk_overlap", type=int, default=20,
                        help="Overlap between fixed-size windows "
                             "(default: %(default)s).")
    parser.add_argument("--unit", choices=("tokens", "chars"), default="tokens",
                        help="Fixed-size window unit (default: %(default)s).")
    parser.add_argument("--structure_max_tokens", type=int, default=None,
                        help="Token budget per structure chunk; default: the "
                             "model window minus the [CLS]/[SEP] margin.")
    parser.add_argument("--structure_max_chars", type=int, default=0,
                        help="Optional secondary cap in characters for "
                             "structure chunks (0 = off).")
    parser.add_argument("--structure_min_chars", type=int, default=200,
                        help="Merge structure chunks below this size "
                             "(default: %(default)s).")
    parser.add_argument("--strategy", choices=("both", "fixed_size", "structure"),
                        default="both", help="Which chunking strategies to run "
                                            "(default: %(default)s).")
    parser.add_argument("--eval", dest="eval_path", type=Path, default=None,
                        help="YAML/JSON file with hand-written benchmark queries.")
    parser.add_argument("--query", dest="questions", action="append", default=None,
                        metavar="TEXT",
                        help="Ask the built index and print the top hits; "
                             "repeatable.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Hits printed per --query (default: %(default)s).")
    parser.add_argument("--reuse", action="store_true",
                        help="Answer --query from the indices already in "
                             "--index_dir: no re-chunking, no re-embedding of "
                             "the corpus (the model still embeds the question).")
    parser.add_argument("--min_section_chars", type=int, default=200,
                        help="Skip auto-generated queries for sections smaller "
                             "than this (default: %(default)s).")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Embedding model (default: {DEFAULT_MODEL}, or the "
                             "model recorded in the index with --reuse).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Embedding batch size (default: %(default)s).")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Stop after N documents (0 = no limit).")
    parser.add_argument("--pdf_max_pages", type=int, default=0,
                        help="Keep only the first N pages of each PDF "
                             "(0 = all pages). The report states how many "
                             "pages were left out.")
    parser.add_argument("--dedup", action="store_true",
                        help="Drop chunks whose text duplicates an earlier chunk.")
    parser.add_argument("--local_only", action="store_true",
                        help="Use only a locally cached model (no download).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.chunk_overlap < args.chunk_size:
        print(f"[error] --chunk_overlap ({args.chunk_overlap}) must be smaller "
              f"than --chunk_size ({args.chunk_size})")
        return 2
    if args.top_k < 1:
        print(f"[error] --top_k must be >= 1 (got {args.top_k})")
        return 2
    if args.reuse and not args.questions:
        print("[error] --reuse only makes sense together with --query")
        return 2
    if args.reuse:
        return _ask_existing(args)
    if args.input_dir is None:
        print("[error] --input_dir is required to build an index")
        return 2
    if args.unit == "chars" and args.chunk_size > 20_000:
        print("[warn] --unit chars with a very large --chunk_size produces "
              "huge chunks for most languages")
    extensions = normalize_extensions(args.extensions)
    try:
        documents, notes = collect_documents(
            input_dir=args.input_dir,
            extensions=extensions,
            max_docs=args.max_docs,
            pdf_max_pages=args.pdf_max_pages,
        )
    except FileNotFoundError as error:
        print(f"[error] {error}")
        return 1
    if not documents:
        print(f"[error] no corpus files with extensions {extensions} "
              f"under {args.input_dir}")
        return 1

    print(f"Корпус: {len(documents)} документов, "
          f"{sum(d.char_count for d in documents)} символов")

    embedder = Embedder(args.model or DEFAULT_MODEL, local_only=args.local_only)
    limit = embedder.max_tokens - 2  # room for [CLS]/[SEP]
    if args.strategy in ("both", "fixed_size") and args.unit == "tokens":
        if args.chunk_size > limit:
            print(f"[warn] --chunk_size {args.chunk_size} exceeds the model "
                  f"window ({limit} tokens): chunk tails would be truncated")
    print(f"Модель: {embedder.name} (dim={embedder.dim}, "
          f"max_tokens={embedder.max_tokens})")

    corpus = corpus_summary(
        documents,
        embedder.tokenizer,
        extra={
            "input_dir": str(args.input_dir),
            "embedding_model": embedder.name,
            "embedding_dim": embedder.dim,
        },
    )
    chunkers = {
        "fixed_size": FixedSizeChunker(
            size=args.chunk_size,
            overlap=args.chunk_overlap,
            unit=args.unit,
            tokenizer=embedder.tokenizer,
        ),
        "structure": StructureChunker(
            max_tokens=structure_max_tokens(
                embedder.max_tokens, args.structure_max_tokens
            ),
            min_chars=args.structure_min_chars,
            max_chars=args.structure_max_chars,
            tokenizer=embedder.tokenizer,
        ),
    }
    wanted = (
        ["fixed_size", "structure"] if args.strategy == "both" else [args.strategy]
    )

    results: list[StrategyResult] = []
    for name in wanted:
        result = run_strategy(
            strategy=name,
            documents=documents,
            chunker=chunkers[name],
            embedder=embedder,
            corpus=corpus,
            batch_size=args.batch_size,
            dedup=args.dedup,
        )
        path = save_index(result.index, result.index_path(args.index_dir))
        result.stats["index_kb"] = round(path.stat().st_size / 1024, 1)
        results.append(result)
        print(f"[{name}] {len(result.chunks)} чанков, "
              f"{result.encode_seconds:.1f}s на эмбеддинги -> {path}")

    queries = build_queries(
        documents, min_section_chars=args.min_section_chars,
        tokenizer=embedder.tokenizer,
    )
    if args.eval_path:
        try:
            queries.extend(load_queries(args.eval_path, documents))
        except (ValueError, OSError) as error:
            # A broken eval file is a user error: report it, do not traceback.
            print(f"[error] {args.eval_path}: {error}")
            return 1
    if queries:
        vectors = embedder.encode([query.text for query in queries], args.batch_size)
        for result in results:
            result.metrics = evaluate(
                result.chunks, result.embeddings, queries, vectors
            )
    else:
        print("[warn] no benchmark queries — pass --eval or use larger sections")

    comparison = {
        "corpus": corpus,
        "queries": [
            dataclasses.asdict(query) for query in queries
        ],
        "strategies": {
            result.strategy: {
                "chunking_params": result.index["chunking_params"],
                "stats": result.stats,
                "metrics": result.metrics,
            }
            for result in results
        },
    }
    comparison_path = args.index_dir / "comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Сравнение: {comparison_path}")

    if args.questions:
        for result in results:
            answers = query_index(
                result.index, embedder, args.questions, top_k=args.top_k
            )
            print(render_answers(result.index, answers, args.questions))
    if args.out:
        report = render_markdown(
            corpus=corpus,
            results=results,
            queries=queries,
            notes=notes,
            command=" ".join(sys.argv),
            index_dir=str(args.index_dir),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Отчёт: {out_path}")
    return 0


def _ask_existing(args: argparse.Namespace) -> int:
    """``--reuse``: answer questions from the stored indices, without rebuilding.

    The model comes from the index (that is the one whose vectors it holds) and
    a ``--model`` that contradicts it is an error rather than a silent mismatch:
    vectors from two models are not comparable, and the scores would be noise.
    """
    questions = args.questions or []
    wanted = (
        ["fixed_size", "structure"] if args.strategy == "both" else [args.strategy]
    )
    embedder: Embedder | None = None
    for name in wanted:
        path = args.index_dir / f"index_{name}.json"
        try:
            index = load_index(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print(f"[error] {error}")
            return 1
        stored_model = str(index["embedding_model"])
        if args.model and args.model != stored_model:
            print(f"[error] --model {args.model!r} but {path} was built with "
                  f"{stored_model!r}: rebuild the index or drop --model")
            return 2
        if embedder is None:
            embedder = Embedder(stored_model, local_only=args.local_only)
        answers = query_index(index, embedder, questions, top_k=args.top_k)
        print(render_answers(index, answers, questions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
