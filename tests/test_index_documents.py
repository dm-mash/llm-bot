"""Tests for :mod:`scripts.index_documents`.

Everything here is offline and deterministic: the real embedding model is
replaced by a fake with a stable hashing embedding, and the real tokenizer by
:class:`CharTokenizer` (whitespace tokens). What is under test is the logic the
model is not responsible for — corpus collection, chunk boundaries, size
budgets, index provenance, retrieval and the report.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import zlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Import ``scripts/index_documents.py`` as a module.

    The script is a CLI, not a package member, so it is loaded by path. It must
    be registered in ``sys.modules`` *before* execution, otherwise the
    ``@dataclass`` decorator cannot resolve its own module globals.
    """
    path = ROOT / "scripts" / "index_documents.py"
    spec = importlib.util.spec_from_file_location("index_documents", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


idx = _load_module()


class FakeEmbedder:
    """Deterministic stand-in for :class:`idx.Embedder`.

    Bags word hashes into ``dim`` buckets and normalizes, so identical texts
    always score 1.0 and a text containing a query word outranks one that does
    not — enough to exercise ranking without downloading a model.

    The bucket uses ``crc32``, not ``hash()``: Python randomises string hashing
    per process, so a ``hash()`` based embedder silently reshuffles rankings
    between runs and any test asserting *which* chunk wins becomes a coin flip.
    """

    def __init__(self, name: str = "fake-model", local_only: bool = False,
                 dim: int = 32, max_tokens: int = 32) -> None:
        self.name = name
        self.dim = dim
        self.max_tokens = max_tokens
        self.tokenizer = idx.CharTokenizer()

    def encode(self, texts, batch_size: int = 0) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in text.lower().split():
                vector[zlib.crc32(word.encode("utf-8")) % self.dim] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([round(value / norm, 5) for value in vector])
        return vectors


@pytest.fixture()
def tokenizer() -> idx.CharTokenizer:
    return idx.CharTokenizer()


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A tiny corpus with a heading tree and two Python symbols."""
    root = tmp_path / "corpus"
    (root / "pkg").mkdir(parents=True)
    (root / "a.md").write_text(
        "# Руководство\n\n"
        "## Установка\n\n"
        "Скачайте репозиторий, поставьте зависимости и проверьте окружение.\n\n"
        "## Запуск\n\n"
        "Запустите тесты командой pytest и убедитесь, что всё зелёное.\n",
        encoding="utf-8",
    )
    (root / "pkg" / "mod.py").write_text(
        "import os\n\n\n"
        "def load(path):\n"
        '    """Read a file."""\n'
        "    with open(path) as handle:\n"
        "        return handle.read()\n\n\n"
        "class Store:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (root / "skip.log").write_text("ignored by extension\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "vendored.py").write_text("raise SystemExit\n", encoding="utf-8")
    return root


@pytest.fixture()
def documents(corpus: Path):
    docs, notes = idx.collect_documents(
        corpus, idx.normalize_extensions(".md", ".py")
    )
    return docs, notes


def _chunks(documents, chunker) -> list[idx.Chunk]:
    return [chunk for i, doc in enumerate(documents) for chunk in chunker.chunk(doc, i)]


def _markdown(text: str) -> idx.Document:
    """A Markdown document with its heading sections already parsed."""
    doc = idx.Document(
        path=Path("t.md"), rel_path="t.md", ext=".md", title="t",
        lines=text.splitlines(),
    )
    return dataclasses.replace(doc, sections=idx.parse_headings(doc.lines))


# --------------------------------------------------------------------------- #
# Corpus collection
# --------------------------------------------------------------------------- #


def test_collect_documents_filters_extensions_and_prunes_venv(documents) -> None:
    docs, notes = documents
    assert sorted(doc.rel_path for doc in docs) == ["a.md", "pkg/mod.py"]
    assert notes == []


def test_collect_documents_reports_only_unreadable_files(corpus: Path) -> None:
    (corpus / "empty.md").write_text("   \n\n", encoding="utf-8")
    docs, notes = idx.collect_documents(corpus, idx.normalize_extensions(".md"))
    # .log and .py are outside the extension filter: not even a note.
    assert [doc.rel_path for doc in docs] == ["a.md"]
    assert notes == [".md: empty or unreadable"]


def test_collect_documents_never_reads_vendored_trees(tmp_path: Path) -> None:
    root = tmp_path / "c"
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "i.js").write_text("x\n", encoding="utf-8")
    (root / "keep.md").write_text("# t\n", encoding="utf-8")
    docs, _ = idx.collect_documents(root, idx.normalize_extensions(".md", ".js"))
    assert [doc.rel_path for doc in docs] == ["keep.md"]


def test_collect_documents_parses_sections(documents) -> None:
    docs, _ = documents
    markdown = next(doc for doc in docs if doc.rel_path == "a.md")
    assert markdown.title == "Руководство"
    assert [section.path for section in markdown.sections] == [
        "Руководство",
        "Руководство > Установка",
        "Руководство > Запуск",
    ]
    code = next(doc for doc in docs if doc.rel_path == "pkg/mod.py")
    assert [section.path for section in code.sections] == ["<module>", "load", "Store"]
    assert [section.title for section in code.sections] == [
        "<module>", "def load", "class Store",
    ]


def test_headings_inside_fenced_code_are_not_sections() -> None:
    lines = ["# real", "", "```python", "# not a heading", "```", ""]
    assert [s.title for s in idx.parse_headings(lines)] == ["real"]


def test_collect_documents_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        idx.collect_documents(tmp_path / "nope", idx.normalize_extensions(".md"))


def test_empty_directory_yields_no_documents(tmp_path: Path) -> None:
    docs, notes = idx.collect_documents(tmp_path, idx.normalize_extensions(".md"))
    assert docs == [] and notes == []


def test_max_docs_stops_the_walk(corpus: Path) -> None:
    docs, _ = idx.collect_documents(
        corpus, idx.normalize_extensions(".md", ".py"), max_docs=1
    )
    assert len(docs) == 1


# --------------------------------------------------------------------------- #
# PDF page limits
# --------------------------------------------------------------------------- #


def _fake_pdf(monkeypatch: pytest.MonkeyPatch, path: Path, pages: list[str]) -> None:
    """Give ``path`` a PDF text layer of ``pages`` without writing a real PDF.

    A four-page document carries a description on page 1 and the *same* terms
    on pages 2-4, so the truncation test must not depend on real PDF bytes; only
    the page slicing under test matters.
    """
    class Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class Reader:
        def __init__(self, _path: str) -> None:
            self.pages = [Page(text) for text in pages]

    module = type(sys)("pypdf")
    module.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", module)


@pytest.fixture()
def pdf_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    first = root / "ID-250-248-463-709 Отчёт Б.pdf"
    first.write_bytes(b"%PDF-1.7 stub")
    _fake_pdf(monkeypatch, first, [
        "Занятие\nКоличество человек: 1\nДлительность: 5-6 часов",
        "Общие условия\nдля всех документов",
        "Общие условия\nдля всех документов",
        "Общие условия\nдля всех документов",
    ])
    return root


def test_pdf_max_pages_keeps_the_leading_pages(pdf_corpus: Path) -> None:
    """The document description must survive while the shared terms go.

    Indexing a prefix silently turns "the fact is not in the corpus" and "the
    fact was never indexed" into the same report line, so the cut has to be
    exact and declared.
    """
    docs, notes = idx.collect_documents(
        pdf_corpus, {".pdf"}, pdf_max_pages=1
    )
    assert len(docs) == 1
    text = docs[0].text
    assert "Длительность: 5-6 часов" in text
    assert "Общие условия" not in text
    assert docs[0].pages_dropped == 3


def test_pdf_max_pages_reports_the_pages_it_dropped(pdf_corpus: Path) -> None:
    docs, notes = idx.collect_documents(pdf_corpus, {".pdf"}, pdf_max_pages=1)
    assert any("страниц не попали в индекс" in note for note in notes)


def test_pdf_without_a_page_limit_keeps_everything(pdf_corpus: Path) -> None:
    docs, notes = idx.collect_documents(pdf_corpus, {".pdf"})
    assert "Общие условия" in docs[0].text
    assert docs[0].pages_dropped == 0
    assert not any("страниц не попали" in note for note in notes)


def test_pdf_page_limit_larger_than_the_file_is_not_an_error(
    pdf_corpus: Path,
) -> None:
    docs, _ = idx.collect_documents(pdf_corpus, {".pdf"}, pdf_max_pages=99)
    assert "Общие условия" in docs[0].text
    assert docs[0].pages_dropped == 0


def test_a_truncated_pdf_is_one_section(pdf_corpus: Path) -> None:
    """One kept page carries no page break, so it chunks as a single section.

    ``page_of_line`` stays empty here: with one page there is nothing to count.
    """
    docs, _ = idx.collect_documents(pdf_corpus, {".pdf"}, pdf_max_pages=1)
    doc = docs[0]
    assert len(doc.sections) == 1
    assert doc.sections[0].start_line == 1
    # A single-page PDF still knows it is page 1, which is what the chunk
    # metadata reports.
    assert idx.page_of_line(doc, 1) == "page 1"


def test_a_whole_pdf_is_split_into_page_sections(pdf_corpus: Path) -> None:
    """Without the limit every page keeps its own label."""
    docs, _ = idx.collect_documents(pdf_corpus, {".pdf"})
    assert [section.title for section in docs[0].sections] == [
        "page 1", "page 2", "page 3", "page 4"
    ]


def test_a_page_break_does_not_hide_the_page_starts(
    pdf_corpus: Path,
) -> None:
    """The marker must survive line splitting or every PDF loses its pages.

    ``str.splitlines()`` treats ``\\f`` as a line boundary, which would drop the
    marker before parsing sees it: the document would come back as one
    unlabelled section and ``page_of_line`` would always answer "".
    """
    docs, _ = idx.collect_documents(pdf_corpus, {".pdf"})
    doc = docs[0]
    # Pages are 3, 2, 2, 2 lines each; the marker line starts no page.
    assert doc.page_starts == (1, 5, 8, 11)
    assert [(s.start_line, s.end_line) for s in doc.sections] == [
        (1, 3), (5, 6), (8, 9), (11, 12)
    ]
    for start, end in zip(doc.sections, doc.sections[1:]):
        assert start.end_line < end.start_line
    assert idx.page_of_line(doc, 1) == "page 1"
    assert idx.page_of_line(doc, doc.sections[1].start_line) == "page 2"
    assert idx.page_of_line(doc, doc.sections[-1].end_line) == "page 4"


def test_page_spans_skip_the_marker_line() -> None:
    """A marker on its own line belongs to no page."""
    lines = ("a", "b", idx.PAGE_BREAK, "c", idx.PAGE_BREAK, "d")
    assert idx.page_spans(lines) == ((1, 2), (4, 4), (6, 6))


def test_page_spans_cover_every_line_once() -> None:
    """No content line may fall between two pages."""
    lines = ("a", "b", idx.PAGE_BREAK, "c", idx.PAGE_BREAK, "d")
    spans = idx.page_spans(lines)
    covered = [n for start, end in spans for n in range(start, end + 1)]
    assert covered == [1, 2, 4, 6]


def test_an_inline_marker_closes_the_page_without_losing_the_tail() -> None:
    """Text after a marker in the same line is the next page, so it must stay."""
    lines = ("a", f"b{idx.PAGE_BREAK}c", "d")
    assert idx.page_spans(lines) == ((1, 2), (2, 3))


def test_split_lines_keeps_the_page_marker() -> None:
    assert idx.split_lines("a\fb\nc") == ["a\fb", "c"]
    # The plain-Python behaviour that hid the marker in the first place.
    assert "a\fb\nc".splitlines() == ["a", "b", "c"]


def test_pdf_max_pages_zero_means_no_limit(pdf_corpus: Path) -> None:
    docs, _ = idx.collect_documents(pdf_corpus, {".pdf"}, pdf_max_pages=0)
    assert docs[0].pages_dropped == 0


def test_normalize_extensions_accepts_a_comma_separated_string() -> None:
    assert idx.normalize_extensions(".md,.py") == (".md", ".py")
    assert idx.normalize_extensions([".MD", "py"]) == (".md", ".py")


# --------------------------------------------------------------------------- #
# Fixed-size chunking
# --------------------------------------------------------------------------- #


def test_fixed_size_respects_the_token_budget(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = _chunks(docs, idx.FixedSizeChunker(size=8, overlap=2, tokenizer=tokenizer))
    assert chunks
    assert max(len(tokenizer.spans(chunk.text)) for chunk in chunks) <= 8


def test_fixed_size_covers_the_document(documents, tokenizer) -> None:
    docs, _ = documents
    for doc in docs:
        chunks = idx.FixedSizeChunker(size=8, tokenizer=tokenizer).chunk(doc, 0)
        assert doc.text.startswith(chunks[0].text)
        assert doc.text.rstrip().endswith(chunks[-1].text.rstrip())


def test_fixed_size_offsets_point_at_the_chunk_text(documents, tokenizer) -> None:
    docs, _ = documents
    by_rel = {doc.rel_path: doc for doc in docs}
    for chunk in _chunks(docs, idx.FixedSizeChunker(size=8, overlap=2, tokenizer=tokenizer)):
        source = by_rel[chunk.source]
        assert source.text[chunk.char_start:chunk.char_end] == chunk.text
        assert chunk.char_end - chunk.char_start == chunk.char_count
        assert chunk.char_count == len(chunk.text)
        assert 1 <= chunk.start_line <= chunk.end_line <= len(source.lines)


def test_fixed_size_never_splits_a_word(tokenizer) -> None:
    doc = idx.Document(
        path=Path("t.txt"), rel_path="t.txt", ext=".txt", title="t",
        lines=["alpha beta gamma delta epsilon zeta"],
    )
    chunks = idx.FixedSizeChunker(size=3, tokenizer=tokenizer).chunk(doc, 0)
    words = [word for chunk in chunks for word in chunk.text.split()]
    assert words == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def test_fixed_size_overlap_repeats_context(documents, tokenizer) -> None:
    docs, _ = documents
    doc = docs[0]
    plain = idx.FixedSizeChunker(size=8, overlap=0, tokenizer=tokenizer).chunk(doc, 0)
    shifted = idx.FixedSizeChunker(size=8, overlap=4, tokenizer=tokenizer).chunk(doc, 0)
    assert len(shifted) > len(plain)
    assert sum(c.char_count for c in shifted) > sum(c.char_count for c in plain)


def test_fixed_size_character_unit_honours_the_char_budget(tokenizer) -> None:
    text = "word " * 60
    doc = idx.Document(
        path=Path("t.txt"), rel_path="t.txt", ext=".txt", title="t",
        lines=[text.rstrip()],
    )
    chunks = idx.FixedSizeChunker(
        size=40, overlap=0, unit="chars", tokenizer=tokenizer
    ).chunk(doc, 0)
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert text.strip().endswith(chunks[-1].text)


def test_fixed_size_params_expose_the_configuration() -> None:
    assert idx.FixedSizeChunker(size=64, overlap=8).params == {
        "unit": "tokens", "chunk_size": 64, "chunk_overlap": 8,
    }


def test_fixed_size_rejects_a_bad_overlap() -> None:
    with pytest.raises(ValueError):
        idx.FixedSizeChunker(size=10, overlap=10)
    with pytest.raises(ValueError):
        idx.FixedSizeChunker(size=0)


# --------------------------------------------------------------------------- #
# Structural chunking
# --------------------------------------------------------------------------- #


def test_structure_keeps_sections_whole(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = _chunks(docs, idx.StructureChunker(max_tokens=64, min_chars=0, tokenizer=tokenizer))
    # "Руководство" holds no text of its own, so it is folded into its first
    # child rather than becoming a title-only chunk that outranks the sections
    # with the actual content.
    assert {chunk.section for chunk in chunks} == {
        "Руководство > Установка",
        "Руководство > Запуск",
        "<module>",
        "load",
        "Store",
    }
    assert any("Руководство" in chunk.text for chunk in chunks)  # title kept
    for chunk in chunks:
        if chunk.section != "<module>":  # a code preamble has no heading line
            assert chunk.section_title in chunk.text


def test_structure_respects_the_token_budget(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = _chunks(docs, idx.StructureChunker(max_tokens=6, min_chars=0, tokenizer=tokenizer))
    for chunk in chunks:
        if len(tokenizer.spans(chunk.text)) > 6:
            # The only allowed exception: a single line cannot be split without
            # corrupting the text (minified code, a Markdown table row).
            assert "\n" not in chunk.text.strip()


def test_structure_splits_oversized_sections_on_line_boundaries(tokenizer) -> None:
    body = "\n\n".join(f"Строка номер {n} про запуск тестов." for n in range(12))
    text = f"# Заголовок\n\n{body}\n"
    doc = _markdown(text)
    chunks = idx.StructureChunker(max_tokens=8, min_chars=0, tokenizer=tokenizer).chunk(doc, 0)
    assert len(chunks) > 1
    assert max(len(tokenizer.spans(c.text)) for c in chunks) <= 8
    assert chunks[0].text.startswith("# Заголовок")
    assert sum(c.text.count("Строка номер") for c in chunks) == 12


def test_structure_chunks_are_exact_line_ranges(documents, tokenizer) -> None:
    docs, _ = documents
    by_rel = {doc.rel_path: doc for doc in docs}
    chunks = _chunks(docs, idx.StructureChunker(max_tokens=64, min_chars=0, tokenizer=tokenizer))
    for chunk in chunks:
        source = by_rel[chunk.source]
        assert source.text[chunk.char_start:chunk.char_end] == chunk.text
        lines = source.lines[chunk.start_line - 1:chunk.end_line]
        assert "\n".join(lines).strip() == chunk.text


def test_structure_merges_tiny_pieces_inside_one_section(tokenizer) -> None:
    doc = _markdown("# t\n\nодин\n\nдва\n\nтри\n")
    chunks = idx.StructureChunker(max_tokens=64, min_chars=50, tokenizer=tokenizer).chunk(doc, 0)
    assert len(chunks) == 1
    assert all(word in chunks[0].text for word in ("один", "два", "три"))


def test_structure_does_not_merge_across_sections(tokenizer) -> None:
    doc = _markdown("# t\n\n## a\n\nраздел а\n\n## б\n\nраздел б\n")
    chunks = idx.StructureChunker(max_tokens=64, min_chars=200, tokenizer=tokenizer).chunk(doc, 0)
    # "# t" has no body and goes into "a"; "a" and "б" both have text of their
    # own, so they stay apart even though both are under the 200-char minimum.
    assert len(chunks) == 2
    assert all(len(chunk.text) < 200 for chunk in chunks)
    assert {chunk.section for chunk in chunks} == {"t > a", "t > б"}
    assert "раздел а" in chunks[0].text and "раздел б" not in chunks[0].text
    assert "раздел б" in chunks[1].text and "раздел а" not in chunks[1].text


def test_structure_never_merges_into_an_oversized_chunk(tokenizer) -> None:
    text = "# t\n\n" + "\n\n".join(f"абзац {n} из нескольких слов" for n in range(8))
    doc = _markdown(text)
    chunks = idx.StructureChunker(max_tokens=6, min_chars=20, tokenizer=tokenizer).chunk(doc, 0)
    for chunk in chunks:
        assert len(tokenizer.spans(chunk.text)) <= 6


def test_structure_falls_back_to_the_document_title_without_headings(
    tokenizer
) -> None:
    """A document with no headings still gets a non-empty ``section``."""
    doc = idx.Document(
        path=Path("plain.md"), rel_path="plain.md", ext=".md", title="Заметки",
        lines=["Индексация документов", "просто текст без заголовков"],
    )
    chunks = idx.StructureChunker(max_tokens=64, min_chars=0, tokenizer=tokenizer).chunk(doc, 0)
    assert len(chunks) == 1
    assert chunks[0].section == "Заметки"
    assert chunks[0].section_title == "Заметки"


def test_structure_rejects_negative_limits() -> None:
    with pytest.raises(ValueError):
        idx.StructureChunker(max_tokens=0)
    with pytest.raises(ValueError):
        idx.StructureChunker(max_tokens=10, min_chars=-1)
    with pytest.raises(ValueError):
        idx.StructureChunker(max_tokens=10, max_chars=-5)


def test_structure_params_expose_the_budget() -> None:
    params = idx.StructureChunker(max_tokens=32, min_chars=64).params
    assert params == {"max_tokens": 32, "min_chars": 64}
    assert "max_chars" in idx.StructureChunker(max_tokens=32, max_chars=500).params


def test_structure_max_chars_is_a_secondary_cap(tokenizer) -> None:
    text = "# t\n\n" + "\n\n".join(f"абзац {n} из нескольких слов" for n in range(8))
    doc = _markdown(text)
    chunks = idx.StructureChunker(
        max_tokens=10_000, min_chars=0, max_chars=200, tokenizer=tokenizer
    ).chunk(doc, 0)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_structure_caps_cannot_split_a_single_line(tokenizer) -> None:
    """A one-line document is one chunk: structural chunks are line ranges."""
    doc = _markdown("# t\n\n" + " ".join(["слово"] * 400) + "\n")
    chunks = idx.StructureChunker(
        max_tokens=10_000, min_chars=0, max_chars=200, tokenizer=tokenizer
    ).chunk(doc, 0)
    oversized = [chunk for chunk in chunks if len(chunk.text) > 200]
    assert len(oversized) == 1
    assert oversized[0].start_line == oversized[0].end_line  # never split mid-line
    # reported through chunk_stats, not silently
    stats = idx.chunk_stats(
        chunks, 10_000, {"max_tokens": 10_000, "min_chars": 0, "max_chars": 200},
        tokenizer=tokenizer, window_tokens=10_000,
    )
    assert stats["over_chars_share"] == len(oversized) / len(chunks) > 0.0
    assert stats["over_budget_share"] == 0.0


# --------------------------------------------------------------------------- #
# Index, provenance and statistics
# --------------------------------------------------------------------------- #


def test_index_is_serializable_and_complete(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = _chunks(docs, idx.FixedSizeChunker(size=8, tokenizer=tokenizer))
    embedder = FakeEmbedder()
    index = idx.build_index(
        chunks=chunks,
        embeddings=embedder.encode([c.text for c in chunks]),
        strategy="fixed_size",
        corpus=idx.corpus_summary(docs, tokenizer),
        params=idx.FixedSizeChunker(size=8).params,
        embedder_name=embedder.name,
        dim=embedder.dim,
    )
    payload = json.loads(json.dumps(index))  # round-trips through JSON
    assert payload["embedding_model"] == "fake-model"
    assert payload["embedding_dim"] == embedder.dim
    assert payload["chunking_strategy"] == "fixed_size"
    assert payload["corpus"]["pages_estimate"] > 0
    assert len(payload["chunks"]) == len(chunks)
    first = payload["chunks"][0]
    assert set(first["metadata"]) >= {
        "chunk_id", "source", "title", "section", "document_id", "chunk_position",
        "start_line", "end_line", "char_start", "char_end", "char_count",
        "token_count", "content_hash", "chunking_strategy",
    }
    assert len(first["embedding"]) == embedder.dim
    assert first["id"] == first["metadata"]["chunk_id"]


def test_build_index_rejects_a_length_mismatch(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.FixedSizeChunker(size=8, tokenizer=tokenizer).chunk(docs[0], 0)
    with pytest.raises(ValueError):
        idx.build_index(
            chunks=chunks, embeddings=[], strategy="fixed_size",
            corpus={}, params={}, embedder_name="fake", dim=8,
        )


def test_save_index_creates_parent_directories(tmp_path: Path, documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.FixedSizeChunker(size=16, tokenizer=tokenizer).chunk(docs[0], 0)
    index = idx.build_index(
        chunks=chunks, embeddings=FakeEmbedder().encode([c.text for c in chunks]),
        strategy="fixed_size", corpus={}, params={}, embedder_name="fake", dim=32,
    )
    path = idx.save_index(index, tmp_path / "deep" / "nested" / "index.json")
    assert path.exists() and path.stat().st_size > 0


def test_embeddings_are_normalized(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.FixedSizeChunker(size=16, tokenizer=tokenizer).chunk(docs[0], 0)
    embedder = FakeEmbedder()
    for vector in embedder.encode([chunk.text for chunk in chunks]):
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-3)
    assert embedder.encode([chunks[0].text])[0] == embedder.encode([chunks[0].text])[0]


def test_chunk_stats_counts_sizes_and_coverage(documents, tokenizer) -> None:
    docs, _ = documents
    doc = docs[0]
    chunks = idx.FixedSizeChunker(size=6, tokenizer=tokenizer).chunk(doc, 0)
    stats = idx.chunk_stats(
        chunks,
        corpus_chars=len(doc.text),
        params=idx.FixedSizeChunker(size=6).params,
        tokenizer=tokenizer,
        window_tokens=16,
    )
    assert stats["chunks"] == len(chunks)
    assert stats["tokens_max"] <= 6
    assert stats["over_budget_share"] == 0.0
    assert stats["over_window_share"] == 0.0
    assert 0.0 < stats["coverage_share"] <= 1.0
    assert stats["duplicate_share"] == 0.0
    assert stats["index_kb"] == 0.0


def test_chunk_stats_reports_oversized_chunks(documents, tokenizer) -> None:
    docs, _ = documents
    doc = docs[0]
    chunks = idx.StructureChunker(max_tokens=4, min_chars=0, tokenizer=tokenizer).chunk(doc, 0)
    stats = idx.chunk_stats(
        chunks, len(doc.text), idx.StructureChunker(max_tokens=4).params,
        tokenizer=tokenizer, window_tokens=4,
    )
    assert stats["over_window_share"] > 0.0


def _stats_for(tmp_path: Path, text: str, max_tokens: int, tokenizer) -> dict:
    path = tmp_path / "long.md"
    path.write_text(text, encoding="utf-8")
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    chunker = idx.StructureChunker(max_tokens=max_tokens, min_chars=0,
                                  tokenizer=tokenizer)
    chunks = chunker.chunk(docs[0], 0)
    for chunk in chunks:
        chunk.token_count = tokenizer.count(chunk.text)
    return idx.chunk_stats(
        chunks, len(docs[0].text), chunker.params,
        tokenizer=tokenizer, window_tokens=max_tokens,
    ), chunks


def test_overflow_from_one_long_line_is_measured_separately(
    tmp_path: Path, tokenizer
) -> None:
    """A line longer than the budget is a deliberate exception, not a leak.

    Both numbers are reported so the exception stays visible, and the strict
    invariant (a *multi-line* chunk over budget) stays at zero.
    """
    row = "| " + " ".join(f"поле-{i}" for i in range(40)) + " |"
    text = f"# Спецификация\n\n{row}\n\n## Обычный раздел\n\n" + "текст " * 20 + "\n"
    stats, chunks = _stats_for(tmp_path, text, max_tokens=24, tokenizer=tokenizer)
    assert stats["over_budget_share"] > 0.0
    assert stats["single_line_overflow_share"] == stats["over_budget_share"]
    assert stats["over_budget_multiline_share"] == 0.0
    over = [c for c in chunks if c.token_count > 24]
    assert len(over) == 1
    assert over[0].text == row  # kept whole, not cut mid-line
    assert over[0].start_line == over[0].end_line


def test_multiline_chunks_never_exceed_the_budget(
    tmp_path: Path, tokenizer
) -> None:
    """The strict invariant, on text that has no single-line excuse."""
    paragraph = " ".join(f"слово-{i}" for i in range(30))
    text = f"# Спецификация\n\n{paragraph}\n\n## Второй\n\n{paragraph}\n\n## Третий\n\n{paragraph}\n"
    stats, chunks = _stats_for(tmp_path, text, max_tokens=40, tokenizer=tokenizer)
    assert stats["over_budget_share"] == 0.0
    assert stats["over_budget_multiline_share"] == 0.0
    assert stats["single_line_overflow_share"] == 0.0
    assert len(chunks) > 1


def test_single_line_overflow_is_named_in_the_run_output(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    row = "| " + " ".join(f"поле-{i}" for i in range(60)) + " |"
    source = tmp_path / "table.md"
    source.write_text(f"# Таблица\n\n{row}\n", encoding="utf-8")
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    assert idx.main([
        "--input_dir", str(tmp_path), "--index_dir", str(tmp_path / "emb"),
        "--out", "", "--strategy", "structure", "--structure_min_chars", "0",
        "--structure_max_tokens", "20", "--min_section_chars", "0",
    ]) == 0
    out = capsys.readouterr().out
    assert "одиночная строка длиннее" in out
    assert "table.md:3" in out


def test_drop_duplicates_keeps_the_first_occurrence(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.StructureChunker(max_tokens=64, min_chars=0, tokenizer=tokenizer).chunk(docs[0], 0)
    assert idx.drop_duplicates(chunks + chunks) == chunks


# --------------------------------------------------------------------------- #
# Retrieval and benchmark
# --------------------------------------------------------------------------- #


def test_search_ranks_by_similarity() -> None:
    embedder = FakeEmbedder()
    corpus = ["котики и собачки", "автомобили и дороги", "котики"]
    vectors = embedder.encode(corpus)
    query = embedder.encode(["котики"])[0]
    ranked = idx.search(vectors, query, top_k=3)
    assert [i for i, _ in ranked] == [2, 0, 1]
    assert ranked[0][1] > ranked[-1][1]


def test_search_handles_an_empty_index() -> None:
    assert idx.search([], [0.1, 0.2], top_k=5) == []


def test_is_hit_uses_exact_char_spans() -> None:
    chunk = idx.Chunk(
        chunk_id="c", text="x" * 10, strategy="fixed_size", document_id=0,
        source="a.md", title="a", section="", section_title="",
        chunk_position=0, start_line=1, end_line=1, char_start=100, char_end=110,
    )
    query = idx.EvalQuery(
        text="q", source="a.md", section="s", start_line=5, end_line=6,
        char_start=105, char_end=120,
    )
    assert idx.is_hit(chunk, query)  # spans overlap
    far = idx.EvalQuery(
        text="q", source="a.md", section="s", start_line=50, end_line=51,
        char_start=5000, char_end=5010,
    )
    assert not idx.is_hit(chunk, far)
    other = idx.EvalQuery(
        text="q", source="b.md", section="s", start_line=1, end_line=1,
        char_start=100, char_end=110,
    )
    assert not idx.is_hit(chunk, other)  # different document


def test_is_hit_falls_back_to_lines_without_offsets() -> None:
    chunk = idx.Chunk(
        chunk_id="c", text="t", strategy="structure", document_id=0, source="a.md",
        title="a", section="s", section_title="s", chunk_position=0,
        start_line=4, end_line=6, char_start=0, char_end=0,
    )
    query = idx.EvalQuery(
        text="q", source="a.md", section="s", start_line=6, end_line=9
    )
    assert idx.is_hit(chunk, query)


def test_build_queries_uses_sections_big_enough_to_be_answers(corpus, tokenizer) -> None:
    docs, _ = idx.collect_documents(corpus, idx.normalize_extensions(".md", ".py"))
    queries = idx.build_queries(docs, min_section_chars=10, tokenizer=tokenizer)
    assert queries
    by_source = {doc.rel_path: doc for doc in docs}
    for query in queries:
        source = by_source[query.source]
        gold = source.text[query.char_start:query.char_end]
        assert gold.splitlines()[0]  # the gold span starts at a heading/symbol
        assert query.section in {s.path for s in source.sections}
        assert query.gold_id == f"{query.source}:{query.start_line}-{query.end_line}"
    tiny = idx.build_queries(docs, min_section_chars=100_000, tokenizer=tokenizer)
    assert tiny == []


def test_structure_chunks_contain_a_whole_section_fixed_size_fragments_it(
    documents, tokenizer
) -> None:
    """The structural guarantee, independent of any embedding model.

    Whenever a section fits the budget, one structure chunk holds the whole gold
    span; fixed-size windows spread the same span over several chunks, which is
    what caps its recall@k — the ranking may still be right, but the answer is
    cut in half.
    """
    docs, _ = documents
    queries = idx.build_queries(docs, min_section_chars=10, tokenizer=tokenizer)
    assert queries
    structural = _chunks(docs, idx.StructureChunker(max_tokens=64, min_chars=0,
                                                    tokenizer=tokenizer))
    fixed = _chunks(docs, idx.FixedSizeChunker(size=8, tokenizer=tokenizer))
    for query in queries:
        containing = [
            chunk
            for chunk in structural
            if idx.is_hit(chunk, query)
            and chunk.char_start <= query.char_start
            and chunk.char_end >= query.char_end
        ]
        assert len(containing) == 1, f"no single chunk for {query.gold_id}"
        assert containing[0].section == query.section
        assert any(idx.is_hit(chunk, query) for chunk in fixed)
    fragmented = sum(
        1
        for query in queries
        if len([c for c in fixed if idx.is_hit(c, query)]) > 1
    )
    assert fragmented > 0


def test_evaluate_reports_zero_for_a_query_with_no_match(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.StructureChunker(max_tokens=64, min_chars=0, tokenizer=tokenizer).chunk(docs[0], 0)
    embedder = FakeEmbedder()
    query = idx.EvalQuery(
        text="отсутствующий запрос", source="нет.md", section="нет",
        start_line=1, end_line=2,
    )
    metrics = idx.evaluate(chunks, embedder.encode([c.text for c in chunks]), [query],
                           embedder.encode([query.text]))
    assert metrics["recall@1"] == 0.0
    assert metrics["mrr@10"] == 0.0
    assert "нет.md" in metrics["misses"]


def test_evaluate_without_queries_is_empty() -> None:
    assert idx.evaluate([], [], [], []) == {}


def test_load_queries_reads_yaml_and_validates_sources(documents, tmp_path: Path) -> None:
    docs, _ = documents
    path = tmp_path / "queries.yaml"
    path.write_text(
        "queries:\n"
        "  - text: как установить\n"
        "    source: a.md\n"
        "    section: Руководство > Установка\n",
        encoding="utf-8",
    )
    queries = idx.load_queries(path, docs)
    assert len(queries) == 1
    assert queries[0].kind == "manual"
    assert queries[0].section == "Руководство > Установка"
    assert queries[0].char_end > queries[0].char_start

    bad = tmp_path / "bad.yaml"
    bad.write_text("queries:\n  - text: t\n    source: nope.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source"):
        idx.load_queries(bad, docs)

    missing = tmp_path / "missing.yaml"
    missing.write_text(
        "queries:\n  - text: t\n    source: a.md\n    section: Нет\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not found"):
        idx.load_queries(missing, docs)


def test_load_queries_resolves_a_bare_leaf_title(documents, tmp_path: Path) -> None:
    """A hand-written query names the section, not its full "A > B" ancestry."""
    docs, _ = documents
    path = tmp_path / "q.yaml"
    path.write_text(
        "queries:\n  - text: t\n    source: a.md\n    section: Установка\n",
        encoding="utf-8",
    )
    query = idx.load_queries(path, docs)[0]
    assert query.section == "Установка"
    section = idx._find_section(docs[0], "Установка")
    assert section is not None and section.title == "Установка"


def test_load_queries_refuses_an_ambiguous_section(tmp_path: Path) -> None:
    """A section name shared by two sections must not resolve arbitrarily."""
    def document(text: str) -> idx.Document:
        return idx.Document(
            path=Path("t.md"), rel_path="t.md", ext=".md", title="t",
            lines=text.splitlines(), sections=idx.parse_headings(text.splitlines()),
        )

    path = tmp_path / "q.yaml"
    path.write_text(
        "queries:\n  - text: t\n    source: t.md\n    section: shared\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found"):
        idx.load_queries(path, [document("# t\n\n## shared\n\nодин\n\n"
                                        "## shared\n\nдва\n")])
    assert idx.load_queries(path, [document("# t\n\n## shared\n\nодин\n")])


def test_main_reports_a_broken_eval_file(corpus: Path, tmp_path: Path,
                                        monkeypatch, capsys) -> None:
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    bad = tmp_path / "bad.yaml"
    bad.write_text("queries:\n  - text: t\n    source: nope.md\n", encoding="utf-8")
    assert idx.main([
        "--input_dir", str(corpus), "--index_dir", str(tmp_path / "emb"),
        "--out", "", "--eval", str(bad),
    ]) == 1
    assert "unknown source" in capsys.readouterr().out


def test_load_queries_defaults_the_gold_span_to_the_whole_document(
    documents, tmp_path: Path
) -> None:
    docs, _ = documents
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"queries": [{"text": "t", "source": "a.md"}]}), encoding="utf-8")
    query = idx.load_queries(path, docs)[0]
    assert query.start_line == 1
    assert query.end_line == len(docs[0].lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parser_defaults_keep_both_budgets_explicit() -> None:
    args = idx.build_parser().parse_args(["--input_dir", "."])
    assert args.questions is None and args.top_k == 5 and args.reuse is False
    assert args.structure_max_tokens is None  # derived from the model window
    assert args.structure_max_chars == 0  # secondary cap off
    assert args.structure_min_chars == 200
    assert args.chunk_size == 120
    assert args.chunk_overlap == 20
    assert args.unit == "tokens"
    assert args.strategy == "both"
    # The indices are megabytes of embeddings: never inside the committed tree.
    assert args.index_dir == Path("data/emb")
    assert args.out.startswith("results/")


def test_main_rejects_an_overlap_larger_than_the_window(capsys) -> None:
    assert idx.main([
        "--input_dir", ".", "--chunk_size", "10", "--chunk_overlap", "10"
    ]) == 2
    assert "--chunk_overlap" in capsys.readouterr().out


def test_parser_requires_input_dir_unless_reusing(capsys) -> None:
    assert idx.main(["--index_dir", "somewhere"]) == 2
    assert "--input_dir is required" in capsys.readouterr().out


def test_structure_max_tokens_defaults_to_the_model_window() -> None:
    assert idx.structure_max_tokens(256, None) == 254
    assert idx.structure_max_tokens(512, 300) == 300
    assert idx.structure_max_tokens(8, None) == 32  # never below a sane floor


def test_end_to_end_run_writes_indices_comparison_and_report(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    index_dir = tmp_path / "emb"
    report = tmp_path / "report.md"
    assert idx.main([
        "--input_dir", str(corpus),
        "--extensions", ".md,.py",
        "--index_dir", str(index_dir),
        "--out", str(report),
        "--chunk_size", "8",
        "--chunk_overlap", "2",
        "--structure_max_tokens", "6",
        "--structure_min_chars", "0",
        "--min_section_chars", "20",
    ]) == 0
    assert report.exists()
    for strategy in ("fixed_size", "structure"):
        path = index_dir / f"index_{strategy}.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["chunks"]
        assert payload["corpus"]["input_dir"] == str(corpus)
    comparison = json.loads((index_dir / "comparison.json").read_text(encoding="utf-8"))
    assert set(comparison["strategies"]) == {"fixed_size", "structure"}
    assert comparison["queries"]
    text = report.read_text(encoding="utf-8")
    assert "# Индексация документов" in text
    assert "recall@1" in text
    assert "Как читать индекс" in text
    assert "Сравнение стратегий" in text


def test_report_snippet_actually_runs(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The «Как читать индекс» snippet must work, not just look right."""
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    index_dir = tmp_path / "emb"
    report = tmp_path / "report.md"
    idx.main([
        "--input_dir", str(corpus), "--index_dir", str(index_dir),
        "--out", str(report), "--structure_min_chars", "0",
        "--structure_max_tokens", "64",
    ])
    text = report.read_text(encoding="utf-8")
    block = text.split("## Как читать индекс")[1].split("```python")[1].split("```")[0]
    capsys.readouterr()  # drop the pipeline's own output
    exec(compile(block, "report-snippet", "exec"), {})  # noqa: S102
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("1.000 ")  # the query chunk ranks first


def test_single_strategy_run_skips_the_other(corpus, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    index_dir = tmp_path / "emb"
    assert idx.main([
        "--input_dir", str(corpus), "--index_dir", str(index_dir),
        "--out", "", "--strategy", "structure",
        "--structure_min_chars", "0", "--structure_max_tokens", "32",
    ]) == 0
    assert (index_dir / "index_structure.json").exists()
    assert not (index_dir / "index_fixed_size.json").exists()


def test_main_reports_a_missing_input_dir(tmp_path: Path, capsys) -> None:
    assert idx.main([
        "--input_dir", str(tmp_path / "nope"),
        "--index_dir", str(tmp_path / "emb"),
        "--out", str(tmp_path / "report.md"),
    ]) == 1
    assert "nope" in capsys.readouterr().out


def test_main_reports_an_empty_corpus(tmp_path: Path, capsys) -> None:
    assert idx.main([
        "--input_dir", str(tmp_path),
        "--index_dir", str(tmp_path / "emb"),
        "--out", "",
    ]) == 1
    assert "no corpus files" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Manual querying
# --------------------------------------------------------------------------- #


def test_query_index_ranks_the_matching_chunk_first(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = _chunks(docs, idx.StructureChunker(max_tokens=64, min_chars=0,
                                                tokenizer=tokenizer))
    embedder = FakeEmbedder()
    index = idx.build_index(
        chunks=chunks, embeddings=embedder.encode([c.text for c in chunks]),
        strategy="structure", corpus={}, params={}, embedder_name=embedder.name,
        dim=embedder.dim,
    )
    question = chunks[1].text
    answers = idx.query_index(index, embedder, [question], top_k=2)
    assert len(answers) == 1
    assert len(answers[0]) == 2
    scores = [score for score, _ in answers[0]]
    assert scores == sorted(scores, reverse=True)  # sorted, and capped at top_k
    # Asking with a chunk's own text is the one case with no ambiguity about
    # which chunk should win: same vector, cosine 1.0, nobody else can tie.
    # Not exactly 1.0 because the index stores vectors rounded to 5 decimals.
    top_score, top = answers[0][0]
    assert top_score == pytest.approx(1.0, abs=2e-4)
    assert top["text"] == question
    assert top["metadata"]["source"] == chunks[1].source
    assert (top["metadata"]["start_line"], top["metadata"]["end_line"]) == (
        chunks[1].start_line, chunks[1].end_line
    )


def test_query_index_respects_top_k_and_empty_index(tokenizer) -> None:
    embedder = FakeEmbedder()
    index = {"chunking_strategy": "fixed_size", "chunks": []}
    assert idx.query_index(index, embedder, ["вопрос"], top_k=3) == [[]]
    assert "(индекс пуст)" in idx.render_answers(index, [[]], ["вопрос"])


def test_render_answers_shows_score_location_and_section(documents, tokenizer) -> None:
    docs, _ = documents
    chunks = idx.StructureChunker(max_tokens=64, min_chars=0,
                                  tokenizer=tokenizer).chunk(docs[0], 0)
    embedder = FakeEmbedder()
    index = idx.build_index(
        chunks=chunks, embeddings=embedder.encode([c.text for c in chunks]),
        strategy="structure", corpus={}, params={}, embedder_name=embedder.name,
        dim=embedder.dim,
    )
    question = chunks[1].text
    text = idx.render_answers(
        index, idx.query_index(index, embedder, [question], top_k=1), [question]
    )
    assert text.startswith(f"[structure] {question[:40]}")
    location = chunks[1]
    assert f"a.md:{location.start_line}-{location.end_line}  " in text
    assert location.section in text
    assert text.rstrip().endswith("...") is False  # a single hit, no truncation
    assert location.text.splitlines()[0][:20] in text


def test_load_index_rejects_a_missing_or_foreign_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build it first"):
        idx.load_index(tmp_path / "nope.json")
    foreign = tmp_path / "index_structure.json"
    foreign.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an index"):
        idx.load_index(foreign)


def test_query_after_building_prints_both_strategies(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    assert idx.main([
        "--input_dir", str(corpus), "--index_dir", str(tmp_path / "emb"), "--out", "",
        "--structure_min_chars", "0", "--structure_max_tokens", "64",
        "--query", "как запустить тесты", "--query", "второй вопрос",
        "--top_k", "2",
    ]) == 0
    out = capsys.readouterr().out
    for strategy in ("fixed_size", "structure"):
        for question in ("как запустить тесты", "второй вопрос"):
            assert out.count(f"[{strategy}] {question}") == 1


def test_reuse_answers_without_rebuilding(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The whole point of --reuse: no corpus walk, no re-embedding."""
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    index_dir = tmp_path / "emb"
    idx.main([
        "--input_dir", str(corpus), "--index_dir", str(index_dir), "--out", "",
        "--strategy", "structure", "--structure_min_chars", "0",
        "--structure_max_tokens", "64",
    ])
    monkeypatch.setattr(
        idx, "collect_documents",
        lambda **_: (_ for _ in ()).throw(AssertionError("re-walked the corpus")),
    )
    monkeypatch.setattr(idx, "Embedder", lambda name, local_only=False: FakeEmbedder(name))
    assert idx.main([
        "--reuse", "--index_dir", str(index_dir), "--strategy", "structure",
        "--query", "как запустить тесты", "--top_k", "1",
    ]) == 0
    assert "[structure] как запустить тесты" in capsys.readouterr().out


def test_reuse_reports_a_missing_index(tmp_path: Path, capsys) -> None:
    assert idx.main([
        "--reuse", "--index_dir", str(tmp_path), "--strategy", "structure",
        "--query", "вопрос",
    ]) == 1
    assert "build it first" in capsys.readouterr().out


def test_reuse_refuses_a_model_mismatch(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Vectors from two models are not comparable — refuse, do not warn."""
    monkeypatch.setattr(idx, "Embedder", FakeEmbedder)
    index_dir = tmp_path / "emb"
    idx.main([
        "--input_dir", str(corpus), "--index_dir", str(index_dir), "--out", "",
        "--strategy", "structure", "--structure_min_chars", "0",
        "--structure_max_tokens", "64",
    ])
    capsys.readouterr()
    assert idx.main([
        "--reuse", "--index_dir", str(index_dir), "--strategy", "structure",
        "--query", "вопрос", "--model", "some-other-model",
    ]) == 2
    assert "some-other-model" in capsys.readouterr().out


def test_reuse_without_query_is_refused(capsys) -> None:
    assert idx.main(["--reuse"]) == 2
    assert "only makes sense together with --query" in capsys.readouterr().out


def test_top_k_must_be_positive(capsys) -> None:
    assert idx.main(["--input_dir", ".", "--top_k", "0", "--query", "x"]) == 2
    assert "--top_k" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Fenced code blocks
# --------------------------------------------------------------------------- #


def test_fenced_ranges_covers_delimiters_and_unterminated_fences() -> None:
    lines = [
        "```yaml", "a: 1", "", "b: 2", "```", "", "~~~", "c", "", "```", "d",
    ]
    assert idx.fenced_ranges(lines) == [(1, 5), (7, len(lines))]


def test_blank_line_runs_does_not_split_inside_protected_ranges() -> None:
    lines = ["# T", "", "```yaml", "a: 1", "", "b: 2", "```", "", "after"]
    protected = idx.fenced_ranges(lines)
    # Without the ranges the blank line at 5 cuts the example in half.
    assert idx.blank_line_runs(lines, 1, len(lines)) == [(1, 1), (3, 4), (6, 7), (9, 9)]
    assert idx.blank_line_runs(lines, 1, len(lines), protected) == [(1, 1), (3, 7), (9, 9)]


def test_fenced_example_stays_whole_in_one_chunk(tmp_path: Path) -> None:
    """The regression that started this: a YAML example split at its own blanks."""
    block = "\n".join(["```yaml", "kind_labels:", "  stack: да", "",
                        "invariants:", "  STACK-1:", "    kind: stack", "```"])
    source = tmp_path / "doc.md"
    source.write_text(
        f"# Спецификация\n\n#### Пример YAML-записи\n\n{block}\n\n"
        "Следующий раздел.\n\n## Хвост\n\nпосле примера.\n",
        encoding="utf-8",
    )
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    doc = docs[0]
    assert doc.verbatim, "the document must report its fenced ranges"
    chunks = idx.StructureChunker(max_tokens=200, min_chars=0,
                                  tokenizer=idx.CharTokenizer()).chunk(doc, 0)
    joined = [c.text for c in chunks if "kind_labels" in c.text]
    assert len(joined) == 1
    assert block in joined[0], "the whole fenced example must be in one chunk"
    # The heading travels with it, so the chunk is findable by its own title.
    assert "Пример YAML-записи" in joined[0]


def test_fence_longer_than_the_budget_is_still_split(tmp_path: Path) -> None:
    """Fence protection is about gratuitous splits, not about unbounded chunks.

    A verbatim block that cannot fit the budget has to be cut, otherwise the
    model would silently drop the tail; the report shows it as over budget.
    """
    body = "\n".join(f"  key_{i}: значение {i}" for i in range(60))
    source = tmp_path / "big.md"
    source.write_text(f"# Большой пример\n\n```python\n{body}\n```\n", encoding="utf-8")
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    chunks = idx.StructureChunker(max_tokens=40, min_chars=0,
                                  tokenizer=idx.CharTokenizer()).chunk(docs[0], 0)
    assert len(chunks) > 1
    joined = "\n".join(chunk.text for chunk in chunks)
    assert joined.count("```") == 2          # both fences survive the split
    # The opening fence leads the code. The document title above it is a
    # heading-only piece and gets folded in, so the fence is no longer at
    # offset 0 — but it still opens the chunk that holds the start of the code.
    fenced = [c for c in chunks if "```python" in c.text]
    assert len(fenced) == 1
    assert fenced[0].text.lstrip().splitlines()[0].endswith("Большой пример")
    assert fenced[0].text.splitlines()[2].strip() == "```python"
    assert "key_0" in joined and "key_59" in joined  # and no line is lost


def test_heading_only_section_is_merged_into_its_child(tmp_path: Path) -> None:
    """A heading with no body of its own must not become a chunk of its own.

    ``## Часы работы`` followed directly by ``### Будни`` makes a chunk that is
    only the title. It embeds as a near-perfect match for "Часы работы кофейни"
    and therefore outranks the subsection that actually holds the answer, so the
    retriever returns a title where the fact should be. The title is kept, but
    as part of the child's chunk.
    """
    source = tmp_path / "kb.md"
    source.write_text(
        "## Часы работы\n\n### Будни\n\n- Понедельник — пятница: 8:00 — 22:00.\n",
        encoding="utf-8",
    )
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    chunks = idx.StructureChunker(max_tokens=200, min_chars=0,
                                  tokenizer=idx.CharTokenizer()).chunk(docs[0], 0)
    assert len(chunks) == 1
    assert "Часы работы" in chunks[0].text       # the title survives
    assert "8:00 — 22:00" in chunks[0].text      # and so does the answer
    assert chunks[0].section.endswith("Будни")   # under the more specific path


def test_a_heading_with_body_of_its_own_stays_its_own_chunk(tmp_path: Path) -> None:
    """The merge must not swallow a real section to remove a lone title."""
    source = tmp_path / "kb.md"
    source.write_text(
        "## Часы работы\n\nКофейня открыта ежедневно.\n\n"
        "### Будни\n\n- Понедельник — пятница: 8:00 — 22:00.\n",
        encoding="utf-8",
    )
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    chunks = idx.StructureChunker(max_tokens=200, min_chars=0,
                                  tokenizer=idx.CharTokenizer()).chunk(docs[0], 0)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("## Часы работы")
    assert "ежедневно" in chunks[0].text


def test_parsers_still_ignore_structure_inside_fences() -> None:
    lines = [
        "# Настоящий заголовок", "", "```md", "# Не заголовок", "",
        "## Тоже не заголовок", "```", "", "## Настоящий раздел", "текст",
    ]
    titles = [section.title for section in idx.parse_headings(lines)]
    assert titles == ["Настоящий заголовок", "Настоящий раздел"]
    code = [
        "import os", "", "```python", "def not_a_symbol():", "```", "",
        "def real_symbol():", "    pass",
    ]
    symbols = [section.title for section in idx.parse_code_blocks(code)]
    assert symbols == ["<module>", "def real_symbol"]


def test_a_heading_is_never_left_as_a_chunk_of_its_own(tmp_path: Path) -> None:
    """A title with no body retrieves on the title and answers nothing.

    The heading is the first piece of its section, so it can only be merged
    forward; without that the exact-title query returns an empty chunk.
    """
    block = "\n".join(["```yaml", "kind: stack", "statement: держим stdlib", "```"])
    source = tmp_path / "doc.md"
    source.write_text(
        f"# План\n\n#### Пример YAML-записи\n\n{block}\n\n"
        + "filler " * 200 + "\n\n## Следующий\n\nхвост.\n",
        encoding="utf-8",
    )
    docs, _ = idx.collect_documents(tmp_path, {".md"})
    chunks = idx.StructureChunker(max_tokens=200, min_chars=200,
                                  tokenizer=idx.CharTokenizer()).chunk(docs[0], 0)
    assert not [c for c in chunks if c.text.strip() == "#### Пример YAML-записи"]
    keeper = next(c for c in chunks if "kind: stack" in c.text)
    assert "Пример YAML-записи" in keeper.text
    assert "```" in keeper.text
