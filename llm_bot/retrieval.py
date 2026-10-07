"""Local vector search over the JSON indices built by ``scripts/index_documents.py``.

The indexing pipeline lives in a script (it is a deliverable with its own CLI,
report and benchmark), but *searching* an index is something the bot itself
needs at runtime. This module is the shared half: the embedder, the cosine
top-k, and the read/query/render helpers around a stored index. The script
imports from here so both sides score hits with exactly the same code — an
index built by the script can therefore be queried by the bot without any
drift between the benchmark and the live agent.

``sentence-transformers`` is imported inside :class:`Embedder.__init__`, never
at module import time, so ``llm_bot`` stays importable (and the bot stays fast)
on a machine with no embedding model and no torch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

try:  # numpy is only needed for the fast matrix multiply in ``search``
    import numpy as np
except ImportError:  # pragma: no cover - the pure-Python fallback is used
    np = None  # type: ignore[assignment]

# Chosen by measuring, not by reputation: on the day-21 corpus (Russian
# questions over Russian+English docs) paraphrase-multilingual-MiniLM-L12-v2
# answered 43% of the hand-written questions at rank 1 against 0% for the
# smaller English-only all-MiniLM-L6-v2. Its window is 128 tokens, not 256,
# hence the smaller --chunk_size default. The E5 family retrieves better still,
# but it wants "query: "/"passage: " prefixes, which this indexer does not add:
# encoding passages as queries would make the numbers wrong rather than merely
# weaker.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Version stamped into the index payload; ``load_index`` accepts it unchanged.
INDEX_VERSION = "1.0"


class ModelTokenizer:
    """Subword tokenizer of a sentence-transformers model, with offsets."""

    name = "subword"

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.name = getattr(tokenizer, "name_or_path", "subword") or "subword"

    def spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False, verbose=False,
        )
        return [
            tuple(pair) for pair in encoded["offset_mapping"] if pair[1] > pair[0]
        ]

    def count(self, text: str) -> int:
        return len(self.spans(text))


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    The model is loaded once per run (loading it per strategy doubled both the
    time and the memory of the pipeline) and every vector is L2-normalized, so
    a dot product is a cosine similarity.
    """

    def __init__(
        self, name: str = DEFAULT_EMBEDDING_MODEL, local_only: bool = False
    ) -> None:
        from llm_bot.progress import quiet_loading

        quiet_loading()
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(name, local_files_only=local_only)
        except Exception:  # noqa: BLE001 - offline run without a warm HF cache
            if not local_only:
                raise
            print(f"[warn] {name} is not cached, falling back to a download")
            self._model = SentenceTransformer(name)
        self.name = name
        dimension = getattr(self._model, "get_embedding_dimension", None)
        if dimension is None:  # sentence-transformers < 5
            dimension = self._model.get_sentence_embedding_dimension
        self.dim = int(dimension())
        self.max_tokens = int(getattr(self._model, "max_seq_length", 256) or 256)
        self.tokenizer = ModelTokenizer(self._model.tokenizer)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[round(float(value), 5) for value in row] for row in vectors]


def search(
    embeddings: Sequence[Sequence[float]], query: Sequence[float], top_k: int = 5
) -> list[tuple[int, float]]:
    """Top-k chunks by cosine similarity (vectors are already normalized)."""
    if not embeddings:
        return []
    top_k = max(1, min(top_k, len(embeddings)))
    if np is not None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        scores = matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]
    scored = [
        (i, math.fsum(a * b for a, b in zip(row, query)))
        for i, row in enumerate(embeddings)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def load_index(path: Path) -> dict[str, object]:
    """Read a stored index back (used by ``--reuse`` and by the RAG retriever)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"index not found: {path} (build it first, or drop --reuse)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("chunks", "embedding_model", "chunking_strategy"):
        if key not in payload:
            raise ValueError(f"{path}: not an index (no {key!r})")
    return payload


def query_index(
    index: dict[str, object],
    embedder: Embedder,
    questions: Sequence[str],
    top_k: int = 5,
) -> list[list[tuple[float, dict[str, object]]]]:
    """Answer ``questions`` against a stored or freshly built index.

    Works on the plain index dict, so the same path serves a freshly built
    index and one loaded from disk by ``--reuse`` — the answers cannot drift
    between the two.
    """
    chunks = index["chunks"]  # type: ignore[assignment]
    if not chunks:
        return [[] for _ in questions]
    embeddings = [chunk["embedding"] for chunk in chunks]
    vectors = embedder.encode(questions)
    answers: list[list[tuple[float, dict[str, object]]]] = []
    for vector in vectors:
        ranked = search(embeddings, vector, top_k=top_k)
        answers.append([(score, chunks[position]) for position, score in ranked])
    return answers


def render_answers(
    index: dict[str, object],
    answers: Sequence[Sequence[tuple[float, dict[str, object]]]],
    questions: Sequence[str],
    width: int = 90,
) -> str:
    """One block per question: score, ``file:line``, section, first line."""
    lines: list[str] = []
    for question, hits in zip(questions, answers):
        lines.append(f"[{index['chunking_strategy']}] {question}")
        if not hits:
            lines.append("  (индекс пуст)")
        for score, record in hits:
            meta = record["metadata"]
            location = (
                f"{meta.get('source', '?')}:{meta.get('start_line', 0)}"
                f"-{meta.get('end_line', 0)}"
            )
            section = meta.get("section") or meta.get("title") or "—"
            head = str(record.get("text", "")).strip().splitlines()
            lines.append(f"  {score:.3f}  {location}  {section}")
            if head:
                lines.append("       " + head[0][:width])
        lines.append("")
    return "\n".join(lines)