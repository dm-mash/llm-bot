"""Mock-CRM "notes" API — the backend wrapped by the MCP server.

A deliberately tiny CRM: notes are stored as a JSON file (``NOTES_DB`` env var,
default ``data/notes.json``). It plays the role of "any API" for the MCP task:
the MCP server exposes these functions as tools, and the agent reaches them
through MCP — never by importing this module directly.

The file is created lazily on the first write and guarded by a thread lock so
concurrent tool calls cannot corrupt it.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable holding the notes database path.
_ENV_NOTES_DB = "NOTES_DB"
DEFAULT_DB_PATH = Path("data") / "notes.json"

_lock = threading.Lock()


def default_db_path() -> Path:
    """Return the notes database path (``NOTES_DB`` or ``data/notes.json``)."""
    raw = os.getenv(_ENV_NOTES_DB)
    return Path(raw) if raw else DEFAULT_DB_PATH


@dataclass(frozen=True)
class Note:
    """One CRM note record."""

    id: int
    text: str
    tags: list[str] = field(default_factory=list)


def _load(path: Path) -> list[dict]:
    """Read the notes list from *path*; missing/corrupt file means empty."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _save(path: Path, notes: list[dict]) -> None:
    """Atomically write the notes list to *path* (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def _normalize_tags(tags: list[str] | None) -> list[str]:
    """Lowercase, strip and deduplicate tags."""
    seen: list[str] = []
    for tag in tags or []:
        clean = tag.strip().lower()
        if clean and clean not in seen:
            seen.append(clean)
    return seen


def add_note(
    text: str, tags: list[str] | None = None, *, db_path: Path | None = None
) -> Note:
    """Append a note and return it with its assigned id."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Текст заметки не может быть пустым.")
    path = db_path or default_db_path()
    clean_tags = _normalize_tags(tags)
    with _lock:
        notes = _load(path)
        note_id = (
            max((int(item.get("id", 0)) for item in notes), default=0) + 1
        )
        notes.append({"id": note_id, "text": text, "tags": clean_tags})
        _save(path, notes)
    return Note(id=note_id, text=text, tags=clean_tags)


def list_notes(
    tag: str | None = None, *, db_path: Path | None = None
) -> list[Note]:
    """Return all notes, optionally filtered by *tag* (case-insensitive)."""
    path = db_path or default_db_path()
    wanted = (tag or "").strip().lower()
    with _lock:
        notes = _load(path)
    result = []
    for item in notes:
        tags = [str(t) for t in item.get("tags", [])]
        if wanted and wanted not in tags:
            continue
        result.append(
            Note(
                id=int(item.get("id", 0)),
                text=str(item.get("text", "")),
                tags=tags,
            )
        )
    return result


def find_notes(
    query: str, *, db_path: Path | None = None
) -> list[Note]:
    """Return notes whose text or tags contain *query* (case-insensitive)."""
    needle = (query or "").strip().lower()
    if not needle:
        raise ValueError("Поисковый запрос не может быть пустым.")
    # Escape so user input is never treated as regex syntax.
    pattern = re.compile(re.escape(needle))
    path = db_path or default_db_path()
    with _lock:
        notes = _load(path)
    result = []
    for item in notes:
        text = str(item.get("text", ""))
        tags = [str(t) for t in item.get("tags", [])]
        if pattern.search(text.lower()) or any(
            pattern.search(tag.lower()) for tag in tags
        ):
            result.append(
                Note(id=int(item.get("id", 0)), text=text, tags=tags)
            )
    return result


def render_notes(notes: list[Note]) -> str:
    """Render notes as a human-readable text block (the tool's return value)."""
    if not notes:
        return "Заметок не найдено."
    lines = []
    for note in notes:
        tags = f" [{', '.join(note.tags)}]" if note.tags else ""
        lines.append(f"id={note.id}{tags}: {note.text}")
    return "\n".join(lines)
