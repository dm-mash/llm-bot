"""Persistence for the layered agent memory.

Storage backend for :mod:`llm_bot.memory`. Two durable layers are persisted:

* **Working memory** — per-session task data, stored under ``<session_id>`` so it
  survives restarts of the same session but is never shared between sessions.
* **Long-term memory** — per-agent, per-owner durable data (profile / decisions /
  knowledge), stored under ``<agent>/<owner>`` so one user's data is isolated
  from another user's (privacy via namespace).

Values are stored as serialized :class:`~llm_bot.memory.MemoryEntry` objects.
The default implementation writes plain JSON files under a single root directory
(``data/memory/``); the working and long-term namespaces are kept in separate
sub-trees so they can be pointed at different volumes later without changing the
API.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, runtime_checkable

from llm_bot.memory import MemoryEntry

# Filesystem-safe id: alphanumerics, dash, underscore, dot.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# Working memory lives here (per session id).
_WORKING_DIR = "working"
# Long-term memory lives here (per agent, then per owner).
_LONG_TERM_DIR = "long_term"


def _entry_to_dict(entry: MemoryEntry) -> dict[str, str]:
    return {"key": entry.key, "value": entry.value, "source": entry.source}


def _dict_to_entry(data: Any, fallback_key: str) -> MemoryEntry:
    if isinstance(data, dict):
        return MemoryEntry(
            key=str(data.get("key", fallback_key)),
            value=str(data.get("value", "")),
            source=str(data.get("source", "assistant")),
        )
    return MemoryEntry(key=fallback_key, value=str(data))


def _assert_safe(value: str, what: str) -> None:
    if not value or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"Invalid {what} {value!r}. Use only letters, digits, dash, "
            "underscore or dot."
        )


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent storage for the durable memory layers."""

    def load_working(self, session_id: str) -> dict[str, MemoryEntry]:
        """Return the working-memory entries for *session_id* (``{}`` if none)."""
        ...

    def save_working(
        self, session_id: str, entries: dict[str, MemoryEntry]
    ) -> None:
        """Persist the working-memory entries for *session_id*."""
        ...

    def load_long_term(
        self, agent: str, owner: str
    ) -> dict[str, MemoryEntry]:
        """Return the long-term entries for *agent*/*owner* (``{}`` if none)."""
        ...

    def save_long_term(
        self, agent: str, owner: str, entries: dict[str, MemoryEntry]
    ) -> None:
        """Persist the long-term entries for *agent*/*owner*."""
        ...


class JsonMemoryStore:
    """JSON-file implementation of :class:`MemoryStore`.

    Directory layout under *root* (default ``data/memory/``)::

        root/working/<session_id>.json
        root/long_term/<agent>/<owner>.json

    Each file is a JSON object mapping entry key -> ``{key, value, source}``.
    """

    def __init__(self, root: str = "data/memory") -> None:
        self.root = root
        os.makedirs(os.path.join(root, _WORKING_DIR), exist_ok=True)

    # -- paths -------------------------------------------------------------- #
    def _working_path(self, session_id: str) -> str:
        _assert_safe(session_id, "session id")
        return os.path.join(
            self.root, _WORKING_DIR, f"{session_id}.json"
        )

    def _long_term_path(self, agent: str, owner: str) -> str:
        _assert_safe(agent, "agent name")
        _assert_safe(owner, "owner")
        directory = os.path.join(self.root, _LONG_TERM_DIR, agent)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{owner}.json")

    # -- low-level helpers -------------------------------------------------- #
    @staticmethod
    def _read(path: str) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write(path: str, entries: dict[str, MemoryEntry]) -> None:
        payload = {k: _entry_to_dict(e) for k, e in entries.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _decode(raw: dict[str, Any] | None) -> dict[str, MemoryEntry]:
        if not raw:
            return {}
        entries: dict[str, MemoryEntry] = {}
        for key, value in raw.items():
            entries[str(key)] = _dict_to_entry(value, str(key))
        return entries

    # -- working ------------------------------------------------------------ #
    def load_working(self, session_id: str) -> dict[str, MemoryEntry]:
        return self._decode(self._read(self._working_path(session_id)))

    def save_working(
        self, session_id: str, entries: dict[str, MemoryEntry]
    ) -> None:
        self._write(self._working_path(session_id), entries)

    # -- long-term ---------------------------------------------------------- #
    def load_long_term(
        self, agent: str, owner: str
    ) -> dict[str, MemoryEntry]:
        return self._decode(self._read(self._long_term_path(agent, owner)))

    def save_long_term(
        self, agent: str, owner: str, entries: dict[str, MemoryEntry]
    ) -> None:
        self._write(self._long_term_path(agent, owner), entries)