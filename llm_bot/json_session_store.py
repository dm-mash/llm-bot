"""JSON-file implementation of :class:`SessionStore`.

Each session is stored as ``<session_id>.json`` inside a single directory
(default ``data/sessions/``). The file holds a conversation's *live* history — a
list of ``{"role", "content"}`` messages — and, when context compression is in
use, a separately-persisted ``summary`` string that replaces the older part of
the dialog that has been folded away (see :mod:`llm_bot.compress`).

Backward compatibility: sessions written before compression was added are plain
JSON *lists* of messages. Such files are still read correctly (``summary`` is
``""``), and the next save upgrades them to the new compound object.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Session ids are kept filesystem-safe: alphanumerics, dash, underscore, dot.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class JsonSessionStore:
    """Persists session histories as JSON files, one per session.

    A session file is either the legacy flat form (a list of messages) or the
    newer compound form ``{"summary": "...", "history": [...]}``. ``load`` /
    ``load_summary`` transparently handle both, so old conversations resume
    unchanged.
    """

    def __init__(self, directory: str = "data/sessions") -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_id: str) -> str:
        if not session_id or not _SAFE_ID.fullmatch(session_id):
            raise ValueError(
                f"Invalid session id {session_id!r}. Use only letters, digits, "
                "dash, underscore or dot."
            )
        return os.path.join(self.directory, f"{session_id}.json")

    @staticmethod
    def _split(data: Any) -> tuple[list[dict[str, str]], str]:
        """Normalise a loaded session file into ``(history, summary)``.

        Accepts both the legacy flat list and the compound
        ``{"history": [...], "summary": "..."}`` form.
        """
        if isinstance(data, dict):
            raw_history = data.get("history", [])
            summary = data.get("summary", "")
            if not isinstance(summary, str):
                summary = ""
        else:
            raw_history = data
            summary = ""
        if not isinstance(raw_history, list):
            raw_history = []
        history = [m for m in raw_history if isinstance(m, dict)]
        return history, summary

    def load(self, session_id: str) -> list[dict[str, str]]:
        """Return the live message history (without the compressed summary)."""
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        history, _summary = self._split(data)
        return history

    def load_summary(self, session_id: str) -> str:
        """Return the persisted compression summary for a session (``""`` if none)."""
        path = self._path(session_id)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _history, summary = self._split(data)
        return summary

    def save(self, session_id: str, history: list[dict[str, str]]) -> None:
        """Persist *history*, preserving any previously stored summary.

        Existing callers that only know about the live history keep working:
        the current summary (if any) is carried over from disk.
        """
        current = self.load_summary(session_id)
        self.save_full(session_id, history, summary=current)

    def save_full(
        self,
        session_id: str,
        history: list[dict[str, str]],
        *,
        summary: str,
    ) -> None:
        """Persist both the live *history* and the compressed *summary* together."""
        path = self._path(session_id)
        payload: dict[str, Any] = {
            "summary": summary,
            "history": history,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def list(self) -> list[str]:
        names = sorted(
            f[: -len(".json")]
            for f in os.listdir(self.directory)
            if f.endswith(".json")
        )
        return names