"""JSON-file implementation of :class:`SessionStore`.

Each session is stored as ``<session_id>.json`` inside a single directory
(default ``data/sessions/``). The history is a list of ``{"role", "content"}``
messages, so a conversation survives process restarts and can be resumed with the
same ``session_id``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Session ids are kept filesystem-safe: alphanumerics, dash, underscore, dot.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class JsonSessionStore:
    """Persists session histories as JSON files, one per session."""

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

    def load(self, session_id: str) -> list[dict[str, str]]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [msg for msg in data if isinstance(msg, dict)]

    def save(self, session_id: str, history: list[dict[str, str]]) -> None:
        path = self._path(session_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)

    def list(self) -> list[str]:
        names = sorted(
            f[: -len(".json")]
            for f in os.listdir(self.directory)
            if f.endswith(".json")
        )
        return names