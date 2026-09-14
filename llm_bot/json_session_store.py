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

    @staticmethod
    def _read_payload(session_id: str, path: str) -> Any | None:
        """Read and decode a session file, returning ``None`` when missing."""
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, session_id: str) -> list[dict[str, str]]:
        """Return the live message history (without the compressed summary)."""
        path = self._path(session_id)
        data = self._read_payload(session_id, path)
        if data is None:
            return []
        history, _summary = self._split(data)
        return history

    def load_summary(self, session_id: str) -> str:
        """Return the persisted compression summary for a session (``""`` if none)."""
        path = self._path(session_id)
        data = self._read_payload(session_id, path)
        if data is None:
            return ""
        _history, summary = self._split(data)
        return summary

    def load_facts(self, session_id: str) -> dict[str, str]:
        """Return the persisted sticky-facts block (``{}`` when there is none)."""
        path = self._path(session_id)
        data = self._read_payload(session_id, path)
        if not isinstance(data, dict):
            return {}
        raw = data.get("facts")
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def load_branches(
        self, session_id: str, *, default_branch: str
    ) -> tuple[dict[str, list[dict[str, str]]], str]:
        """Return ``(branches, current)`` from a session file.

        When no branching state is stored, falls back to a single
        ``default_branch`` holding the plain live history (legacy sessions).
        """
        path = self._path(session_id)
        data = self._read_payload(session_id, path)
        if not isinstance(data, dict):
            return {default_branch: self.load(session_id)}, default_branch
        raw_branches = data.get("branches")
        if isinstance(raw_branches, dict) and raw_branches:
            branches: dict[str, list[dict[str, str]]] = {}
            for name, hist in raw_branches.items():
                if isinstance(hist, list):
                    branches[str(name)] = [
                        m for m in hist if isinstance(m, dict)
                    ]
            current = data.get("current_branch", default_branch)
            if current not in branches:
                current = next(iter(branches), default_branch)
            return branches, str(current)
        # No branching state: treat the flat history as the single default branch.
        history, _summary = self._split(data)
        return {default_branch: history}, default_branch

    def save(self, session_id: str, history: list[dict[str, str]]) -> None:
        """Persist *history*, preserving any previously stored summary/facts.

        Existing callers that only know about the live history keep working:
        the current summary (if any) is carried over from disk.
        """
        current = self.load_summary(session_id)
        facts = self.load_facts(session_id)
        self._save_compound(
            session_id,
            history,
            summary=current,
            facts=facts,
            branches=None,
            current_branch=None,
        )

    def save_full(
        self,
        session_id: str,
        history: list[dict[str, str]],
        *,
        summary: str,
    ) -> None:
        """Persist both the live *history* and the compressed *summary* together."""
        self._save_compound(
            session_id,
            history,
            summary=summary,
            facts=self.load_facts(session_id),
            branches=None,
            current_branch=None,
        )

    def save_facts(self, session_id: str, facts: dict[str, str]) -> None:
        """Persist the sticky-facts block alongside the existing history."""
        data = self._read_payload(session_id, self._path(session_id))
        history, summary = self._split(data)
        self._save_compound(
            session_id,
            history,
            summary=summary,
            facts=facts,
            branches=self._load_branches_from(data),
            current_branch=self._load_current_from(data),
        )

    def save_branches(
        self,
        session_id: str,
        branches: dict[str, list[dict[str, str]]],
        current: str,
    ) -> None:
        """Persist the branching state alongside the existing history/summary."""
        data = self._read_payload(session_id, self._path(session_id))
        history, summary = self._split(data)
        self._save_compound(
            session_id,
            history,
            summary=summary,
            facts=self.load_facts(session_id),
            branches=branches,
            current_branch=current,
        )

    @staticmethod
    def _load_branches_from(data: Any) -> dict[str, list[dict[str, str]]] | None:
        if not isinstance(data, dict):
            return None
        raw = data.get("branches")
        if not isinstance(raw, dict) or not raw:
            return None
        return {
            str(name): [m for m in hist if isinstance(m, dict)]
            for name, hist in raw.items()
            if isinstance(hist, list)
        }

    @staticmethod
    def _load_current_from(data: Any) -> str | None:
        if isinstance(data, dict) and isinstance(data.get("current_branch"), str):
            return data["current_branch"]
        return None

    def _save_compound(
        self,
        session_id: str,
        history: list[dict[str, str]],
        *,
        summary: str,
        facts: dict[str, str] | None,
        branches: dict[str, list[dict[str, str]]] | None,
        current_branch: str | None,
    ) -> None:
        """Write the full compound session payload, preserving all state."""
        path = self._path(session_id)
        payload: dict[str, Any] = {
            "summary": summary,
            "history": history,
        }
        if facts:
            payload["facts"] = facts
        if branches:
            payload["branches"] = branches
        if current_branch:
            payload["current_branch"] = current_branch
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def list(self) -> list[str]:
        names = sorted(
            f[: -len(".json")]
            for f in os.listdir(self.directory)
            if f.endswith(".json")
        )
        return names