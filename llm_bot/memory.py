"""Explicit layered memory model for an agent.

The agent's memory is split into three independent layers, each with its own
lifetime, storage backend and scope:

* :class:`ShortTermMemory` — the *current dialog*. Held in RAM for the lifetime
  of a session only; it is the live ``history`` stack that is sent to the model
  on each turn and persisted (as the session history) by the session itself.

* :class:`WorkingMemory` — *data of the current task*. Persisted per-session so
  it survives a process restart of the same ``session_id``, but never shared
  with other sessions. Holds goal, constraints, steps, decisions of the task at
  hand.

* :class:`LongTermMemory` — *profile / decisions / knowledge*. Persisted
  per-agent and isolated per-**owner** (a user/namespace) for privacy, so one
  user's profile and choices are never visible to another user's session. This
  is the only layer that carries information across different sessions of the
  same agent.

A single :class:`MemoryLayers` facade ties the three layers together and lets a
caller *explicitly* choose which layer a value is written to / read from (see
:meth:`MemoryLayers.write_to` / :meth:`MemoryLayers.read_from`) instead of
relying on implicit placement. It also renders each non-empty layer as a marked
prompt block (see :meth:`MemoryLayers.prefix_messages`) that is injected ahead
of the agent's system prompt, so the model sees the separation of durable memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

logger = logging.getLogger(__name__)

Layer = Literal["short", "working", "long"]

# Section markers rendered into the prompt so the model can distinguish layers.
_LONG_HEADER = "Долговременная память (профиль, решения, знания):"
_WORKING_HEADER = "Рабочая память (данные текущей задачи):"


@dataclass(frozen=True)
class MemoryEntry:
    """A single remembered fact.

    Attributes:
        key: Unique key of the fact within its layer.
        value: The fact's content.
        source: Where the fact came from (``"user"``, ``"assistant"``,
            ``"system"``, ``"derive"``).
    """

    key: str
    value: str
    source: str = "assistant"


# --------------------------------------------------------------------------- #
# Layer 1: short-term (in-memory, per-session)
# --------------------------------------------------------------------------- #


class ShortTermMemory:
    """The current dialog, held in RAM for the lifetime of the session.

    Each turn pushes a user message and the assistant reply. The full stack is
    exposed via :meth:`messages`, and :meth:`recent` returns the last ``n``
    messages for a sliding-window view. Nothing here is persisted by this class
    itself — persisting the live dialog is the session's responsibility.
    """

    def __init__(self, messages: list[dict[str, str]] | None = None) -> None:
        self._messages = [dict(m) for m in messages] if messages else []

    def push(self, message: dict[str, str]) -> None:
        """Append a ``{"role", "content"}`` message to the current dialog."""
        self._messages.append({"role": message.get("role", "user"),
                               "content": message.get("content", "")})

    def messages(self) -> list[dict[str, str]]:
        """Read-only copy of the full current dialog."""
        return [dict(m) for m in self._messages]

    def recent(self, n: int) -> list[dict[str, str]]:
        """Return the last ``n`` messages (or fewer if the dialog is shorter)."""
        return [dict(m) for m in self._messages[-max(0, int(n)):]]

    def clear(self) -> None:
        """Discard the in-memory dialog."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# --------------------------------------------------------------------------- #
# Layers 2 & 3: persistent key/value memory
# --------------------------------------------------------------------------- #


class _PersistentKeyValueMemory:
    """Shared base for working and long-term key/value memory.

    Subclasses provide persistence via :meth:`_load` / :meth:`_save` backed by
    a store. Writes go through :meth:`remember`, reads through :meth:`recall`.
    """

    def __init__(self, initial: dict[str, MemoryEntry] | None = None) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        if initial:
            for key, entry in initial.items():
                self._entries[key] = self._coerce(entry, key)

    @staticmethod
    def _coerce(entry: MemoryEntry | tuple | dict, key: str) -> MemoryEntry:
        if isinstance(entry, MemoryEntry):
            return entry
        if isinstance(entry, tuple):
            value, source = entry[0], entry[1] if len(entry) > 1 else "assistant"
            return MemoryEntry(key=key, value=str(value), source=str(source))
        if isinstance(entry, dict):
            return MemoryEntry(
                key=str(entry.get("key", key)),
                value=str(entry.get("value", "")),
                source=str(entry.get("source", "assistant")),
            )
        return MemoryEntry(key=key, value=str(entry))

    # -- persistence hooks (implemented by concrete layers) ----------------- #
    def _load(self) -> dict[str, MemoryEntry]:
        raise NotImplementedError

    def _save(self, entries: dict[str, MemoryEntry]) -> None:
        raise NotImplementedError

    # -- public API --------------------------------------------------------- #
    def remember(self, key: str, value: str, source: str = "assistant") -> None:
        """Write *value* under *key*, then persist the whole layer."""
        self._entries[key] = MemoryEntry(key=key, value=value, source=source)
        self._save(self._entries)

    def recall(self, key: str) -> str | None:
        """Return the value stored under *key*, or ``None`` when absent."""
        entry = self._entries.get(key)
        return entry.value if entry is not None else None

    def forget(self, key: str) -> None:
        """Remove *key* from this layer, then persist."""
        self._entries.pop(key, None)
        self._save(self._entries)

    def snapshot(self) -> dict[str, MemoryEntry]:
        """Read-only mapping of the current layer contents."""
        return {k: e for k, e in self._entries.items()}

    def render(self) -> str:
        """Render the layer as ``key: value`` lines (newest first)."""
        ordered = list(self._entries.items())
        ordered.reverse()
        return "\n".join(f"{k}: {e.value}" for k, e in ordered)

    def __len__(self) -> int:
        return len(self._entries)


class WorkingMemory(_PersistentKeyValueMemory):
    """Per-session memory of the current task.

    Persisted alongside the session (see :mod:`llm_bot.memory_store`) so it
    survives restarts of the same ``session_id``, but never leaks into other
    sessions.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        session_id: str = "",
        entries: dict[str, MemoryEntry] | None = None,
    ) -> None:
        super().__init__(entries)
        self._store = store
        self._session_id = session_id
        if self._store is not None and self._session_id:
            for key, entry in self._store.load_working(session_id).items():
                self._entries[key] = self._coerce(entry, key)

    def _load(self) -> dict[str, MemoryEntry]:
        if self._store is None or not self._session_id:
            return {}
        return self._store.load_working(self._session_id)

    def _save(self, entries: dict[str, MemoryEntry]) -> None:
        if self._store is None or not self._session_id:
            return
        self._store.save_working(self._session_id, entries)


class LongTermMemory(_PersistentKeyValueMemory):
    """Per-agent, per-owner durable memory: profile, decisions, knowledge.

    Isolated by an *owner* (a user/namespace) so one user's personal data is
    never served to another user's session. Owned by the agent and shared across
    all sessions of that agent *for the same owner*.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        agent: str = "",
        owner: str = "default",
        entries: dict[str, MemoryEntry] | None = None,
    ) -> None:
        super().__init__(entries)
        self._store = store
        self._agent = agent
        self._owner = owner
        if self._store is not None and self._agent:
            for key, entry in self._store.load_long_term(agent, owner).items():
                self._entries[key] = self._coerce(entry, key)

    @property
    def owner(self) -> str:
        """The namespace this memory is isolated to."""
        return self._owner

    def _load(self) -> dict[str, MemoryEntry]:
        if self._store is None or not self._agent:
            return {}
        return self._store.load_long_term(self._agent, self._owner)

    def _save(self, entries: dict[str, MemoryEntry]) -> None:
        if self._store is None or not self._agent:
            return
        self._store.save_long_term(self._agent, self._owner, entries)


# --------------------------------------------------------------------------- #
# The explicit-choice facade
# --------------------------------------------------------------------------- #


class MemoryLayers:
    """Facade over the three layers with explicit read/write targeting.

    Callers *choose* the destination layer explicitly via :meth:`write_to` /
    :meth:`read_from`, or through the named helpers (:meth:`remember_working`,
    :meth:`remember_long_term`). :meth:`prefix_messages` renders the durable
    layers (long + working) as marked system-message blocks to inject ahead of
    the system prompt.
    """

    def __init__(
        self,
        *,
        short: ShortTermMemory,
        working: WorkingMemory,
        long: LongTermMemory,
    ) -> None:
        self.short = short
        self.working = working
        self.long = long

    # -- explicit layer targeting ------------------------------------------ #
    def write_to(
        self, layer: Layer, key: str, value: str, source: str = "assistant"
    ) -> None:
        """Explicitly write *value* under *key* into the chosen *layer*."""
        if layer == "short":
            raise ValueError(
                "Слой 'short' хранит сообщения диалога; используйте short.push()."
            )
        target = self.working if layer == "working" else self.long
        target.remember(key, value, source=source)

    def read_from(self, layer: Layer, key: str) -> str | None:
        """Explicitly read *key* from the chosen *layer*."""
        if layer == "short":
            return None
        target = self.working if layer == "working" else self.long
        return target.recall(key)

    # -- named helpers (still explicit about the destination) --------------- #
    def remember_working(self, key: str, value: str, source: str = "assistant") -> None:
        """Explicitly store a fact in the working (current-task) layer."""
        self.working.remember(key, value, source=source)
        logger.debug("[memory] working[%s] = %r", key, value)

    def remember_long_term(
        self, key: str, value: str, source: str = "assistant"
    ) -> None:
        """Explicitly store a fact in the long-term (profile/decisions) layer."""
        self.long.remember(key, value, source=source)
        logger.debug("[memory] long_term[%s] = %r", key, value)

    def forget_working(self, key: str) -> None:
        self.working.forget(key)
        logger.debug("[memory] working forget %s", key)

    def forget_long_term(self, key: str) -> None:
        self.long.forget(key)
        logger.debug("[memory] long_term forget %s", key)

    # -- rendering ---------------------------------------------------------- #
    def prefix_messages(self) -> list[dict[str, str]]:
        """Render durable memory as marked system-message blocks.

        Order is long-term first, then working, so profile/knowledge precedes
        task data. Empty layers are omitted. These blocks are injected *before*
        the agent's system prompt.
        """
        prefix: list[dict[str, str]] = []
        if len(self.long) > 0:
            prefix.append({
                "role": "system",
                "content": _LONG_HEADER + "\n" + self.long.render(),
            })
        if len(self.working) > 0:
            prefix.append({
                "role": "system",
                "content": _WORKING_HEADER + "\n" + self.working.render(),
            })
        return prefix

    def summary(self) -> str:
        """Human-readable diagnostic summary of all three layers."""
        parts = [
            f"short={len(self.short)}",
            f"working={len(self.working)}",
            f"long={len(self.long)}",
        ]
        return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Explicit per-turn memory extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MemoryEvent:
    """Diagnostics for one automatic memory-extraction call.

    Attributes:
        working_written: Keys written to the working layer this turn.
        long_term_written: Keys written to the long-term layer this turn.
        request_tokens: Tokens spent on the classifier request.
        reply_tokens: Tokens spent on the classifier reply.
        recognized: Whether the classifier reply was valid JSON and parsed.
    """

    working_written: list[str] = field(default_factory=list)
    long_term_written: list[str] = field(default_factory=list)
    request_tokens: int = 0
    reply_tokens: int = 0
    recognized: bool = True

    @property
    def total_tokens(self) -> int:
        """Total tokens spent on this extraction call."""
        return self.request_tokens + self.reply_tokens


_EXTRACT_PROMPT = (
    "You are an agent memory classifier. You are given a fragment of a dialogue. "
    "Split the extracted facts into TWO layers:\n"
    "- working: data of the CURRENT task (goal, constraints, steps, deadlines, "
    "in-progress decisions). It only survives within this session.\n"
    "- long_term: user profile, their preferences, general knowledge, and "
    "long-lived decisions/agreements. It carries across sessions.\n\n"
    "Return ONLY JSON, no explanations, no markdown:\n"
    '{{"working": {{"key": "value", ...}}, "long_term": {{"key": "value", ...}}}}\n'
    "An empty layer is an empty object. Keys are short, meaningful, in Russian or "
    "English.\n\nDialogue:\n{transcript}"
)


def extract_memory(
    memory: MemoryLayers,
    user_msg: dict[str, str],
    reply: str,
    chat: Callable[[list[dict[str, str]]], str],
) -> MemoryEvent:
    """Classify a completed turn into working / long-term facts and store them.

    Makes one small LLM call via *chat* to decide **explicitly** which facts
    belong in the working layer vs the long-term layer, then writes them through
    :meth:`MemoryLayers.remember_working` / :meth:`MemoryLayers.remember_long_term`
    (so the destination is always explicit). Existing entries with the same key
    are overwritten.

    Returns a :class:`MemoryEvent` describing what was written and the tokens the
    classifier call cost (for diagnostics / token accounting).
    """
    from llm_bot.tokens import count_message_tokens, count_messages_tokens

    transcript_lines = [
        f"Пользователь: {user_msg.get('content', '')}",
        f"Ассистент: {reply}",
    ]
    request = [{
        "role": "user",
        "content": _EXTRACT_PROMPT.format(transcript="\n".join(transcript_lines)),
    }]
    request_tokens = count_messages_tokens(request)
    output = chat(request)
    reply_tokens = count_message_tokens({"role": "assistant", "content": output})

    payload: dict[str, dict[str, str]] = {}
    recognized = True
    try:
        parsed = _parse_json_object(output)
        raw_working = parsed.get("working")
        raw_long = parsed.get("long_term")
        if isinstance(raw_working, dict):
            payload["working"] = _stringify(raw_working)
        if isinstance(raw_long, dict):
            payload["long_term"] = _stringify(raw_long)
    except Exception:  # noqa: BLE001 - never break a turn on extraction failure
        recognized = False
        logger.warning("[memory] не удалось распознать ответ классификатора: %r",
                       output)

    working_written: list[str] = []
    long_term_written: list[str] = []
    for key, value in payload.get("working", {}).items():
        memory.remember_working(key, value, source="derive")
        working_written.append(key)
    for key, value in payload.get("long_term", {}).items():
        memory.remember_long_term(key, value, source="derive")
        long_term_written.append(key)

    event = MemoryEvent(
        working_written=working_written,
        long_term_written=long_term_written,
        request_tokens=request_tokens,
        reply_tokens=reply_tokens,
        recognized=recognized,
    )
    logger.debug(
        "[memory] extract -> working=%d long=%d, %d tokens, recognized=%s",
        len(working_written),
        len(long_term_written),
        event.total_tokens,
        recognized,
    )
    return event


def _stringify(mapping: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in mapping.items()}


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first ``{...}`` JSON object found in *text* (tolerant)."""
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])