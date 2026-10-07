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
# The wording matters more than it looks. With a bare header the memory block
# reads as one more context, and the model duly quotes from it: in a 14-turn run
# every answer came back quoting the customer's own stored words in «…», which
# the quote audit then had to reject one turn at a time. Memory is what the user
# said, not a document, so both blocks say so up front and tell the model not to
# source anything from them.
_NOT_A_SOURCE = (
    "Это не источник. Это слова самого пользователя из прошлых ходов, "
    "не документы: не цитируй их кавычками и не указывай номера источников "
    "на них. Опирайся на них как на контекст разговора."
)
_LONG_HEADER = "Долговременная память (профиль, решения, знания). " + _NOT_A_SOURCE
_WORKING_HEADER = "Рабочая память (данные текущей задачи). " + _NOT_A_SOURCE


@dataclass(frozen=True)
class MemoryEntry:
    """A single remembered fact.

    Attributes:
        key: Unique key of the fact within its layer.
        value: The fact's content.
        source: Where the fact came from (``"user"``, ``"assistant"``,
            ``"system"``, ``"derive"``).
        evidence: The user's own words the fact was taken from. Set by
            automatic extraction, where it is what the fact was allowed to
            exist on the strength of.
        turn: Ordinal number of the dialogue turn the fact came from, so a
            stored fact can be pointed back at the place it was said.
    """

    key: str
    value: str
    source: str = "assistant"
    evidence: str = ""
    turn: int = 0


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
    def remember(
        self,
        key: str,
        value: str,
        source: str = "assistant",
        evidence: str = "",
        turn: int = 0,
    ) -> None:
        """Write *value* under *key*, then persist the whole layer."""
        self._entries[key] = MemoryEntry(
            key=key, value=value, source=source, evidence=evidence, turn=turn
        )
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
    def remember_working(
        self,
        key: str,
        value: str,
        source: str = "assistant",
        evidence: str = "",
        turn: int = 0,
    ) -> None:
        """Explicitly store a fact in the working (current-task) layer."""
        self.working.remember(key, value, source=source, evidence=evidence, turn=turn)
        logger.debug("[memory] working[%s] = %r", key, value)

    def remember_long_term(
        self,
        key: str,
        value: str,
        source: str = "assistant",
        evidence: str = "",
        turn: int = 0,
    ) -> None:
        """Explicitly store a fact in the long-term (profile/decisions) layer."""
        self.long.remember(key, value, source=source, evidence=evidence, turn=turn)
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
    #: Facts the classifier proposed that did not survive verification, with the
    #: reason. This is the number that says whether automatic extraction can be
    #: trusted at all: a classifier that invents facts shows up here rather than
    #: in a report about the bot being wrong much later.
    rejected: list[tuple[str, str]] = field(default_factory=list)
    request_tokens: int = 0
    reply_tokens: int = 0
    recognized: bool = True

    @property
    def total_tokens(self) -> int:
        """Total tokens spent on this extraction call."""
        return self.request_tokens + self.reply_tokens

    @property
    def verified_rate(self) -> float:
        """Share of proposed facts that reached memory (1.0 when none proposed)."""
        proposed = len(self.working_written) + len(self.long_term_written) + len(
            self.rejected
        )
        if not proposed:
            return 1.0
        return (len(self.working_written) + len(self.long_term_written)) / proposed


_EXTRACT_PROMPT = (
    "You are an agent memory classifier. You are given a fragment of a dialogue. "
    "Split the extracted facts into TWO layers:\n"
    "- working: data of the CURRENT task (goal, constraints, steps, deadlines, "
    "in-progress decisions). It only survives within this session.\n"
    "- long_term: user profile, their preferences, and long-lived "
    "decisions/agreements they stated. It carries across sessions.\n\n"
    "Return ONLY JSON, no explanations, no markdown. Every fact is an object with "
    "the fact itself and the words it came from:\n"
    '{{"working": {{"key": {{"value": "...", "evidence": "..."}}}}, '
    '"long_term": {{"key": {{"value": "...", "evidence": "..."}}}}}}\n'
    '"evidence" MUST be a verbatim substring of what the USER wrote in this '
    "dialogue — copy the words exactly, do not translate, reword or fix them. An "
    "empty layer is an empty object. Keys are short, meaningful, in Russian or "
    "English.\n\n"
    "Rules that override the layer descriptions above:\n"
    "- Store ONLY what the USER asserted. Never turn the assistant's own answers, "
    "conclusions or summaries into facts, however certain they sound. An assistant "
    "claim is not evidence.\n"
    "- Never store the contents of source documents as knowledge. Retrieved "
    "documents are re-read on demand, so copying their text into memory only "
    "creates stale claims that later contradict the sources.\n"
    "- If the user merely asked something and the assistant answered, store nothing "
    "from that exchange unless the user asserted a preference, a constraint or a "
    "fact about themselves.\n"
    "- Prefer an empty layer over a fact you cannot attribute to the user.\n\n"
    "A fact whose evidence is not in the user's words will be dropped, so "
    '"evidence" is not a formality: it is the only thing that keeps a fact.\n\n'
    "Dialogue:\n{transcript}"
)


@dataclass(frozen=True)
class MemoryAudit:
    """What survived verification on one extraction, and what did not.

    The kept list is what was written; the rejected list carries the reason
    because "the fact was dropped" is useless for improving anything, while
    "dropped: no evidence in the user's words" says which failure to look for.
    """

    kept: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def clean(self) -> bool:
        return not self.rejected

    def __len__(self) -> int:
        return len(self.kept) + len(self.rejected)


def _normalise(text: str) -> str:
    """Collapse whitespace and case so a copied quote can be located.

    Nothing else is loosened: punctuation stays, because stripping it would let
    "budget is 400" match a user who wrote "budget is 4000", and a memory that
    survives on a technicality is worse than one that was dropped.
    """
    return " ".join(text.split()).casefold()


def _verified(value: object, evidence: object, spoken: str) -> bool:
    """True when *evidence* is a verbatim substring of what the user said.

    The classifier is the one doing the copying, so this only checks that it did
    not invent the quote — which is the whole failure mode: a fact the user never
    stated, written into a layer that is injected into every later turn.
    """
    if not isinstance(value, str) or not isinstance(evidence, str):
        return False
    needle = _normalise(evidence)
    return bool(needle) and needle in spoken


def extract_memory(
    memory: MemoryLayers,
    user_msg: dict[str, str],
    reply: str,
    chat: Callable[[list[dict[str, str]]], str],
    turn: int = 0,
) -> MemoryEvent:
    """Classify a completed turn into working / long-term facts and store them.

    Makes one small LLM call via *chat* to decide **explicitly** which facts
    belong in the working layer vs the long-term layer, then writes them through
    :meth:`MemoryLayers.remember_working` / :meth:`MemoryLayers.remember_long_term`
    (so the destination is always explicit). Existing entries with the same key
    are overwritten.

    *turn* is recorded on every written fact so a stored value can be traced back
    to the turn it was taken from.

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

    payload: dict[str, dict[str, Any]] = {}
    recognized = True
    try:
        parsed = _parse_json_object(output)
        raw_working = parsed.get("working")
        raw_long = parsed.get("long_term")
        if isinstance(raw_working, dict):
            payload["working"] = raw_working
        if isinstance(raw_long, dict):
            payload["long_term"] = raw_long
    except Exception as exc:  # noqa: BLE001 - never break a turn on extraction
        recognized = False
        # Not the whole reply. It runs to hundreds of characters of JSON, and the
        # interesting part is why it did not parse — one real reply was missing a
        # closing brace, and the log buried that under six invented facts about
        # the customer. The full text goes to the debug log instead.
        logger.warning(
            "[memory] ответ классификатора не разобран (%s): %s…",
            type(exc).__name__,
            output[:120],
        )
        logger.debug("[memory] полный ответ классификатора: %r", output)

    # Only the user's own words may become durable. A fact the user never stated
    # used to be written on the classifier's word alone and then injected into
    # every later turn: a turn of small talk could leave «аллергия на орехи» in
    # the profile forever, and nothing downstream could tell it from something
    # the customer actually said.
    spoken = _normalise(user_msg.get("content", ""))
    working_written: list[str] = []
    long_term_written: list[str] = []
    rejected: list[tuple[str, str]] = []
    for layer, target, written in (
        ("working", memory.remember_working, working_written),
        ("long_term", memory.remember_long_term, long_term_written),
    ):
        for key, raw in payload.get(layer, {}).items():
            name = str(key)
            label = f"{layer}:{name}"
            if not isinstance(raw, dict):
                rejected.append((label, "нет доказательства"))
                continue
            value, evidence = raw.get("value"), raw.get("evidence")
            if not _verified(value, evidence, spoken):
                rejected.append((label, "доказательства нет в словах пользователя"))
                continue
            target(name, str(value), source="derive", evidence=str(evidence), turn=turn)
            written.append(name)

    # Not a warning per fact. A single turn can produce half a dozen, and they
    # printed over the conversation while being the expected outcome: the
    # classifier reaching for facts the user never said is what the check is for.
    # The count is on the normal memory line; the detail belongs to -v.
    for label, reason in rejected:
        logger.debug("[memory] факт отброшен (%s): %s", label, reason)
    if rejected:
        logger.info(
            "[memory] отброшено фактов без доказательства: %d", len(rejected)
        )

    event = MemoryEvent(
        working_written=working_written,
        long_term_written=long_term_written,
        rejected=rejected,
        request_tokens=request_tokens,
        reply_tokens=reply_tokens,
        recognized=recognized,
    )
    logger.debug(
        "[memory] extract -> working=%d long=%d отброшено=%d, %d tokens, "
        "recognized=%s",
        len(working_written),
        len(long_term_written),
        len(rejected),
        event.total_tokens,
        recognized,
    )
    return event


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first ``{...}`` JSON object found in *text* (tolerant)."""
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])