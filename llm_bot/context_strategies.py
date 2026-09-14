"""Pluggable context-management strategies (without summary).

A ``ContextStrategy`` controls **what part of a conversation is actually sent to
the model** on each turn, and how durable memory (sticky facts, dialogue branches)
is maintained. Strategies are alternatives to the rolling-summary compressor in
:mod:`llm_bot.compress`; unlike it, none of the strategies in this module fold the
old dialog into an LLM-generated summary.

Three strategies are provided:

* :class:`SlidingWindow` — only the last ``N`` messages go to the model; older ones
  are excluded from the request (but never deleted from persistent storage).
* :class:`StickyFacts` — a key/value ``facts`` block (goal, constraints,
  preferences, decisions, agreements) is kept durable and updated after every
  turn via a small LLM call; the request carries ``facts`` + the last ``N``
  messages.
* :class:`Branching` — a conversation can fork: ``/branch <name>`` snapshots the
  current position as an automatic checkpoint and starts a new independent line,
  and you can switch between branches.

Every strategy owns its own full history (see :attr:`ContextStrategy.history`) so
that, even when the *request* is trimmed, nothing is permanently lost and a
conversation can be resumed or branched.

Lifecycle driven by :class:`~llm_bot.agent.Session` for one turn:

1. ``prepare(user_msg)`` — build the request stack from the durable history plus
   the new user message, **without mutating anything** (so an overflow check can
   abort cleanly);
2. the request is sent and the reply arrives;
3. ``on_turn_end(user_msg, reply)`` — atomically record the completed turn and
   update durable memory (facts / active branch);
4. ``save_state(store, session_id)`` — persist the full history and strategy state.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Shared value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContextEvent:
    """Diagnostics for one turn under a context strategy.

    Attributes:
        strategy: The strategy name that produced the event.
        dropped: How many messages were excluded from the request this turn.
        extra_tokens: Tokens the strategy spent on its own overhead (e.g. the
            LLM call that refreshes sticky facts), on top of the main turn.
        details: Free-form human-readable note (e.g. a branch switch).
    """

    strategy: str
    dropped: int = 0
    extra_tokens: int = 0
    details: str = ""


@dataclass
class PreparedTurn:
    """The request view produced by a strategy for one turn.

    Attributes:
        request_history: The slice of the full history to send to the model.
        prefix: Extra context messages placed *before* the agent's system prompt
            (e.g. the rendered ``facts`` block) so durable memory survives a small
            sliding window.
        dropped: How many messages were excluded from the request.
        event: Optional diagnostics event describing this turn.
    """

    request_history: list[dict[str, str]] = field(default_factory=list)
    prefix: list[dict[str, str]] = field(default_factory=list)
    dropped: int = 0
    event: ContextEvent | None = None


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #


class ContextStrategy(ABC):
    """Base class for a pluggable context-management strategy.

    A strategy owns the full history of the conversation it manages (see
    :attr:`history`) and decides what subset of it — plus any durable-memory
    prefix such as sticky facts — is sent to the model on each turn.
    """

    #: Machine name of the strategy (used in config/CLI, e.g. ``"sliding"``).
    name: str = ""

    @abstractmethod
    def prepare(self, user_msg: dict[str, str]) -> PreparedTurn:
        """Return the request view for the current history plus *user_msg*.

        Must not mutate any state, so the caller can run a context-overflow
        check and abort without side effects.
        """

    @abstractmethod
    def on_turn_end(self, user_msg: dict[str, str], reply: str) -> None:
        """Atomically record a completed turn and update durable memory."""

    @abstractmethod
    def load_state(self, store: Any, session_id: str) -> None:
        """Restore durable state (history, facts, branches) from *store*."""

    @abstractmethod
    def save_state(self, store: Any, session_id: str) -> None:
        """Persist the full history and any durable state to *store*."""

    @property
    @abstractmethod
    def history(self) -> list[dict[str, str]]:
        """Read-only full history managed by this strategy."""

    @property
    def extra_tokens(self) -> int:
        """Total tokens the strategy spent on its own overhead across all turns."""
        return 0


def _tail(history: list[dict[str, str]], window: int) -> tuple[list[dict[str, str]], int]:
    """Return ``(last ``window`` messages, dropped_count)`` from *history*."""
    if len(history) > window:
        return history[-window:], len(history) - window
    return list(history), 0


# --------------------------------------------------------------------------- #
# Strategy 1: Sliding Window
# --------------------------------------------------------------------------- #


class SlidingWindow(ContextStrategy):
    """Only the last ``window_size`` messages are sent to the model.

    Older messages are excluded from the request (reducing token spend) but the
    full history is still kept in persistent storage, so nothing is lost and the
    conversation can be resumed or branched later.
    """

    name = "sliding"

    def __init__(self, window_size: int) -> None:
        self._window = max(1, int(window_size))
        self._history: list[dict[str, str]] = []

    @property
    def window_size(self) -> int:
        """The sliding-window size (how many recent messages go to the model)."""
        return self._window

    def prepare(self, user_msg: dict[str, str]) -> PreparedTurn:
        request_history, dropped = _tail([*self._history, user_msg], self._window)
        return PreparedTurn(
            request_history=request_history,
            dropped=dropped,
            event=ContextEvent(strategy=self.name, dropped=dropped),
        )

    def on_turn_end(self, user_msg: dict[str, str], reply: str) -> None:
        self._history.append(user_msg)
        self._history.append({"role": "assistant", "content": reply})

    def load_state(self, store: Any, session_id: str) -> None:
        self._history = store.load(session_id)

    def save_state(self, store: Any, session_id: str) -> None:
        store.save(session_id, self._history)

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)


# --------------------------------------------------------------------------- #
# Strategy 2: Sticky Facts (key/value memory)
# --------------------------------------------------------------------------- #

# Instruction used to refresh the sticky-facts block. The model returns lines of
# the form "key: value" which are parsed back into a dict.
_FACTS_UPDATE_PROMPT = (
    "Ты — система ведения памятки (key-value facts). Ниже — текущая памятка и "
    "новый фрагмент переписки. Обнови памятку: добавь новые важные факты, "
    "поправь изменившиеся, удали устаревшие. Используй ключи вроде: цель, "
    "ограничения, предпочтения, решения, договорённости, срок, бюджет, стек. "
    "Отвечай ТОЛЬКО строками вида «ключ: значение», по одному факту на строку, "
    "без пояснений и без маркдауна.\n\n"
    "Текущая памятка:\n{existing}\n\n"
    "Новый фрагмент переписки:\n{transcript}"
)

# Rendered as a system message prefix so durable facts survive a small window.
_FACTS_BLOCK_HEADER = (
    "Известные факты о задаче/разговоре (durable memory, ключ: значение):\n"
)


class StickyFacts(ContextStrategy):
    """Key/value durable memory refreshed after every turn.

    The request carries the last ``window_size`` messages *plus* a rendered
    ``facts`` block as a system-message prefix, so important details survive even
    when the sliding window drops the older turns that first mentioned them.
    """

    name = "facts"

    def __init__(
        self,
        window_size: int,
        max_facts: int = 20,
        chat: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self._window = max(1, int(window_size))
        self._max_facts = max(1, int(max_facts))
        self._chat = chat
        self._history: list[dict[str, str]] = []
        self._facts: dict[str, str] = {}
        self._extra_tokens = 0

    # -- state -------------------------------------------------------------- #

    @property
    def facts(self) -> dict[str, str]:
        """Read-only snapshot of the durable key/value memory."""
        return dict(self._facts)

    @property
    def window_size(self) -> int:
        """The sliding-window size (how many recent messages go to the model)."""
        return self._window

    @property
    def max_facts(self) -> int:
        """Hard cap on the number of sticky facts kept in memory."""
        return self._max_facts

    @property
    def extra_tokens(self) -> int:
        return self._extra_tokens

    # -- interface ---------------------------------------------------------- #

    def prepare(self, user_msg: dict[str, str]) -> PreparedTurn:
        request_history, dropped = _tail([*self._history, user_msg], self._window)
        prefix: list[dict[str, str]] = []
        if self._facts:
            prefix = [
                {
                    "role": "system",
                    "content": _FACTS_BLOCK_HEADER + render_facts(self._facts),
                }
            ]
        return PreparedTurn(
            request_history=request_history,
            prefix=prefix,
            dropped=dropped,
            event=ContextEvent(strategy=self.name, dropped=dropped),
        )

    def on_turn_end(self, user_msg: dict[str, str], reply: str) -> None:
        self._history.append(user_msg)
        self._history.append({"role": "assistant", "content": reply})
        if self._chat is None:
            return
        transcript = render_transcript([user_msg, {"role": "assistant", "content": reply}])
        prompt = _FACTS_UPDATE_PROMPT.format(
            existing=render_facts(self._facts) or "(пусто)",
            transcript=transcript,
        )
        from llm_bot.tokens import count_message_tokens, count_messages_tokens

        request = [{"role": "user", "content": prompt}]
        self._extra_tokens += count_messages_tokens(request)
        output = self._chat(request)
        self._extra_tokens += count_message_tokens(
            {"role": "assistant", "content": output}
        )
        self._facts = parse_facts(output, self._max_facts)

    def load_state(self, store: Any, session_id: str) -> None:
        self._history = store.load(session_id)
        self._facts = store.load_facts(session_id)

    def save_state(self, store: Any, session_id: str) -> None:
        store.save_facts(session_id, self._facts)
        store.save(session_id, self._history)

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)


def render_facts(facts: dict[str, str]) -> str:
    """Render a facts dict as ``key: value`` lines."""
    return "\n".join(f"{k}: {v}" for k, v in facts.items())


def render_transcript(messages: list[dict[str, str]]) -> str:
    """Render a list of messages into a readable transcript for the LLM."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


_FACT_LINE_RE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(.+?)\s*$")


def parse_facts(text: str, max_facts: int) -> dict[str, str]:
    """Parse ``key: value`` lines (e.g. from a facts-refresh reply) into a dict.

    Non-matching lines are ignored; at most *max_facts* entries are kept, taking
    the *last* encountered (most recently stated) entries first so the newest
    facts win over a full stale memory.
    """
    facts: dict[str, str] = {}
    for raw in text.splitlines():
        m = _FACT_LINE_RE.match(raw)
        if not m:
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if key:
            facts[key] = value
    if len(facts) > max_facts:
        ordered = list(facts.items())
        facts = dict(ordered[-max_facts:])
    return facts


# --------------------------------------------------------------------------- #
# Strategy 3: Branching
# --------------------------------------------------------------------------- #


class Branching(ContextStrategy):
    """Fork a conversation into independent lines from a checkpoint.

    A branch is created with :meth:`branch`, which records the current position as
    an automatic checkpoint and starts a new line sharing the trunk history. Each
    branch keeps its own full history; :meth:`switch` moves the active line.
    """

    name = "branching"
    _DEFAULT_TRUNK = "main"

    def __init__(self, window_size: int | None = None) -> None:
        self._window = window_size
        self._branches: dict[str, list[dict[str, str]]] = {
            self._DEFAULT_TRUNK: []
        }
        self._current = self._DEFAULT_TRUNK

    # -- introspection ------------------------------------------------------ #

    @property
    def current_branch(self) -> str:
        return self._current

    @property
    def branches(self) -> dict[str, list[dict[str, str]]]:
        """Read-only mapping of branch name -> its full history."""
        return {k: list(v) for k, v in self._branches.items()}

    # -- operations --------------------------------------------------------- #

    def branch(self, name: str) -> None:
        """Create a new branch forking from the current position (auto checkpoint).

        The new branch inherits the entire history of the current branch up to now
        (the shared trunk) and becomes active. Creating a branch with an existing
        name raises ``ValueError``.
        """
        if not name or name in self._branches:
            raise ValueError(f"Ветка {name!r} уже существует или имя пустое.")
        self._branches[name] = list(self._branches[self._current])
        self._current = name

    def switch(self, name: str) -> None:
        """Make *name* the active branch."""
        if name not in self._branches:
            raise KeyError(name)
        self._current = name

    # -- interface ---------------------------------------------------------- #

    def prepare(self, user_msg: dict[str, str]) -> PreparedTurn:
        history = self._branches[self._current]
        if self._window is not None:
            request_history, dropped = _tail([*history, user_msg], self._window)
        else:
            request_history = [*history, user_msg]
            dropped = 0
        return PreparedTurn(
            request_history=request_history,
            dropped=dropped,
            event=ContextEvent(strategy=self.name, dropped=dropped),
        )

    def on_turn_end(self, user_msg: dict[str, str], reply: str) -> None:
        branch = self._branches[self._current]
        branch.append(user_msg)
        branch.append({"role": "assistant", "content": reply})

    def load_state(self, store: Any, session_id: str) -> None:
        self._branches, self._current = store.load_branches(
            session_id, default_branch=self._DEFAULT_TRUNK
        )

    def save_state(self, store: Any, session_id: str) -> None:
        store.save_branches(session_id, self._branches, self._current)

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._branches[self._current])