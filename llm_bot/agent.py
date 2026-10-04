"""The Agent and Session entities.

``Agent`` encapsulates everything about *what* the model says: its role (system
prompt) and its generation settings. It does not hold any conversation state.

``Session`` is a concrete conversation with an agent. It owns the message
history, appends each user turn and the agent's reply, sends the full stack to
the LLM via the client, and persists the history through a
:class:`~llm_bot.stores.SessionStore`. To the outside world it exposes only the
reply text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from llm_bot.client import ContextOverflowError, ContextTooLargeError, LLMClient
from llm_bot.compress import (
    CompressionEvent,
    CompressionSettings,
    ContextCompressor,
    summarize_prompt,
)
from llm_bot.context_strategies import ContextStrategy
from llm_bot.invariants import (
    Invariant,
    InvariantAuditEvent,
    InvariantRegistry,
    InvariantViolationError,
    audit_reply,
)
from llm_bot.mcp_tools import DEFAULT_MAX_ROUNDS, MCPEvent, MCPRouter
from llm_bot.memory import MemoryEvent, MemoryLayers, extract_memory
from llm_bot.rag import (
    CitationAudit,
    RagEvent,
    Retriever,
    audit_citations,
    sources_from_text,
    strip_citations,
)
from llm_bot.stores import AgentConfig, SessionStore
from llm_bot.task_state import (
    TaskDetectionEvent,
    TaskState,
    TaskStateMachine,
    detect_task_turn,
)
from llm_bot.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    ChatResult,
    TokenUsage,
    count_message_tokens,
    count_messages_tokens,
    merge_provider_usage,
)

logger = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    """Replace lone surrogates so the text is safely UTF-8 encodable.

    Text read from an interactive terminal or piped stdin can contain lone
    surrogate code points (U+DC80-U+DCFF). This happens because CPython decodes
    stdin bytes with the ``surrogateescape`` error handler: a byte that is not
    valid UTF-8 (e.g. a UTF-8 continuation byte left over after a Backspace split
    a multi-byte character) is mapped to a surrogate. Those surrogates are valid
    in memory but cannot be encoded back to UTF-8, which crashes httpx's JSON
    serialization with ``UnicodeEncodeError``. Here we map each lone surrogate to
    the Unicode replacement character ``U+FFFD`` so the message can be sent.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_messages(messages: list[dict[str, str]]) -> None:
    """Mutate *messages* in place, replacing lone surrogates in every content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and any(
            0xD800 <= ord(ch) <= 0xDFFF for ch in content
        ):
            msg["content"] = _sanitize_text(content)


# The service feedback the tool loop appends after every REAL call has the
# exact form «Результат инструмента <server>__<tool>:\n...». A live model
# (gpt-oss-120b, 2026-09-28) once MIMICKED that format in a plain reply —
# it fabricated aggregates for currency_yahoo__analyze_rates without calling
# it and put the real directive into an inaccessible reasoning field. The
# loop saw no directive, broke, and the fabricated "result" went to the
# user as the final answer. This pattern recognizes the impersonation so
# the loop can push back instead of breaking.
_FABRICATED_FEEDBACK_RE = re.compile(
    r"^\s*Результат инструмента\s+`?\*{0,2}"
    r"([A-Za-z0-9_]+__[A-Za-z0-9_]+)\*{0,2}`?\s*:"
)

_FABRICATION_CORRECTION = (
    "Эту строку сгенерировал ты сам: формат «Результат инструмента …» —"
    " служебный, такие сообщения пишет только система после РЕАЛЬНОГО"
    " вызова инструмента через директиву call_tool. Выдавать выдуманные"
    " результаты за вызов нельзя. Верни директиву call_tool для следующего"
    " невыполненного шага цепочки (или, если все шаги уже выполнены —"
    " финальный ответ пользователю обычным текстом)."
)


def _looks_like_fabricated_feedback(reply: str) -> bool:
    """True when the reply impersonates the system's tool-result format."""
    return _FABRICATED_FEEDBACK_RE.match(reply) is not None


def summary_budget_chars(
    settings: CompressionSettings,
    context_window: int | None,
) -> int | None:
    """Return the maximum summary length in chars, or ``None`` for no cap.

    The budget is the smaller of the explicit ``max_summary_tokens`` cap and the
    model's context window times ``max_summary_ratio`` (a fraction of the window
    kept for the summary so it never crowds out the system prompt and live
    history). When neither is known, returns ``None`` (no cap).
    """
    budget_tokens = settings.max_summary_tokens
    if context_window is not None:
        by_window = int(context_window * settings.max_summary_ratio)
        budget_tokens = (
            by_window
            if budget_tokens is None
            else min(budget_tokens, by_window)
        )
    if budget_tokens is None or budget_tokens <= 0:
        return None
    # The local token estimator approximates ~4 chars per token.
    return budget_tokens * 4


class Agent:
    """An immutable role bound to a specific model/client.

    An agent holds the behaviour settings from its :class:`AgentConfig` and the
    transport client it talks through. It builds the ``messages`` stack (system
    prompt + history) but does not store any history itself — a :class:`Session`
    is created from an agent for each conversation.

    Attributes:
        config: The behavioural definition (role, system prompt, references).
        client: The underlying LLM transport used for calls.
    """

    def __init__(self, config: AgentConfig, *, client: LLMClient) -> None:
        self.config = config
        self.client = client

    @property
    def name(self) -> str:
        return self.config.name

    def build_messages(
        self,
        history: list[dict[str, str]],
        *,
        summary: str = "",
        prefix: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Return the full ``messages`` stack for a request.

        Prepends the assembled system prompt (if any) to the given *history*.
        Durable context can be injected *before* the role prompt in two ways:

        * *summary* — the rolling-summary string, injected as the very first
          system message (compression path);
        * *prefix* — a list of system messages carrying durable memory produced
          by a context strategy (e.g. a rendered sticky-facts block).

        Order: ``[summary?, prefix..., system, ...history]`` so durable memory
        is always visible ahead of the agent's role prompt.
        """
        messages: list[dict[str, str]] = []
        if summary:
            messages.append({"role": "system", "content": summary})
        if prefix:
            messages.extend(prefix)
        system_prompt = self.config.effective_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        return messages

    @property
    def context_window(self) -> int | None:
        """The model's input context window in tokens, or ``None`` if unknown."""
        return self.client.config.context_window

    @property
    def max_request_tokens(self) -> int | None:
        """Hard per-request token ceiling, or ``None`` if no such extra limit.

        This reflects the account-tier size cap (e.g. Groq's TPM ceiling), which
        is typically smaller than :attr:`context_window`.
        """
        return self.client.config.max_request_tokens

    @property
    def compression_settings(self) -> CompressionSettings | None:
        """Compression settings for sessions of this agent, or ``None``.

        Delegates to :attr:`~llm_bot.stores.AgentConfig.compression_settings`;
        a session created from this agent will auto-enable compression when this
        is not ``None``.
        """
        return self.config.compression_settings


class Session:
    """A single conversation with an :class:`Agent`, owning its own history.

    Multiple sessions can share the same agent (different topics, different
    users); each keeps its own independent message stack. History is loaded on
    construction and persisted to the :class:`SessionStore` after every turn, so
    a conversation survives process restarts.

    Attributes:
        session_id: Stable identifier used to persist/load the history.
        agent: The agent this session talks to.
    """

    def __init__(
        self,
        session_id: str,
        agent: Agent,
        *,
        store: SessionStore,
        history: list[dict[str, str]] | None = None,
        compression: CompressionSettings | None = None,
        on_compress: Callable[[CompressionEvent], None] | None = None,
        strategy: ContextStrategy | None = None,
        memory: MemoryLayers | None = None,
        memory_auto_extract: bool = True,
        task: TaskStateMachine | None = None,
        task_auto_detect: bool = True,
        invariants: InvariantRegistry | None = None,
        audit_invariants_warn: bool = False,
        mcp: MCPRouter | None = None,
        mcp_max_rounds: int | None = None,
        retriever: Retriever | None = None,
        rag_reuse_evidence: bool = False,
        rag_cite: bool = True,
    ) -> None:
        self.session_id = session_id
        self.agent = agent
        self._store = store
        self._memory = memory
        # MCP tool integration: an optional router over MCP servers. When set,
        # its tools block is injected into every request and a model reply in
        # the {"call_tool": ...} directive form triggers a real tool call,
        # whose result is fed back as a service turn (bounded by mcp_max_rounds).
        # Real-world beta finding: after a creation-time validation error
        # (e.g. an invented action name) the model needs one extra round to
        # correct itself, so the budget must leave room for that.
        self._mcp = mcp
        # RAG: an optional retriever over a local document index. When set,
        # every user question is answered *also* by the top chunks of that
        # index, and the rendered block is injected as a prefix system
        # message. It is deliberately not history: the block is rebuilt per
        # question and never persisted, so the dialog never accumulates
        # yesterday's evidence (same rule as tool results under ARCH-2).
        self._retriever = retriever
        self._rag_events: list[RagEvent] = []
        self._citation_audits: list[CitationAudit] = []
        # Evidence this dialog has already seen, kept only when the owner asks
        # for it. Off by default: the block is rebuilt per turn on purpose, and
        # remembering it would undo that.
        self._rag_reuse_evidence = rag_reuse_evidence
        self._rag_cite = rag_cite
        self._rag_earlier: list[str] = []
        # Round budget: an explicit value wins; otherwise the router's
        # ``max_rounds`` (configurable via the ``max_rounds`` key of the MCP
        # config for long cross-server chains), otherwise the default.
        if mcp_max_rounds is None:
            mcp_max_rounds = (
                mcp.max_rounds if mcp is not None else DEFAULT_MAX_ROUNDS
            )
        self._mcp_max_rounds = max(1, int(mcp_max_rounds))
        self._mcp_events: list[MCPEvent] = []
        # When enabled, after each turn the reply is classified via a small LLM
        # call and the extracted facts are written explicitly into the working /
        # long-term layers (see ``extract_memory``).
        self._memory_auto_extract = memory_auto_extract
        self._memory_events: list[MemoryEvent] = []
        self._history = (
            list(history) if history is not None else store.load(session_id)
        )
        if memory is not None:
            memory.short.clear()
            for msg in self._history:
                memory.short.push(msg)
        # Token usage of the most recent turn (see ``chat`` / ``chat_with_details``).
        self.last_usage: TokenUsage | None = None
        # Context compression. When enabled, the older part of the history is
        # folded into a running summary that is injected at the front of each
        # request instead of resending the full old dialog.
        self._compression = compression
        self._summary: str = ""
        self._on_compress = on_compress
        self._compression_events: list[CompressionEvent] = []
        if compression is not None:
            self._summary = store.load_summary(session_id)
            self._compressor = ContextCompressor(
                compression,
                summarize=self._summarize_block,
                max_chars=summary_budget_chars(
                    compression, agent.context_window
                ),
            )
        else:
            self._compressor = None
        # Pluggable context-management strategy (sliding window / sticky facts /
        # branching). When set it owns the conversation history and request
        # assembly; it is mutually exclusive with rolling-summary compression.
        self._strategy = strategy
        if self._strategy is not None:
            self._strategy.load_state(store, session_id)
        # Formalized task state (stage / step / expected action). When enabled,
        # the machine loads its persisted snapshot via a change-hook and injects
        # a rendered block into the request prefix so the model can continue the
        # task without re-explanation after a pause or a process restart.
        self._task = task
        self._task_events: list[TaskDetectionEvent] = []
        self._task_auto_detect = task_auto_detect
        if self._task is not None:
            saved = store.load_task_state(session_id)
            if saved is not None:
                self._task.load(TaskState.from_dict(saved))
            self._task._on_change = self._persist_task_state
        # Invariants: hard constraints stored SEPARATELY from the dialog. A
        # registry passed by the factory already carries the global (YAML)
        # invariants; here the session-scoped ones (persisted under the
        # session file's ``invariants`` key) are merged in. Every user message
        # passes a deterministic regex gate BEFORE the request is assembled,
        # and the rendered block is injected ahead of the role prompt on
        # every turn.
        self._invariants = invariants
        self._invariant_events: list[InvariantAuditEvent] = []
        self._audit_invariants_warn = audit_invariants_warn
        if self._invariants is not None:
            for entry in store.load_invariants(session_id):
                try:
                    self._invariants.add(
                        Invariant.from_dict(entry, source="session")
                    )
                except (ValueError, KeyError):
                    continue  # a corrupt entry never breaks session restore
            self._invariants._on_change = self._persist_invariants

    def _persist_task_state(self, state: TaskState) -> None:
        """Persist the task snapshot whenever the machine changes it."""
        if self._task is not None:
            self._store.save_task_state(self.session_id, state.to_dict())

    def _persist_invariants(self) -> None:
        """Persist session-scoped invariants whenever the registry changes."""
        if self._invariants is not None:
            self._store.save_invariants(
                self.session_id,
                [
                    item.to_dict()
                    for item in self._invariants.items()
                    if item.source == "session"
                ],
            )

    @property
    def history(self) -> list[dict[str, str]]:
        """Read-only view of the conversation history.

        With a pluggable strategy the history is owned by the strategy (e.g. the
        currently active branch); otherwise it is the session's own stack.
        """
        if self._strategy is not None:
            return self._strategy.history
        return list(self._history)

    @property
    def strategy(self) -> ContextStrategy | None:
        """The active context-management strategy, or ``None`` (full history)."""
        return self._strategy

    @property
    def memory(self) -> MemoryLayers | None:
        """The session's explicit layered memory (short / working / long).

        ``None`` when the session was created without memory wiring. When set,
        durable blocks (working + long-term) are injected ahead of the system
        prompt on every turn and each dialog message is mirrored into the
        short-term layer.
        """
        return self._memory

    @property
    def memory_auto_extract(self) -> bool:
        """Whether each turn's reply is auto-classified into working/long-term."""
        return self._memory_auto_extract

    @property
    def memory_events(self) -> list[MemoryEvent]:
        """Every automatic memory-extraction event recorded in this session."""
        return list(self._memory_events)

    @property
    def last_memory_event(self) -> MemoryEvent | None:
        """The most recent memory-extraction event, or ``None`` if none happened."""
        return self._memory_events[-1] if self._memory_events else None

    @property
    def total_memory_extractions(self) -> int:
        """How many turns were classified into durable memory."""
        return len(self._memory_events)

    @property
    def total_memory_tokens(self) -> int:
        """Total tokens spent on memory-classification calls across the session."""
        return sum(e.total_tokens for e in self._memory_events)

    # -- Invariants (hard constraints) ---------------------------------------- #

    @property
    def invariants(self) -> InvariantRegistry | None:
        """The session's invariant registry, or ``None`` when not wired."""
        return self._invariants

    # -- MCP tools ------------------------------------------------------------- #

    @property
    def mcp(self) -> MCPRouter | None:
        """The session's MCP tool router, or ``None`` when not wired."""
        return self._mcp

    @property
    def mcp_events(self) -> list[MCPEvent]:
        """Every MCP tool round-trip recorded in this session."""
        return list(self._mcp_events)

    @property
    def last_mcp_event(self) -> MCPEvent | None:
        """The most recent MCP tool event, or ``None``."""
        return self._mcp_events[-1] if self._mcp_events else None

    @property
    def rag_events(self) -> list[RagEvent]:
        """Every retrieval recorded in this session (one per question)."""
        return list(self._rag_events)

    @property
    def last_rag_event(self) -> RagEvent | None:
        """The most recent retrieval event, or ``None``."""
        return self._rag_events[-1] if self._rag_events else None

    @property
    def citation_audits(self) -> list[CitationAudit]:
        """One citation check per grounded turn, in order."""
        return list(self._citation_audits)

    def _invariant_ids(self) -> tuple[str, ...]:
        """Ids rendered as ``[STACK-1]`` in the prompt — never source citations."""
        if self._invariants is None:
            return ()
        return tuple(item.id for item in self._invariants.items())

    @property
    def last_citation_audit(self) -> CitationAudit | None:
        """The most recent citation check, or ``None``."""
        return self._citation_audits[-1] if self._citation_audits else None

    def _rag_subject(self, *, limit: int = 2) -> list[str]:
        """Documents this dialog has already answered from, newest first.

        A follow-up often names no product at all («а он долго действует?»),
        and in a corpus of near-duplicate documents every file repeats the same
        wording — so plain retrieval answers from whichever unrelated file the
        reranker happened to like. Passing the files already cited keeps the
        block on the subject the customer is actually asking about. Read from the
        visible history only: the RAG block is not stored there, so these are the
        citations the model itself produced.
        """
        names: list[str] = []
        for message in reversed(self._history):
            if message.get("role") != "assistant":
                continue
            for name in sources_from_text(str(message.get("content", "")), limit=2):
                if name not in names:
                    names.append(name)
            if len(names) >= limit:
                break
        return names[:limit]

    @property
    def invariant_events(self) -> list[InvariantAuditEvent]:
        """Every post-reply audit event recorded in this session."""
        return list(self._invariant_events)

    @property
    def last_invariant_event(self) -> InvariantAuditEvent | None:
        """The most recent invariant-audit event, or ``None``."""
        return (
            self._invariant_events[-1] if self._invariant_events else None
        )

    @property
    def total_invariant_tokens(self) -> int:
        """Total tokens spent on invariant-audit calls across the session."""
        return sum(e.total_tokens for e in self._invariant_events)

    def add_invariant(
        self,
        invariant_id: str,
        kind: str,
        statement: str,
        rationale: str = "",
        forbidden_patterns: list[str] | None = None,
    ) -> Invariant:
        """Add a session-scoped invariant and persist it.

        The invariant is stored under the session file (never in the dialog
        history) and survives restarts. Raises ``ValueError`` on duplicate
        ids (including global ones) or invalid fields.
        """
        if self._invariants is None:
            raise ValueError(
                "Инварианты не подключены к этой сессии."
            )
        item = Invariant.from_dict(
            {
                "id": invariant_id,
                "kind": kind,
                "statement": statement,
                "rationale": rationale,
                "forbidden_patterns": forbidden_patterns or [],
            },
            source="session",
        )
        self._invariants.add(item)
        return item

    def drop_invariant(self, invariant_id: str) -> Invariant:
        """Remove a session-scoped invariant by id and persist the change.

        Global invariants (from ``data/invariants.yaml``) are protected:
        attempting to drop one raises ``ValueError`` — owner configuration is
        immutable from the dialog.
        """
        if self._invariants is None:
            raise ValueError(
                "Инварианты не подключены к этой сессии."
            )
        if self._invariants.is_protected(invariant_id):
            raise ValueError(
                f"Инвариант {invariant_id!r} глобальный и не может быть "
                "удалён из диалога."
            )
        return self._invariants.drop(invariant_id)

    # -- Task state machine -------------------------------------------------- #

    @property
    def task(self) -> TaskStateMachine | None:
        """The session's task state machine, or ``None`` when not enabled."""
        return self._task

    @property
    def task_state(self) -> TaskState | None:
        """The current task snapshot, or ``None`` when the machine is off."""
        return self._task.state if self._task is not None else None

    @property
    def task_events(self) -> list[TaskDetectionEvent]:
        """Every auto-detection event recorded in this session."""
        return list(self._task_events)

    @property
    def last_task_event(self) -> TaskDetectionEvent | None:
        """The most recent task auto-detection event, or ``None``."""
        return self._task_events[-1] if self._task_events else None

    @property
    def total_task_extractions(self) -> int:
        """How many turns were classified into the task state machine."""
        return len(self._task_events)

    @property
    def total_task_tokens(self) -> int:
        """Total tokens spent on task-classification calls across the session."""
        return sum(e.total_tokens for e in self._task_events)

    # -- Branching helpers (only meaningful for the "branching" strategy) ----- #

    def branch(self, name: str) -> None:
        """Create a new dialogue branch forking the current position.

        Only valid when the session uses the ``branching`` strategy. The new
        branch becomes active and the state is persisted.
        """
        if self._strategy is None:
            raise ValueError("Сессия не использует стратегию 'branching'.")
        self._strategy.branch(name)
        self._strategy.save_state(self._store, self.session_id)

    def switch_branch(self, name: str) -> None:
        """Make *name* the active dialogue branch (branching strategy only)."""
        if self._strategy is None:
            raise ValueError("Сессия не использует стратегию 'branching'.")
        self._strategy.switch(name)
        self._strategy.save_state(self._store, self.session_id)

    @property
    def summary(self) -> str:
        """The running context-compression summary (``""`` when none/disabled)."""
        return self._summary

    @property
    def compression_enabled(self) -> bool:
        """True when context compression is active for this session."""
        return self._compression is not None

    @property
    def compression_events(self) -> list[CompressionEvent]:
        """Every compression round recorded so far in this session."""
        return list(self._compression_events)

    @property
    def last_compression_event(self) -> CompressionEvent | None:
        """The most recent compression round, or ``None`` if none happened."""
        return self._compression_events[-1] if self._compression_events else None

    @property
    def total_compressions(self) -> int:
        """How many times the history has been folded into the summary."""
        return len(self._compression_events)

    @property
    def total_messages_folded(self) -> int:
        """Total messages moved into the summary across all compressions."""
        return sum(e.messages_folded for e in self._compression_events)

    def _summarize_block(self, existing: str, block: str, max_chars: int | None) -> str:
        """Summarise *existing* + *block* via the session's own LLM client.

        Used as the compressor's ``summarize`` callback so summary generation
        needs no extra credentials or transport — it reuses the same client the
        session already talks through. *max_chars* is passed into the prompt as a
        soft size limit; the hard cap is applied by the compressor afterwards.
        """
        prompt = summarize_prompt(existing, block, max_chars)
        reply = self.agent.client.chat([{"role": "user", "content": prompt}])
        return _sanitize_text(reply.strip())

    def chat(self, user_message: str, *, service_turn: bool = False) -> str:
        """Send a user message and return only the assistant's reply text.

        Equivalent to :meth:`chat_with_details`, but returns just the reply
        string for callers that do not care about token accounting. Token stats
        remain available on :attr:`last_usage`. *service_turn* is forwarded —
        machine-generated turns skip the task auto-detector.
        """
        return self.chat_with_details(user_message, service_turn=service_turn).reply

    def chat_with_details(
        self, user_message: str, *, service_turn: bool = False
    ) -> ChatResult:
        """Send a user message and return a :class:`ChatResult` with token usage.

        The message is appended to the history, the whole stack is sent to the
        LLM, and the assistant's reply is stored. Before the request goes out we
        count the tokens for:

            * the current request (``request_tokens``);
            * the entire dialog history (``history_tokens``);
            * the context actually sent to the model (``context_tokens``,
              history + current request);
            * the model's reply (``reply_tokens``).

        If the assembled context would exceed the model's ``context_window``, a
        :class:`~llm_bot.client.ContextOverflowError` is raised and nothing is
        sent — this is where the caller should trim history or start a new
        session.

        *service_turn* marks a machine-generated message (e.g. the CLI
        auto-turn after a stage change). Such turns update the dialog history
        but are NOT fed to the task auto-detector: otherwise the classifier
        would read the machine's own stage report («проверка пройдена,
        дефектов нет») and could autonomously advance the pipeline
        (``validation → done``) without any user decision.
        """
        user_message = _sanitize_text(user_message.strip())
        if not user_message:
            raise ValueError("Message must not be empty.")

        # Invariant gate: a deterministic pre-flight check that runs BEFORE
        # anything else — a refusal costs zero tokens, never touches the
        # provider and leaves the history untouched (the user message is not
        # even appended yet).
        if self._invariants is not None:
            violated = self._invariants.check_request(user_message)
            if violated is not None:
                matched = next(
                    (
                        pattern
                        for pattern in violated.forbidden_patterns
                        if re.search(pattern, user_message, re.IGNORECASE)
                    ),
                    "",
                )
                raise InvariantViolationError(
                    violated,
                    matched_pattern=matched,
                    refusal=violated.refusal_text(matched),
                )

        user_msg = {"role": "user", "content": user_message}
        request_tokens = count_message_tokens(user_msg)

        # Project the history that will actually be sent. There are two paths:
        #
        # 1. A pluggable context strategy (sliding / sticky-facts / branching)
        #    computes the request view from its durable history plus this turn's
        #    user message, WITHOUT mutating anything — so an overflow check below
        #    can abort leaving the session and the store untouched.
        # 2. Rolling-summary compression folds the oldest messages into a running
        #    summary *in memory* here; the result is only committed after the
        #    budget checks pass.
        # Mirror the user message into the short-term (current dialog) layer.
        if self._memory is not None:
            self._memory.short.push(user_msg)

        # Invariants block: hard constraints are injected as the FIRST prefix
        # system message (ahead of memory and task state) so the model
        # explicitly reasons within them on every turn.
        invariant_prefix = []
        if self._invariants is not None:
            block = self._invariants.render_prompt_block()
            if block:
                invariant_prefix = [
                    {"role": "system", "content": block}
                ]

        # Durable memory (long-term profile/decisions + working task data) is
        # injected as a prefix so it is visible ahead of the system prompt on
        # every turn, even under a small sliding window or a fresh session.
        memory_prefix = (
            self._memory.prefix_messages() if self._memory is not None else []
        )

        # Formalized task state is rendered as a prefix system message so the
        # model sees stage/step/expected-action ahead of the role prompt on
        # every turn (this is what makes "continue" work without re-explaining).
        task_block = (
            self._task.render_prompt_block() if self._task is not None else ""
        )
        task_prefix = (
            [{"role": "system", "content": task_block}] if task_block else []
        )

        # MCP tools block: advertise available external tools and the call
        # directive. Injected after the task block, before the role prompt, so
        # the model always knows which tools it can reach this turn.
        mcp_prefix = []
        if self._mcp is not None:
            block = self._mcp.render_tools_block()
            if block:
                mcp_prefix = [{"role": "system", "content": block}]

        # RAG: the top chunks of the local index for THIS question. Placed
        # after the tools block and before the role prompt: the model already
        # knows what it can call, and the evidence sits immediately ahead of the
        # request it has to answer. Because it is a prefix message it counts
        # towards the budget checks below like any other part of the request,
        # so a big index cannot silently blow the context window.
        rag_prefix = []
        rag_event: RagEvent | None = None
        if self._retriever is not None:
            try:
                rag_block, rag_event = self._retriever.render(
                    user_message, prefer_sources=self._rag_subject()
                )
            except Exception as exc:  # noqa: BLE001 - retrieval must never
                # take the turn down: the bot answers without grounding rather
                # than failing (same policy as memory/task extraction below).
                logger.warning("RAG retrieval failed, answering without it: %s", exc)
            else:
                if rag_block:
                    rag_prefix = [{"role": "system", "content": rag_block}]
                self._rag_events.append(rag_event)

        prepared = None
        if self._strategy is not None:
            prepared = self._strategy.prepare(user_msg)
            # Combine the strategy's own durable prefix (e.g. sticky facts) with
            # the layered-memory prefix and the task-state block, durable state
            # first.
            prefix = [
                *invariant_prefix,
                *memory_prefix,
                *task_prefix,
                *mcp_prefix,
                *rag_prefix,
                *prepared.prefix,
            ]
            messages = self.agent.build_messages(
                prepared.request_history, prefix=prefix
            )
        else:
            projected_history = [*self._history, user_msg]
            projected_summary = self._summary
            folded_messages: list[dict[str, str]] = []
            if self._compressor is not None:
                projected_history, projected_summary, folded_messages = (
                    self._compressor.compress(projected_history, projected_summary)
                )

            # Build the exact stack that would go to the model (summary + system +
            # (possibly compressed) history), injecting durable memory and the
            # task-state block as prefix.
            messages = self.agent.build_messages(
                projected_history,
                summary=projected_summary,
                prefix=[
                    *invariant_prefix,
                    *memory_prefix,
                    *task_prefix,
                    *mcp_prefix,
                    *rag_prefix,
                ]
                or None,
            )
        # Recency-bias footer: inject a short invariant reminder as a
        # system message right before the current user message so the
        # model sees the constraint immediately before the request it
        # must check.  This complements the full invariant block at the
        # top of the context (which may be thousands of tokens away).
        if self._invariants is not None:
            footer = self._invariants.render_footer()
            if footer and messages:
                # Insert before the last message (the current user msg).
                messages.insert(-1, {"role": "system", "content": footer})
        # Defensive guard: history may also carry surrogates from earlier loads,
        # so sanitize the whole stack before handing it to the HTTP client.
        _sanitize_messages(messages)

        context_tokens = count_messages_tokens(messages)
        context_window = (
            self.agent.context_window
            if self.agent.context_window is not None
            else DEFAULT_CONTEXT_WINDOW
        )
        max_request_tokens = self.agent.max_request_tokens

        # Pre-flight budget checks: refuse to send a request the provider would
        # reject. Two independent ceilings apply:
        #   * the model's context window;
        #   * the account-tier per-request size cap, which is often smaller
        #     (e.g. Groq's TPM ceiling) and cannot be retried away.
        if (
            max_request_tokens is not None
            and context_tokens > max_request_tokens
        ):
            raise ContextTooLargeError(
                f"Запрос превышает лимит размера аккаунта: {context_tokens} "
                f"токенов > {max_request_tokens}. Сократите сообщение или "
                f"начните новый сеанс.",
                status_code=413,
                requested_tokens=context_tokens,
                limit_tokens=max_request_tokens,
            )
        if context_tokens > context_window:
            raise ContextOverflowError(
                f"Диалог превышает контекст модели: {context_tokens} токенов > "
                f"лимит {context_window}. Завершите тему или начните новый сеанс.",
                context_tokens=context_tokens,
                context_window=context_window,
            )

        # Commit the projected history and summary now that the budget checks
        # passed and the request is about to be sent. This applies only to the
        # rolling-summary path; a pluggable strategy defers its (atomic) history
        # update to ``on_turn_end`` after the reply arrives.
        if self._strategy is None:
            self._history = projected_history
            self._summary = projected_summary

            if folded_messages:
                event = CompressionEvent(
                    messages_folded=len(folded_messages),
                    folded_tokens=count_messages_tokens(folded_messages),
                    summary_chars=len(projected_summary),
                    history_before=len(folded_messages) + len(projected_history),
                    history_after=len(projected_history),
                )
                self._compression_events.append(event)
                if self._on_compress is not None:
                    self._on_compress(event)

        reply = self.agent.client.chat(messages)

        # MCP tool loop: the model may answer with a {"call_tool": ...}
        # directive (single call or a batch array) instead of a user-facing
        # reply. Execute every call through the router, append the directive
        # + all results to THIS turn's request view, and let the model produce
        # the final answer. Bounded by mcp_max_rounds (rounds = model replies)
        # so a misbehaving model cannot loop forever; several calls in ONE
        # reply do not consume extra rounds.
        if self._mcp is not None:
            for round_index in range(self._mcp_max_rounds):
                directives = self._mcp.parse_directives(reply)
                if not directives:
                    # Guard against the impersonation above: instead of
                    # delivering a fabricated «Результат инструмента …»
                    # reply, append a corrective service turn and let the
                    # model redo the missed step — but only while the round
                    # budget allows (at the budget edge the reply stands,
                    # same documented boundary as a leftover directive).
                    if (
                        round_index < self._mcp_max_rounds - 1
                        and _looks_like_fabricated_feedback(reply)
                    ):
                        messages.append(
                            {"role": "assistant", "content": reply}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": _FABRICATION_CORRECTION,
                            }
                        )
                        reply = self.agent.client.chat(messages)
                        context_tokens = count_messages_tokens(messages)
                        continue
                    break
                feedbacks = []
                for directive in directives:
                    event = self._mcp.call_tool(
                        directive.name, directive.arguments
                    )
                    self._mcp_events.append(event)
                    if event.ok:
                        feedbacks.append(
                            f"Результат инструмента {directive.name}:\n"
                            f"{event.result}"
                        )
                    else:
                        feedbacks.append(
                            f"Инструмент {directive.name} завершился ошибкой: "
                            f"{event.error}"
                        )
                messages.append({"role": "assistant", "content": reply})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "\n\n".join(feedbacks)
                            + "\nИспользуй эти результаты и дай финальный"
                            " ответ пользователю обычным текстом. Если часть"
                            " действий ещё не выполнена (а не отвергнута),"
                            " снова верни директиву call_tool."
                        ),
                    }
                )
                reply = self.agent.client.chat(messages)
            if self._mcp_events:
                # Token accounting must reflect what was actually sent across
                # the whole turn, including tool round-trips.
                context_tokens = count_messages_tokens(messages)

        # Every citation in the answer has to be backed by the block that was
        # actually sent. Done before the reply reaches memory and the session
        # store, so a rewritten source number is not persisted as "what we told
        # the customer" and then reused as evidence on the next turn.
        if rag_event is not None and rag_event.retrieved:
            if self._rag_cite:
                reply, audit = audit_citations(
                    reply,
                    rag_event.retrieved,
                    also_backed=self._rag_earlier if self._rag_reuse_evidence else (),
                    ignore=self._invariant_ids(),
                )
                self._citation_audits.append(audit)
                if self._rag_reuse_evidence and audit.clean:
                    self._rag_earlier.extend(rag_event.retrieved)
                if audit.dropped:
                    logger.warning(
                        "RAG citations not in the retrieved block: %s",
                        ", ".join(audit.dropped),
                    )
            else:
                reply = strip_citations(reply, ignore=self._invariant_ids())
                # Reusing earlier evidence is gated on a clean audit because a
                # citation contradicting the answer is what makes a block unsafe
                # to lean on again. A reply with no citations contradicts
                # nothing, so the block is reused. Otherwise switching
                # --rag-no-cite on for readability would quietly change what
                # gets retrieved on the follow-up turn, and the two modes would
                # no longer be comparable.
                if self._rag_reuse_evidence:
                    self._rag_earlier.extend(rag_event.retrieved)

        # Mirror the assistant reply into the short-term (current dialog) layer.
        if self._memory is not None:
            self._memory.short.push({"role": "assistant", "content": reply})

        # Optionally classify this completed turn into working / long-term memory
        # via a small LLM call, writing extracted facts EXPLICITLY to their layers.
        if (
            self._memory is not None
            and self._memory_auto_extract
        ):
            try:
                event = extract_memory(
                    self._memory,
                    user_msg,
                    reply,
                    self.agent.client.chat,
                )
            except Exception as exc:  # noqa: BLE001 - never let memory break the turn
                # One concise line for the user; the full traceback goes to
                # the debug log (visible with -v) instead of the terminal.
                logger.error("[memory] авто-классификация не удалась: %s", exc)
                logger.debug("[memory] авто-классификация trace:", exc_info=True)
                event = MemoryEvent(recognized=False)
            self._memory_events.append(event)

        # Optionally drive the task state machine from this completed turn via a
        # small LLM call (task setup / stage hints / pause & resume phrases).
        # Skip detection while paused: the task is explicitly stopped, and the
        # detection call would waste tokens and risk rate limits.
        # Skip detection on SERVICE turns (CLI auto-turn after a stage change):
        # the classifier would read the machine's own stage report and could
        # autonomously advance the pipeline (validation → done) without any
        # user decision — only the real dialog drives the machine.
        if (
            self._task is not None
            and self._task_auto_detect
            and not self._task.state.is_paused
            and not service_turn
        ):
            try:
                task_event = detect_task_turn(
                    self._task,
                    user_msg,
                    reply,
                    self.agent.client.chat,
                )
            except Exception as exc:  # noqa: BLE001 - never let the task FSM break a turn
                # One concise line for the user; the full traceback goes to
                # the debug log (visible with -v) instead of the terminal.
                logger.error("[task] авто-детект не удался: %s", exc)
                logger.debug("[task] авто-детект trace:", exc_info=True)
                task_event = TaskDetectionEvent(recognized=False)
            self._task_events.append(task_event)

        # Post-reply audit: one small LLM call checks the reply against the
        # invariants. By default this is a HARD gate — a violating reply is
        # refused (InvariantViolationError raised, history rolled back) just
        # like the pre-flight request gate. When ``audit_invariants_warn`` is
        # True the violation is only logged to stderr (soft mode for
        # debugging). An audit call failure never breaks the turn.
        if self._invariants is not None and len(self._invariants) > 0:
            try:
                audit_event = audit_reply(
                    self._invariants,
                    reply,
                    self.agent.client.chat,
                    user_message=user_message,
                )
            except Exception as exc:  # noqa: BLE001 - never break a turn
                logger.error("[invariants] аудит не удался: %s", exc)
                logger.debug("[invariants] аудит trace:", exc_info=True)
                audit_event = InvariantAuditEvent(recognized=False)
            self._invariant_events.append(audit_event)
            if audit_event.violated and not self._audit_invariants_warn:
                # HARD gate: roll back the turn and refuse the reply.
                if self._strategy is None:
                    # Roll back the assistant reply from in-memory history.
                    if (
                        self._history
                        and self._history[-1].get("role") == "assistant"
                        and self._history[-1].get("content") == reply
                    ):
                        self._history.pop()
                    # Roll back the user message too.
                    if (
                        self._history
                        and self._history[-1].get("role") == "user"
                        and self._history[-1].get("content") == user_message
                    ):
                        self._history.pop()
                    # Persist the rolled-back history.
                    if self._compressor is not None:
                        self._store.save_full(
                            self.session_id, self._history,
                            summary=self._summary,
                        )
                    else:
                        self._store.save(self.session_id, self._history)
                violated_item = self._invariants.get(audit_event.violated_id)
                raise InvariantViolationError(
                    violated_item,
                    refusal=(
                        f"Ответ отклонён: он нарушает инвариант "
                        f"{audit_event.violated_id}"
                        f" ({violated_item.kind if violated_item else '?'}). "
                        f"{audit_event.rationale} "
                        f"Инвариант: "
                        f"{violated_item.statement if violated_item else ''} "
                        "Переформулируйте запрос так, чтобы ответ не "
                        "противоречил ограничению."
                    ),
                )

        provider_usage = self.agent.client.last_usage
        reply_tokens = count_message_tokens({"role": "assistant", "content": reply})

        usage = merge_provider_usage(
            TokenUsage(
                request_tokens=request_tokens,
                history_tokens=context_tokens - request_tokens,
                context_tokens=context_tokens,
                reply_tokens=reply_tokens,
                total_tokens=context_tokens + reply_tokens,
                context_window=context_window,
                estimated=provider_usage is None,
            ),
            provider_usage,
        )
        self.last_usage = usage

        if self._strategy is not None:
            # Atomically record the completed turn in the strategy's durable
            # history/memory and persist it (with any facts/branches).
            self._strategy.on_turn_end(user_msg, reply)
            self._strategy.save_state(self._store, self.session_id)
        else:
            self._history.append({"role": "assistant", "content": reply})
            if self._compressor is not None:
                # Persist both the compressed history and the running summary.
                self._store.save_full(
                    self.session_id, self._history, summary=self._summary
                )
            else:
                self._store.save(self.session_id, self._history)
        return ChatResult(reply=reply, usage=usage)