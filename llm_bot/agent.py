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

from typing import Any, Callable

from llm_bot.client import ContextOverflowError, ContextTooLargeError, LLMClient
from llm_bot.compress import (
    CompressionEvent,
    CompressionSettings,
    ContextCompressor,
    summarize_prompt,
)
from llm_bot.context_strategies import ContextStrategy
from llm_bot.stores import AgentConfig, SessionStore
from llm_bot.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    ChatResult,
    TokenUsage,
    count_message_tokens,
    count_messages_tokens,
    merge_provider_usage,
)


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
    ) -> None:
        self.session_id = session_id
        self.agent = agent
        self._store = store
        self._history = (
            list(history) if history is not None else store.load(session_id)
        )
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

    def chat(self, user_message: str) -> str:
        """Send a user message and return only the assistant's reply text.

        Equivalent to :meth:`chat_with_details`, but returns just the reply
        string for callers that do not care about token accounting. Token stats
        remain available on :attr:`last_usage`.
        """
        return self.chat_with_details(user_message).reply

    def chat_with_details(self, user_message: str) -> ChatResult:
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
        """
        user_message = _sanitize_text(user_message.strip())
        if not user_message:
            raise ValueError("Message must not be empty.")

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
        prepared = None
        if self._strategy is not None:
            prepared = self._strategy.prepare(user_msg)
            messages = self.agent.build_messages(
                prepared.request_history, prefix=prepared.prefix
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
            # (possibly compressed) history).
            messages = self.agent.build_messages(
                projected_history, summary=projected_summary
            )
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