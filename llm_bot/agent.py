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

from typing import Any

from llm_bot.client import ContextOverflowError, ContextTooLargeError, LLMClient
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

    def build_messages(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return the full ``messages`` stack for a request.

        Prepends the assembled system prompt (if any) to the given *history*.
        """
        messages: list[dict[str, str]] = []
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
    ) -> None:
        self.session_id = session_id
        self.agent = agent
        self._store = store
        self._history = (
            list(history) if history is not None else store.load(session_id)
        )
        # Token usage of the most recent turn (see ``chat`` / ``chat_with_details``).
        self.last_usage: TokenUsage | None = None

    @property
    def history(self) -> list[dict[str, str]]:
        """Read-only view of the conversation history."""
        return list(self._history)

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

        # Build the exact stack that would go to the model (system + all history).
        messages = self.agent.build_messages([*self._history, user_msg])
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

        self._history.append(user_msg)
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

        self._history.append({"role": "assistant", "content": reply})
        self._store.save(self.session_id, self._history)
        return ChatResult(reply=reply, usage=usage)