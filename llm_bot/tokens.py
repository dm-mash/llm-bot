"""Deterministic token accounting for the agent.

This module answers the three questions the agent needs to reason about its
budget *before* it talks to the model:

    1. How many tokens does the *current request* consume?
    2. How many tokens does the *whole dialog history* consume?
    3. How many tokens did the *model's reply* use?

Providers differ in exactly how they split text into tokens (and some, like
local Ollama setups, do not report ``usage`` at all). Requiring ``tiktoken``
would pull in a heavyweight BPE dependency that downloads model files on first
use and only knows OpenAI's vocabularies. Instead we use a small, deterministic
estimator that works for *any* provider and never needs the network:

    tokens(text) ≈ ceil(len(chars) / 4)

plus a fixed per-message overhead of 4 tokens (the same convention OpenAI uses
for its ``{"role", "content"}`` message framing). This is accurate enough to
show how a dialog grows and when it approaches a context window.

When the provider *does* report usage (via ``extract_usage``), we merge those
authoritative numbers in so the reported totals reflect reality rather than
our estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# OpenAI's documented per-message overhead for the {"role","content"} framing.
MESSAGE_OVERHEAD_TOKENS = 4

# Default fallback context window (tokens) used when neither the config nor the
# model profile specifies one. Chosen conservatively so overflow is visible in
# demos without a real 128k window hiding the problem.
DEFAULT_CONTEXT_WINDOW = 8192


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens *text* takes, using ``ceil(chars / 4)``.

    A rough BPE approximation: roughly 4 characters per token for mixed
    English/Russian text. Guaranteed deterministic and offline.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def count_message_tokens(message: dict[str, Any]) -> int:
    """Count tokens for a single ``{"role", "content"}`` message.

    Includes the fixed per-message overhead plus the content estimate.
    """
    content = message.get("content", "")
    if not isinstance(content, str):
        content = ""
    return MESSAGE_OVERHEAD_TOKENS + estimate_tokens(content)


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens across a full list of messages (system + history + user)."""
    return sum(count_message_tokens(m) for m in messages)


@dataclass(frozen=True)
class TokenUsage:
    """A snapshot of token consumption for one turn.

    Attributes:
        request_tokens: Tokens for the current user request alone.
        history_tokens: Tokens for the entire dialog history sent to the model
            (system prompt + all prior turns, excluding the live reply).
        context_tokens: Full request context sent to the model
            (``history_tokens`` including the current request).
        reply_tokens: Tokens the model spent on its answer (estimate from text,
            or authoritative ``completion_tokens`` when the provider reports it).
        total_tokens: Grand total for the turn (``context_tokens + reply_tokens``).
        context_window: The model's configured context window in tokens, or
            ``None`` if unknown.
        estimated: True if any of the numbers were produced by the local
            heuristic rather than by the provider's ``usage`` field.
    """

    request_tokens: int = 0
    history_tokens: int = 0
    context_tokens: int = 0
    reply_tokens: int = 0
    total_tokens: int = 0
    context_window: int | None = None
    estimated: bool = True

    @property
    def overflow(self) -> bool:
        """True if the request context exceeds the model's context window."""
        if self.context_window is None:
            return False
        return self.context_tokens > self.context_window

    @property
    def fill_percent(self) -> float | None:
        """How much of the context window the request fills (0..100), or None."""
        if not self.context_window:
            return None
        return (self.context_tokens / self.context_window) * 100.0


def merge_provider_usage(usage: TokenUsage, provider: dict[str, int] | None) -> TokenUsage:
    """Return *usage* with authoritative numbers substituted from *provider*.

    *provider* is whatever :func:`~llm_bot.diagnostics.extract_usage` returned.
    When the provider reports ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` we trust those over our estimates, and mark the result as
    not estimated.
    """
    if not provider:
        return usage

    context_tokens = int(provider.get("prompt_tokens", usage.context_tokens))
    reply_tokens = int(provider.get("completion_tokens", usage.reply_tokens))
    total_tokens = int(provider.get("total_tokens", context_tokens + reply_tokens))

    estimated = bool(
        context_tokens == usage.context_tokens
        and reply_tokens == usage.reply_tokens
        and total_tokens == usage.total_tokens
    )
    return TokenUsage(
        request_tokens=usage.request_tokens,
        history_tokens=usage.history_tokens,
        context_tokens=context_tokens,
        reply_tokens=reply_tokens,
        total_tokens=total_tokens,
        context_window=usage.context_window,
        estimated=estimated,
    )


@dataclass(frozen=True)
class ChatResult:
    """The outcome of one :meth:`~llm_bot.agent.Session.chat` call.

    Carries both the reply text and the token accounting so callers (CLI, web,
    tests) can surface cost and budget without re-sending anything.

    Attributes:
        reply: The assistant's reply text.
        usage: Token usage for this turn.
        truncated: True if the model was cut off by its token limit
            (``finish_reason='length'``); implies ``reply`` is partial.
    """

    reply: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    truncated: bool = False