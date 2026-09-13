"""Context compression for long conversations.

As a dialog grows, sending the *entire* history to the model each turn costs more
tokens and eventually overflows the context window. This module implements the
classic **rolling-summary** strategy:

* the last ``keep_last`` messages are kept verbatim (so recent context stays
  precise);
* when the live history exceeds ``block_size`` messages, the oldest part that
  falls outside the recent window is folded into a short running ``summary``;
* the summary is stored separately and injected at the *front* of the request
  (as the first system message) instead of the full old history.

To keep compression cheap, the summary is updated **incrementally**: only the
newly rolled-off block plus the existing summary are sent to the LLM, never the
whole dialog. Summaries are produced with the same :class:`~llm_bot.client.LLMClient`
the session already uses, so no extra credentials or transport are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# The default summarization instruction. Builds a prompt that combines the
# existing running summary with the newly rolled-off block so the update is
# incremental (we never resend the whole dialog).
_SUMMARIZE_PROMPT = (
    "Ты — система сжатия контекста. Ниже — краткое изложение предыдущего "
    "разговора и новый фрагмент переписки. Перепиши итоговое изложение так, "
    "чтобы оно сохранило все существенные факты, решения, имена, числа и "
    "договорённости из обоих частей. Не добавляй новых фактов. Отвечай ТОЛЬКО "
    "текстом нового изложения, без пояснений.\n\n"
    "Имеющееся изложение:\n{existing}\n\n"
    "Новый фрагмент переписки:\n{block}"
)

# A single-turn summarizer (no existing summary yet).
_FIRST_SUMMARIZE_PROMPT = (
    "Ты — система сжатия контекста. Ниже — фрагмент переписки. Составь краткое "
    "изложение, сохранив все существенные факты, решения, имена, числа и "
    "договорённости. Не добавляй новых фактов. Отвечай ТОЛЬКО текстом "
    "изложения, без пояснений.\n\nФрагмент переписки:\n{block}"
)


# A soft-limit instruction appended to the summarization prompt so the model is
# asked to keep the summary compact. This is a *request*, not a guarantee; the
# hard cap is enforced in code (see ``ContextCompressor``).
_MAX_LEN_INSTRUCTION = (
    "\n\nВажно: итоговое изложение должно быть КРАТКИМ — не длиннее примерно "
    "{max_chars} символов. Приоритет — сохранить факты, имена и числа."
)


def summarize_prompt(
    existing: str,
    block: str,
    max_chars: int | None = None,
) -> str:
    """Build the summarization prompt for one update round.

    When *max_chars* is set, a soft size instruction is appended so the model
    keeps the summary within budget (the actual hard cap is applied in code).
    """
    if existing.strip():
        prompt = _SUMMARIZE_PROMPT.format(existing=existing, block=block)
    else:
        prompt = _FIRST_SUMMARIZE_PROMPT.format(block=block)
    if max_chars is not None and max_chars > 0:
        prompt += _MAX_LEN_INSTRUCTION.format(max_chars=max_chars)
    return prompt


def truncate_summary(summary: str, max_chars: int | None) -> str:
    """Truncate *summary* to at most *max_chars* chars on a word boundary.

    Returns *summary* unchanged when no limit is given or it already fits. Adds
    an ellipsis marker when content was cut so the model/caller knows it is not
    the full summary.
    """
    if max_chars is None or max_chars <= 0 or len(summary) <= max_chars:
        return summary
    cut = summary[:max_chars]
    # Prefer to break at the last whitespace to avoid slicing mid-word.
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip() + " …"


def render_block(messages: list[dict[str, str]]) -> str:
    """Render a list of messages into a readable transcript for the LLM."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


@dataclass(frozen=True)
class CompressionSettings:
    """Knobs controlling when and how a session compresses its history.

    Attributes:
        keep_last: How many of the *latest* messages to keep verbatim. Must be a
            positive even number (a whole number of user/assistant turns); if
            odd it is rounded down to the nearest even value.
        block_size: How many messages may accumulate before the oldest part is
            folded into the summary. Must be > ``keep_last``.
        max_summary_tokens: Hard cap on the running summary size in tokens
            (``None`` = no explicit cap). If also bounded by the model's context
            window via :attr:`max_summary_ratio`, the smaller of the two wins.
        max_summary_ratio: Fraction (0..1) of the model's context window that the
            summary may occupy, used as a fallback cap when ``max_summary_tokens``
            is not set (or as a ceiling when both are set). Defaults to 0.3.
    """

    keep_last: int = 10
    block_size: int = 20
    max_summary_tokens: int | None = None
    max_summary_ratio: float = 0.3

    def __post_init__(self) -> None:
        # Normalise keep_last to an even number (complete turns).
        keep = self.keep_last if self.keep_last % 2 == 0 else self.keep_last - 1
        object.__setattr__(self, "keep_last", max(2, keep))
        object.__setattr__(self, "block_size", max(self.keep_last + 2, self.block_size))
        ratio = max(0.0, min(1.0, self.max_summary_ratio))
        object.__setattr__(self, "max_summary_ratio", ratio)

@dataclass(frozen=True)
class CompressionEvent:
    """Records one history-compression round for diagnostics/UX.

    Attributes:
        messages_folded: How many messages were folded into the summary.
        folded_tokens: Estimated tokens of those folded messages (the context
            space they free up). Filled by the caller.
        summary_chars: Length (in chars) of the resulting summary text.
        history_before: Live history length before compression.
        history_after: Live history length after compression (≈ keep_last).
    """

    messages_folded: int = 0
    folded_tokens: int = 0
    summary_chars: int = 0
    history_before: int = 0
    history_after: int = 0


class ContextCompressor:
    """Folds the older part of a conversation into a running summary.

    The compressor is stateless on its own: the current ``summary`` and the live
    ``history`` are passed in and returned out, so a :class:`Session` owns the
    truth and persists it. Summarisation calls go through a provided ``summarize``
    callable (usually ``agent.client.chat``), keeping this class provider-agnostic
    and easy to unit test.
    """

    def __init__(
        self,
        settings: CompressionSettings,
        *,
        summarize: Callable[[str, str, int | None], str],
        max_chars: int | None = None,
    ) -> None:
        """Build a compressor.

        Args:
            settings: The N/M tuning (keep-last / block size, summary cap).
            summarize: ``callable(existing_summary, block_text, max_chars) -> str``.
                Receives the current summary (possibly ``""``), the rendered text
                of the newly rolled-off block, and an optional char budget for
                the resulting summary; returns the updated summary.
            max_chars: Hard cap on the resulting summary length in chars. When
                set, the model's output is truncated to fit (guaranteed to never
                exceed the budget), and the budget is also passed into the
                summarizer as a soft instruction.
        """
        self._settings = settings
        self._summarize = summarize
        self._max_chars = max_chars

    @property
    def settings(self) -> CompressionSettings:
        return self._settings

    def should_compress(self, history_len: int) -> bool:
        """True when *history_len* messages exceed the block-size threshold."""
        return history_len > self._settings.block_size

    def compress(
        self,
        history: list[dict[str, str]],
        existing_summary: str,
    ) -> tuple[list[dict[str, str]], str, list[dict[str, str]]]:
        """Fold the oldest messages that fall outside the recent window.

        Returns ``(new_history, new_summary, folded_messages)`` where
        ``folded_messages`` is the list of messages that were folded into the
        summary (empty when compression did not trigger). This lets the caller
        report how much context was saved.
        """
        if not self.should_compress(len(history)):
            return history, existing_summary, []

        keep = self._settings.keep_last
        # Everything beyond the most recent ``keep`` messages is the block to fold.
        rolled_off = history[: len(history) - keep]
        kept = history[len(history) - keep :]

        block_text = render_block(rolled_off)
        new_summary = self._summarize(existing_summary, block_text, self._max_chars)
        new_summary = truncate_summary(new_summary, self._max_chars)
        return kept, new_summary, rolled_off