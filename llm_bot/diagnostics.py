"""Structured diagnostics for LLM requests.

``LLMClient`` emits these events so any interface (the console today, a web app
later) can surface the same information — which URL the request goes to, which
model is used, how the request body looks, and (when the provider reports it)
token usage.

This module deliberately contains no rendering/UI code: consumers plug their own
formatter into the :class:`DetailListener` hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# The fields of a chat-completions ``usage`` object that are most commonly
# reported by providers. We surface them verbatim; unknown/extra keys are kept too.
_TOKEN_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def extract_usage(data: dict[str, Any]) -> dict[str, int] | None:
    """Extract token usage from a chat-completions response.

    Not all providers report usage, so this returns ``None`` when it is absent or
    not shaped as expected.
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    known = {
        key: value
        for key, value in usage.items()
        if key in _TOKEN_USAGE_KEYS and isinstance(value, int)
    }
    return known if known else None


@dataclass(frozen=True)
class RequestDetails:
    """Information about a request just before it is sent.

    Attributes:
        method: HTTP method (always ``POST`` for chat completions).
        url: The full URL the request is sent to.
        model: The model identifier used for the request.
        payload: The JSON request body as it was formed and sent.
    """

    method: str
    url: str
    model: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseDetails:
    """Information about one HTTP response attempt.

    Attributes:
        status_code: The HTTP status code returned by the API.
        usage: Parsed token usage, or ``None`` if the provider did not report it.
        elapsed_ms: Round-trip time of the attempt in milliseconds.
        attempt: 1-based index of this attempt (1 on the first try, 2 on the
            first retry, and so on).
        body: The raw parsed JSON body of the response, or ``None`` when no body
            was available (e.g. transient/error attempts emitted with an empty dict).
    """

    status_code: int
    usage: dict[str, int] | None = None
    elapsed_ms: float | None = None
    attempt: int = 1
    body: dict[str, Any] | None = None


@runtime_checkable
class DetailListener(Protocol):
    """Receives structured diagnostics from an :class:`~llm_bot.client.LLMClient`.

    A console formatter implements this to print details to stderr; a web app
    could implement it to store or return the metadata alongside a reply.
    """

    def on_request(self, details: RequestDetails) -> None:
        """Called once per request, right before it is sent."""
        ...

    def on_response(self, details: ResponseDetails) -> None:
        """Called after each HTTP attempt (including transient/error ones)."""
        ...