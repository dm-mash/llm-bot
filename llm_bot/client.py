"""Provider-agnostic LLM client service layer.

This module is the heart of the application. ``LLMClient`` speaks the
OpenAI-compatible ``/chat/completions`` protocol, so it works with OpenAI,
Ollama, LM Studio, LocalAI, and other compatible servers just by changing
the ``base_url`` and ``model`` in the config.

Keeping this logic in a standalone class means both the console entry point
(``llm_bot.cli``) and a future web interface can share the same code without
duplication.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from llm_bot.config import LLMConfig
from llm_bot.diagnostics import (
    DetailListener,
    RequestDetails,
    ResponseDetails,
    extract_usage,
)

logger = logging.getLogger(__name__)

# HTTP status codes that indicate a transient problem worth retrying.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Chat-completions endpoint path appended to the configured base URL.
_CHAT_ENDPOINT = "/chat/completions"


class LLMError(Exception):
    """Base exception for all LLM client errors."""


class LLMRequestError(LLMError):
    """Raised when a request fails with a non-retryable error."""


class LLMRetryExhaustedError(LLMRequestError):
    """Raised when all retries are exhausted for a transient failure."""


class LLMTruncatedError(LLMRequestError):
    """Raised when the model stopped because it hit the token limit.

    A reasoning model may fill its entire completion budget with chain-of-thought
    and be cut off (``finish_reason="length"``) before any visible ``content`` is
    produced. We surface this explicitly instead of returning a blank reply.
    """


class ContextOverflowError(LLMError):
    """Raised before a request is sent when the dialog exceeds the context window.

    The agent counts the assembled history and refuses to talk to the model if it
    would overflow the configured ``context_window``. This prevents the provider
    from rejecting (or silently truncating) an oversized request, and is the
    point at which the caller should trim the history or start a new session.

    Attributes:
        context_tokens: The estimated token count of the request that was blocked.
        context_window: The model's configured context window in tokens.
    """

    def __init__(
        self,
        message: str,
        *,
        context_tokens: int,
        context_window: int,
    ) -> None:
        super().__init__(message)
        self.context_tokens = context_tokens
        self.context_window = context_window


class ContextTooLargeError(LLMRequestError):
    """Raised when the provider refuses a request that exceeds a size ceiling.

    Distinct from :class:`ContextOverflowError` in one important way:
    ``ContextOverflowError`` is raised *before* a request is sent, from our own
    token estimate. ``ContextTooLargeError`` is raised when the provider itself
    rejects the request (e.g. an HTTP 413 "request too large") because it exceeds
    a hard per-request token ceiling — such as a TPM *size* cap on Groq's
    ``on_demand`` tier, which refuses any single request larger than the limit
    regardless of how full the rate-limit bucket is.

    This is a **permanent** condition for the given request: retrying with the
    same payload can never succeed. The only remedies are to shrink the request
    (trim history, shorten the prompt) or raise the account tier.

    Attributes:
        status_code: The HTTP status returned by the provider (e.g. 413).
        requested_tokens: The size of the rejected request in tokens, if known.
        limit_tokens: The provider's per-request ceiling in tokens, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        requested_tokens: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.requested_tokens = requested_tokens
        self.limit_tokens = limit_tokens


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient, retryable failure."""
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _classify_request_too_large(
    status_code: int,
    text: str,
) -> ContextTooLargeError | None:
    """Classify a client error as a hard per-request size ceiling, or None.

    Some providers (e.g. Groq on the ``on_demand`` tier) reject any single
    request larger than their token-per-minute cap even with a full rate-limit
    bucket. The response is an HTTP 413 whose body reports the requested size
    against the limit, e.g.::

        Request too large ... Limit 8000, Requested 8104 ...

    Such a request can **never** succeed on retry, because the payload itself is
    above the ceiling — shrinking it is the only remedy. This helper recognises
    that shape and returns a :class:`ContextTooLargeError`; otherwise it returns
    ``None`` so the caller falls back to the generic error path.
    """
    if status_code != 413:
        return None

    requested = _extract_size(text, "Requested")
    limit = _extract_size(text, "Limit")
    if requested is None or limit is None or requested <= limit:
        # Either the sizes aren't reported, or the request fits the ceiling and
        # the 413 was caused by something else (e.g. a genuinely oversized body).
        return None

    message = (
        f"Запрос превышает жёсткий лимит размера: {requested} токенов при "
        f"лимите {limit}. Сократите сообщение или начните новый сеанс — "
        f"повторная отправка того же запроса не поможет."
    )
    return ContextTooLargeError(
        message,
        status_code=status_code,
        requested_tokens=requested,
        limit_tokens=limit,
    )


def _extract_size(text: str, label: str) -> int | None:
    """Return the integer following *label* in *text* (e.g. ``Requested 8104``)."""
    marker = f"{label} "
    idx = text.find(marker)
    if idx == -1:
        return None
    token = text[idx + len(marker):].split(",")[0].strip()
    try:
        return int(token)
    except ValueError:
        return None


@runtime_checkable
class TokenProvider(Protocol):
    """Provides a bearer access token to be sent as ``Authorization: Bearer ...``.

    A pluggable source of auth tokens lets providers that need an OAuth2 exchange
    (e.g. GigaChat) reuse :class:`LLMClient` unchanged.
    """

    def get_token(self) -> str:
        """Return the current valid access token, refreshing it if necessary."""
        ...


class LLMClient:
    """A small, provider-agnostic client for chat-completions APIs.

    The client is lightweight and reusable: it is safe to create one per request
    (as the console does) or keep a long-lived instance (as a web app would).

    Authentication is resolved by priority:
        1. A provided ``token_provider`` (bearer token obtained externally).
        2. Otherwise the static ``config.api_key`` (also sent as a bearer token).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        token_provider: TokenProvider | None = None,
        detail_listener: DetailListener | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Configuration; defaults to ``LLMConfig.from_env()``.
            transport: Optional custom transport (useful for tests with
                ``httpx.MockTransport``).
            token_provider: Optional source of the bearer access token. When given,
                it takes precedence over ``config.api_key``.
            detail_listener: Optional consumer of structured request/response
                details (URL, model, payload, token usage). Kept interface-agnostic
                so a CLI formatter and a future web handler can both use it.
        """
        self.config = config or LLMConfig.from_env()
        # A timeout is required even when a custom transport is supplied.
        self._transport = transport
        self._timeout = httpx.Timeout(self.config.timeout)
        self._token_provider = token_provider
        self._detail_listener = detail_listener
        # Token usage reported by the provider for the most recent successful call,
        # or None if the provider does not report usage. Updated by chat/send_prompt.
        self.last_usage: dict[str, int] | None = None

    def _resolve_token(self) -> str:
        """Return the bearer token to use, or ``""`` if none is configured."""
        if self._token_provider is not None:
            return self._token_provider.get_token()
        return self.config.api_key

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self._resolve_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _build_system_prompt(self) -> str:
        """Return the effective system prompt for this request.

        The effective prompt is assembled from three optional parts, joined with a
        blank line when more than one is present:

            1. ``default_system_prompt``  — always prepended when set (global
               behavior such as "reply in the user's language").
            2. ``system_prompt``          — the request-specific instruction.
            3. max_response_words hint    — an instruction to keep the reply brief.

        The brevity hint is expressed as a prompt instruction rather than an
        API-level token cap (``max_tokens``), because providers handle
        ``max_tokens`` inconsistently (some truncate, others return an empty
        response).
        """
        parts = [
            self.config.default_system_prompt.strip(),
            self.config.system_prompt.strip(),
        ]
        if self.config.max_response_words is not None:
            parts.append(
                f"Важно: отвечай кратко — не более примерно "
                f"{self.config.max_response_words} слов."
            )
        non_empty = [p for p in parts if p]
        return "\n\n".join(non_empty)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Assemble the message stack for a single-prompt request.

        Kept for backward compatibility with :meth:`send_prompt`. The system
        prompt is prepended here so callers who pass a raw prompt still get the
        configured system behavior.
        """
        messages: list[dict[str, str]] = []
        system_prompt = self._build_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        return payload

    def _build_url(self) -> str:
        """Return the full chat-completions URL for the configured base URL."""
        return self.config.base_url.rstrip("/") + _CHAT_ENDPOINT

    def _emit_request(self, url: str, payload: dict[str, Any]) -> None:
        """Notify the detail listener (if any) that a request is about to be sent."""
        if self._detail_listener is None:
            return
        details = RequestDetails(
            method="POST",
            url=url,
            model=self.config.model,
            payload=payload,
        )
        self._detail_listener.on_request(details)

    def _emit_response(self, status_code: int, attempt: int, elapsed_ms: float, data: dict[str, Any]) -> None:
        """Notify the detail listener (if any) of one HTTP response attempt."""
        if self._detail_listener is None:
            return
        details = ResponseDetails(
            status_code=status_code,
            usage=extract_usage(data),
            elapsed_ms=round(elapsed_ms, 1),
            attempt=attempt,
            body=data,
        )
        self._detail_listener.on_response(details)

    def _request(
        self,
        client: httpx.Client,
        url: str,
        payload: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        response = client.post(
            url,
            headers=self._build_headers(),
            json=payload,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if response.status_code == 429 or response.status_code >= 500:
            # Transient server/rate-limit error; caller decides whether to retry.
            self._emit_response(response.status_code, attempt, elapsed_ms, {})
            raise _TransientHTTPError(response.status_code, response.text)
        if response.status_code >= 400:
            # Permanent client error (e.g. bad key, invalid model) — do not retry.
            self._emit_response(response.status_code, attempt, elapsed_ms, {})
            too_large = _classify_request_too_large(
                response.status_code, response.text
            )
            if too_large is not None:
                raise too_large
            raise LLMRequestError(
                f"LLM API returned HTTP {response.status_code}: {response.text}"
            )
        response.raise_for_status()
        data = response.json()
        self._emit_response(response.status_code, attempt, elapsed_ms, data)
        return data

    def _execute_with_retry(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = self._build_url()
        payload = self._build_payload(messages)
        self._emit_request(url, payload)
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(
                    transport=self._transport, timeout=self._timeout
                ) as client:
                    return self._request(client, url, payload, attempt + 1)
            except _TransientHTTPError as exc:
                last_error = exc
                logger.warning("Transient HTTP error %s (attempt %d)", exc, attempt + 1)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning("Transient network error %s (attempt %d)", exc, attempt + 1)
            except LLMRequestError:
                # Permanent error — do not retry.
                raise

            if attempt < self.config.max_retries:
                self._sleep_between_retries(attempt)

        raise LLMRetryExhaustedError(
            f"LLM request failed after {self.config.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _sleep_between_retries(self, attempt: int) -> None:
        """Exponential backoff: backoff * 2^attempt seconds."""
        delay = self.config.retry_backoff * (2**attempt)
        logger.info("Retrying in %.2fs...", delay)
        time.sleep(delay)

    def send_prompt(self, prompt: str) -> str:
        """Send a prompt and return the text of the model's reply.

        Args:
            prompt: The user prompt to send.

        Returns:
            The assistant's response text.

        Raises:
            LLMRequestError: On permanent (non-retryable) failures.
            LLMRetryExhaustedError: When transient failures persist past max_retries.
        """
        data = self._execute_with_retry(self._build_messages(prompt))
        self.last_usage = extract_usage(data)
        return self._extract_text(data)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a pre-built message stack and return the text of the model's reply.

        Unlike :meth:`send_prompt`, this accepts a fully formed ``messages`` list
        (e.g. an agent-managed conversation history) and does not inject any
        system prompt itself. Callers that need a system prompt must include it
        as the first message.

        Args:
            messages: The complete list of ``{"role", "content"}`` messages to
                send, in order.

        Returns:
            The assistant's response text.

        Raises:
            LLMRequestError: On permanent (non-retryable) failures.
            LLMRetryExhaustedError: When transient failures persist past max_retries.
        """
        data = self._execute_with_retry(messages)
        self.last_usage = extract_usage(data)
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract the assistant message text from a chat-completions response.

        Some reasoning models (e.g. gpt-oss-120b on Groq) put their chain-of-thought
        in a separate field (``reasoning_content`` / ``reasoning``) and leave
        ``content`` empty or ``None``.

        A ``finish_reason="length"`` always means the model hit its token limit and
        the output was cut off — regardless of whether there is partial text in
        ``content`` or only chain-of-thought in a ``reasoning*`` field. We raise
        :class:`LLMTruncatedError` so the caller knows generation was interrupted,
        reporting the partial-content size (if any) and the reasoning size (if any).

        Otherwise we fall back to a ``reasoning*`` field when ``content`` is empty,
        so a blank reply is not silently returned.
        """
        try:
            choices = data["choices"]
            choice = choices[0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Unexpected response shape from LLM API: {data!r}") from exc

        content = message.get("content")
        reasoning = next(
            (
                candidate
                for key, candidate in message.items()
                if "reason" in key.lower()
                and isinstance(candidate, str)
                and candidate.strip()
            ),
            None,
        )

        # finish_reason='length' => generation cut off by the token limit.
        if finish_reason == "length":
            details = []
            if isinstance(content, str) and content.strip():
                details.append(f"неполный ответ в content ({len(content)} chars)")
            if reasoning:
                details.append(f"reasoning {len(reasoning)} chars")
            suffix = (": " + "; ".join(details)) if details else ""
            raise LLMTruncatedError(
                "Генерация прервана из-за превышения лимита токенов"
                f" (finish_reason='length'){suffix}"
            )

        if isinstance(content, str) and content.strip():
            return content

        # Fallback: prefer the first reasoning-style field that has visible text.
        for key in sorted(message):
            if "reason" in key.lower():
                candidate = message.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate

        raise LLMRequestError(f"LLM API returned empty content: {data!r}")


class _TransientHTTPError(Exception):
    """Internal marker for retryable HTTP statuses (429, 5xx)."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"HTTP {self.status_code}"