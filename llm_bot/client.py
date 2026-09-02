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


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient, retryable failure."""
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


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
    ) -> None:
        """Initialize the client.

        Args:
            config: Configuration; defaults to ``LLMConfig.from_env()``.
            transport: Optional custom transport (useful for tests with
                ``httpx.MockTransport``).
            token_provider: Optional source of the bearer access token. When given,
                it takes precedence over ``config.api_key``.
        """
        self.config = config or LLMConfig.from_env()
        # A timeout is required even when a custom transport is supplied.
        self._transport = transport
        self._timeout = httpx.Timeout(self.config.timeout)
        self._token_provider = token_provider

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

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.config.model,
            "messages": messages,
        }

    def _request(self, client: httpx.Client, prompt: str) -> dict[str, Any]:
        response = client.post(
            self.config.base_url.rstrip("/") + _CHAT_ENDPOINT,
            headers=self._build_headers(),
            json=self._build_payload(prompt),
        )
        if response.status_code == 429 or response.status_code >= 500:
            # Transient server/rate-limit error; caller decides whether to retry.
            raise _TransientHTTPError(response.status_code, response.text)
        if response.status_code >= 400:
            # Permanent client error (e.g. bad key, invalid model) — do not retry.
            raise LLMRequestError(
                f"LLM API returned HTTP {response.status_code}: {response.text}"
            )
        response.raise_for_status()
        return response.json()

    def _execute_with_retry(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(
                    transport=self._transport, timeout=self._timeout
                ) as client:
                    return self._request(client, prompt)
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
        data = self._execute_with_retry(prompt)
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract the assistant message text from a chat-completions response."""
        try:
            choices = data["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Unexpected response shape from LLM API: {data!r}") from exc
        if content is None:
            raise LLMRequestError(f"LLM API returned empty content: {data!r}")
        return content


class _TransientHTTPError(Exception):
    """Internal marker for retryable HTTP statuses (429, 5xx)."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"HTTP {self.status_code}"