"""GigaChat (Sber) integration.

GigaChat exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, but its
authorization is a two-step process:

1. Obtain an OAuth2 **access token** by exchanging your credentials
   (``client_id`` + ``client_secret``) via the ``client_credentials`` grant at
   the token endpoint.
2. Send that token as ``Authorization: Bearer <token>`` to the chat endpoint.

The token is short-lived (~30 min), so :class:`GigaChatTokenProvider` caches it
and refreshes it automatically before expiry. Because it implements the
:class:`~llm_bot.client.TokenProvider` protocol, it plugs straight into
:class:`~llm_bot.client.LLMClient` with no changes to the request logic.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from typing import Any, Callable

import httpx

from llm_bot.client import LLMClient, LLMRequestError
from llm_bot.config import LLMConfig
from llm_bot.diagnostics import DetailListener

logger = logging.getLogger(__name__)

# Refresh the token this many seconds before it actually expires.
_REFRESH_EARLY_SECONDS = 60.0

# If the API does not return an expiry, assume this lifetime.
_DEFAULT_TOKEN_LIFETIME_SECONDS = 1800.0

# Milisecond epoch timestamps are ~13 digits; seconds ~10 digits.
_MS_EPOCH_THRESHOLD = 10**12


def _oauth_authorization(config: LLMConfig) -> str:
    """Build the ``Authorization`` header for the OAuth token request.

    Prefers an explicitly configured ``gigachat_basic_auth`` value; otherwise
    constructs ``Basic base64(client_id:client_secret)``.
    """
    if config.gigachat_basic_auth:
        return config.gigachat_basic_auth
    raw = f"{config.gigachat_client_id}:{config.gigachat_client_secret}"
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def _parse_expires_at(data: dict[str, Any], now: float) -> float:
    """Determine token expiry (epoch seconds) from the OAuth response."""
    if "expires_at" in data:
        try:
            value = float(data["expires_at"])
            # Convert milliseconds to seconds if needed.
            if value > _MS_EPOCH_THRESHOLD:
                value = value / 1000.0
            return value
        except (TypeError, ValueError):
            pass
    if "expires_in" in data:
        try:
            return now + float(data["expires_in"])
        except (TypeError, ValueError):
            pass
    return now + _DEFAULT_TOKEN_LIFETIME_SECONDS


class GigaChatTokenProvider:
    """Caches and refreshes a GigaChat OAuth2 access token.

    Implements the :class:`~llm_bot.client.TokenProvider` protocol, so it can be
    passed directly to :class:`~llm_bot.client.LLMClient`.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the provider.

        Args:
            config: Configuration carrying GigaChat OAuth credentials.
            transport: Optional custom transport (useful for tests).
            time_fn: Callable returning current epoch seconds (defaults to ``time.time``).
        """
        self.config = config
        self._transport = transport
        self._time_fn = time_fn
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        """Return a valid access token, fetching/refreshing if needed."""
        now = self._time_fn()
        if self._access_token and now < self._expires_at - _REFRESH_EARLY_SECONDS:
            return self._access_token
        self._access_token, self._expires_at = self._fetch_token()
        return self._access_token

    def _fetch_token(self) -> tuple[str, float]:
        """Perform the OAuth2 ``client_credentials`` exchange and return token + expiry."""
        if not self.config.gigachat_client_secret and not self.config.gigachat_basic_auth:
            raise LLMRequestError(
                "GigaChat client_secret is not configured (GIGACHAT_CLIENT_SECRET)."
            )

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": _oauth_authorization(self.config),
        }
        body = {
            "scope": self.config.gigachat_scope,
            "grant_type": "client_credentials",
        }

        timeout = httpx.Timeout(self.config.timeout)
        with httpx.Client(transport=self._transport, timeout=timeout) as client:
            response = client.post(self.config.gigachat_oauth_url, headers=headers, data=body)

        if response.status_code >= 400:
            raise LLMRequestError(
                f"GigaChat OAuth failed with HTTP {response.status_code}: {response.text}"
            )
        try:
            payload = response.json()
            token = payload["access_token"]
        except (KeyError, ValueError) as exc:
            raise LLMRequestError(
                f"Unexpected GigaChat OAuth response: {response.text!r}"
            ) from exc

        expires_at = _parse_expires_at(payload, self._time_fn())
        logger.info("GigaChat token acquired, expires at %.0f", expires_at)
        return token, expires_at


def build_gigachat_client(
    config: LLMConfig | None = None,
    *,
    detail_listener: DetailListener | None = None,
) -> LLMClient:
    """Build an :class:`~llm_bot.client.LLMClient` configured for GigaChat.

    Args:
        config: Configuration; defaults to ``LLMConfig.from_env()``. The
            ``base_url`` should point at GigaChat's OpenAI-compatible endpoint
            (default ``https://gigachat.devices.sberbank.ru/api/v1``).
        detail_listener: Optional consumer of structured request/response details
            (URL, model, payload, token usage); forwarded to the client.

    Returns:
        A fully wired :class:`~llm_bot.client.LLMClient` that obtains and
        refreshes GigaChat tokens automatically.
    """
    config = config or LLMConfig.from_env()
    token_provider = GigaChatTokenProvider(config)
    return LLMClient(config, token_provider=token_provider, detail_listener=detail_listener)