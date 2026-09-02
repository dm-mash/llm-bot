"""Application configuration.

``LLMConfig`` centralizes all settings needed to talk to the LLM API. Values are
read from environment variables (with optional ``.env`` support via python-dotenv),
so the exact same configuration object can be reused by the console entry point
today and a web interface later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load variables from a local .env file if present (no-op if the file is missing).
load_dotenv()

# Environment variable names used by this module.
_ENV_BASE_URL = "LLM_BASE_URL"
_ENV_API_KEY = "LLM_API_KEY"
_ENV_MODEL = "LLM_MODEL"
_ENV_MAX_RETRIES = "LLM_MAX_RETRIES"
_ENV_RETRY_BACKOFF = "LLM_RETRY_BACKOFF"
_ENV_TIMEOUT = "LLM_TIMEOUT"
_ENV_SYSTEM_PROMPT = "LLM_SYSTEM_PROMPT"

# GigaChat (Sber) OAuth2 client_credentials settings.
_ENV_GIGACHAT_OAUTH_URL = "GIGACHAT_OAUTH_URL"
_ENV_GIGACHAT_CLIENT_ID = "GIGACHAT_CLIENT_ID"
_ENV_GIGACHAT_CLIENT_SECRET = "GIGACHAT_CLIENT_SECRET"
_ENV_GIGACHAT_SCOPE = "GIGACHAT_SCOPE"
_ENV_GIGACHAT_BASIC_AUTH = "GIGACHAT_BASIC_AUTH"


def _get_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to *default* on absence/invalid value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float env var, falling back to *default* on absence/invalid value."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM API.

    Attributes:
        base_url: Base URL of the chat completions endpoint (e.g. ``https://api.openai.com/v1``).
        api_key: API key; may be empty for local providers that need no auth.
        model: Model identifier to use.
        max_retries: How many times to retry transient failures.
        retry_backoff: Base delay (seconds) for exponential backoff between retries.
        timeout: Request timeout in seconds.
        system_prompt: Optional system prompt sent as a ``system`` message before
            the user prompt (e.g. to request a specific JSON response schema).
    """

    base_url: str = field(default_factory=lambda: os.getenv(_ENV_BASE_URL, "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.getenv(_ENV_API_KEY, ""))
    model: str = field(default_factory=lambda: os.getenv(_ENV_MODEL, "gpt-4o-mini"))
    max_retries: int = field(default_factory=lambda: _get_int(_ENV_MAX_RETRIES, 3))
    retry_backoff: float = field(default_factory=lambda: _get_float(_ENV_RETRY_BACKOFF, 1.0))
    timeout: float = field(default_factory=lambda: _get_float(_ENV_TIMEOUT, 30.0))
    system_prompt: str = field(default_factory=lambda: os.getenv(_ENV_SYSTEM_PROMPT, ""))

    # --- GigaChat (Sber) OAuth2 client_credentials settings ---
    gigachat_oauth_url: str = field(
        default_factory=lambda: os.getenv(
            _ENV_GIGACHAT_OAUTH_URL,
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        )
    )
    gigachat_client_id: str = field(default_factory=lambda: os.getenv(_ENV_GIGACHAT_CLIENT_ID, ""))
    gigachat_client_secret: str = field(
        default_factory=lambda: os.getenv(_ENV_GIGACHAT_CLIENT_SECRET, "")
    )
    gigachat_scope: str = field(
        default_factory=lambda: os.getenv(_ENV_GIGACHAT_SCOPE, "GIGACHAT_API_PERS")
    )
    # Optional pre-encoded "Basic ..." value if a provider variant needs a different credential.
    gigachat_basic_auth: str = field(default_factory=lambda: os.getenv(_ENV_GIGACHAT_BASIC_AUTH, ""))

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build an :class:`LLMConfig` from the current environment (and .env file)."""
        return cls()

    def with_overrides(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> "LLMConfig":
        """Return a copy of this config with any provided fields overridden.

        Useful for CLI flags (``--base-url``, ``--model``, etc.) without mutating
        the original object.
        """
        return LLMConfig(
            base_url=base_url or self.base_url,
            api_key=self.api_key if api_key is None else api_key,
            model=model or self.model,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            timeout=self.timeout,
            system_prompt=self.system_prompt if system_prompt is None else system_prompt,
            gigachat_oauth_url=self.gigachat_oauth_url,
            gigachat_client_id=self.gigachat_client_id,
            gigachat_client_secret=self.gigachat_client_secret,
            gigachat_scope=self.gigachat_scope,
            gigachat_basic_auth=self.gigachat_basic_auth,
        )