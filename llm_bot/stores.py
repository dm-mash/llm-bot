"""Repository interfaces and configuration dataclasses.

This module defines the *contracts* for reading model credentials, agent
definitions and session histories. Business logic depends only on these
interfaces (via :class:`Protocol`), so the concrete storage backend (YAML/JSON
today, a database later) can be swapped without touching the rest of the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _resolve_env(value: str) -> str:
    """Resolve ``${VAR}`` placeholders in a config string from the environment.

    ``${VAR}`` is replaced by the value of the ``VAR`` environment variable. An
    undefined variable resolves to an empty string, so local providers that need
    no credentials work without setting anything. Dollar signs that are not part
    of a ``${...}`` placeholder are passed through unchanged.
    """
    if "${" not in value:
        return value
    resolved = value
    while "${" in resolved:
        start = resolved.find("${")
        end = resolved.find("}", start)
        if end == -1:
            break
        name = resolved[start + 2 : end]
        replacement = os.getenv(name, "")
        resolved = resolved[:start] + replacement + resolved[end + 1 :]
    return resolved


@dataclass(frozen=True)
class ModelConfig:
    """Connection credentials for a single LLM provider.

    This is transport-level information only — it says *how* to reach the API.
    Behavioural settings (system prompts, temperature, ...) live on the
    :class:`AgentConfig`.
    """

    name: str
    provider: str = "openai"  # "openai" | "gigachat"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # GigaChat OAuth2 credentials (only used when provider == "gigachat").
    client_id: str = ""
    client_secret: str = ""
    scope: str = "GIGACHAT_API_PERS"
    basic_auth: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModelConfig":
        """Build a :class:`ModelConfig`, resolving ``${VAR}`` placeholders."""
        return cls(
            name=name,
            provider=str(data.get("provider", "openai")),
            base_url=_resolve_env(str(data.get("base_url", ""))),
            api_key=_resolve_env(str(data.get("api_key", ""))),
            model=str(data.get("model", "")),
            client_id=_resolve_env(str(data.get("client_id", ""))),
            client_secret=_resolve_env(str(data.get("client_secret", ""))),
            scope=str(data.get("scope", "GIGACHAT_API_PERS")),
            basic_auth=str(data.get("basic_auth", "")),
        )


@dataclass(frozen=True)
class AgentConfig:
    """Behavioural definition of an agent.

    An agent describes *what* to say (system prompt, generation settings) and
    *which* model to use (a reference to a :class:`ModelConfig` by name). It does
    NOT hold conversation history — history belongs to a :class:`Session`.
    """

    name: str
    model: str  # name of the referenced ModelConfig
    system_prompt: str = ""
    default_system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    max_response_words: int | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "AgentConfig":
        """Build an :class:`AgentConfig` from a raw YAML/dict entry."""
        temperature = data.get("temperature")
        max_tokens = data.get("max_tokens")
        max_response_words = data.get("max_response_words")
        return cls(
            name=name,
            model=str(data["model"]),
            system_prompt=str(data.get("system_prompt", "")),
            default_system_prompt=str(data.get("default_system_prompt", "")),
            temperature=_opt_float(temperature),
            max_tokens=_opt_int(max_tokens),
            max_response_words=_opt_int(max_response_words),
        )

    def effective_system_prompt(self) -> str:
        """Assemble the full system prompt from default + specific + brevity hint."""
        parts = [
            self.default_system_prompt.strip(),
            self.system_prompt.strip(),
        ]
        if self.max_response_words is not None:
            parts.append(
                f"Важно: отвечай кратко — не более примерно "
                f"{self.max_response_words} слов."
            )
        return "\n\n".join(p for p in parts if p)


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@runtime_checkable
class ModelStore(Protocol):
    """Read access to LLM model/provider credentials by name."""

    def get(self, name: str) -> ModelConfig: ...
    def list(self) -> list[str]: ...


@runtime_checkable
class AgentStore(Protocol):
    """Read access to agent definitions by name."""

    def get(self, name: str) -> AgentConfig: ...
    def list(self) -> list[str]: ...


@runtime_checkable
class SessionStore(Protocol):
    """Persistent storage for session (conversation) histories."""

    def load(self, session_id: str) -> list[dict[str, str]]: ...
    def save(self, session_id: str, history: list[dict[str, str]]) -> None: ...
    def list(self) -> list[str]: ...