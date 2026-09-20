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

from llm_bot.compress import CompressionSettings


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
    # Maximum input context size (tokens) this model supports. Used by the agent
    # to detect when a dialog overflows the budget before sending a request.
    context_window: int | None = None
    # Hard per-request token ceiling enforced by the account tier (often smaller
    # than the model's context window, e.g. Groq's TPM size cap). Requests above
    # this are refused even with a full rate-limit bucket, so the agent treats it
    # as an extra budget.
    max_request_tokens: int | None = None
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
            context_window=_opt_int(data.get("context_window")),
            max_request_tokens=_opt_int(data.get("max_request_tokens")),
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

    Context compression can be enabled per-agent via ``keep_last_messages`` and
    ``summarize_messages_threshold`` (see :attr:`compression_settings`).
    """

    name: str
    model: str  # name of the referenced ModelConfig
    system_prompt: str = ""
    default_system_prompt: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    max_response_words: int | None = None
    # Context compression: N (how many recent messages stay verbatim) and M (the
    # block size that triggers folding the oldest part into a summary). Both must
    # be set (and M > N) for compression to be active.
    keep_last_messages: int | None = None
    summarize_messages_threshold: int | None = None
    # Cap on the running summary size. max_summary_tokens is an explicit token
    # limit; max_summary_ratio caps the summary as a fraction of the model's
    # context window (fallback/ceiling when the token cap is absent). Both are
    # optional; without either the summary is unbounded.
    max_summary_tokens: int | None = None
    max_summary_ratio: float | None = None
    # Pluggable context-management strategy (without summary). One of:
    #   * "sliding"   — only the last ``context_window_messages`` messages are sent;
    #   * "facts"     — a durable key/value facts block + the last
    #                   ``context_window_messages`` messages;
    #   * "branching" — fork the dialog into independent lines.
    # When unset, the agent falls back to its compression settings (if any), else
    # to the full-history behaviour.
    context_strategy: str | None = None
    # Task state machine: track the formal stage (planning/execution/validation/
    # done + paused) of the work being done in the session and inject it into the
    # system context on every turn. See :mod:`llm_bot.task_state`.
    task_state: bool = False
    # When the task state machine is on, auto-detect task setup / stage hints /
    # pause & resume phrases from the dialog after each turn via a small LLM
    # call (mirrors the memory auto-extraction behaviour).
    task_auto_detect: bool = True
    # How many of the most recent MESSAGES the strategy keeps in the request
    # (sliding window size N) for the "sliding"/"facts" strategies (and, when set,
    # as a cap for "branching"). Measured in messages, not tokens.
    context_window_messages: int | None = None
    context_max_facts: int | None = None     # cap for the "facts" strategy

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
            keep_last_messages=_opt_int(data.get("keep_last_messages")),
            summarize_messages_threshold=_opt_int(
                data.get("summarize_messages_threshold")
            ),
            max_summary_tokens=_opt_int(data.get("max_summary_tokens")),
            max_summary_ratio=_opt_float(data.get("max_summary_ratio")),
            context_strategy=_opt_str(data.get("context_strategy")),
            context_window_messages=_opt_int(data.get("context_window_messages")),
            context_max_facts=_opt_int(data.get("context_max_facts")),
            task_state=_opt_bool(data.get("task_state"), default=False),
            task_auto_detect=_opt_bool(
                data.get("task_auto_detect"), default=True
            ),
        )

    @property
    def compression_settings(self) -> CompressionSettings | None:
        """Compression settings when configured, or ``None`` when disabled.

        Compression is active only when both ``keep_last_messages`` and
        ``summarize_messages_threshold`` are set (and the threshold exceeds the
        keep-last count, which :class:`CompressionSettings` normalises anyway).
        """
        if (
            self.keep_last_messages is None
            or self.summarize_messages_threshold is None
        ):
            return None
        return CompressionSettings(
            keep_last=self.keep_last_messages,
            block_size=self.summarize_messages_threshold,
            max_summary_tokens=self.max_summary_tokens,
            max_summary_ratio=(
                self.max_summary_ratio
                if self.max_summary_ratio is not None
                else CompressionSettings.max_summary_ratio
            ),
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


def _opt_str(value: Any) -> str | None:
    """Return a non-empty stripped string, or ``None`` for missing/blank values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_bool(value: Any, *, default: bool) -> bool:
    """Interpret a YAML value as a bool, falling back to *default*.

    Accepts real booleans and the usual string forms ("true"/"yes"/"on"/"1").
    Missing/blank values fall back to *default*.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "on", "1"}


@dataclass(frozen=True)
class ProfileConfig:
    """Orchestration-level personalization: HOW the agent behaves for a user.

    A profile is static configuration (declared in ``data/profiles.yaml``), NOT
    memory: it never stores learned dialog data. It is composed onto an
    :class:`AgentConfig` at composition time via
    :func:`llm_bot.profiles.apply_profile`.

    Attributes:
        name: Profile key (same as the YAML mapping key).
        style: Communication style (``formal`` / ``casual`` / ``technical`` /
            ``friendly`` / ``professional``).
        format: Response format (``concise`` / ``detailed`` / ``structured`` /
            ``conversational`` / ``bullet_points``).
        expertise: User expertise level (``beginner`` / ``intermediate`` /
            ``expert``).
        language: Response language (ISO code or name). Empty = no override.
        max_response_words: Brevity cap in words (overrides the agent's value
            when set).
        temperature: Generation temperature override (when set).
        extra_instructions: Free-form additional directives.
        forbidden_topics: Topics the assistant must not discuss.
        interests: User interests, used to tailor examples.
    """

    name: str
    style: str = ""
    format: str = ""
    expertise: str = ""
    language: str = ""
    max_response_words: int | None = None
    temperature: float | None = None
    extra_instructions: str = ""
    forbidden_topics: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ProfileConfig":
        """Build a :class:`ProfileConfig` from a raw YAML/dict entry."""
        raw_topics = data.get("forbidden_topics") or []
        raw_interests = data.get("interests") or []
        return cls(
            name=name,
            style=str(data.get("style", "")).strip(),
            format=str(data.get("format", "")).strip(),
            expertise=str(data.get("expertise", "")).strip(),
            language=str(data.get("language", "")).strip(),
            max_response_words=_opt_int(data.get("max_response_words")),
            temperature=_opt_float(data.get("temperature")),
            extra_instructions=str(data.get("extra_instructions", "")).strip(),
            forbidden_topics=[str(t) for t in raw_topics if str(t).strip()],
            interests=[str(i) for i in raw_interests if str(i).strip()],
        )


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
class ProfileStore(Protocol):
    """Read access to personalization profile definitions by name."""

    def get(self, name: str) -> ProfileConfig: ...
    def list(self) -> list[str]: ...


@runtime_checkable
class SessionStore(Protocol):
    """Persistent storage for session (conversation) histories.

    In addition to the live ``history`` (a list of messages), a store may persist
    a context-compression ``summary`` string alongside it. The summary replaces
    the older part of the dialog that has been folded away by the compressor (see
    :mod:`llm_bot.compress`). Stores that do not support summaries can rely on
    the default implementations below, which treat the summary as permanently
    empty.
    """

    def load(self, session_id: str) -> list[dict[str, str]]:
        """Return the live message history for *session_id* (without the summary)."""
        ...

    def save(self, session_id: str, history: list[dict[str, str]]) -> None:
        """Persist *history*, preserving any previously stored summary."""
        ...

    def load_summary(self, session_id: str) -> str:
        """Return the persisted compression summary (``""`` when there is none)."""
        return ""

    def save_full(
        self,
        session_id: str,
        history: list[dict[str, str]],
        *,
        summary: str,
    ) -> None:
        """Persist both the live *history* and the compressed *summary* together."""
        # Default: stores without summary support just keep the live history.
        self.save(session_id, history)

    # --- Pluggable context strategies (sticky facts / branches) ----------- #
    # These are used by the context strategies in :mod:`llm_bot.context_strategies`.
    # Stores that do not support them can rely on the defaults below (permanently
    # empty facts / a single default branch), so a plain store keeps working even
    # when a session uses a strategy.

    def load_facts(self, session_id: str) -> dict[str, str]:
        """Return the persisted sticky-facts block (``{}`` when there is none)."""
        return {}

    def save_facts(self, session_id: str, facts: dict[str, str]) -> None:
        """Persist the sticky-facts block for a session (no-op by default)."""
        # Keep any existing state; the default store does not persist facts.
        del facts

    def load_branches(
        self, session_id: str, *, default_branch: str
    ) -> tuple[dict[str, list[dict[str, str]]], str]:
        """Return ``(branches, current)``; defaults to a single ``default_branch``."""
        return {default_branch: self.load(session_id)}, default_branch

    def save_branches(
        self,
        session_id: str,
        branches: dict[str, list[dict[str, str]]],
        current: str,
    ) -> None:
        """Persist the branching state (no-op by default)."""
        # Without branch support we persist only the active branch's history.
        if current in branches:
            self.save(session_id, branches[current])

    # --- Task state machine ------------------------------------------------ #
    # Used by :class:`llm_bot.task_state.TaskStateMachine` so the formal task
    # stage/step/expected-action survive process restarts ("continue without
    # re-explaining"). Stores that do not support it can rely on the defaults
    # below (permanently absent), so a plain store keeps working.

    def load_task_state(self, session_id: str) -> dict[str, Any] | None:
        """Return the persisted task-state snapshot (``None`` when there is none)."""
        return None

    def save_task_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist the task-state snapshot for a session (no-op by default)."""
        del state

    def list(self) -> list[str]: ...