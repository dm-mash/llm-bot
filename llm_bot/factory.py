"""Wiring: assemble :class:`Agent` and :class:`Session` from storage backends.

This module is the composition root. It maps a :class:`ModelConfig` onto an
:class:`~llm_bot.config.LLMConfig` (transport + auth), applies the agent's
generation settings (temperature / max_tokens), builds the right client (plain
or GigaChat), and hands back a ready-to-use :class:`Session`.

Swapping YAML for a database is purely a matter of passing different store
instances; the wiring here does not change.
"""

from __future__ import annotations

from typing import Callable

import httpx

from llm_bot.agent import Agent, Session
from llm_bot.client import LLMClient
from llm_bot.compress import CompressionEvent, CompressionSettings
from llm_bot.config import LLMConfig
from llm_bot.context_strategies import (
    Branching,
    ContextStrategy,
    SlidingWindow,
    StickyFacts,
)
from llm_bot.diagnostics import DetailListener
from llm_bot.gigachat import GigaChatTokenProvider
from llm_bot.stores import AgentConfig, AgentStore, ModelConfig, ModelStore, SessionStore


def llm_config_for(
    model: ModelConfig,
    agent: AgentConfig | None = None,
) -> LLMConfig:
    """Build an :class:`LLMConfig` from a :class:`ModelConfig`, applying agent overrides."""
    config = LLMConfig(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model,
        context_window=model.context_window,
        max_request_tokens=model.max_request_tokens,
        temperature=agent.temperature if agent else None,
        max_tokens=agent.max_tokens if agent else None,
        system_prompt=agent.system_prompt if agent else "",
        default_system_prompt=agent.default_system_prompt if agent else "",
        max_response_words=agent.max_response_words if agent else None,
        gigachat_oauth_url=LLMConfig.from_env().gigachat_oauth_url,
        gigachat_client_id=model.client_id,
        gigachat_client_secret=model.client_secret,
        gigachat_scope=model.scope,
        gigachat_basic_auth=model.basic_auth,
    )
    return config


def build_client(
    model: ModelConfig,
    agent: AgentConfig | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    detail_listener: DetailListener | None = None,
) -> LLMClient:
    """Build an :class:`LLMClient` for the given model and agent settings."""
    config = llm_config_for(model, agent)
    token_provider = None
    if model.provider == "gigachat":
        token_provider = GigaChatTokenProvider(config, transport=transport)
    return LLMClient(
        config,
        transport=transport,
        token_provider=token_provider,
        detail_listener=detail_listener,
    )


def make_agent(
    agent_name: str,
    *,
    model_store: ModelStore,
    agent_store: AgentStore,
    transport: httpx.BaseTransport | None = None,
    detail_listener: DetailListener | None = None,
) -> Agent:
    """Build an :class:`Agent` by resolving its config and referenced model."""
    agent_config = agent_store.get(agent_name)
    model_config = model_store.get(agent_config.model)
    client = build_client(
        model_config,
        agent_config,
        transport=transport,
        detail_listener=detail_listener,
    )
    return Agent(agent_config, client=client)


def make_strategy(
    config: AgentConfig,
    *,
    chat: Callable[[list[dict[str, str]]], str],
    override: str | None = None,
    window_messages: int | None = None,
) -> ContextStrategy | None:
    """Build a context-management strategy from an agent config (+ optional CLI override).

    *override* (e.g. ``"sliding"`` / ``"facts"`` / ``"branching"``) takes
    precedence over the agent's configured ``context_strategy``. *window_messages*
    (the sliding-window size in MESSAGES) takes precedence over the config's
    ``context_window_messages``. When neither selects a strategy, returns ``None``
    (full-history behaviour).
    """
    kind = override or config.context_strategy
    if kind is None:
        return None
    window = (
        window_messages
        if window_messages is not None
        else config.context_window_messages
    )
    if kind == "sliding":
        if not window:
            raise ValueError(
                "Стратегии 'sliding' нужен параметр context_window_messages "
                "(размер окна в сообщениях)."
            )
        return SlidingWindow(window)
    if kind == "facts":
        if not window:
            raise ValueError(
                "Стратегии 'facts' нужен параметр context_window_messages "
                "(размер окна в сообщениях)."
            )
        return StickyFacts(window, max_facts=config.context_max_facts or 20, chat=chat)
    if kind == "branching":
        return Branching(window_size=window)
    raise ValueError(
        f"Неизвестная стратегия контекста: {kind!r}. "
        "Доступны: sliding, facts, branching."
    )


def make_session(
    session_id: str,
    agent_name: str,
    *,
    model_store: ModelStore,
    agent_store: AgentStore,
    session_store: SessionStore,
    transport: httpx.BaseTransport | None = None,
    detail_listener: DetailListener | None = None,
    history: list[dict[str, str]] | None = None,
    compression: CompressionSettings | None = None,
    on_compress: Callable[[CompressionEvent], None] | None = None,
    strategy: ContextStrategy | None = None,
    strategy_override: str | None = None,
    window_messages: int | None = None,
) -> Session:
    """Build a :class:`Session` for the given agent, ready to chat.

    Two complementary ways to manage context are supported:

    * **Rolling-summary compression** is enabled automatically when the agent
      config declares ``keep_last_messages`` / ``summarize_messages_threshold``
      (see :attr:`~llm_bot.stores.AgentConfig.compression_settings`). Pass
      *compression* explicitly to override or force-enable it.
    * **Pluggable context strategies** (sliding window / sticky facts / branching)
      are derived from the agent config's ``context_strategy`` (see
      :func:`make_strategy`), or passed in directly via *strategy*. A strategy
      takes precedence over compression when both are present.

    When compression triggers, *on_compress* (if given) is called with a
    :class:`~llm_bot.compress.CompressionEvent` so callers (e.g. a CLI) can print
    a service message about the fold and its token impact.
    """
    agent = make_agent(
        agent_name,
        model_store=model_store,
        agent_store=agent_store,
        transport=transport,
        detail_listener=detail_listener,
    )
    effective = (
        compression if compression is not None else agent.config.compression_settings
    )
    effective_strategy = strategy
    if effective_strategy is None:
        effective_strategy = make_strategy(
            agent.config,
            chat=agent.client.chat,
            override=strategy_override,
            window_messages=window_messages,
        )
    return Session(
        session_id,
        agent,
        store=session_store,
        history=history,
        compression=effective,
        on_compress=on_compress,
        strategy=effective_strategy,
    )