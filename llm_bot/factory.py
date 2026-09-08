"""Wiring: assemble :class:`Agent` and :class:`Session` from storage backends.

This module is the composition root. It maps a :class:`ModelConfig` onto an
:class:`~llm_bot.config.LLMConfig` (transport + auth), applies the agent's
generation settings (temperature / max_tokens), builds the right client (plain
or GigaChat), and hands back a ready-to-use :class:`Session`.

Swapping YAML for a database is purely a matter of passing different store
instances; the wiring here does not change.
"""

from __future__ import annotations

import httpx

from llm_bot.agent import Agent, Session
from llm_bot.client import LLMClient
from llm_bot.config import LLMConfig
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
) -> Session:
    """Build a :class:`Session` for the given agent, ready to chat."""
    agent = make_agent(
        agent_name,
        model_store=model_store,
        agent_store=agent_store,
        transport=transport,
        detail_listener=detail_listener,
    )
    return Session(
        session_id,
        agent,
        store=session_store,
        history=history,
    )