"""Wiring: assemble :class:`Agent` and :class:`Session` from storage backends.

This module is the composition root. It maps a :class:`ModelConfig` onto an
:class:`~llm_bot.config.LLMConfig` (transport + auth), applies the agent's
generation settings (temperature / max_tokens), builds the right client (plain
or GigaChat), and hands back a ready-to-use :class:`Session`.

Swapping YAML for a database is purely a matter of passing different store
instances; the wiring here does not change.
"""

from __future__ import annotations

from pathlib import Path
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
from llm_bot.mcp_tools import MCPRouter, router_from_config
from llm_bot.gigachat import GigaChatTokenProvider
from llm_bot.invariants import Invariant, InvariantRegistry
from llm_bot.memory import (
    LongTermMemory,
    MemoryLayers,
    ShortTermMemory,
    WorkingMemory,
)
from llm_bot.memory_store import JsonMemoryStore, MemoryStore
from llm_bot.profiles import apply_profile
from llm_bot.rag import DEFAULT_TOP_K as DEFAULT_RAG_TOP_K
from llm_bot.rag import Retriever
from llm_bot.stores import (
    AgentConfig,
    AgentStore,
    ModelConfig,
    ModelStore,
    ProfileStore,
    SessionStore,
)
from llm_bot.task_state import TaskStateMachine


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


def make_agent_from_config(
    agent_config: AgentConfig,
    *,
    model_store: ModelStore,
    transport: httpx.BaseTransport | None = None,
    detail_listener: DetailListener | None = None,
) -> Agent:
    """Build an :class:`Agent` from an (already composed) :class:`AgentConfig`.

    This is the low-level half of :func:`make_agent`; it exists so callers that
    pre-compose the config (e.g. with a personalization profile via
    :func:`llm_bot.profiles.apply_profile`) can reuse the client wiring without
    re-reading the agent store.
    """
    model_config = model_store.get(agent_config.model)
    client = build_client(
        model_config,
        agent_config,
        transport=transport,
        detail_listener=detail_listener,
    )
    return Agent(agent_config, client=client)


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
    return make_agent_from_config(
        agent_config,
        model_store=model_store,
        transport=transport,
        detail_listener=detail_listener,
    )


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
    owner_id: str = "default",
    memory_store: MemoryStore | None = None,
    memory_auto_extract: bool = True,
    profile: str | None = None,
    profile_store: ProfileStore | None = None,
    task_state: bool | None = None,
    task_auto_detect: bool = True,
    invariants_file: str | None = None,
    invariants: bool | InvariantRegistry | None = None,
    audit_invariants_warn: bool = False,
    mcp_servers: list[str] | None = None,
    mcp_config_file: str | None = None,
    mcp_router: MCPRouter | None = None,
    rag_index: str | Path | None = None,
    rag_top_k: int | None = None,
    rag_max_context_tokens: int | None = None,
    retriever: Retriever | None = None,
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

    Personalization: when *profile* names an entry from a
    :class:`~llm_bot.stores.ProfileStore` (``data/profiles.yaml`` by default),
    it is composed onto the agent config BEFORE the client is built (see
    :func:`llm_bot.profiles.apply_profile`). This is orchestration config, not
    memory — the profile only changes the composed agent behaviour; no profile
    data is ever written to or read from the memory layers.

    Task state machine: when *task_state* is true (or the agent config declares
    ``task_state: true``), a :class:`~llm_bot.task_state.TaskStateMachine` is
    attached to the session. It persists its snapshot (stage / step / expected
    action) through the session store and injects a rendered block into the
    request prefix so the model can continue the task without re-explanation
    after a pause or a process restart. When auto-detection is on (default),
    the machine also drives itself from the dialog after each turn via a small
    LLM call (see :func:`llm_bot.task_state.detect_task_turn`).

    RAG: passing *rag_index* (a JSON index built by
    ``scripts/index_documents.py``) attaches a
    :class:`~llm_bot.rag.Retriever`, so every question is answered against the
    top chunks of that index and the model is told to cite them. *rag_top_k* and
    *rag_max_context_tokens* tune how much of it gets into the prompt; the
    block's ceiling also falls back to a share of the model's context window
    (see :func:`llm_bot.rag.rag_budget_tokens`). Pass a ready-made *retriever*
    instead to skip construction (tests, custom retrieval).
    """
    agent_config = agent_store.get(agent_name)
    if profile is not None:
        if profile_store is None:
            from llm_bot.yaml_stores import YamlProfileStore

            profile_store = YamlProfileStore()
        profile_config = profile_store.get(profile)
        agent_config = apply_profile(agent_config, profile_config)
    agent = make_agent_from_config(
        agent_config,
        model_store=model_store,
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
    # Build the explicit layered memory (short / working / long) when a store is
    # provided. Long-term memory is isolated per (agent, owner) for privacy, so
    # one user's durable data never leaks into another user's session.
    memory = None
    if memory_store is not None:
        memory = MemoryLayers(
            short=ShortTermMemory(),
            working=WorkingMemory(
                store=memory_store, session_id=session_id
            ),
            long=LongTermMemory(
                store=memory_store, agent=agent_name, owner=owner_id
            ),
        )

    # Task state machine: enabled explicitly via *task_state* or via the agent
    # config; the explicit flag wins (mirrors strategy_override semantics).
    task_enabled = (
        task_state
        if task_state is not None
        else agent.config.task_state
    )

    # Invariants: merge global ones (data/invariants.yaml by default) with the
    # session-scoped subset persisted under the session file. Three modes:
    #   * an InvariantRegistry instance is used directly (programmatic use /
    #     tests) — the file is not read;
    #   * False disables the whole layer (CLI --no-invariants);
    #   * None (default) loads the global YAML; a missing file simply means
    #     no global invariants. Labels come from the optional ``kind_labels``
    #     section of the same file. Session-scoped entries always merge
    #     inside Session.
    invariant_registry: InvariantRegistry | None = None
    if isinstance(invariants, InvariantRegistry):
        invariant_registry = invariants
    elif invariants is not False:
        labels: dict[str, str] = {}
        global_entries: list[dict] = []
        try:
            from llm_bot.yaml_stores import YamlInvariantStore

            store = YamlInvariantStore(
                invariants_file or "data/invariants.yaml"
            )
            labels = store.kind_labels()
            global_entries = [store.get(name) for name in store.list()]
        except FileNotFoundError:
            global_entries = []
        if global_entries:
            try:
                invariant_registry = InvariantRegistry(
                    [
                        Invariant.from_dict(entry, source="global")
                        for entry in global_entries
                    ],
                    kind_labels=labels,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Конфигурация инвариантов некорректна: {exc}"
                ) from exc
        # When no global invariants exist and no explicit file was given,
        # leave invariant_registry as None so the audit does not fire on
        # every turn. Session-scoped invariants added via /invariant add
        # will still work because Session loads them from the store.

    # RAG: the index is read (and validated) here, so a missing or foreign file
    # fails at startup rather than on the user's first question. The embedding
    # model itself is still loaded lazily on the first query.
    effective_retriever = retriever
    if effective_retriever is None and rag_index is not None:
        effective_retriever = Retriever(
            rag_index,
            top_k=rag_top_k if rag_top_k is not None else DEFAULT_RAG_TOP_K,
            max_context_tokens=rag_max_context_tokens,
            context_window=agent.context_window,
        )

    return Session(
        session_id,
        agent,
        store=session_store,
        history=history,
        compression=effective,
        on_compress=on_compress,
        strategy=effective_strategy,
        memory=memory,
        memory_auto_extract=memory_auto_extract,
        task=(
            TaskStateMachine()
            if task_enabled
            else None
        ),
        task_auto_detect=(
            task_auto_detect and agent.config.task_auto_detect
        ),
        invariants=invariant_registry,
        audit_invariants_warn=audit_invariants_warn,
        mcp=(
            mcp_router
            if mcp_router is not None
            else router_from_config(
                mcp_servers,
                Path(mcp_config_file) if mcp_config_file else None,
            )
        ),
        retriever=effective_retriever,
    )