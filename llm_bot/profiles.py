"""Profile composition: personalization as orchestration config.

A profile is NOT memory. Memory accumulates *learned* data during a dialog
(facts, decisions — see :mod:`llm_bot.memory`). A profile is static
configuration that declares *how* an agent behaves for a given user or task:
communication style, response format, constraints.

Profiles live in ``data/profiles.yaml`` (mirroring ``agents.yaml`` /
``models.yaml``) and are applied at **composition time**:
:func:`apply_profile` takes the agent's :class:`~llm_bot.stores.AgentConfig`
plus a :class:`~llm_bot.stores.ProfileConfig` and returns a *new* personalized
``AgentConfig``. Everything downstream (client, session, token accounting,
compression, context strategies, memory) works unchanged — the profile is just
a different lens on the same agent.

Example::

    from llm_bot.stores import AgentConfig
    from llm_bot.profiles import apply_profile

    agent = AgentConfig(name="assistant", model="gpt4o",
                        system_prompt="Ты полезный помощник.")
    profile = ProfileConfig(name="developer", style="technical",
                            format="concise", expertise="expert",
                            max_response_words=200)
    personalized = apply_profile(agent, profile)
    personalized.system_prompt  # base prompt + profile directives
    personalized.max_response_words  # 200
"""

from __future__ import annotations

from llm_bot.stores import AgentConfig, ProfileConfig

# Human-readable rendering of the declared style / format / expertise enums.
# Keys mirror the accepted YAML values; values are model-facing directives.
_STYLE_DIRECTIVES: dict[str, str] = {
    "formal": "Используй формальный, сдержанный деловой язык.",
    "casual": "Общайся непринуждённо и просто, как с другом.",
    "technical": "Используй точный технический язык и профессиональную терминологию.",
    "friendly": "Отвечай тепло, доброжелательно и поддерживающе.",
    "professional": "Держи деловой, профессиональный тон.",
}

_FORMAT_DIRECTIVES: dict[str, str] = {
    "concise": "Отвечай кратко и по существу, без воды.",
    "detailed": "Давай развёрнутые, подробные объяснения.",
    "structured": "Структурируй ответ: заголовки, разделы, логичная организация.",
    "conversational": "Веди естественный, живой диалог.",
    "bullet_points": "Оформляй ответы списками для читаемости.",
}

_EXPERTISE_DIRECTIVES: dict[str, str] = {
    "beginner": "Объясняй просто, без жаргона, с примерами и аналогиями.",
    "intermediate": "Соблюдай баланс глубины и ясности, учитывай базовые знания.",
    "expert": "Используй продвинутые концепции и технические детали без упрощений.",
}

# Directives are emitted in a stable order so prompts are deterministic.
_PROMPT_HEADER = "Профиль пользователя (персонализация ответов):"


def profile_prompt_block(profile: ProfileConfig) -> str:
    """Render a :class:`ProfileConfig` as a deterministic prompt fragment.

    The fragment is appended to the agent's system prompt by
    :func:`apply_profile`. Only non-empty profile fields produce directives, so
    a minimal profile does not clutter the prompt.
    """
    parts: list[str] = []
    if profile.style:
        directive = _STYLE_DIRECTIVES.get(profile.style)
        if directive:
            parts.append(f"- Стиль: {directive}")
    if profile.format:
        directive = _FORMAT_DIRECTIVES.get(profile.format)
        if directive:
            parts.append(f"- Формат: {directive}")
    if profile.expertise:
        directive = _EXPERTISE_DIRECTIVES.get(profile.expertise)
        if directive:
            parts.append(f"- Уровень пользователя: {directive}")
    if profile.language:
        parts.append(f"- Отвечай на языке: {profile.language}")
    if profile.forbidden_topics:
        topics = ", ".join(profile.forbidden_topics)
        parts.append(f"- Не обсуждай темы: {topics}.")
    if profile.interests:
        interests = ", ".join(profile.interests)
        parts.append(f"- Интересы пользователя (учитывай при примерах): {interests}.")
    if profile.extra_instructions:
        parts.append(f"- {profile.extra_instructions}")
    if not parts:
        return ""
    return _PROMPT_HEADER + "\n" + "\n".join(parts)


def apply_profile(
    agent: AgentConfig, profile: ProfileConfig
) -> AgentConfig:
    """Compose an :class:`AgentConfig` with a :class:`ProfileConfig`.

    Returns a NEW ``AgentConfig`` (the input is frozen and left untouched) in
    which:

    * ``system_prompt`` — the agent's own prompt extended with the profile's
      rendered directives (profile *adds* to the role, never replaces it);
    * ``max_response_words`` — profile cap wins when set (a profile-level
      constraint tightens the agent default);
    * ``temperature`` — profile override wins when set (e.g. a technical
      profile asking for deterministic answers).

    Everything else (model reference, compression, context strategy) is copied
    verbatim so the personalized agent behaves identically apart from the
    declared personalization.
    """
    block = profile_prompt_block(profile)
    system_prompt = (
        f"{agent.system_prompt}\n\n{block}" if block else agent.system_prompt
    )
    max_response_words = (
        profile.max_response_words
        if profile.max_response_words is not None
        else agent.max_response_words
    )
    temperature = (
        profile.temperature
        if profile.temperature is not None
        else agent.temperature
    )
    return AgentConfig(
        name=agent.name,
        model=agent.model,
        system_prompt=system_prompt,
        default_system_prompt=agent.default_system_prompt,
        temperature=temperature,
        max_tokens=agent.max_tokens,
        max_response_words=max_response_words,
        keep_last_messages=agent.keep_last_messages,
        summarize_messages_threshold=agent.summarize_messages_threshold,
        max_summary_tokens=agent.max_summary_tokens,
        max_summary_ratio=agent.max_summary_ratio,
        context_strategy=agent.context_strategy,
        context_window_messages=agent.context_window_messages,
        context_max_facts=agent.context_max_facts,
    )
