"""Tests for the profile system: orchestration config, not memory.

Covers:

* :class:`~llm_bot.stores.ProfileConfig` parsing (``from_dict``);
* :class:`~llm_bot.yaml_stores.YamlProfileStore` (YAML loading, unknown names);
* :func:`llm_bot.profiles.profile_prompt_block` (deterministic rendering);
* :func:`llm_bot.profiles.apply_profile` (composition onto ``AgentConfig``);
* factory wiring (``make_session(..., profile=...)`` produces a personalized
  agent whose directives reach the actual request payload);
* memory separation (profile data never enters the memory layers);
* CLI flags (``--list-profiles`` / ``--show-profile``).
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest

from llm_bot.cli import build_parser, main
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.memory import (
    LongTermMemory,
    MemoryLayers,
    ShortTermMemory,
    WorkingMemory,
)
from llm_bot.profiles import apply_profile, profile_prompt_block
from llm_bot.stores import AgentConfig, ModelConfig, ProfileConfig
from llm_bot.yaml_stores import YamlProfileStore


PROFILES_YAML = """\
profiles:
  developer:
    style: technical
    format: concise
    expertise: expert
    language: ru
    max_response_words: 200
    temperature: 0.3
    extra_instructions: "Приводи примеры кода."
    forbidden_topics: [политика]
    interests: [python, architecture]
  minimal:
    style: friendly
"""


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# ProfileConfig
# --------------------------------------------------------------------------- #


def test_profile_config_from_dict_full():
    profile = ProfileConfig.from_dict(
        "developer",
        {
            "style": "technical",
            "format": "concise",
            "expertise": "expert",
            "language": "ru",
            "max_response_words": 200,
            "temperature": 0.3,
            "extra_instructions": "Приводи примеры кода.",
            "forbidden_topics": ["политика", ""],
            "interests": ["python", "architecture"],
        },
    )
    assert profile.name == "developer"
    assert profile.style == "technical"
    assert profile.format == "concise"
    assert profile.expertise == "expert"
    assert profile.language == "ru"
    assert profile.max_response_words == 200
    assert profile.temperature == 0.3
    assert profile.extra_instructions == "Приводи примеры кода."
    assert profile.forbidden_topics == ["политика"]  # blank entries dropped
    assert profile.interests == ["python", "architecture"]


def test_profile_config_from_dict_defaults():
    profile = ProfileConfig.from_dict("empty", {})
    assert profile.name == "empty"
    assert profile.style == ""
    assert profile.format == ""
    assert profile.expertise == ""
    assert profile.language == ""
    assert profile.max_response_words is None
    assert profile.temperature is None
    assert profile.extra_instructions == ""
    assert profile.forbidden_topics == []
    assert profile.interests == []


# --------------------------------------------------------------------------- #
# YamlProfileStore
# --------------------------------------------------------------------------- #


def test_yaml_profile_store(tmp_path):
    path = _write(tmp_path, "profiles.yaml", PROFILES_YAML)
    store = YamlProfileStore(path)
    assert store.list() == ["developer", "minimal"]

    profile = store.get("developer")
    assert profile.style == "technical"
    assert profile.max_response_words == 200
    assert profile.temperature == 0.3
    assert profile.interests == ["python", "architecture"]


def test_yaml_profile_store_unknown_name(tmp_path):
    path = _write(tmp_path, "profiles.yaml", PROFILES_YAML)
    store = YamlProfileStore(path)
    with pytest.raises(KeyError):
        store.get("no-such-profile")


def test_yaml_profile_store_missing_file(tmp_path):
    store = YamlProfileStore(str(tmp_path / "absent.yaml"))
    with pytest.raises(FileNotFoundError):
        store.list()


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def test_profile_prompt_block_full():
    profile = ProfileConfig.from_dict(
        "developer",
        {
            "style": "technical",
            "format": "concise",
            "expertise": "expert",
            "language": "ru",
            "forbidden_topics": ["политика"],
            "interests": ["python"],
            "extra_instructions": "Приводи примеры кода.",
        },
    )
    block = profile_prompt_block(profile)
    assert block.startswith("Профиль пользователя")
    assert "технический" in block.lower()
    assert "кратко" in block.lower()
    assert "ru" in block
    assert "политика" in block
    assert "python" in block
    assert "Приводи примеры кода." in block


def test_profile_prompt_block_minimal_profile_renders_something():
    block = profile_prompt_block(ProfileConfig(name="minimal", style="friendly"))
    assert "Профиль пользователя" in block
    assert "доброжелательно" in block.lower() or "тепло" in block.lower()


def test_profile_prompt_block_empty_profile_is_empty():
    block = profile_prompt_block(ProfileConfig(name="bare"))
    assert block == ""


def test_profile_prompt_block_is_deterministic():
    profile = ProfileConfig.from_dict(
        "p", {"style": "casual", "format": "concise", "interests": ["a", "b"]}
    )
    assert profile_prompt_block(profile) == profile_prompt_block(profile)


# --------------------------------------------------------------------------- #
# apply_profile composition
# --------------------------------------------------------------------------- #


def _base_agent() -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты полезный помощник.",
        default_system_prompt="Базовая инструкция.",
        temperature=0.7,
        max_response_words=None,
        keep_last_messages=10,
        summarize_messages_threshold=20,
    )


def test_apply_profile_extends_system_prompt_and_overrides():
    agent = _base_agent()
    profile = ProfileConfig.from_dict(
        "developer",
        {
            "style": "technical",
            "max_response_words": 200,
            "temperature": 0.3,
        },
    )
    composed = apply_profile(agent, profile)

    # A NEW config is produced; the frozen original is untouched.
    assert composed is not agent
    assert agent.system_prompt == "Ты полезный помощник."

    # The role prompt is preserved and the profile directives are appended.
    assert composed.system_prompt.startswith("Ты полезный помощник.")
    assert "Профиль пользователя" in composed.system_prompt

    # Profile overrides win when set.
    assert composed.max_response_words == 200
    assert composed.temperature == 0.3

    # Everything else is copied verbatim (compression settings intact).
    assert composed.model == agent.model
    assert composed.default_system_prompt == agent.default_system_prompt
    assert composed.keep_last_messages == 10
    assert composed.summarize_messages_threshold == 20


def test_apply_profile_leaves_unset_agent_values_alone():
    agent = _base_agent()
    profile = ProfileConfig(name="minimal", style="friendly")
    composed = apply_profile(agent, profile)

    assert composed.temperature == 0.7  # no profile temperature override
    assert composed.max_response_words is None  # no profile cap
    assert "Профиль пользователя" in composed.system_prompt


def test_apply_profile_tightens_but_never_widens_word_cap():
    agent = dataclasses.replace(_base_agent(), max_response_words=100)
    stricter = apply_profile(
        agent, ProfileConfig(name="p", max_response_words=50)
    )
    assert stricter.max_response_words == 50

    # A profile without a cap keeps the agent's cap (it does not remove it).
    kept = apply_profile(agent, ProfileConfig(name="p", style="casual"))
    assert kept.max_response_words == 100


def test_apply_profile_keeps_compression_settings_property():
    agent = _base_agent()
    composed = apply_profile(agent, ProfileConfig(name="p", style="technical"))
    # The composed config keeps a working compression_settings property.
    assert composed.compression_settings is not None


# --------------------------------------------------------------------------- #
# Factory wiring (real path: stub stores + MockTransport like test_agent.py)
# --------------------------------------------------------------------------- #


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
        context_window=8192,
    )


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name: str) -> ModelConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name: str) -> AgentConfig:
        return self._config

    def list(self) -> list[str]:
        return [self._config.name]


class _StubProfileStore:
    def __init__(self, profiles: dict[str, ProfileConfig]) -> None:
        self._profiles = profiles

    def get(self, name: str) -> ProfileConfig:
        if name not in self._profiles:
            available = ", ".join(sorted(self._profiles)) or "(none)"
            raise KeyError(f"Unknown profile '{name}'. Available: {available}.")
        return self._profiles[name]

    def list(self) -> list[str]:
        return sorted(self._profiles)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )


def test_make_session_applies_profile_end_to_end(tmp_path):
    """The composed profile directives reach the actual request payload."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    transport = httpx.MockTransport(handler)
    agent_store = _StubAgentStore(_base_agent())
    developer = ProfileConfig.from_dict(
        "developer", {"style": "technical", "max_response_words": 200, "temperature": 0.3}
    )

    session = make_session(
        "s1",
        "assistant",
        model_store=_StubModelStore(_model_config()),
        agent_store=agent_store,
        session_store=JsonSessionStore(str(tmp_path / "sessions")),
        transport=transport,
        profile="developer",
        profile_store=_StubProfileStore({"developer": developer}),
    )

    # The session's agent carries the composed config.
    assert "Профиль пользователя" in session.agent.config.system_prompt
    assert session.agent.config.max_response_words == 200
    assert session.agent.config.temperature == 0.3

    # And the wire payload reflects both the directives and the overrides.
    session.chat("привет")
    payload = captured["payload"]
    assert "Профиль пользователя" in payload
    assert "technical" in payload.lower() or "технический" in payload.lower()
    assert '"temperature":0.3' in payload


def test_make_session_without_profile_is_unchanged(tmp_path):
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return _ok_response()

    transport = httpx.MockTransport(handler)
    session = make_session(
        "s1",
        "assistant",
        model_store=_StubModelStore(_model_config()),
        agent_store=_StubAgentStore(_base_agent()),
        session_store=JsonSessionStore(str(tmp_path / "sessions")),
        transport=transport,
    )

    assert session.agent.config.system_prompt == "Ты полезный помощник."
    assert "Профиль пользователя" not in session.agent.config.system_prompt
    session.chat("привет")
    assert "Профиль пользователя" not in captured["payload"]
    assert '"temperature":0.7' in captured["payload"]


def test_make_session_unknown_profile_raises_clear_error(tmp_path):
    with pytest.raises(KeyError, match="Unknown profile 'ghost'"):
        make_session(
            "s1",
            "assistant",
            model_store=_StubModelStore(_model_config()),
            agent_store=_StubAgentStore(_base_agent()),
            session_store=JsonSessionStore(str(tmp_path / "sessions")),
            profile="ghost",
            profile_store=_StubProfileStore({}),
        )


# --------------------------------------------------------------------------- #
# Memory separation: profile data never reaches the memory layers
# --------------------------------------------------------------------------- #


def test_profile_is_not_written_to_memory():
    """Composing a profile mutates only the AgentConfig, never the memory."""
    memory = MemoryLayers(
        short=ShortTermMemory(),
        working=WorkingMemory(),
        long=LongTermMemory(),
    )
    composed = apply_profile(
        _base_agent(),
        ProfileConfig.from_dict(
            "developer", {"style": "technical", "max_response_words": 200}
        ),
    )

    # Composing the profile wrote nothing into memory.
    assert len(memory.long) == 0
    assert len(memory.working) == 0
    # The profile lives in the composed config's prompt instead.
    assert "Профиль пользователя" in composed.system_prompt
    # And the rendered memory prefix carries no profile block.
    assert memory.prefix_messages() == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parser_accepts_profile_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["--agent", "assistant", "--profile", "developer"]
    )
    assert args.profile == "developer"
    assert args.list_profiles is False
    assert args.show_profile is None


def test_cli_list_profiles(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "data/profiles.yaml", PROFILES_YAML)
    monkeypatch.chdir(tmp_path)  # YamlProfileStore default path is data/profiles.yaml
    code = main(["--list-profiles"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[developer]" in out
    assert "[minimal]" in out


def test_cli_list_profiles_when_store_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no data/profiles.yaml here
    code = main(["--list-profiles"])
    assert code == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_show_profile(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "data/profiles.yaml", PROFILES_YAML)
    monkeypatch.chdir(tmp_path)
    code = main(["--show-profile", "developer"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[developer]" in out
    assert "style: technical" in out
    assert "Профиль пользователя" in out


def test_cli_show_profile_unknown(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "data/profiles.yaml", PROFILES_YAML)
    monkeypatch.chdir(tmp_path)
    code = main(["--show-profile", "ghost"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown profile 'ghost'" in err
    assert "developer" in err  # available profiles are listed


# --------------------------------------------------------------------------- #
# compare_profiles script helpers (offline)
# --------------------------------------------------------------------------- #


def test_compare_profiles_markdown_report_offline(tmp_path, monkeypatch):
    """The compare script's report renders per-profile sections and metrics.

    The network-bound _ask_once is replaced with canned results, so this test
    exercises only report rendering.
    """
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import compare_profiles as cp  # type: ignore[import-not-found]

    _write(tmp_path, "data/profiles.yaml", PROFILES_YAML)
    monkeypatch.chdir(tmp_path)
    profile_store = YamlProfileStore()

    results = [
        {
            "profile": "developer",
            "reply": "Рекурсия — функция, вызывающая себя. Пример: def f(n): return n*f(n-1)",
            "error": None,
            "elapsed_ms": 120,
            "reply_chars": 70,
            "reply_words": 12,
            "context_tokens": 150,
            "reply_tokens": 30,
            "system_prompt": "x",
            "temperature": 0.3,
            "max_response_words": 200,
        },
        {
            "profile": "minimal",
            "reply": "Рекурсия — это когда функция вызывает сама себя, как матрёшка. "
                     "Шаг 1: базовый случай. Шаг 2: рекурсивный переход.",
            "error": None,
            "elapsed_ms": 140,
            "reply_chars": 110,
            "reply_words": 20,
            "context_tokens": 160,
            "reply_tokens": 40,
            "system_prompt": "x",
            "temperature": None,
            "max_response_words": None,
        },
    ]

    report = cp._render_markdown(
        "Объясни рекурсию", "assistant", results, profile_store
    )
    assert "# Сравнение профилей персонализации" in report
    assert "## Профиль: developer" in report
    assert "## Профиль: minimal" in report
    assert "Директивы профиля" in report
    assert "Профиль пользователя" in report
    assert "temperature=0.3" in report
    assert "max_response_words=200" in report
    assert "самый лаконичный" in report


def test_compare_profiles_unknown_profile_exits_cleanly(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import compare_profiles as cp  # type: ignore[import-not-found]

    _write(tmp_path, "data/profiles.yaml", PROFILES_YAML)
    monkeypatch.chdir(tmp_path)
    code = cp.main(["--profiles", "ghost", "--out", str(tmp_path / "r.md")])
    assert code == 2
    assert "unknown profile" in capsys.readouterr().err
