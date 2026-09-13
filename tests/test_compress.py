"""Tests for context compression: settings, compressor, and session integration."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_bot.agent import Session, summary_budget_chars
from llm_bot.compress import (
    CompressionSettings,
    ContextCompressor,
    render_block,
    summarize_prompt,
    truncate_summary,
)
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig


def _agent_config(*, max_tokens: int = 64) -> AgentConfig:
    return AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
        temperature=0.5,
        max_tokens=max_tokens,
    )


def _model_config(*, context_window: int | None = None) -> ModelConfig:
    return ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
        context_window=context_window,
    )


class _StubAgentStore:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def get(self, name):
        return self._config

    def list(self):
        return [self._config.name]


class _StubModelStore:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def get(self, name):
        return self._config

    def list(self):
        return [self._config.name]


def _session(
    transport,
    tmp_path,
    *,
    agent_cfg=None,
    model_cfg=None,
    compression=None,
    on_compress=None,
    session_id="s1",
):
    agent_cfg = agent_cfg or _agent_config()
    model_cfg = model_cfg or _model_config()
    session_store = JsonSessionStore(str(tmp_path / "sessions"))
    return make_session(
        session_id,
        agent_cfg.name,
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(agent_cfg),
        session_store=session_store,
        transport=transport,
        compression=compression,
        on_compress=on_compress,
    )


def _ok_handler(captured=None):
    """Return a handler that records the JSON payload and echoes a short reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    return handler


# --------------------------------------------------------------------------- #
# CompressionSettings
# --------------------------------------------------------------------------- #


def test_settings_normalise_keep_last_to_even():
    s = CompressionSettings(keep_last=11, block_size=20)
    assert s.keep_last == 10
    assert s.block_size > s.keep_last


def test_settings_block_size_at_least_keep_last_plus_two():
    s = CompressionSettings(keep_last=4, block_size=5)
    assert s.block_size >= s.keep_last + 2


# --------------------------------------------------------------------------- #
# Summarizer prompt helpers
# --------------------------------------------------------------------------- #


def test_summarize_prompt_first_round_has_no_existing_section():
    p = summarize_prompt("", "block text")
    assert "Имеющееся изложение" not in p
    assert "block text" in p


def test_summarize_prompt_incremental_includes_existing():
    p = summarize_prompt("old summary", "new block")
    assert "old summary" in p
    assert "new block" in p


def test_render_block_labels_roles():
    msgs = [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    text = render_block(msgs)
    assert "Пользователь: вопрос" in text
    assert "Ассистент: ответ" in text


# --------------------------------------------------------------------------- #
# ContextCompressor (pure logic)
# --------------------------------------------------------------------------- #


def _capture_summarizer(calls):
    """Return a summarizer that records (existing, block) and echoes a marker."""

    def summarize(existing: str, block: str, max_chars: int | None) -> str:
        calls.append((existing, block, max_chars))
        return f"summary({len(existing)})"

    return summarize


def test_compressor_noop_below_block_size():
    calls = []
    comp = ContextCompressor(
        CompressionSettings(keep_last=4, block_size=6), summarize=_capture_summarizer(calls)
    )
    history = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    new_hist, new_summary, folded = comp.compress(history, "")
    assert calls == []  # nothing summarized yet
    assert new_hist == history
    assert new_summary == ""
    assert folded == []


def test_compressor_folds_oldest_messages_when_over_block():
    calls = []
    comp = ContextCompressor(
        CompressionSettings(keep_last=4, block_size=6), summarize=_capture_summarizer(calls)
    )
    history = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    new_hist, new_summary, folded = comp.compress(history, "")
    # Only the last 4 are kept; the oldest 4 are folded into the summary.
    assert [m["content"] for m in new_hist] == ["m4", "m5", "m6", "m7"]
    assert [m["content"] for m in folded] == ["m0", "m1", "m2", "m3"]
    assert new_summary == "summary(0)"
    assert len(calls) == 1
    # The summarizer received the rolled-off block text.
    assert "m0" in calls[0][1] and "m3" in calls[0][1]


def test_compressor_is_incremental_reuses_previous_summary():
    calls = []
    comp = ContextCompressor(
        CompressionSettings(keep_last=4, block_size=6), summarize=_capture_summarizer(calls)
    )
    # Existing summary + a new block to fold.
    history = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    new_hist, new_summary, _folded = comp.compress(history, "PREVIOUS")
    assert new_summary == "summary(8)"
    assert calls[0][0] == "PREVIOUS"
    # Only the rolled-off block (the oldest messages) is sent, plus the prior
    # summary — never the whole dialog. The kept messages are NOT resent.
    block = calls[0][1]
    assert "m0" in block and "m3" in block
    assert "m4" not in block


def test_should_compress_boundary():
    comp = ContextCompressor(
        CompressionSettings(keep_last=4, block_size=6), summarize=lambda e, b, m: b
    )
    assert comp.should_compress(6) is False
    assert comp.should_compress(7) is True


# --------------------------------------------------------------------------- #
# Session integration
# --------------------------------------------------------------------------- #


def test_session_compression_injects_summary_and_trims_history(tmp_path):
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode())
        # Reply is the model's answer; when the summarizer is asked, echo
        # "SUMMARY" so we can detect the summary message in later requests.
        content = "SUMMARY"
        captured.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "SUMMARY" if content else "ok",
                        }
                    }
                ]
            },
        )

    # keep_last=4, block_size=6: after 6 messages a 7th triggers folding.
    session = _session(
        httpx.MockTransport(handler),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
    )

    # Send 8 user turns (each produces a user+assistant pair => 16 messages).
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")

    # The summary was produced (the assistant reply to the summary prompt) and
    # injected as the first system message in subsequent requests.
    last_payload = captured[-1]
    roles = [m["role"] for m in last_payload["messages"]]
    assert roles[0] == "system"  # summary frame
    # First message should carry the summary content.
    assert last_payload["messages"][0]["content"] == "SUMMARY"
    # The live history was trimmed well below the full 16 messages: after the
    # final turn it holds keep_last (4) kept messages plus the appended reply.
    assert len(session.history) == 5
    assert len(session.history) < 16
    # The persisted summary is stored separately.
    assert session.summary == "SUMMARY"
    assert session._store.load_summary("s1") == "SUMMARY"


def test_session_compression_disabled_by_default(tmp_path):
    captured = []
    session = _session(httpx.MockTransport(_ok_handler(captured)), tmp_path)
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")
    # No compression: all 16 messages present, no summary.
    assert len(session.history) == 16
    assert session.compression_enabled is False
    assert session.summary == ""
    # The request stack starts with the agent's system prompt (no summary frame).
    first_role = captured[0]["messages"][0]["role"]
    assert first_role == "system"
    assert captured[0]["messages"][0]["content"] == "Ты помощник."


def test_session_compression_persists_and_round_trips(tmp_path):
    captured = []
    session = _session(
        httpx.MockTransport(_ok_handler(captured)),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
    )
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")
    assert session.summary

    # A new session on the same store resumes with the summary loaded.
    resumed = _session(
        httpx.MockTransport(_ok_handler(captured)),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
    )
    assert resumed.summary == session.summary
    assert resumed.history == session.history


def test_session_legacy_flat_store_reads_with_empty_summary(tmp_path):
    """A legacy flat-list session file (pre-compression) still loads cleanly."""
    directory = str(tmp_path / "sessions")
    store = JsonSessionStore(directory)
    store.save("legacy", [{"role": "user", "content": "hi"}])

    agent_cfg = _agent_config()
    model_cfg = _model_config()
    session = make_session(
        "legacy",
        agent_cfg.name,
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(agent_cfg),
        session_store=store,
        transport=httpx.MockTransport(_ok_handler()),
        compression=CompressionSettings(keep_last=4, block_size=6),
    )
    assert session.history == [{"role": "user", "content": "hi"}]
    assert session.summary == ""


def test_session_compression_saves_token_reduce_history_tokens(tmp_path):
    """With compression, later turns send a smaller context than without."""
    def make_session_under(compression, tag):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.read().decode())
            captured.append(payload)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reply text here",
                            }
                        }
                    ]
                },
            )

        sess = _session(
            httpx.MockTransport(handler),
            tmp_path,
            compression=compression,
            session_id=tag,
        )
        return sess

    # Plain session: grows unbounded.
    plain = make_session_under(None, "plain")
    for i in range(8):
        plain.chat_with_details(f"long question number {i} " + "x" * 60)
    plain_context = plain.last_usage.context_tokens

    # Compressed session: same dialog, but history is folded.
    comp = make_session_under(
        CompressionSettings(keep_last=4, block_size=6), "comp"
    )
    for i in range(8):
        comp.chat_with_details(f"long question number {i} " + "x" * 60)
    comp_context = comp.last_usage.context_tokens

    # The compressed session sends a strictly smaller context on the final turn.
    assert comp_context < plain_context
    assert comp.last_usage.history_tokens < plain.last_usage.history_tokens


# --------------------------------------------------------------------------- #
# Per-agent config fields (agents.yaml): keep_last_messages / threshold.
# --------------------------------------------------------------------------- #


def test_agent_config_compression_disabled_when_absent():
    cfg = AgentConfig(name="assistant", model="openai")
    assert cfg.compression_settings is None


def test_agent_config_compression_settings_from_fields():
    cfg = AgentConfig(
        name="assistant",
        model="openai",
        keep_last_messages=10,
        summarize_messages_threshold=20,
    )
    settings = cfg.compression_settings
    assert settings is not None
    assert settings.keep_last == 10
    assert settings.block_size == 20


def test_agent_config_from_dict_parses_compression_fields():
    cfg = AgentConfig.from_dict(
        "assistant",
        {"model": "openai", "keep_last_messages": 8, "summarize_messages_threshold": 16},
    )
    assert cfg.keep_last_messages == 8
    assert cfg.summarize_messages_threshold == 16
    assert cfg.compression_settings is not None


def test_agent_config_from_dict_compression_defaults_off():
    cfg = AgentConfig.from_dict("assistant", {"model": "openai"})
    assert cfg.keep_last_messages is None
    assert cfg.summarize_messages_threshold is None
    assert cfg.compression_settings is None


def test_make_session_auto_enables_from_agent_config(tmp_path):
    """A session made via make_session turns on compression if the agent config says so."""
    captured = []
    agent_cfg = _agent_config()
    agent_cfg = AgentConfig(
        name=agent_cfg.name,
        model=agent_cfg.model,
        system_prompt=agent_cfg.system_prompt,
        temperature=agent_cfg.temperature,
        max_tokens=agent_cfg.max_tokens,
        keep_last_messages=4,
        summarize_messages_threshold=6,
    )
    session = _session(
        httpx.MockTransport(_ok_handler(captured)),
        tmp_path,
        agent_cfg=agent_cfg,
    )
    # No explicit compression passed -> derived from the agent config.
    assert session.compression_enabled is True
    assert session.summary == ""


def test_make_session_compression_still_off_by_default(tmp_path):
    session = _session(httpx.MockTransport(_ok_handler()), tmp_path)
    assert session.compression_enabled is False


# --------------------------------------------------------------------------- #
# Compression events / service message
# --------------------------------------------------------------------------- #


def test_compression_events_recorded_and_counters(tmp_path):
    session = _session(
        httpx.MockTransport(_ok_handler()),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
    )
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")

    # With keep_last=4 / block_size=6, compression triggers on turns 4, 6 and 8.
    assert session.total_compressions == 3
    # Messages folded per round: 3 + 4 + 4.
    assert session.total_messages_folded == 11
    last = session.last_compression_event
    assert last is not None
    assert last.messages_folded == 4
    assert last.folded_tokens > 0
    assert last.history_before == 8
    assert last.history_after == 4
    assert last.summary_chars > 0
    assert len(session.compression_events) == 3


def test_compression_events_empty_when_disabled(tmp_path):
    session = _session(httpx.MockTransport(_ok_handler()), tmp_path)
    session.chat_with_details("привет")
    assert session.compression_events == []
    assert session.total_compressions == 0
    assert session.total_messages_folded == 0
    assert session.last_compression_event is None


def test_on_compress_callback_fired(tmp_path):
    fired = []

    def on_compress(event):
        fired.append(event)

    session = _session(
        httpx.MockTransport(_ok_handler()),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
        on_compress=on_compress,
    )
    for i in range(8):
        session.chat_with_details(f"вопрос {i}")

    assert len(fired) == session.total_compressions == 3
    assert all(e.messages_folded > 0 for e in fired)


def test_cli_print_compression_message(capsys, tmp_path):
    """The CLI helper emits a service message to stderr after a fold."""
    from llm_bot.cli import _print_compression

    session = _session(
        httpx.MockTransport(_ok_handler()),
        tmp_path,
        compression=CompressionSettings(keep_last=4, block_size=6),
    )
    for i in range(4):  # triggers the first fold (turn 4)
        session.chat_with_details(f"вопрос {i}")

    assert session.total_compressions >= 1
    _print_compression(session)
    err = capsys.readouterr().err
    assert "[compression]" in err
    assert "свёрнуто" in err
    assert "токенов" in err
    assert "история" in err


def test_cli_print_compression_noop_without_compression(capsys, tmp_path):
    from llm_bot.cli import _print_compression

    session = _session(httpx.MockTransport(_ok_handler()), tmp_path)
    session.chat_with_details("привет")
    _print_compression(session)
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# Summary size limits
# --------------------------------------------------------------------------- #


def test_summarize_prompt_appends_soft_limit_instruction():
    p = summarize_prompt("", "block", max_chars=500)
    assert "не длиннее примерно 500 символов" in p


def test_summarize_prompt_no_limit_when_max_chars_none():
    p = summarize_prompt("", "block")
    assert "не длиннее" not in p


def test_truncate_summary_noop_when_fits():
    assert truncate_summary("короткий текст", 100) == "короткий текст"
    assert truncate_summary("текст", None) == "текст"


def test_truncate_summary_cuts_on_word_boundary_and_marks():
    long = "один два три четыре пять шесть семь восемь"
    out = truncate_summary(long, 20)
    assert len(out) <= 20 + 2  # ellipsis may add a couple chars
    assert out.endswith("…")
    # The cut happens at a whitespace, not mid-word (whole words are preserved).
    prefix = out[:-1].rstrip()
    assert prefix == "один два три четыре"


def test_compressor_hard_truncates_summary_result():
    def summarize(existing, block, max_chars):
        return "очень длинное изложение которое модель прислала целиком" * 10

    comp = ContextCompressor(
        CompressionSettings(keep_last=4, block_size=6),
        summarize=summarize,
        max_chars=30,
    )
    history = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    _, new_summary, _folded = comp.compress(history, "")
    # Hard cap guarantees the summary stays within budget (word-boundary cut).
    assert len(new_summary) <= 30 + 2
    assert new_summary.endswith("…")


def test_summary_budget_chars_from_tokens_without_window():
    s = CompressionSettings(max_summary_tokens=50)
    assert summary_budget_chars(s, None) == 200  # ~4 chars per token


def test_summary_budget_chars_from_window_ratio():
    s = CompressionSettings()  # max_summary_tokens None, ratio default 0.3
    assert summary_budget_chars(s, 1000) == int(1000 * 0.3) * 4


def test_summary_budget_chars_takes_min_of_token_and_window():
    s = CompressionSettings(max_summary_tokens=100, max_summary_ratio=0.3)
    # window gives 300 tokens but explicit cap is 100 -> use 100.
    assert summary_budget_chars(s, 1000) == 100 * 4


def test_summary_budget_chars_none_when_no_window_and_no_tokens():
    assert summary_budget_chars(CompressionSettings(), None) is None


def test_agent_config_parses_max_summary_fields():
    cfg = AgentConfig.from_dict(
        "assistant",
        {
            "model": "openai",
            "keep_last_messages": 10,
            "summarize_messages_threshold": 20,
            "max_summary_tokens": 500,
            "max_summary_ratio": 0.25,
        },
    )
    assert cfg.max_summary_tokens == 500
    assert cfg.max_summary_ratio == 0.25
    settings = cfg.compression_settings
    assert settings is not None
    assert settings.max_summary_tokens == 500
    assert settings.max_summary_ratio == pytest.approx(0.25)