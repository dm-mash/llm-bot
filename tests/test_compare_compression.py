"""Unit tests for the pure helpers in scripts/compare_compression.py (no network).

The script is importable without side effects (its CLI only runs under
``__main__``), so we can test the fact handling, retention model, summarization
detection and the mock handler directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

# Make scripts/compare_compression.py importable as a module. It is not a real
# package, so we add its folder to sys.path; Pylance can't statically resolve it,
# hence the type-ignore. At runtime this works because the CLI only runs under
# ``__main__``.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import compare_compression as cc  # noqa: E402  # type: ignore[import-not-found]


def _request(messages):
    """Build an httpx.Request carrying a chat-completions body."""
    return httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        json={"model": "gpt-demo", "messages": messages},
    )


# --- Fact encoding / extraction -------------------------------------------- #


def test_plant_fact_and_extract_roundtrip():
    # _FACT_TEXT is 0-indexed: index 3 is the fourth description.
    text = cc.plant_fact(3)
    assert cc.facts_in_text(text) == {3: "команда из трёх человек"}


def test_facts_in_text_parses_multiple_lines():
    text = "ФАКТ 1: а\nФАКТ 5: б\nобычный текст"
    assert cc.facts_in_text(text) == {1: "а", 5: "б"}


def test_facts_in_messages_unions_across_stack():
    messages = [
        {"role": "system", "content": "ФАКТ 1: один"},
        {"role": "user", "content": "ФАКТ 2: два"},
        {"role": "assistant", "content": "ФАКТ 1: один (повтор)"},
    ]
    facts = cc.facts_in_messages(messages)
    assert set(facts) == {1, 2}
    # dict.update is last-wins across the stack.
    assert facts[1] == "один (повтор)"


# --- Retention model -------------------------------------------------------- #


def test_retain_facts_lossless():
    facts = {1: "a", 2: "b", 3: "c"}
    assert cc._retain_facts(facts, 1.0) == facts


def test_retain_facts_zero_keeps_none():
    facts = {1: "a", 2: "b"}
    assert cc._retain_facts(facts, 0.0) == {}


def test_retain_facts_keeps_oldest_first():
    # 3 facts at retention 0.67 -> keep ~2 oldest.
    facts = {1: "a", 2: "b", 3: "c"}
    kept = cc._retain_facts(facts, 0.67)
    assert set(kept) == {1, 2}  # lowest ids retained


# --- Detection helpers ------------------------------------------------------ #


def test_is_summarization_request_true_for_marker():
    prompt = cc.summarize_prompt("", "block")
    assert cc._is_summarization_request([{"role": "user", "content": prompt}]) is True


def test_is_summarization_request_false_for_normal():
    messages = [{"role": "user", "content": "обычный вопрос"}]
    assert cc._is_summarization_request(messages) is False


def test_is_recall_question():
    assert cc._is_recall_question([{"role": "user", "content": "перечисли факты"}]) is True
    assert cc._is_recall_question([{"role": "user", "content": "как дела?"}]) is False


# --- Mock handler ----------------------------------------------------------- #


def test_handler_echoes_visible_facts_on_recall():
    handler = cc.build_handler(retention=1.0)
    messages = [
        {"role": "system", "content": "ФАКТ 1: один\nФАКТ 2: два"},
        {"role": "user", "content": "Вопрос: перечисли все факты"},
    ]
    resp = handler(_request(messages))
    data = json.loads(resp.read())
    answer = data["choices"][0]["message"]["content"]
    assert cc.facts_in_text(answer) == {1: "один", 2: "два"}


def test_handler_summarization_respects_retention_loss():
    """Retention loss happens during summarization, not at recall time."""
    from llm_bot.compress import summarize_prompt

    handler = cc.build_handler(retention=0.5)
    prompt = summarize_prompt("", "ФАКТ 1: один\nФАКТ 2: два\nФАКТ 3: три")
    resp = handler(_request([{"role": "user", "content": prompt}]))
    data = json.loads(resp.read())
    summary = data["choices"][0]["message"]["content"]
    kept = cc.facts_in_text(summary)
    # 3 facts at retention 0.5 -> keeps the oldest 2 (ids 1, 2); fact 3 lost.
    assert set(kept) == {1, 2}
    assert 3 not in kept


def test_handler_recall_echoes_all_visible_facts_regardless_of_retention():
    # Recall simply reports what is present in context; if the summary already
    # lost a fact, that fact is gone, but recall itself does no extra filtering.
    handler = cc.build_handler(retention=1.0)
    messages = [
        {"role": "system", "content": "ФАКТ 1: один"},
        {"role": "user", "content": "Вопрос: перечисли все факты"},
    ]
    resp = handler(_request(messages))
    data = json.loads(resp.read())
    answer = data["choices"][0]["message"]["content"]
    assert cc.facts_in_text(answer) == {1: "один"}


def test_handler_tracks_summary_tokens():
    handler = cc.build_handler(retention=1.0)
    from llm_bot.compress import summarize_prompt

    prompt = summarize_prompt("", "ФАКТ 9: девять")
    resp = handler(_request([{"role": "user", "content": prompt}]))
    assert resp.status_code == 200
    assert handler.counters["summary_requests"] == 1
    assert handler.counters["summary_prompt_tokens"] > 0


# --- Token savings ---------------------------------------------------------- #


def test_percent_saved():
    plain = cc.DialogResult(
        mode="plain",
        turns=2,
        facts_per_turn=1,
        total_facts=2,
        recalled=2,
        recall_quality=1.0,
        main_context_tokens=1000,
        summary_context_tokens=0,
        total_tokens=1000,
        summary_requests=0,
    )
    compressed = cc.DialogResult(
        mode="compressed",
        turns=2,
        facts_per_turn=1,
        total_facts=2,
        recalled=2,
        recall_quality=1.0,
        main_context_tokens=400,
        summary_context_tokens=100,
        total_tokens=500,
        summary_requests=2,
        settings=cc.CompressionSettings(keep_last=4, block_size=6),
    )
    assert cc._percent_saved(plain, compressed) == 50.0