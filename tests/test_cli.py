"""Tests for the interactive CLI chat loop and its slash commands."""

from __future__ import annotations

from llm_bot.cli import _interactive_loop, _print_history


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Minimal stand-in for llm_bot.agent.Session used by the CLI loop."""

    def __init__(self, session_id: str, agent_name: str, history: list[dict]) -> None:
        self.session_id = session_id
        self.agent = _FakeAgent(agent_name)
        self.history = list(history)
        self.chat_calls: list[str] = []

    def chat(self, text: str) -> str:
        self.chat_calls.append(text)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": "reply-ok"})
        return "reply-ok"


def test_interactive_loop_prints_history_via_slash_commands(capsys, monkeypatch):
    """/history and /история must print prior messages without calling the LLM."""
    session = _FakeSession(
        "s1",
        "assistant",
        [
            {"role": "user", "content": "old-q"},
            {"role": "assistant", "content": "old-a"},
        ],
    )
    inputs = iter(["привет", "/history", "/история", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    code = _interactive_loop(session)

    assert code == 0
    # Only the real message "привет" reached the model; slash commands did not.
    assert session.chat_calls == ["привет"]

    out = capsys.readouterr().out
    # Both aliases print the transcript: old + new turns.
    assert "old-q" in out
    assert "old-a" in out
    assert "reply-ok" in out
    assert out.count("History of session 's1':") >= 2


def test_interactive_loop_history_empty_message(capsys, monkeypatch):
    """/history on a fresh session prints a friendly empty note."""
    session = _FakeSession("s1", "assistant", [])
    inputs = iter(["/история", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    _interactive_loop(session)

    err = capsys.readouterr().err
    assert "history is empty" in err
    assert session.chat_calls == []


def test_print_history_renders_roles_and_content(capsys):
    session = _FakeSession(
        "s9",
        "translator",
        [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Hi"},
        ],
    )

    _print_history(session)

    out = capsys.readouterr().out
    assert "History of session 's9':" in out
    assert "[you] Привет" in out
    assert "[assistant] Hi" in out


def test_print_history_empty_session(capsys):
    _print_history(_FakeSession("s9", "translator", []))
    err = capsys.readouterr().err
    assert "history is empty" in err