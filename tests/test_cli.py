"""Tests for the interactive CLI chat loop and its slash commands."""

from __future__ import annotations

from llm_bot.cli import _interactive_loop, _print_history, _read_input


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
    monkeypatch.setattr(
        "llm_bot.cli._read_input", lambda _prompt, **kwargs: next(inputs)
    )

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
    monkeypatch.setattr(
        "llm_bot.cli._read_input", lambda _prompt, **kwargs: next(inputs)
    )

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


def test_read_input_uses_prompt_toolkit_on_tty(monkeypatch):
    """On an interactive terminal _read_input must delegate to prompt_toolkit,
    which edits on whole Unicode characters (so Backspace cannot split a
    multi-byte character into an invalid surrogate)."""
    monkeypatch.setattr(
        "sys.stdin",
        type("TtyStdin", (), {"isatty": lambda self: True})(),
    )
    captured = {}
    import prompt_toolkit

    def fake_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "привет \udcd1"

    monkeypatch.setattr(prompt_toolkit, "prompt", fake_prompt)

    result = _read_input("> ")

    assert result == "привет \udcd1"
    assert captured["prompt"] == "> "
    # Completion must always be wired up on a TTY.
    assert captured["kwargs"]["completer"] is not None


def test_read_input_passes_session_history_on_tty(monkeypatch):
    """The per-session in-memory history must be forwarded to prompt_toolkit."""
    monkeypatch.setattr(
        "sys.stdin",
        type("TtyStdin", (), {"isatty": lambda self: True})(),
    )
    import prompt_toolkit
    from prompt_toolkit.history import InMemoryHistory

    captured = {}
    history = InMemoryHistory()

    def fake_prompt(prompt, **kwargs):
        captured["history"] = kwargs.get("history")
        return "x"

    monkeypatch.setattr(prompt_toolkit, "prompt", fake_prompt)

    _read_input("> ", history=history)

    assert captured["history"] is history


def test_command_completer_suggests_commands():
    """The tab-completer must offer the slash commands and exit/quit words."""
    from llm_bot.cli import _command_completer, _COMMAND_WORDS

    completer = _command_completer()
    assert set(completer.words) == set(_COMMAND_WORDS)


def test_read_input_falls_back_to_builtin_input(monkeypatch):
    """On non-interactive stdin (pipes, tests) _read_input uses input()."""
    monkeypatch.setattr(
        "sys.stdin",
        type("PipedStdin", (), {"isatty": lambda self: False})(),
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: f"echo:{prompt}"
    )
    assert _read_input("? ") == "echo:? "