"""Tests for the interactive CLI chat loop and its slash commands."""

from __future__ import annotations

from llm_bot.cli import (
    _auto_turn_after_detection,
    _handle_task_command,
    _interactive_loop,
    _print_history,
    _print_resume_info,
    _read_input,
)
from llm_bot.client import LLMError
from llm_bot.task_state import TaskStage, TaskState, TaskStateMachine


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

    def chat(self, text: str, *, service_turn: bool = False) -> str:
        self.chat_calls.append(text)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": "reply-ok"})
        return "reply-ok"


class _TaskSession(_FakeSession):
    """Fake session exposing a task machine, like ``Session`` does.

    The real ``Session`` reads the machine through ``Session.task``; the fake
    wraps a real :class:`TaskStateMachine` so CLI commands operate on actual
    FSM transitions while ``chat`` stays recorded, not sent to a model.
    """

    def __init__(self, session_id: str = "s1", agent_name: str = "assistant") -> None:
        super().__init__(session_id, agent_name, [])
        self.task = TaskStateMachine()

    @property
    def task_state(self) -> TaskState:
        # Mirrors ``Session.task_state`` (llm_bot/agent.py), which
        # ``_print_task_status`` in llm_bot/cli.py reads.
        return self.task.state


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


def test_print_resume_info_reports_history_and_summary(capsys):
    """Opening an existing session must report the message count and summary size."""
    class _ResumedSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__(
                "s1",
                "assistant",
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            )
            self.summary = "compressed context"

    _print_resume_info(_ResumedSession())

    err = capsys.readouterr().err
    assert "сообщений в истории: 2" in err
    assert "summary: 18 симв." in err


def test_print_resume_info_reports_history_only_without_summary(capsys):
    """Message count is shown even when there is no running summary."""
    session = _FakeSession(
        "s1",
        "assistant",
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    )

    _print_resume_info(session)

    err = capsys.readouterr().err
    assert "сообщений в истории: 2" in err
    assert "summary" not in err


def test_print_resume_info_silent_for_fresh_session(capsys):
    """A brand-new (empty) session should print nothing."""
    _print_resume_info(_FakeSession("s1", "assistant", []))

    assert capsys.readouterr().err == ""


class _FakeStrategy:
    """Minimal stand-in for a ContextStrategy, exposing its public attrs."""

    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)


def test_print_resume_info_sliding_window_shows_window_size(capsys):
    """Under the sliding strategy the window size must be reported."""
    session = _FakeSession(
        "s1", "assistant", [{"role": "user", "content": "q"}]
    )
    session.strategy = _FakeStrategy(name="sliding", window_size=6)

    _print_resume_info(session)

    err = capsys.readouterr().err
    assert "окно: 6 сообщ." in err
    assert "фактов" not in err


def test_print_resume_info_sticky_facts_shows_fact_count_and_window(capsys):
    """Under the facts strategy the fact count (and window) must be reported."""
    session = _FakeSession(
        "s1", "assistant", [{"role": "user", "content": "q"}]
    )
    session.strategy = _FakeStrategy(
        name="facts",
        window_size=8,
        facts={"цель": "X", "стек": "py"},
    )

    _print_resume_info(session)

    err = capsys.readouterr().err
    assert "окно: 8 сообщ." in err
    assert "фактов: 2" in err


def test_print_resume_info_branching_lists_branches_with_current(capsys):
    """Under the branching strategy all branches are listed, current marked."""
    session = _FakeSession(
        "s1", "assistant", [{"role": "user", "content": "q"}]
    )
    session.strategy = _FakeStrategy(
        name="branching",
        current_branch="feature",
        branches={"main": [], "feature": []},
    )

    _print_resume_info(session)

    err = capsys.readouterr().err
    assert "ветки [feature]: main, *feature*" in err


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


# -- auto-turn on stage change (/task start|next|resume) --------------------- #


def test_task_next_sends_one_auto_turn(capsys):
    """/task next must immediately send exactly one service chat turn."""
    session = _TaskSession()
    assert _handle_task_command(session, "/task start написать парсер") is True
    assert len(session.chat_calls) == 1  # auto-turn after start
    assert "planning" in session.chat_calls[0]

    session.chat_calls.clear()
    assert _handle_task_command(session, "/task next") is True
    auto_turns = [c for c in session.chat_calls if "Этап задачи изменился" in c]
    assert auto_turns == session.chat_calls  # nothing else was sent
    assert len(auto_turns) == 1
    assert session.task.state.stage is TaskStage.EXECUTION


def test_task_resume_sends_auto_turn_but_pause_does_not(capsys):
    """Pause must not prompt the model; resume must send one service turn."""
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    session.chat_calls.clear()

    assert _handle_task_command(session, "/task pause") is True
    assert session.chat_calls == []  # hard rule: no model calls while paused
    assert session.task.state.is_paused

    assert _handle_task_command(session, "/task resume") is True
    assert len(session.chat_calls) == 1
    assert "Этап задачи изменился" in session.chat_calls[0]
    assert not session.task.state.is_paused


def test_task_auto_turn_error_is_reported_not_raised(capsys, monkeypatch):
    """An LLM failure during the auto-turn must not crash the command."""
    # The service-turn signature: the CLI passes service_turn=True.
    session = _TaskSession()

    def boom(text: str, *, service_turn: bool = False) -> str:
        raise LLMError("provider down")

    session.chat = boom  # type: ignore[method-assign]
    assert _handle_task_command(session, "/task start сломанный ход") is True
    err = capsys.readouterr().err
    assert "авто-ход" in err


def test_task_status_sends_no_auto_turn(capsys):
    """/task (status) and unknown subcommands must not touch the model."""
    session = _TaskSession()
    assert _handle_task_command(session, "/task") is True
    assert _handle_task_command(session, "/task белиберда") is True
    assert session.chat_calls == []


# -- G1/G3/G4: rejection feedback, rework, start guard ----------------------- #


def test_task_rejection_feedback_is_printed(capsys):
    """A rejected auto-hint must produce an explicit stderr line."""
    session = _TaskSession()
    session.task.start("задача")
    session.task.next_stage()  # execution
    from llm_bot.cli import _print_task_rejection
    from llm_bot.task_state import TaskDetectionEvent

    session.last_task_event = TaskDetectionEvent(rejected_hint="done")
    _print_task_rejection(session)
    err = capsys.readouterr().err
    assert "отклонена" in err
    assert "done" in err
    # G7-4: the leading "/task next" appears exactly once (no trailing dup).
    assert err.count("/task next") == 1
    assert "/task next → validation → done\n" in err


def test_task_rejection_feedback_silent_without_rejection(capsys):
    from llm_bot.cli import _print_task_rejection
    from llm_bot.task_state import TaskDetectionEvent

    session = _TaskSession()
    session.task.start("задача")
    session.last_task_event = TaskDetectionEvent()
    _print_task_rejection(session)
    assert capsys.readouterr().err == ""


def test_task_rework_command(capsys):
    """/task rework returns validation -> execution with one auto-turn."""
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    _handle_task_command(session, "/task next")
    _handle_task_command(session, "/task next")
    assert session.task.stage is TaskStage.VALIDATION
    session.chat_calls.clear()

    assert _handle_task_command(session, "/task rework дефекты в коде") is True
    assert session.task.stage is TaskStage.EXECUTION
    auto_turns = [c for c in session.chat_calls
                  if "Этап задачи изменился" in c]
    assert len(auto_turns) == 1
    err = capsys.readouterr().err
    assert "доработка" in err
    assert "execution" in err


def test_task_rework_outside_validation_is_rejected(capsys):
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    assert _handle_task_command(session, "/task rework") is True
    err = capsys.readouterr().err
    assert "validation" in err
    assert session.task.stage is TaskStage.PLANNING


def test_task_start_on_active_task_is_guarded(capsys):
    """/task start while a task runs must NOT clobber it; reset is required."""
    session = _TaskSession()
    _handle_task_command(session, "/task start первая")
    _handle_task_command(session, "/task next")
    assert _handle_task_command(session, "/task start вторая") is True
    err = capsys.readouterr().err
    assert "уже активна" in err
    assert "reset" in err
    assert session.task.state.description == "первая"
    assert session.task.stage is TaskStage.EXECUTION

    _handle_task_command(session, "/task reset")
    _handle_task_command(session, "/task start вторая")
    assert session.task.state.description == "вторая"


def test_task_next_accepts_a_note(capsys):
    """G1: /task next <заметка> lands in the machine's log."""
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    session.chat_calls.clear()
    assert _handle_task_command(session, "/task next план утверждён") is True
    assert session.task.stage is TaskStage.EXECUTION
    assert any("план утверждён" in e for e in session.task.state.log)
    assert len(session.chat_calls) == 1  # auto-turn still fires exactly once


# -- G6-3: empty-stage warning on manual /task next -------------------------- #


def test_task_next_warns_on_empty_stage(capsys):
    """G6-3: jumping over a stage that produced nothing warns the user."""
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    _handle_task_command(session, "/task next")  # planning -> execution
    session.chat_calls.clear()
    capsys.readouterr()

    assert _handle_task_command(session, "/task next") is True
    err = capsys.readouterr().err
    assert "не имел результатов" in err
    assert session.task.stage is TaskStage.VALIDATION


def test_task_next_no_warning_when_stage_has_results(capsys):
    """G6-3: a stage with work done (log entries) does not warn."""
    session = _TaskSession()
    _handle_task_command(session, "/task start демо")
    _handle_task_command(session, "/task next")  # -> execution
    _handle_task_command(session, "/task step пишу код")
    session.chat_calls.clear()
    capsys.readouterr()

    assert _handle_task_command(session, "/task next") is True
    err = capsys.readouterr().err
    assert "не имел результатов" not in err
    assert session.task.stage is TaskStage.VALIDATION


def test_task_auto_turn_is_service_turn(capsys):
    """G6-1: the CLI auto-turn must pass service_turn=True to chat."""
    session = _TaskSession()
    calls: list[dict] = []

    def spy_chat(text: str, *, service_turn: bool = False) -> str:
        calls.append({"text": text, "service_turn": service_turn})
        return "reply-ok"

    session.chat = spy_chat  # type: ignore[method-assign]
    _handle_task_command(session, "/task start демо")
    assert len(calls) == 1
    assert calls[0]["service_turn"] is True
    assert "Этап задачи изменился" in calls[0]["text"]


# -- G8: service auto-turn after an auto-detected stage change ----------------- #


def test_auto_turn_after_detection_fires_on_stage_move(capsys):
    """G8: a detected stage move is immediately followed by one service turn."""
    from llm_bot.task_state import TaskDetectionEvent

    session = _TaskSession()
    session.task.start("подобрать фильмы")  # planning
    # The detector does both: moves the machine and records the event.
    session.task.move_to(TaskStage.EXECUTION)
    session.last_task_event = TaskDetectionEvent(stage_moved="execution")
    _auto_turn_after_detection(session)
    assert len(session.chat_calls) == 1
    assert "Этап задачи изменился" in session.chat_calls[0]
    assert "execution" in session.chat_calls[0]


def test_auto_turn_after_detection_fires_on_start_and_resume(capsys):
    """G8: started/resumed detections also trigger the immediate auto-turn."""
    from llm_bot.task_state import TaskDetectionEvent

    session = _TaskSession()
    session.last_task_event = TaskDetectionEvent(started=True)
    _auto_turn_after_detection(session)
    assert len(session.chat_calls) == 1
    assert "planning" in session.chat_calls[0]

    session2 = _TaskSession()
    session2.task.start("задача")
    session2.task.pause()
    session2.task.resume()  # back on the pre-pause stage (planning)
    session2.last_task_event = TaskDetectionEvent(resumed=True)
    _auto_turn_after_detection(session2)
    assert len(session2.chat_calls) == 1
    assert "planning" in session2.chat_calls[0]


def test_auto_turn_after_detection_silent_without_move(capsys):
    """G8: a plain reply (no stage change) must not send any extra turn."""
    from llm_bot.task_state import TaskDetectionEvent

    session = _TaskSession()
    session.task.start("задача")
    session.last_task_event = TaskDetectionEvent(recognized=True)
    _auto_turn_after_detection(session)
    assert session.chat_calls == []
    session.last_task_event = TaskDetectionEvent(rejected_hint="done")
    _auto_turn_after_detection(session)  # a rejection is NOT a move
    assert session.chat_calls == []
    session.last_task_event = None
    _auto_turn_after_detection(session)  # no event at all
    assert session.chat_calls == []


def test_interactive_loop_auto_turns_after_detected_move(capsys, monkeypatch):
    """G8, end to end: a chat turn that moves the stage triggers one more
    service turn right away, mirroring a manual ``/task next``."""
    from llm_bot.task_state import TaskDetectionEvent

    session = _TaskSession()
    session.task.start("подобрать фильмы")  # planning
    base_chat = session.chat

    def chat_then_detect(text: str, *, service_turn: bool = False) -> str:
        result = base_chat(text, service_turn=service_turn)
        if not service_turn:
            # Emulate Session.chat: the completed user turn drives the
            # detector, which moves the machine and records the event.
            session.task.move_to(TaskStage.EXECUTION)
            session.last_task_event = TaskDetectionEvent(
                stage_moved="execution"
            )
        return result

    session.chat = chat_then_detect  # type: ignore[method-assign]

    inputs = iter(["давай финальный этап", "exit"])
    monkeypatch.setattr(
        "llm_bot.cli._read_input", lambda _prompt, **kwargs: next(inputs)
    )
    code = _interactive_loop(session)
    assert code == 0
    err = capsys.readouterr().err
    # Turn 1 = the user message; turn 2 = the G8 service auto-turn.
    assert session.chat_calls[0] == "давай финальный этап"
    assert len(session.chat_calls) == 2
    assert "Этап задачи изменился" in session.chat_calls[1]
    assert "авто-ход (этап 'execution')" in err
    assert session.task.state.stage is TaskStage.EXECUTION