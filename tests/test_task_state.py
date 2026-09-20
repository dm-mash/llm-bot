"""Tests for the task state machine (llm_bot.task_state) and its integration.

Covers the ТЗ requirements:

* formal transitions ``planning → execution → validation → done`` (illegal
  jumps rejected);
* pause from ANY stage (planning / execution / validation; not from done);
* continuation without re-explanation — the snapshot survives process restart
  via the session store and is injected into the request prefix;
* auto-detection drives the machine from the dialog, but never applies illegal
  transitions and never breaks a turn on classifier failure.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.stores import AgentConfig, ModelConfig
from llm_bot.task_state import (
    TRANSITIONS,
    TaskDetectionEvent,
    TaskIllegalTransitionError,
    TaskStage,
    TaskState,
    TaskStateMachine,
    detect_task_turn,
)


# --------------------------------------------------------------------------- #
# Pure state machine
# --------------------------------------------------------------------------- #


def test_pipeline_forward_transitions_are_legal():
    task = TaskStateMachine()
    task.start("сделать фичу")
    assert task.stage is TaskStage.PLANNING
    task.next_stage()
    assert task.stage is TaskStage.EXECUTION
    task.next_stage()
    assert task.stage is TaskStage.VALIDATION
    task.next_stage()
    assert task.stage is TaskStage.DONE


def test_illegal_transition_is_rejected():
    from llm_bot.task_state import next_stage

    # No skipping edges in the table: planning -> validation / done are absent.
    assert TRANSITIONS[TaskStage.PLANNING] == (TaskStage.EXECUTION,)
    assert TaskStage.VALIDATION not in TRANSITIONS[TaskStage.PLANNING]
    assert TaskStage.DONE not in TRANSITIONS[TaskStage.PLANNING]
    # Paused and done are dead ends for forward movement.
    with pytest.raises(TaskIllegalTransitionError):
        next_stage(TaskStage.DONE)
    with pytest.raises(TaskIllegalTransitionError):
        next_stage(TaskStage.PAUSED)
    # An unknown stage name is a bad input (ValueError family).
    with pytest.raises(ValueError):
        next_stage("неизвестный-этап")


def test_machine_stays_on_legal_edges_only():
    task = TaskStateMachine()
    task.start("задача")
    task.pause()
    # Forward movement is impossible while paused: resume first.
    with pytest.raises(TaskIllegalTransitionError):
        task.next_stage()
    assert task.stage is TaskStage.PAUSED
    task.resume()
    task.next_stage()  # planning -> execution
    assert task.stage is TaskStage.EXECUTION


def test_stage_directives_change_the_prompt_per_stage():
    task = TaskStateMachine()
    task.start("задача")

    task.next_stage(note="x")  # -> execution
    block = task.render_prompt_block()
    assert "ВЫПОЛНЕНИЯ" in block
    assert "ПЛАНИРОВАНИЯ" not in block

    task.next_stage()  # -> validation
    block = task.render_prompt_block()
    assert "ПРОВЕРКИ" in block
    assert "ВЫПОЛНЕНИЯ" not in block

    task.next_stage()  # -> done
    block = task.render_prompt_block()
    assert "ЗАВЕРШЕНА" in block


def test_next_stage_autofills_expected_action_when_unset():
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    assert task.state.expected_action == "выполнить текущий шаг"
    task.next_stage()
    assert task.state.expected_action == "проверить результат по критериям"
    task.next_stage()
    assert task.state.expected_action == "подготовить итоговую сводку"


def test_next_stage_keeps_user_expected_action():
    task = TaskStateMachine()
    task.start("задача")
    task.set_expected_action("моё действие")
    task.next_stage()
    assert task.state.expected_action == "моё действие"


def test_next_from_done_is_illegal():
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    task.next_stage()
    task.next_stage()
    assert task.stage is TaskStage.DONE
    with pytest.raises(TaskIllegalTransitionError):
        task.next_stage()


def test_pause_from_any_non_terminal_stage():
    for stage in (TaskStage.PLANNING, TaskStage.EXECUTION, TaskStage.VALIDATION):
        task = TaskStateMachine()
        task.start("задача")
        while task.stage is not stage:
            task.next_stage()
        task.pause()
        assert task.stage is TaskStage.PAUSED
        assert task.state.paused_from is stage


def test_pause_from_done_is_illegal():
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    task.next_stage()
    task.next_stage()
    with pytest.raises(TaskIllegalTransitionError):
        task.pause()


def test_pause_twice_is_illegal_and_log_is_not_duplicated():
    task = TaskStateMachine()
    task.start("задача")
    task.pause()
    with pytest.raises(TaskIllegalTransitionError):
        task.pause()
    pauses = [e for e in task.state.log if e.startswith("пауза")]
    assert pauses == [
        "пауза на этапе 'planning'"
    ], "repeated pause must not append a second log entry"


def test_resume_returns_to_the_original_stage_and_keeps_axes():
    task = TaskStateMachine()
    task.start("задача про автомат")
    task.set_step("пишу тесты")
    task.set_expected_action("запустить pytest")
    task.next_stage()  # -> execution
    task.pause()

    task.resume()
    assert task.stage is TaskStage.EXECUTION
    assert task.state.paused_from is None
    assert task.state.step == "пишу тесты"
    assert task.state.expected_action == "запустить pytest"
    assert any("продолжение" in entry for entry in task.state.log)


def test_resume_only_from_paused():
    task = TaskStateMachine()
    task.start("задача")
    with pytest.raises(TaskIllegalTransitionError):
        task.resume()


def test_snapshot_roundtrip():
    task = TaskStateMachine()
    task.start("задача")
    task.set_step("шаг 1")
    task.set_expected_action("что-то сделать")
    task.pause()

    data = task.state.to_dict()
    restored = TaskState.from_dict(data)
    assert restored.stage is TaskStage.PAUSED
    assert restored.paused_from is TaskStage.PLANNING
    assert restored.step == "шаг 1"
    assert restored.expected_action == "что-то сделать"
    assert restored.log == task.state.log


def test_transition_table_shape():
    assert TRANSITIONS[TaskStage.PLANNING] == (TaskStage.EXECUTION,)
    assert TRANSITIONS[TaskStage.EXECUTION] == (TaskStage.VALIDATION,)
    assert TRANSITIONS[TaskStage.VALIDATION] == (TaskStage.DONE,)
    assert TRANSITIONS[TaskStage.DONE] == ()
    assert TRANSITIONS[TaskStage.PAUSED] == ()


def test_prompt_block_contains_axes():
    task = TaskStateMachine()
    assert task.render_prompt_block() == ""  # inactive -> nothing injected
    task.start("автомат")
    task.set_step("шаг X")
    task.set_expected_action("действие Y")
    block = task.render_prompt_block()
    assert "planning" in block
    assert "шаг X" in block
    assert "действие Y" in block
    # Active (not paused) -> work directive, no pause directive.
    assert "Продолжай работу" in block
    assert "ПРИОСТАНОВЛЕНА" not in block


def test_prompt_block_on_pause_has_strict_directive():
    task = TaskStateMachine()
    task.start("автомат")
    task.pause()
    block = task.render_prompt_block()
    # Strict pause directive is present...
    assert "ПРИОСТАНОВЛЕНА" in block
    assert "Не выполняй" in block
    assert "/task resume" in block
    # ...and the contradictory "keep working" line is NOT present.
    assert "Продолжай работу" not in block


def test_on_change_hook_fires_on_every_change():
    changes: list[TaskState] = []
    task = TaskStateMachine(on_change=changes.append)
    task.start("задача")
    task.pause()
    task.resume()
    task.next_stage()
    assert len(changes) == 4


# --------------------------------------------------------------------------- #
# Auto-detection
# --------------------------------------------------------------------------- #


class _ScriptedChat:
    """Chat callable returning canned replies (records the requests)."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.replies.pop(0)


def test_detect_starts_task_from_dialog():
    chat = _ScriptedChat([
        json.dumps(
            {
                "new_task": "написать скрипт сортировки",
                "step": "уточнение требований",
                "expected_action": "согласовать ТЗ",
            },
            ensure_ascii=False,
        )
    ])
    task = TaskStateMachine()
    event = detect_task_turn(
        task,
        {"role": "user", "content": "напиши скрипт сортировки"},
        "хорошо, уточню требования",
        chat,
    )
    assert event.started
    assert task.stage is TaskStage.PLANNING
    assert task.state.description == "написать скрипт сортировки"
    assert task.state.step == "уточнение требований"
    assert task.state.expected_action == "согласовать ТЗ"


def test_detect_applies_only_legal_stage_hint():
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "execution"}, ensure_ascii=False),
        json.dumps({"stage_hint": "done"}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("задача")
    event = detect_task_turn(task, {"role": "user", "content": "u"}, "r", chat)
    assert event.stage_moved == "execution"
    assert task.stage is TaskStage.EXECUTION

    # execution -> done skips validation: rejected.
    event = detect_task_turn(task, {"role": "user", "content": "u"}, "r", chat)
    assert event.stage_moved == ""
    assert event.rejected_hint == "done"
    assert task.stage is TaskStage.EXECUTION


def test_detect_pause_and_resume_phrases():
    chat = _ScriptedChat([
        json.dumps({"paused": True}, ensure_ascii=False),
        json.dumps({"resumed": True}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("задача")
    detect_task_turn(task, {"role": "user", "content": "давай паузу"}, "ok", chat)
    assert task.stage is TaskStage.PAUSED
    detect_task_turn(task, {"role": "user", "content": "продолжай"}, "ok", chat)
    assert task.stage is TaskStage.PLANNING


def test_detect_classifier_failure_does_not_break_turn():
    chat = _ScriptedChat(["мусор без JSON"])
    task = TaskStateMachine()
    task.start("задача")
    event = detect_task_turn(task, {"role": "user", "content": "u"}, "r", chat)
    assert not event.recognized
    assert task.stage is TaskStage.PLANNING  # untouched


def test_detect_event_defaults():
    event = TaskDetectionEvent()
    assert event.total_tokens == 0
    assert not event.started


# --------------------------------------------------------------------------- #
# Session integration (factory + store persistence)
# --------------------------------------------------------------------------- #


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "reply"}}]},
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


def _make_session(session_id, directory, transport, **kwargs):
    agent_cfg = AgentConfig(
        name="assistant",
        model="openai",
        system_prompt="Ты помощник.",
    )
    model_cfg = ModelConfig(
        name="openai",
        base_url="https://example.test/v1",
        api_key="k",
        model="gpt",
    )
    return make_session(
        session_id,
        agent_cfg.name,
        model_store=_StubModelStore(model_cfg),
        agent_store=_StubAgentStore(agent_cfg),
        session_store=JsonSessionStore(str(directory)),
        transport=transport,
        **kwargs,
    )


def test_session_task_disabled_by_default(tmp_path):
    transport = httpx.MockTransport(lambda request: _ok_response())
    session = _make_session("t0", tmp_path / "s", transport)
    assert session.task is None
    assert session.task_state is None


def test_session_injects_task_block_into_request(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.read().decode())
        return _ok_response()

    transport = httpx.MockTransport(handler)
    session = _make_session(
        "t1", tmp_path / "s", transport, task_state=True,
        task_auto_detect=False,
    )
    task = session.task
    assert task is not None
    task.start("сортировка")
    task.set_step("выбор алгоритма")

    session.chat("продолжаем")

    system_contents = [
        m["content"] for m in captured["payload"]["messages"]
        if m["role"] == "system"
    ]
    assert any("Состояние задачи" in c for c in system_contents)
    assert any("сортировка" in c for c in system_contents)
    assert any("выбор алгоритма" in c for c in system_contents)
    # Manual change was persisted immediately (on_change hook).
    saved = JsonSessionStore(str(tmp_path / "s")).load_task_state("t1")
    assert saved is not None
    assert saved["description"] == "сортировка"


def test_pause_restart_resume_without_reexplaining(tmp_path):
    """The core ТЗ scenario: pause -> process restart -> continue.

    A fresh Session (new process) restores the paused state from the store and
    injects it into the request, so a bare "продолжай" needs no re-explanation.
    """
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.read().decode())
        return _ok_response()

    dir_ = str(tmp_path / "s")

    # Turn 1: start the task, move forward, pause.
    transport = httpx.MockTransport(handler)
    session = _make_session(
        "t2", dir_, transport, task_state=True, task_auto_detect=False
    )
    task = session.task
    task.start("рефакторинг модуля")
    task.next_stage()  # execution
    task.set_step("правлю тесты")
    task.set_expected_action("прогнать pytest")
    task.pause()
    assert session.task_state.stage is TaskStage.PAUSED

    # "Process restart": a brand-new session object over the same store.
    captured.clear()
    session2 = _make_session(
        "t2", dir_, httpx.MockTransport(handler),
        task_state=True, task_auto_detect=False,
    )
    assert session2.task_state is not None
    assert session2.task_state.stage is TaskStage.PAUSED
    assert session2.task_state.paused_from is TaskStage.EXECUTION
    assert session2.task_state.step == "правлю тесты"

    session2.chat("продолжай")

    system_contents = [
        m["content"] for m in captured["payload"]["messages"]
        if m["role"] == "system"
    ]
    joined = "\n".join(system_contents)
    assert "paused" in joined
    assert "execution" in joined
    assert "правлю тесты" in joined
    assert "прогнать pytest" in joined

    # Resuming after the restart works and persists.
    session2.task.resume()
    assert session2.task_state.stage is TaskStage.EXECUTION
    saved = JsonSessionStore(dir_).load_task_state("t2")
    assert saved["stage"] == "execution"


def test_session_auto_detect_runs_after_turn(tmp_path):
    """With auto-detect on, the machine starts the task from the dialog."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        is_detection = any(
            "классификатор хода работы" in m.get("content", "")
            for m in body["messages"]
            if m["role"] == "user"
        )
        if is_detection:
            calls["n"] += 1
            content = json.dumps(
                {
                    "new_task": "сортировка массива",
                    "step": "обсуждение алгоритма",
                },
                ensure_ascii=False,
            )
        else:
            content = "хорошо, обсудим алгоритм"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant",
                                            "content": content}}]},
        )

    transport = httpx.MockTransport(handler)
    session = _make_session("t3", tmp_path / "s", transport, task_state=True)
    session.chat("давай напишем сортировку массива")

    assert calls["n"] == 1
    assert session.task_state is not None
    assert session.task_state.description == "сортировка массива"
    assert session.total_task_extractions == 1
    assert session.last_task_event.started
    # The detection event was persisted too.
    saved = JsonSessionStore(str(tmp_path / "s")).load_task_state("t3")
    assert saved is not None
    assert saved["step"] == "обсуждение алгоритма"
