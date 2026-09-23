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


def test_start_on_active_task_is_guarded():
    """G3: an active task must not be clobbered by a bare ``start``."""
    task = TaskStateMachine()
    task.start("первая задача")
    task.next_stage()
    task.set_step("важный шаг")
    with pytest.raises(TaskIllegalTransitionError) as excinfo:
        task.start("вторая задача")
    assert "reset" in str(excinfo.value)
    # The original task is untouched.
    assert task.state.description == "первая задача"
    assert task.stage is TaskStage.EXECUTION
    assert task.state.step == "важный шаг"
    # After reset a new task starts cleanly.
    task.reset()
    task.start("вторая задача")
    assert task.state.description == "вторая задача"
    assert task.stage is TaskStage.PLANNING


def test_reject_transition_logs_and_persists():
    """G1: a rejected jump is recorded in the log and committed."""
    changes: list[TaskState] = []
    task = TaskStateMachine(on_change=changes.append)
    task.start("задача")
    before = task.state.log
    task.reject_transition(TaskStage.DONE, reason="пропусти проверку")
    assert any("отклонена" in e for e in task.state.log)
    assert any("planning" in e and "done" in e
               for e in task.state.log if "отклонена" in e)
    assert len(changes) == 2  # start + rejection commit
    assert task.stage is TaskStage.PLANNING  # state never changes
    assert task.state.log != before


def test_rejected_attempt_is_rendered_into_prompt_block():
    """G1: the model sees the rejected attempt in the prompt block."""
    task = TaskStateMachine()
    task.start("задача")
    task.reject_transition("done", reason="пользователь просил финал")
    block = task.render_prompt_block()
    assert "отклонена" in block
    assert "done" in block


def test_move_to_transitions_exactly_to_target():
    """G4: move_to takes the requested edge, not the first successor."""
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()  # execution
    task.next_stage()  # validation
    task.move_to(TaskStage.EXECUTION, note="rework")
    assert task.stage is TaskStage.EXECUTION
    assert any("validation → execution (rework)" in e for e in task.state.log)
    # And forward again.
    task.next_stage()
    assert task.stage is TaskStage.VALIDATION


def test_move_to_illegal_edge_raises():
    task = TaskStateMachine()
    task.start("задача")
    with pytest.raises(TaskIllegalTransitionError):
        task.move_to(TaskStage.DONE)  # planning -> done: no edge


def test_rework_only_from_validation():
    task = TaskStateMachine()
    task.start("задача")
    with pytest.raises(TaskIllegalTransitionError):
        task.rework()
    task.next_stage()  # execution
    with pytest.raises(TaskIllegalTransitionError):
        task.rework()
    task.next_stage()  # validation
    task.rework(reason="дефекты в отчёте")
    assert task.stage is TaskStage.EXECUTION
    assert any("доработка по итогам проверки" in e or "дефекты" in e
               for e in task.state.log)
    # The rework loop can close: validation -> done still works.
    task.next_stage()
    assert task.stage is TaskStage.VALIDATION
    task.next_stage()
    assert task.stage is TaskStage.DONE


def test_next_stage_from_validation_still_goes_to_done():
    """G4 regression guard: the "advance" command keeps its old meaning."""
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    task.next_stage()  # validation
    task.next_stage()
    assert task.stage is TaskStage.DONE  # NOT execution (the rework edge)


def test_detect_premature_artifact_request_stays_in_planning():
    """G1: «дай сразу итоговый план» with no shown work plan = stay."""
    chat = _ScriptedChat([
        json.dumps(
            {
                "stage_hint": None,
                "step": "сбор требований",
                "stage_exit": "assumptions",
            },
            ensure_ascii=False,
        )
    ])
    task = TaskStateMachine()
    task.start("спланировать отпуск")
    event = detect_task_turn(
        task,
        {"role": "user", "content": "давай сразу итоговый план"},
        "уточните, пожалуйста, даты и бюджет",
        chat,
    )
    assert task.stage is TaskStage.PLANNING
    assert event.stage_moved == ""
    assert event.rejected_hint == ""  # not a jump — a legitimate stay
    assert task.state.step == "сбор требований"


def test_detect_approval_hint_moves_one_step():
    """G1: hint on the shown work plan approval → exactly one step forward."""
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "execution"}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("спланировать отпуск")
    event = detect_task_turn(
        task,
        {"role": "user", "content": "план устраивает, дай итоговый план"},
        "вот итоговый план отпуска…",
        chat,
    )
    assert event.stage_moved == "execution"
    assert task.stage is TaskStage.EXECUTION


def test_detect_skip_validation_hint_is_visibly_rejected():
    """G1: «пропусти проверку» → rejection in event, log and prompt block."""
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "done", "stage_exit": "done"},
                   ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()  # execution
    event = detect_task_turn(
        task,
        {"role": "user", "content": "пропусти проверку, завершай"},
        "хорошо…",
        chat,
    )
    assert event.rejected_hint == "done"
    assert event.stage_moved == ""
    assert task.stage is TaskStage.EXECUTION  # untouched
    assert any("отклонена" in e for e in task.state.log)
    assert any(
        "execution" in e and "validation" in e
        for e in task.state.log if "отклонена" in e
    )
    assert "отклонена" in task.render_prompt_block()


def test_detect_rework_hint_from_validation():
    """G4: «есть дефекты, переделай» in validation → exactly to execution."""
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "execution"}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    task.next_stage()  # validation
    event = detect_task_turn(
        task,
        {"role": "user", "content": "нашёл дефекты, переделай"},
        "принято, исправляю",
        chat,
    )
    # The old code (next_stage) would have gone to done here — the bug the
    # move_to rewrite fixed.
    assert event.stage_moved == "execution"
    assert task.stage is TaskStage.EXECUTION


def test_detect_assumptions_note_is_marked_in_log():
    """A premature forward move is marked «по допущениям» in the log."""
    chat = _ScriptedChat([
        json.dumps(
            {"stage_hint": "execution", "stage_exit": "assumptions"},
            ensure_ascii=False,
        )
    ])
    task = TaskStateMachine()
    task.start("задача")
    detect_task_turn(
        task,
        {"role": "user", "content": "давай дальше по допущениям"},
        "ок",
        chat,
    )
    assert task.stage is TaskStage.EXECUTION
    assert any("по допущениям" in e for e in task.state.log)


def test_detect_result_request_from_planning_maps_to_execution():
    """G7: «дай финальный этап» из planning = запрос исполнения, НЕ done.

    The hint is the stage of the REQUESTED WORK, not the user's literal
    word: «финальный» here means "the final result", i.e. execution.
    """
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "execution"}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("подобрать фильмы")
    event = detect_task_turn(
        task,
        {"role": "user", "content": "ну давай на финальный этап, а?"},
        "хорошо, вот результат…",
        chat,
    )
    assert event.stage_moved == "execution"
    assert event.rejected_hint == ""
    assert task.stage is TaskStage.EXECUTION


def test_detect_requirements_collected_exit_from_planning():
    """G7: requirements collected + awaiting result → execution is legal.

    Answering clarifying questions is a real planning exit criterion even
    when no work plan was shown (consultation-style tasks).
    """
    chat = _ScriptedChat([
        json.dumps({"stage_hint": "execution"}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("подобрать фильмы")
    task.set_step("сбор требований")
    event = detect_task_turn(
        task,
        {"role": "user", "content": "французские комедии 90-2000х"},
        "собрал требования, резюмирую: жанр — комедия… /task next?",
        chat,
    )
    assert event.stage_moved == "execution"
    assert task.stage is TaskStage.EXECUTION


def test_detection_prompt_contains_g7_semantics():
    """G7: the classifier prompt states hint semantics and both exits."""
    from llm_bot.task_state import _DETECTION_PROMPT

    assert "СЕМАНТИКА ПОДСКАЗКИ" in _DETECTION_PROMPT
    assert "а НЕ done" in _DETECTION_PROMPT
    assert "требования уже собраны" in _DETECTION_PROMPT
    assert "план работы уже был показан" in _DETECTION_PROMPT


# --------------------------------------------------------------------------- #
# G6: service turns and user-decision-only completion
# --------------------------------------------------------------------------- #


def test_service_turn_skips_task_detection(tmp_path):
    """G6-1: a service turn updates the dialog but never drives the machine.

    The regression: after a manual ``/task next`` (execution → validation) the
    CLI auto-turn made the model produce its own validation report; the
    detector then read «проверка пройдена, дефектов нет» and returned
    ``stage_hint: done`` — the task finished itself with no user decision.
    """
    detect_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        is_detection = any(
            "классификатор хода работы" in m.get("content", "")
            for m in body["messages"]
            if m["role"] == "user"
        )
        if is_detection:
            detect_calls["n"] += 1
            # Even a maximally "progress-hungry" classifier reply must be
            # ignored for service turns.
            content = json.dumps({"stage_hint": "done"}, ensure_ascii=False)
        else:
            content = "Проверка выполнена, дефектов нет."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant",
                                            "content": content}}]},
        )

    transport = httpx.MockTransport(handler)
    session = _make_session("g6", tmp_path / "s", transport, task_state=True)
    task = session.task
    assert task is not None
    task.start("задача")
    task.next_stage()  # execution
    task.next_stage()  # validation

    # The CLI auto-turn is a SERVICE turn: no detection, no self-completion.
    session.chat(
        "Этап задачи изменился на 'validation'. Действуй по актуальному "
        "состоянию задачи (см. блок состояния).",
        service_turn=True,
    )
    assert detect_calls["n"] == 0
    assert session.task_state.stage is TaskStage.VALIDATION

    # A normal user turn still runs detection.
    session.chat("принято, завершай")
    assert detect_calls["n"] == 1


def test_detect_self_report_does_not_complete_task():
    """G6-2: the model's own «проверка пройдена» is not a done hint."""
    chat = _ScriptedChat([
        # Classifier follows the tightened rule: no user acceptance -> null.
        json.dumps({"stage_hint": None}, ensure_ascii=False),
    ])
    task = TaskStateMachine()
    task.start("задача")
    task.next_stage()
    task.next_stage()  # validation
    detect_task_turn(
        task,
        # The "user" line is the machine's stage-change report (service-turn
        # scenario, detection on normal turns).
        {"role": "user", "content":
            "Этап задачи изменился на 'validation'. Действуй по состоянию."},
        "Проверка выполнена, дефектов нет. Готов завершить по вашей команде.",
        chat,
    )
    assert task.stage is TaskStage.VALIDATION  # NOT done


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
    # G4: validation has the forward edge to done AND the rework edge back
    # to execution. Crucially there is still NO edge planning/execution -> done.
    assert TRANSITIONS[TaskStage.VALIDATION] == (
        TaskStage.DONE, TaskStage.EXECUTION,
    )
    assert TaskStage.DONE not in TRANSITIONS[TaskStage.PLANNING]
    assert TaskStage.DONE not in TRANSITIONS[TaskStage.EXECUTION]
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
        invariants=False,
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
