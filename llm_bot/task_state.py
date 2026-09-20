"""Task state machine: formalized progress of work on a task.

The state machine tracks three axes of a task:

* **stage** — a formal pipeline ``planning → execution → validation → done``
  enforced by a transition table (illegal jumps are rejected);
* **step** — free text describing what is being done right now;
* **expected_action** — free text describing what should happen next.

``paused`` is a first-class state reachable from any non-terminal stage; it
remembers the originating stage (``paused_from``) so :meth:`TaskStateMachine.resume`
can return the machine exactly where it was. Together with persistence this gives
the required behaviour: *pause at any stage* and *continue without re-explaining* —
after a restart the full snapshot is restored from the store and re-injected into
the model's system context on every request.

Two ways to drive the machine:

* **manual** — CLI slash-commands (``/task start``, ``/task pause``, ...) call the
  methods directly and take priority;
* **auto-detect** — :func:`detect_task_turn` makes a small LLM call after each
  turn (the same technique as :func:`llm_bot.memory.extract_memory` and the
  ``StickyFacts`` refresher) to recognize a task being set, hints of stage
  progress, and pause/resume phrases. Only *legal* transitions from
  :data:`TRANSITIONS` are ever applied, so the formalism survives LLM noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskIllegalTransitionError(ValueError):
    """Raised when a transition is not allowed by the transition table."""


# --------------------------------------------------------------------------- #
# Stages and the transition table
# --------------------------------------------------------------------------- #


class TaskStage(str, Enum):
    """Formal stages of the task pipeline.

    ``paused`` is a state, not a pipeline stage: it is entered via
    :meth:`TaskStateMachine.pause` from any non-terminal stage and remembers
    where to return via :attr:`TaskState.paused_from`.
    """

    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"
    PAUSED = "paused"


#: Legal forward transitions of the pipeline. Anything not listed is illegal.
TRANSITIONS: dict[TaskStage, tuple[TaskStage, ...]] = {
    TaskStage.PLANNING: (TaskStage.EXECUTION,),
    TaskStage.EXECUTION: (TaskStage.VALIDATION,),
    TaskStage.VALIDATION: (TaskStage.DONE,),
    TaskStage.DONE: (),
    TaskStage.PAUSED: (),
}

#: Stages from which ``pause`` is legal (any non-terminal, non-paused stage).
_PAUSABLE = (TaskStage.PLANNING, TaskStage.EXECUTION, TaskStage.VALIDATION)

#: Stages a ``resume`` may legally return to.
_RESUMABLE = _PAUSABLE

#: How many log entries are rendered into the prompt block (keep it short).
_PROMPT_LOG_TAIL = 3

#: Per-stage behaviour directives injected into the prompt block. These turn
#: a stage label into an actual instruction the model follows, so changing
#: stages via ``next_stage`` / ``resume`` observably changes behaviour.
_STAGE_DIRECTIVES: dict[TaskStage, str] = {
    TaskStage.PLANNING: (
        "Ты на этапе ПЛАНИРОВАНИЯ. Разбивай задачу на шаги, уточняй "
        "требования и предлагай план. Не реализовывай и не пиши код; жди "
        "утверждения плана (переход дальше — команда /task next)."
    ),
    TaskStage.EXECUTION: (
        "Ты на этапе ВЫПОЛНЕНИЯ. Реализуй текущий шаг; результат — "
        "конкретный артефакт (код, текст, команды), а не обсуждение. "
        "Один шаг за раз; закончив — предложи /task next."
    ),
    TaskStage.VALIDATION: (
        "Ты на этапе ПРОВЕРКИ. Проверь результат по критериям задачи, "
        "перечисли дефекты и риски. Ничего нового не добавляй; по итогам — "
        "команда /task next завершает задачу."
    ),
    TaskStage.DONE: (
        "Задача ЗАВЕРШЕНА. Давай только итоговую сводку; новую работу "
        "не начинай."
    ),
}

#: Default expected_action set automatically when entering a stage via
#: ``next_stage`` (only if the user has not set one). Keeps the state machine
#: self-describing: after a transition the model knows what is expected next.
_STAGE_DEFAULT_ACTIONS: dict[TaskStage, str] = {
    TaskStage.PLANNING: "утвердить план с пользователем",
    TaskStage.EXECUTION: "выполнить текущий шаг",
    TaskStage.VALIDATION: "проверить результат по критериям",
    TaskStage.DONE: "подготовить итоговую сводку",
}


def _stage(value: str | TaskStage) -> TaskStage:
    """Coerce *value* into a :class:`TaskStage` (accepts the raw string)."""
    if isinstance(value, TaskStage):
        return value
    try:
        return TaskStage(str(value).strip().lower())
    except ValueError:
        raise ValueError(
            f"Неизвестный этап задачи: {value!r}. "
            f"Доступны: {', '.join(s.value for s in TaskStage)}."
        ) from None


def next_stage(stage: TaskStage) -> TaskStage:
    """Return the pipeline successor of *stage* per :data:`TRANSITIONS`.

    Raises :class:`TaskIllegalTransitionError` when *stage* is terminal
    (``done``) or not a pipeline stage.
    """
    current = _stage(stage)
    targets = TRANSITIONS.get(current, ())
    if not targets:
        raise TaskIllegalTransitionError(
            f"Из этапа '{current.value}' нет перехода вперёд."
        )
    return targets[0]


# --------------------------------------------------------------------------- #
# Snapshot (persisted value object)
# --------------------------------------------------------------------------- #


@dataclass
class TaskState:
    """An immutable snapshot of the machine, used for persistence/rendering."""

    stage: TaskStage = TaskStage.PLANNING
    step: str = ""
    expected_action: str = ""
    description: str = ""
    paused_from: TaskStage | None = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "stage": self.stage.value,
            "step": self.step,
            "expected_action": self.expected_action,
            "description": self.description,
            "paused_from": self.paused_from.value if self.paused_from else None,
            "log": list(self.log),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        """Restore a snapshot from :meth:`to_dict` output (tolerant)."""
        if not isinstance(data, dict):
            return cls()
        paused_raw = data.get("paused_from")
        log = data.get("log")
        return cls(
            stage=_stage(data.get("stage", TaskStage.PLANNING.value)),
            step=str(data.get("step", "")),
            expected_action=str(data.get("expected_action", "")),
            description=str(data.get("description", "")),
            paused_from=_stage(paused_raw) if paused_raw else None,
            log=[str(entry) for entry in log] if isinstance(log, list) else [],
        )

    # -- convenience --------------------------------------------------------- #

    @property
    def is_paused(self) -> bool:
        """True when the machine is in the ``paused`` state."""
        return self.stage is TaskStage.PAUSED

    @property
    def is_done(self) -> bool:
        """True when the machine reached the terminal ``done`` state."""
        return self.stage is TaskStage.DONE

    @property
    def active_stage(self) -> TaskStage:
        """The pipeline stage the machine is effectively at.

        For a paused machine this is the remembered ``paused_from`` stage.
        """
        if self.stage is TaskStage.PAUSED and self.paused_from is not None:
            return self.paused_from
        return self.stage


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #

_TASK_HEADER = "Состояние задачи (task state machine):"


class TaskStateMachine:
    """A finite-state machine for one task, owned by a :class:`~llm_bot.agent.Session`.

    The machine enforces the transition table, keeps a rolling log of changes and
    renders a compact prompt block for injection into the system context so the
    model can continue the task without the user re-explaining it.
    """

    def __init__(
        self,
        state: TaskState | None = None,
        *,
        on_change: Callable[[TaskState], None] | None = None,
    ) -> None:
        self._state = state if state is not None else TaskState()
        self._on_change = on_change

    # -- reading ------------------------------------------------------------- #

    @property
    def state(self) -> TaskState:
        """The current snapshot (a copy; mutate via the machine's methods)."""
        return TaskState(
            stage=self._state.stage,
            step=self._state.step,
            expected_action=self._state.expected_action,
            description=self._state.description,
            paused_from=self._state.paused_from,
            log=list(self._state.log),
        )

    @property
    def stage(self) -> TaskStage:
        """The current stage (see :attr:`TaskState.stage`)."""
        return self._state.stage

    @property
    def is_active(self) -> bool:
        """True when a task has been started (any state, incl. paused/done)."""
        return bool(self._state.description) or self._state.stage is not TaskStage.PLANNING

    # -- internal helpers ---------------------------------------------------- #

    def _log(self, entry: str) -> None:
        self._state.log.append(entry)

    def _commit(self) -> None:
        """Notify the persistence hook, if wired."""
        if self._on_change is not None:
            self._on_change(self.state)

    # -- transitions --------------------------------------------------------- #

    def start(self, description: str) -> TaskState:
        """Start a new task: stage becomes ``planning``; clears previous state."""
        description = description.strip()
        if not description:
            raise ValueError("Описание задачи не может быть пустым.")
        self._state = TaskState(description=description)
        self._log(f"задача запущена: {description}")
        self._commit()
        return self.state

    def next_stage(self, *, note: str = "") -> TaskState:
        """Advance to the next pipeline stage per :data:`TRANSITIONS`.

        Uses the *actual* current stage (not the paused-from one): forward
        movement while ``paused`` is illegal — call :meth:`resume` first.

        When the expected action is empty *or* still holds a previous stage's
        auto-filled default, it is refreshed from :data:`_STAGE_DEFAULT_ACTIONS`
        for the new stage, so the machine stays self-describing after every
        transition. A user-set action (``set_expected_action``) is preserved.
        """
        current = self._state.stage
        target = next_stage(current)  # raises on illegal jump
        self._state.stage = target
        self._state.paused_from = None
        default = _STAGE_DEFAULT_ACTIONS.get(target)
        stale = self._state.expected_action in set(
            _STAGE_DEFAULT_ACTIONS.values()
        )
        if default and (not self._state.expected_action or stale):
            self._state.expected_action = default
        suffix = f" ({note.strip()})" if note.strip() else ""
        self._log(f"{current.value} → {target.value}{suffix}")
        self._commit()
        return self.state

    def pause(self) -> TaskState:
        """Pause from any non-terminal stage; remembers where to return.

        Pausing an already-paused machine is illegal (the origin is already
        remembered), so repeated ``/task pause`` cannot duplicate log entries.
        """
        if self._state.stage is TaskStage.PAUSED:
            raise TaskIllegalTransitionError(
                "Задача уже на паузе (этап "
                f"'{self._state.active_stage.value}'). Используйте /task resume."
            )
        current = self._state.active_stage
        if current not in _PAUSABLE:
            raise TaskIllegalTransitionError(
                f"Пауза невозможна из этапа '{current.value}'."
            )
        self._state.stage = TaskStage.PAUSED
        self._state.paused_from = current
        self._log(f"пауза на этапе '{current.value}'")
        self._commit()
        return self.state

    def resume(self) -> TaskState:
        """Return from ``paused`` to the remembered stage."""
        if self._state.stage is not TaskStage.PAUSED:
            raise TaskIllegalTransitionError(
                "Продолжение возможно только из состояния 'paused'."
            )
        target = self._state.paused_from
        if target is None or target not in _RESUMABLE:
            raise TaskIllegalTransitionError(
                "Сохранённый этап паузы отсутствует или недопустим."
            )
        self._state.stage = target
        self._state.paused_from = None
        self._log(f"продолжение: возврат на этап '{target.value}'")
        self._commit()
        return self.state

    # -- free-form axes ------------------------------------------------------ #

    def set_step(self, text: str) -> TaskState:
        """Set the current step (what is being done right now)."""
        self._state.step = text.strip()
        if self._state.step:
            self._log(f"шаг: {self._state.step}")
        self._commit()
        return self.state

    def set_expected_action(self, text: str) -> TaskState:
        """Set the expected action (what should happen next)."""
        self._state.expected_action = text.strip()
        if self._state.expected_action:
            self._log(f"ожидаемое действие: {self._state.expected_action}")
        self._commit()
        return self.state

    def reset(self) -> TaskState:
        """Drop the task entirely (back to the initial idle state)."""
        self._state = TaskState()
        self._commit()
        return self.state

    def load(self, state: TaskState) -> None:
        """Replace the internal state with *state* (used on session restore)."""
        self._state = TaskState(
            stage=state.stage,
            step=state.step,
            expected_action=state.expected_action,
            description=state.description,
            paused_from=state.paused_from,
            log=list(state.log),
        )
        self._commit()

    # -- prompt rendering ---------------------------------------------------- #

    def render_prompt_block(self) -> str:
        """Render the state as a system-prompt fragment.

        Injected ahead of the agent's role prompt on every request, this lets the
        model continue the task without the user re-explaining anything: the
        block names the stage, the pause flag, the current step, the expected
        action and the last log entries.
        """
        state = self._state
        if not self.is_active:
            return ""
        lines = [_TASK_HEADER]
        stage_label = state.stage.value
        if state.is_paused and state.paused_from is not None:
            lines.append(
                f"Этап: {stage_label} (пауза; до паузы — '{state.paused_from.value}')"
            )
        else:
            lines.append(f"Этап: {stage_label}")
        if state.description:
            lines.append(f"Задача: {state.description}")
        if state.step:
            lines.append(f"Текущий шаг: {state.step}")
        if state.expected_action:
            lines.append(f"Ожидаемое действие: {state.expected_action}")
        if state.log:
            lines.append("Журнал (последние события):")
            lines.extend(f"- {entry}" for entry in state.log[-_PROMPT_LOG_TAIL:])
        if state.is_paused:
            # Strict, non-contradictory directive: on pause the model must NOT
            # keep working the task (this line is what a soft hint lacked).
            lines.append(
                "Работа над задачей ПРИОСТАНОВЛЕНА пользователем. Не выполняй "
                "шаги задачи, не предлагай следующие шаги и не продолжай "
                "работу. На вопрос о задаче ответь одним предложением, что "
                "она на паузе и возобновляется командой «/task resume»."
            )
        else:
            # Stage directive: the label becomes an instruction, so advancing
            # the stage observably changes what the model does.
            directive = _STAGE_DIRECTIVES.get(state.stage)
            if directive:
                lines.append(directive)
            lines.append(
                "Продолжай работу по состоянию задачи, не требуя повторных "
                "объяснений."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Auto-detection (one small LLM call per turn, like extract_memory)
# --------------------------------------------------------------------------- #

_DETECTION_PROMPT = (
    "Ты — классификатор хода работы над задачей в диалоге. Ниже — текущее "
    "состояние задачи и новый фрагмент переписки (реплика пользователя и ответ "
    "ассистента).\n\n"
    "Правила:\n"
    "1. Если в состоянии написано «(задача не начата)» И пользователь просит "
    "что-то сделать, составить, спроектировать, подготовить, написать, "
    "обсудить план — ЭТО постановка задачи. Тогда верни объект вида "
    "{{\"new_task\": \"<краткое описание задачи>\", \"step\": \"<текущий шаг>\", "
    "\"expected_action\": \"<что делать дальше>\"}}. Обязательны \"new_task\" и "
    "\"step\"; это самый важный случай.\n"
    "2. Если задача уже идёт — верни JSON-объект с полями:\n"
    "   \"stage_hint\": \"planning\"|\"execution\"|\"validation\"|\"done\"|null "
    "(null = этап не менялся);\n"
    "   \"step\": \"<текущий шаг>\"|null;\n"
    "   \"expected_action\": \"<ожидаемое действие>\"|null;\n"
    "   \"paused\": true — пользователь просит паузу/отложить;\n"
    "   \"resumed\": true — пользователь просит продолжить («продолжай», "
    "\"давай дальше\").\n"
    "3. Не меняй то, о чём в реплике нет речи (оставляй null/false).\n"
    "4. Отвечай ТОЛЬКО одним JSON-объектом, без пояснений и маркдауна.\n\n"
    "Текущее состояние:\n{state}\n\n"
    "Фрагмент переписки:\n{transcript}"
)


@dataclass(frozen=True)
class TaskDetectionEvent:
    """Diagnostics of one auto-detection round (mirrors ``MemoryEvent``)."""

    recognized: bool = False
    started: bool = False
    stage_moved: str = ""
    paused: bool = False
    resumed: bool = False
    step_updated: bool = False
    action_updated: bool = False
    rejected_hint: str = ""
    request_tokens: int = 0
    reply_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens the detection call cost."""
        return self.request_tokens + self.reply_tokens


def detect_task_turn(
    task: TaskStateMachine,
    user_msg: dict[str, str],
    reply: str,
    chat: Callable[[list[dict[str, str]]], str],
) -> TaskDetectionEvent:
    """Classify a completed turn and drive the machine accordingly.

    Makes one small LLM call via *chat* (same technique as
    :func:`llm_bot.memory.extract_memory`): recognizes a task being set, legal
    stage hints, step/action updates and pause/resume phrases. Only transitions
    allowed by :data:`TRANSITIONS` are applied — an illegal hint is recorded in
    the event's ``rejected_hint`` and dropped, keeping the formalism intact.
    Never raises on classifier failures: a broken reply yields a
    ``recognized=False`` event and the machine is left untouched.
    """
    from llm_bot.tokens import count_message_tokens, count_messages_tokens

    state_block = task.render_prompt_block() or "(задача не начата)"
    transcript = (
        f"Пользователь: {user_msg.get('content', '')}\nАссистент: {reply}"
    )
    request = [{
        "role": "user",
        "content": _DETECTION_PROMPT.format(state=state_block, transcript=transcript),
    }]
    request_tokens = count_messages_tokens(request)
    output = chat(request)
    reply_tokens = count_message_tokens({"role": "assistant", "content": output})

    payload: dict[str, Any] = {}
    recognized = True
    try:
        payload = _parse_json_object(output)
    except Exception:  # noqa: BLE001 - never break a turn on detection failure
        recognized = False
        logger.warning(
            "[task] не удалось распознать ответ классификатора: %r", output
        )

    event = TaskDetectionEvent(
        recognized=recognized,
        request_tokens=request_tokens,
        reply_tokens=reply_tokens,
    )
    if not recognized:
        return event

    # 1. A task may be born from this turn.
    new_task = payload.get("new_task")
    if not task.is_active and isinstance(new_task, str) and new_task.strip():
        task.start(new_task)
        event = _replace(event, started=True)
        # Fall through: the same reply may already carry step/action hints.

    if not task.is_active:
        return event

    # 2. Pause / resume phrases win over stage movement.
    if payload.get("paused") is True and not task.state.is_paused:
        try:
            task.pause()
            event = _replace(event, paused=True)
        except TaskIllegalTransitionError:
            pass
    elif payload.get("resumed") is True and task.state.is_paused:
        try:
            task.resume()
            event = _replace(event, resumed=True)
        except TaskIllegalTransitionError:
            pass
    else:
        # 3. Stage hint — applied only when it is a legal forward transition.
        hint = payload.get("stage_hint")
        if isinstance(hint, str) and hint.strip():
            try:
                hinted = _stage(hint)
            except ValueError:
                hinted = None
            current = task.state.active_stage
            if hinted is not None and hinted in TRANSITIONS.get(current, ()):
                task.next_stage()
                event = _replace(event, stage_moved=hinted.value)
            elif hinted is not None and hinted is not current:
                event = _replace(event, rejected_hint=hinted.value)

    # 4. Free-form axes are updated freely.
    step = payload.get("step")
    if isinstance(step, str) and step.strip():
        task.set_step(step)
        event = _replace(event, step_updated=True)
    action = payload.get("expected_action")
    if isinstance(action, str) and action.strip():
        task.set_expected_action(action)
        event = _replace(event, action_updated=True)

    return event


def _replace(event: TaskDetectionEvent, **changes: Any) -> TaskDetectionEvent:
    """Return a copy of a frozen event with *changes* applied."""
    return TaskDetectionEvent(**{**event.__dict__, **changes})


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first ``{...}`` JSON object found in *text* (tolerant)."""
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])
