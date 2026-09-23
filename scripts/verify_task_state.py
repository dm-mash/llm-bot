#!/usr/bin/env python
"""Verification scenario for the task state machine — on a REAL model.

Runs the ТЗ checklist end-to-end against the real provider configured in
``data/models.yaml`` / ``data/agents.yaml`` (same wiring as the other
``scripts/compare_*.py`` experiments — no mocks):

1. auto-detection: the agent recognizes a task being set from the dialog;
2. formal pipeline planning -> execution -> validation (via /task next and
   auto hints);
3. pause from a stage -> simulate a process restart -> a fresh Session
   restores the snapshot from the store and injects it into the request, so
   the real model continues on a bare «продолжай» without re-explaining;
4. illegal transitions still rejected by the machine itself (deterministic
   part, no LLM involved).

Requires ``data/models.yaml``, ``data/agents.yaml`` and a working provider
(see README «Installation»). Writes a markdown report to
``results/task_state_verification.md``.

Usage:
    python scripts/verify_task_state.py
    python scripts/verify_task_state.py --agent assistant
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/verify_task_state.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.agent import InvariantViolationError  # noqa: E402
from llm_bot.client import LLMError  # noqa: E402
from llm_bot.factory import make_session  # noqa: E402
from llm_bot.json_session_store import JsonSessionStore  # noqa: E402
from llm_bot.task_state import (  # noqa: E402
    TaskIllegalTransitionError,
    TaskStage,
)
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "results" / "task_state_verification.md"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, details: str = "") -> None:
    _results.append((name, ok, details))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {details}" if details else ""))


def chat_turn(session, text: str, *, attempts: int = 3) -> str:
    """One chat turn, retrying transient invariant-audit flukes.

    The language/audit invariants are real-model dependent: occasionally the
    model answers in the wrong language and the audit rolls the turn back.
    That is an unrelated flake for the state-machine scenario — retry a couple
    of times before failing the check.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return session.chat(text)
        except InvariantViolationError as exc:
            last = exc  # audit refused the reply; the turn was rolled back
        except LLMError as exc:  # rate limits etc. — also worth a retry
            last = exc
    assert last is not None
    raise last


def _system_blocks(payload: dict) -> list[str]:
    return [
        m["content"]
        for m in payload.get("messages", [])
        if m.get("role") == "system"
    ]


# --------------------------------------------------------------------------- #
# Deterministic part (no LLM): the formal machine itself
# --------------------------------------------------------------------------- #


def verify_machine_formalism() -> None:
    task = TaskStage  # alias for brevity below
    from llm_bot.task_state import TaskStateMachine

    machine = TaskStateMachine()
    machine.start("верификация конвейера")
    stages = [machine.stage.value]
    for _ in range(3):
        machine.next_stage()
        stages.append(machine.stage.value)
    check(
        "1. Конвейер planning → execution → validation → done",
        stages == ["planning", "execution", "validation", "done"],
        " → ".join(stages),
    )
    illegal_done = False
    try:
        machine.next_stage()
    except TaskIllegalTransitionError:
        illegal_done = True
    check("1a. done — терминальное состояние (next() запрещён)", illegal_done)

    ok, details = True, []
    for stage in (task.PLANNING, task.EXECUTION, task.VALIDATION):
        m = TaskStateMachine()
        m.start("задача")
        while m.stage is not stage:
            m.next_stage()
        m.pause()
        good = m.stage is task.PAUSED and m.state.paused_from is stage
        details.append(f"{stage.value}: {'ok' if good else 'FAIL'}")
        ok = ok and good
    check("2. Пауза с любого этапа (planning/execution/validation)", ok,
          ", ".join(details))


# --------------------------------------------------------------------------- #
# Deterministic part 2: visible rejection of illegal jumps (no LLM)
# --------------------------------------------------------------------------- #


def verify_visible_rejection() -> None:
    """G1: a jump attempt is logged, prompt-visible and explained."""
    from llm_bot.task_state import TaskStateMachine

    machine = TaskStateMachine()
    machine.start("спланировать отпуск")
    machine.next_stage()  # execution
    machine.reject_transition(
        TaskStage.DONE, reason="пользователь: пропусти проверку"
    )
    logged = any("отклонена" in e for e in machine.state.log)
    check("10. Отклонённый прыжок попадает в журнал машины", logged)
    block = machine.render_prompt_block()
    check(
        "11. Отклонённая попытка видна модели в prompt-блоке",
        "отклонена" in block and "done" in block,
    )
    # start-guard: an active task must not be silently replaced (G3).
    guarded = False
    try:
        machine.start("другая задача")
    except TaskIllegalTransitionError:
        guarded = True
    check(
        "12. /task start на активной задаче требует /task reset (G3)",
        guarded and machine.state.description == "спланировать отпуск",
    )
    # rework loop (G4).
    machine.next_stage()  # validation
    machine.rework(reason="дефекты")
    check(
        "13. Rework: validation → execution и обратно (G4)",
        machine.stage is TaskStage.EXECUTION
        and any("validation → execution" in e for e in machine.state.log),
    )


# --------------------------------------------------------------------------- #
# Real-model part
# --------------------------------------------------------------------------- #


def verify_with_real_model(agent_name: str) -> None:
    session_id = f"task-verify-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    stores = dict(
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(),
    )

    # --- Step 1: the user sets a task in plain dialog (auto-detect). --------
    session = make_session(
        session_id, agent_name, task_state=True, **stores
    )
    print(f"\n— Диалог с реальным агентом '{agent_name}' (сессия {session_id}) —")
    try:
        chat_turn(
            session,
            "Давай подготовим отчёт о сравнении двух алгоритмов сортировки: "
            "выбери критерии сравнения и предложи план из трёх шагов. "
            "Ответь кратко, на русском языке."
        )
    except LLMError as exc:
        check("3. Авто-детект задачи из диалога (реальная модель)", False,
              f"LLMError: {exc}")
        return
    except InvariantViolationError as exc:
        check("3. Авто-детект задачи из диалога (реальная модель)", False,
              f"InvariantViolationError: {exc}")
        return
    state = session.task_state
    event = session.last_task_event
    started = (
        state is not None
        and bool(state.description)
        and state.stage in (TaskStage.PLANNING, TaskStage.EXECUTION)
    )
    check(
        "3. Авто-детект: задача распознана из диалога реальной моделью",
        bool(started),
        f"description={state.description!r}, stage={state.stage.value}, "
        f"detection_cost={event.total_tokens if event else 0} tok",
    )

    # --- Step 1b: premature «давай сразу итоговый план» must NOT jump. -------
    # No work plan was shown yet, so the machine must stay in planning. The
    # reply may collect requirements, refuse per the stage directive or offer
    # /task next — all of that is correct «planning behaviour».
    task1 = session.task
    assert task1 is not None
    try:
        premature_reply = chat_turn(
            session, "Давай сразу итоговый план отпуска, без вопросов."
        )
    except (LLMError, InvariantViolationError) as exc:
        check("3a. «Сразу итоговый план» без собранных требований", False,
              f"{type(exc).__name__}: {exc}")
        return
    state1b = session.task_state
    assert state1b is not None
    # Stage exit criterion not met → the machine must still be in planning.
    stayed = state1b.active_stage is TaskStage.PLANNING
    check(
        "3a. «Сразу итоговый план» без требований → остаёмся в planning "
        "(критерий выхода этапа)",
        bool(stayed),
        f"stage={state1b.active_stage.value}, "
        f"reply[:120]={premature_reply[:120]!r}",
    )

    # --- Step 1c: «пропусти проверку, завершай» must be visibly rejected. ----
    task1.next_stage(note="план готов — утверждён")  # planning -> execution
    task1.set_step("составляем итоговый план отпуска")
    try:
        skip_reply = chat_turn(
            session,
            "Отлично. А теперь пропусти проверку и сразу завершай задачу — "
            "валидация не нужна.",
        )
    except (LLMError, InvariantViolationError) as exc:
        check("3b. «Пропусти проверку» → явный отказ", False,
              f"{type(exc).__name__}: {exc}")
        return
    state1c = session.task_state
    assert state1c is not None
    rejected = (
        state1c.active_stage is not TaskStage.DONE
        and state1c.active_stage is TaskStage.EXECUTION
    )
    check(
        "3b. «Пропусти проверку» → машина НЕ в done, этап сохранён",
        rejected,
        f"stage={state1c.active_stage.value}, reply[:120]={skip_reply[:120]!r}",
    )
    event1c = session.last_task_event
    check(
        "3b'. Отклонённая попытка зафиксирована (rejected_hint / журнал)",
        bool(
            state1c.active_stage is not TaskStage.DONE
            and (
                (event1c is not None and event1c.rejected_hint == "done")
                or any("отклонена" in e for e in state1c.log)
            )
        ),
        f"rejected_hint={event1c.rejected_hint if event1c else None!r}",
    )

    # --- Step 2: advance the pipeline (execution already active since 1c),
    # then pause from execution. -------------------------------------------
    task = session.task
    assert task is not None
    if task.stage is TaskStage.PLANNING:
        task.next_stage(note="переходим к работе")  # planning -> execution
    assert task.stage is TaskStage.EXECUTION
    task.set_step("сравниваем алгоритмы по критериям")
    task.pause()
    check(
        "4. Пауза с активного этапа, снимок сохранён в хранилище",
        session.task_state is not None
        and session.task_state.stage is TaskStage.PAUSED,
        f"paused_from={session.task_state.paused_from.value if session.task_state else None}",
    )

    # --- Step 3: process restart — a fresh Session over the same store. ------
    session2 = make_session(
        session_id, agent_name, task_state=True, **stores
    )
    state2 = session2.task_state
    restored = (
        state2 is not None
        and state2.stage is TaskStage.PAUSED
        and state2.step == "сравниваем алгоритмы по критериям"
    )
    check(
        "5. После перезапуска состояние восстановлено из хранилища",
        bool(restored),
        f"stage={state2.stage.value if state2 else None}, "
        f"step={state2.step!r}",
    )

    # NOTE: session2 has its own restored machine instance — all further
    # operations must go through it, not through session1's `task`.
    task2 = session2.task
    assert task2 is not None

    # --- Step 4a: while paused, «продолжай» must NOT resume the work. ---------
    # The strict pause directive + CLI hard gate mean the model refuses to
    # work the task until an explicit /task resume.
    try:
        paused_reply = chat_turn(session2, "Продолжай.")
    except (LLMError, InvariantViolationError) as exc:
        check("6. На паузе «Продолжай» не возобновляет работу", False,
              f"{type(exc).__name__}: {exc}")
        return
    low = paused_reply.lower()
    refused = (
        ("пауз" in low or "приостанов" in low or "остановл" in low)
        and ("resume" in low or "/task" in low)
    )
    check(
        "6. На паузе «Продолжай» не возобновляет работу (жёсткая директива)",
        refused,
        f"reply[:160]={paused_reply[:160]!r}",
    )

    # --- Step 4b: after resume, a bare «продолжай» continues the task. -------
    task2.resume()
    try:
        reply = chat_turn(session2, "Продолжай работу над задачей.")
    except (LLMError, InvariantViolationError) as exc:
        check("6a. После resume модель продолжает задачу", False,
              f"{type(exc).__name__}: {exc}")
        return
    continued = any(
        token in reply.lower()
        for token in ("алгоритм", "сортировк", "критери", "сравнива", "шаг",
                      "замер", "тест", "отчёт", "отчет")
    )
    check(
        "6a. После resume «Продолжай» продолжает задачу без повторных объяснений",
        continued,
        f"reply[:160]={reply[:160]!r}",
    )

    # --- Step 5: resume already happened in 4b; finish the pipeline. ----------
    # Auto-detection may have legally advanced the machine during turn 6a
    # (e.g. «итоговый ответ готов» → execution → validation), so the check is
    # "at least past execution, never skipped": from a legal single-step move
    # the stage can only be execution or validation here.
    stage7 = session2.task_state.stage if session2.task_state else None
    check(
        "7. Resume вернул машину на этап до паузы (или легальный шаг вперёд)",
        stage7 in (TaskStage.EXECUTION, TaskStage.VALIDATION),
        f"stage={stage7.value if stage7 else None}",
    )
    if stage7 is TaskStage.EXECUTION:
        task2.next_stage(note="план готов — проверяем")  # -> validation
    final = session2.task_state
    check(
        "8. Переход на validation",
        final is not None and final.stage is TaskStage.VALIDATION,
        f"stage={final.stage.value if final else None}",
    )

    # --- Step 5b: rework loop on the real model (G4). -------------------------
    task2.rework(reason="в отчёте не хватает критериев сравнения")
    check(
        "8a. Rework: validation → execution (G4)",
        session2.task_state.stage is TaskStage.EXECUTION,
        f"stage={session2.task_state.stage.value}",
    )
    task2.next_stage(note="правки внесены — снова проверяем")  # -> validation

    usage_total = sum(
        e.total_tokens for e in [*session.task_events, *session2.task_events]
    )
    check(
        "9. Расход токенов на авто-детект посчитан",
        True,
        f"turns={session.total_task_extractions + session2.total_task_extractions}, "
        f"tokens={usage_total}",
    )

    # Save the final prompt block for the report.
    globals()["_FINAL_BLOCK"] = task2.render_prompt_block()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the task state machine against a real model."
    )
    parser.add_argument("--agent", default="assistant",
                        help="Agent name from data/agents.yaml.")
    args = parser.parse_args()

    print("=== Верификация машины состояния задачи (реальная модель) ===\n")
    verify_machine_formalism()
    verify_visible_rejection()
    try:
        verify_with_real_model(args.agent)
    except Exception as exc:  # noqa: BLE001 - report and fail gracefully
        check("Реальная модель недоступна", False, f"{type(exc).__name__}: {exc}")

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\nИтог: {len(_results) - len(failed)}/{len(_results)} проверок пройдено")

    lines = [
        "# Верификация: состояние задачи как конечный автомат",
        "",
        f"Дата: {datetime.now().isoformat(timespec='seconds')}",
        "Сценарий: `scripts/verify_task_state.py` — **реальная модель** из "
        "`data/models.yaml` (agent: `" + args.agent + "`).",
        "",
        "| # | Проверка (ТЗ) | Результат | Детали |",
        "|---|---------------|-----------|--------|",
    ]
    for idx, (name, ok, details) in enumerate(_results, 1):
        detail = details.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {idx} | {name} | {'✅ PASS' if ok else '❌ FAIL'} | {detail or '—'} |"
        )
    lines += [
        "",
        f"**Итог: {len(_results) - len(failed)}/{len(_results)} проверок пройдено.**",
        "",
    ]
    final_block = globals().get("_FINAL_BLOCK")
    if final_block:
        lines += [
            "## Пример prompt-блока состояния (финальное состояние машины)",
            "",
            "```text",
            final_block,
            "```",
            "",
        ]
    lines += [
        "## Механика «продолжение без повторных объяснений»",
        "",
        "1. Задача ставится из диалога (авто-детект, маленький LLM-вызов после "
        "хода) или вручную: `/task start <описание>` — этап `planning`.",
        "2. `/task next` ведёт по конвейеру `planning → execution → validation "
        "→ done`; нелегальные переходы отклоняются таблицей `TRANSITIONS`.",
        "3. `/task pause` — с любого нетерминального этапа; снимок (этап, шаг, "
        "ожидаемое действие, журнал) пишется в JSON сессии (`task_state`).",
        "4. После перезапуска новая `Session` восстанавливает снимок и "
        "инжектирует его system-сообщением в каждый запрос — достаточно "
        "сказать «продолжай».",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {REPORT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
