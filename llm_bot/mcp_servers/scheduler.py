"""MCP server: the control panel for scheduled background tasks.

Through these tools the chat agent creates delayed/periodic jobs (reminders,
periodic data collection, regular summaries), inspects them and reads
aggregated results. Execution itself happens in a separate long-lived process:

    python scripts/scheduler_daemon.py

Tools (stateless — all state lives in the JSON store):

* ``schedule_task(title, action, payload, delay_seconds? | every_seconds? |
  daily_at?, group?)`` → task id;
* ``list_tasks(status?)`` — task table;
* ``cancel_task`` / ``pause_task`` / ``resume_task``;
* ``get_task_results(task_id, limit?)`` — run journal;
* ``task_digest(group? | task_id? | last_n?)`` — aggregated result view:
  run counters, fresh summaries, next fire times.

Speaks MCP over stdio: ONLY protocol frames go to stdout, diagnostics to
stderr. Store path comes from ``SCHEDULER_DB`` (default ``data/scheduler.json``).

Run from the project root for a smoke check (it will wait on stdin):
    python -m llm_bot.mcp_servers.scheduler
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from llm_bot.scheduler import (  # noqa: E402
    KIND_DAILY,
    KIND_INTERVAL,
    KIND_ONCE,
    SchedulerError,
    build_digest,
)
from llm_bot.scheduler_actions import validate_action  # noqa: E402

mcp = FastMCP("scheduler")


def _error(exc: Exception) -> str:
    """Uniform tool-error text (FastMCP turns raised exceptions into isError)."""
    raise ValueError(str(exc)) from exc


def _store():
    from llm_bot.scheduler import JsonSchedulerStore

    return JsonSchedulerStore()


@mcp.tool()
def schedule_task(
    title: str,
    action: str,
    payload: dict | None = None,
    delay_seconds: float | None = None,
    every_seconds: float | None = None,
    daily_at: str | None = None,
    group: str = "",
) -> str:
    """Создать фоновую задачу и вернуть её id.

    Ровно один вариант расписания из трёх:
    * delay_seconds  — однократно через N секунд (reminder и т.п.);
    * every_seconds  — периодически каждые N секунд (сбор данных);
    * daily_at       — ежедневно в HH:MM по локальному времени (сводки).

    action — ТОЛЬКО одно из: 'mcp_call', 'reminder', 'llm_summary'.
    Придуманные имена действий (например collect_usd_rate) отвергаются
    сразу при создании. НЕ указывайте имя инструмента в action —
    инструмент передаётся в payload:

    «reminder» — ТОЛЬКО когда пользователь явно просит напомнить.
    Для регулярного сбора данных нужен action='mcp_call'; если подходящего
    MCP-сервера нет — так и скажите пользователю, НЕ подменяя сбор данных
    напоминанием.

    * reminder:    {"action": "reminder", "payload": {"text": "позвонить"}};
    * add заметку через MCP: {"action": "mcp_call", "payload":
      {"server": "notes", "tool": "add_note",
       "arguments": {"text": "..."}}};
    * сводка:      {"action": "llm_summary", "payload": {"sources": "all"}}.

    ВАЖНО про llm_summary: он суммирует ЖУРНАЛ РЕЗУЛЬТАТОВ других задач
    (payload.sources: "all" | группа | id) и сам данные не собирает.
    Поэтому запрос «выдавай сводку по X каждые N» — это СВЯЗКА из двух
    задач: сборщик mcp_call(данные X, every_seconds=N) + llm_summary
    с sources на него (LLM-вызовы дорогие — сводку делают реже сбора).
    Создав обе задачи, объясните пользователю эту схему.

    Название задачи должно честно отражать действие: mcp_call-сбор
    называйте «сбор/снимок …», а не «сводка».

    group — свободная метка для выборок дайджеста («финансы», «сериалы»).
    """
    schedule = _schedule_from(
        delay_seconds=delay_seconds,
        every_seconds=every_seconds,
        daily_at=daily_at,
    )
    try:
        validate_action(action, payload)
        task = _store().add_task(
            title, action, schedule, payload or {}, group=group
        )
    except SchedulerError as exc:
        _error(exc)
    return (
        f"Задача создана: id={task.id}, «{task.title}», действие {task.action},"
        f" первый запуск {task.next_run}"
    )


def _schedule_from(
    *,
    delay_seconds: float | None,
    every_seconds: float | None,
    daily_at: str | None,
) -> dict:
    """Build the schedule dict; exactly one selector must be set."""
    given = [
        name
        for name, value in (
            (KIND_ONCE, delay_seconds),
            (KIND_INTERVAL, every_seconds),
            (KIND_DAILY, daily_at),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise SchedulerError(
            "Укажите РОВНО ОДНО расписание: delay_seconds, every_seconds "
            "или daily_at=HH:MM."
        )
    kind = given[0]
    if kind == KIND_ONCE:
        return {"kind": KIND_ONCE, "delay_seconds": delay_seconds}
    if kind == KIND_INTERVAL:
        return {"kind": KIND_INTERVAL, "every_seconds": every_seconds}
    return {"kind": KIND_DAILY, "at": str(daily_at)}


@mcp.tool()
def list_tasks(status: str | None = None) -> str:
    """Список задач (при указании *status* — отфильтрованный): id, название,
    действие, расписание, статус, следующий запуск."""
    tasks = _store().list_tasks(status=status)
    if not tasks:
        return "Задач не найдено."
    lines = []
    for t in tasks:
        sched = t.schedule
        if sched.get("kind") == KIND_DAILY:
            human = f"ежедневно в {sched.get('at')}"
        elif sched.get("kind") == KIND_INTERVAL:
            human = f"каждые {sched.get('every_seconds')} с"
        else:
            human = f"однократно через {sched.get('delay_seconds')} с"
        lines.append(
            f"id={t.id} [{t.status}] «{t.title}» — {t.action}, {human}"
            + (f", группа: {t.group}" if t.group else "")
            + (f", следующий запуск: {t.next_run}" if t.next_run else "")
        )
    return "\n".join(lines)


@mcp.tool()
def cancel_task(task_id: str) -> str:
    """Отменить задачу по id (выполняться больше не будет)."""
    try:
        task = _store().set_status(task_id, "cancelled")
    except SchedulerError as exc:
        _error(exc)
    return f"Задача {task.id} «{task.title}» отменена."


@mcp.tool()
def pause_task(task_id: str) -> str:
    """Поставить задачу на паузу (можно возобновить resume_task)."""
    try:
        task = _store().set_status(task_id, "paused")
    except SchedulerError as exc:
        _error(exc)
    return f"Задача {task.id} «{task.title}» на паузе."


@mcp.tool()
def resume_task(task_id: str) -> str:
    """Возобновить задачу после паузы (пропущенное время не догоняется)."""
    try:
        task = _store().set_status(task_id, "active")
    except SchedulerError as exc:
        _error(exc)
    return f"Задача {task.id} «{task.title}» снова активна."


@mcp.tool()
def get_task_results(task_id: str, limit: int = 10) -> str:
    """Журнал прогонов задачи: последние *limit* результатов (новые в конце)."""
    try:
        task = _store().get_task(task_id)
        results = _store().list_results(task_id=task_id, limit=limit)
    except SchedulerError as exc:
        _error(exc)
    if not results:
        return f"У задачи «{task.title}» пока нет результатов."
    lines = [f"Результаты «{task.title}» (показаны последние {len(results)}):"]
    for res in results:
        mark = "ок" if res.ok else "ОШИБКА"
        lines.append(f"- [{mark} {res.run_at}] {res.summary}")
    return "\n".join(lines)


@mcp.tool()
def task_digest(
    group: str | None = None,
    task_id: str | None = None,
    last_n: int = 3,
) -> str:
    """Агрегированная сводка по фоновым задачам: сколько прогонов, ок/ошибки,
    свежие результаты и ближайшие запуски. Без аргументов — по всем задачам;
    *group* — срез по группе, *task_id* — по одной задаче, *last_n* — сколько
    свежих результатов показывать на задачу."""
    store = _store()
    if task_id:
        try:
            tasks = [store.get_task(task_id)]
        except SchedulerError as exc:
            _error(exc)
    else:
        tasks = store.list_tasks(group=group)
    results = store.list_results()
    known = {t.id for t in tasks}
    results = [r for r in results if r.task_id in known]
    return build_digest(tasks, results, limit=last_n)


if __name__ == "__main__":
    # stdio transport is the FastMCP default; stdout is reserved for the
    # protocol, so keep every diagnostic message on stderr.
    mcp.run()
