"""Task scheduler core: models, schedule math and the JSON store.

The scheduler lets the agent create **delayed or periodic tasks** (reminders,
periodic data collection, regular summaries) and returns **aggregated results**
on demand. Architecture is two independent processes over one shared store:

* a stateless MCP server ("control panel") — creates/inspects tasks;
* a long-lived daemon ("executor") — runs due tasks 24/7 and writes results.

Because those processes are separate, the JSON store guards every mutation
with an inter-process ``fcntl.flock`` (thread locks are not enough), and every
write is atomic (tmp file + rename), following :mod:`llm_bot.notes_api`.

All timestamps are stored as UTC ISO strings; a ``daily`` schedule's ``at``
time (``HH:MM``) is interpreted in the *local* timezone of the process that
computes the next run (the daemon), which is what a human user expects.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Environment variable holding the scheduler database path.
_ENV_SCHEDULER_DB = "SCHEDULER_DB"
DEFAULT_DB_PATH = Path("data") / "scheduler.json"

# Task statuses.
STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"  # a once-task exhausted its retries without success

# Retry policy for failed ``once`` tasks (interval/daily retry at their next
# scheduled slot by design).
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 30.0

# Schedule kinds.
KIND_ONCE = "once"
KIND_INTERVAL = "interval"
KIND_DAILY = "daily"

# Action names.
ACTION_MCP_CALL = "mcp_call"
ACTION_REMINDER = "reminder"
ACTION_LLM_SUMMARY = "llm_summary"


class SchedulerError(Exception):
    """Raised for scheduler user errors (bad schedule, unknown task id...)."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledTask:
    """One schedulable task definition."""

    id: str
    title: str
    action: str  # mcp_call | reminder | llm_summary
    schedule: dict  # {"kind": ..., ...} — see next_run_after()
    payload: dict = field(default_factory=dict)
    group: str = ""  # free-form grouping for digests
    status: str = STATUS_ACTIVE
    created_at: str = ""
    next_run: str = ""  # empty until the daemon first computes it
    last_run: str = ""
    attempts: int = 0  # execution attempts so far (for once-task retries)
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        schedule = data.get("schedule")
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            action=str(data.get("action", "")),
            schedule=dict(schedule) if isinstance(schedule, dict) else {},
            payload=dict(data.get("payload") or {}),
            group=str(data.get("group", "")),
            status=str(data.get("status", STATUS_ACTIVE)),
            created_at=str(data.get("created_at", "")),
            next_run=str(data.get("next_run", "")),
            last_run=str(data.get("last_run", "")),
            attempts=int(data.get("attempts", 0) or 0),
            max_attempts=int(data.get("max_attempts", DEFAULT_MAX_ATTEMPTS) or DEFAULT_MAX_ATTEMPTS),
        )


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one execution attempt (success or failure)."""

    task_id: str
    run_at: str  # UTC ISO at execution time
    ok: bool
    summary: str = ""  # human-readable aggregate / reminder text
    details: dict = field(default_factory=dict)  # raw run data

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskResult":
        return cls(
            task_id=str(data.get("task_id", "")),
            run_at=str(data.get("run_at", "")),
            ok=bool(data.get("ok", False)),
            summary=str(data.get("summary", "")),
            details=dict(data.get("details") or {}),
        )


# ---------------------------------------------------------------------------
# Time helpers (pure functions — injectable clock for tests)
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    """Format a datetime as an ISO string (UTC, seconds precision)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime | None:
    """Parse an ISO string back to an aware UTC datetime (``None`` if bad)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def new_task_id() -> str:
    """Short filesystem-safe task id (12 hex chars)."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Schedule math (pure, testable — no sleeps anywhere)
# ---------------------------------------------------------------------------


def next_run_after(schedule: dict, now: datetime | None = None) -> datetime:
    """Compute the first run moment strictly after *now* for a schedule.

    Supported kinds:

    * ``{"kind": "once", "delay_seconds": N}`` — one run N seconds from now;
    * ``{"kind": "interval", "every_seconds": N}`` — every N seconds;
    * ``{"kind": "daily", "at": "HH:MM"}`` — every day at local HH:MM.

    Raises :class:`SchedulerError` on unknown kinds or bad values so a typo in
    a payload becomes a visible user error, not a silently dead task.
    """
    now = now or utc_now()
    kind = str(schedule.get("kind", "")).strip().lower()
    if kind == KIND_ONCE or kind == KIND_INTERVAL:
        key = "delay_seconds" if kind == KIND_ONCE else "every_seconds"
        seconds = schedule.get(key)
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise SchedulerError(
                f"Расписание {kind!r}: {key} должно быть положительным числом."
            )
        return now + timedelta(seconds=seconds)
    if kind == KIND_DAILY:
        raw = str(schedule.get("at", "")).strip()
        parts = raw.split(":")
        if len(parts) != 2:
            raise SchedulerError(
                "Расписание daily: поле at должно быть в формате HH:MM."
            )
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise SchedulerError(
                f"Расписание daily: нечисловое время {raw!r}."
            ) from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise SchedulerError(
                f"Расписание daily: время {raw!r} вне диапазона 00:00–23:59."
            )
        local = now.astimezone()  # daemon's local zone
        candidate = local.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate <= local:
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)
    raise SchedulerError(
        f"Неизвестный вид расписания {kind!r}. "
        f"Доступны: {KIND_ONCE}, {KIND_INTERVAL}, {KIND_DAILY}."
    )


def validate_schedule(schedule: dict) -> None:
    """Raise :class:`SchedulerError` unless *schedule* is well-formed."""
    if not isinstance(schedule, dict):
        raise SchedulerError("Расписание должно быть словарём.")
    next_run_after(schedule)  # any unsupported kind / value raises here


def is_due(task: ScheduledTask, now: datetime | None = None) -> bool:
    """True when an active task's ``next_run`` has arrived (or was missed)."""
    if task.status != STATUS_ACTIVE:
        return False
    nxt = parse_iso(task.next_run)
    if nxt is None:
        return False
    return nxt <= (now or utc_now())


def advance_after_run(
    task: ScheduledTask,
    now: datetime | None = None,
    *,
    ok: bool = True,
) -> ScheduledTask:
    """Return the task state after one execution attempt at *now*.

    ``attempts`` counts *consecutive failures* and is reset by any success —
    a healthy periodic task stays at 0 instead of drifting past its retry
    budget (live beta: an every-minute task showed «попытка 4/3»).

    * ``once`` + success → status ``done``, ``next_run`` cleared;
    * ``once`` + failure → retry: ``next_run`` = now + retry delay while
      attempts remain, otherwise terminal status ``failed`` (never a fake
      ``done`` — a failed run must stay visibly unfinished);
    * ``interval``→ ``next_run = now + every_seconds`` (catch-up: a missed
      backlog during downtime is NOT replayed — the next run is counted from
      the moment of execution; failed runs simply wait for the next slot);
    * ``daily``   → ``next_run`` = tomorrow at the same local HH:MM.
    """
    now = now or utc_now()
    kind = str(task.schedule.get("kind", "")).lower()
    attempts = 0 if ok else task.attempts + 1
    if kind == KIND_ONCE:
        if ok:
            return replace(
                task,
                status=STATUS_DONE,
                attempts=attempts,
                last_run=iso(now),
                next_run="",
            )
        if attempts >= max(1, task.max_attempts):
            return replace(
                task,
                status=STATUS_FAILED,
                attempts=attempts,
                last_run=iso(now),
                next_run="",
            )
        retry_at = now + timedelta(seconds=DEFAULT_RETRY_DELAY_SECONDS)
        return replace(
            task, attempts=attempts, last_run=iso(now), next_run=iso(retry_at)
        )
    nxt = next_run_after(task.schedule, now)
    return replace(task, attempts=attempts, last_run=iso(now), next_run=iso(nxt))


def replace(task: ScheduledTask, **changes) -> ScheduledTask:
    """Return a copy of *task* with the given fields replaced."""
    data = task.to_dict()
    data.update(changes)
    return ScheduledTask.from_dict(data)


# ---------------------------------------------------------------------------
# Digest (aggregated result view)
# ---------------------------------------------------------------------------


def build_digest(
    tasks: list[ScheduledTask],
    results: list[TaskResult],
    *,
    limit: int = 10,
) -> str:
    """Render an aggregated human-readable digest over tasks and their runs."""
    if not tasks:
        return "Задач не найдено."
    by_task: dict[str, list[TaskResult]] = {}
    for res in results:
        by_task.setdefault(res.task_id, []).append(res)
    lines: list[str] = []
    for task in tasks:
        runs = by_task.get(task.id, [])
        ok_n = sum(1 for r in runs if r.ok)
        fail_n = len(runs) - ok_n
        head = f"▪ {task.title} [{task.status}]"
        if task.group:
            head += f" ({task.group})"
        lines.append(head)
        lines.append(
            f"  прогонов: {len(runs)} (ок {ok_n}, ошибок {fail_n})"
        )
        nxt = parse_iso(task.next_run)
        if nxt and task.status == STATUS_ACTIVE:
            lines.append(f"  следующий запуск: {nxt.astimezone():%Y-%m-%d %H:%M}")
        if runs:
            fresh = runs[-limit:]
            for res in reversed(fresh):
                mark = "✔" if res.ok else "✖"
                when = res.run_at
                first = (res.summary or "").splitlines()
                text = first[0] if first else ""
                if len(text) > 120:
                    text = text[:117] + "…"
                lines.append(f"  {mark} {when}: {text}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Store (inter-process safe JSON file)
# ---------------------------------------------------------------------------


class JsonSchedulerStore:
    """JSON-file store shared by the MCP server and the daemon.

    File layout: a single JSON object ``{"tasks": [...], "results": [...]}``.
    Every mutation takes an exclusive ``fcntl.flock`` on a sidecar lock file,
    so the stateless MCP server and the 24/7 daemon (separate processes) never
    interleave read-modify-write cycles. Writes are atomic (tmp + rename).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        raw = os.getenv(_ENV_SCHEDULER_DB) if path is None else str(path)
        self.path = Path(raw) if raw else DEFAULT_DB_PATH
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = threading.Lock()  # within-process safety too

    # -- low-level ------------------------------------------------------------

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"tasks": [], "results": []}
        if not isinstance(data, dict):
            return {"tasks": [], "results": []}
        data.setdefault("tasks", [])
        data.setdefault("results", [])
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    class _Guard:
        """Context manager: exclusive inter-process lock on the lock file."""

        def __init__(self, store: "JsonSchedulerStore") -> None:
            self._store = store

        def __enter__(self) -> "JsonSchedulerStore":
            self._store.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._store.lock_path, "a+", encoding="utf-8")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            return self._store

        def __exit__(self, *exc) -> None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()

    def _locked(self) -> "JsonSchedulerStore._Guard":
        return JsonSchedulerStore._Guard(self)

    # -- tasks -----------------------------------------------------------------

    def add_task(
        self,
        title: str,
        action: str,
        schedule: dict,
        payload: dict | None = None,
        *,
        group: str = "",
        now: datetime | None = None,
    ) -> ScheduledTask:
        """Validate inputs, create an active task and persist it."""
        if not (title or "").strip():
            raise SchedulerError("Название задачи не может быть пустым.")
        if not (action or "").strip():
            raise SchedulerError("Не указано действие задачи (action).")
        validate_schedule(schedule)
        now = now or utc_now()
        task = ScheduledTask(
            id=new_task_id(),
            title=title.strip(),
            action=action.strip(),
            schedule=dict(schedule),
            payload=dict(payload or {}),
            group=(group or "").strip(),
            status=STATUS_ACTIVE,
            created_at=iso(now),
            next_run=iso(next_run_after(schedule, now)),
        )
        with self._thread_lock, self._locked() as _:
            data = self._read()
            data["tasks"].append(task.to_dict())
            self._write(data)
        return task

    def get_task(self, task_id: str) -> ScheduledTask:
        """Return the task with *task_id* or raise :class:`SchedulerError`."""
        with self._thread_lock, self._locked() as _:
            data = self._read()
        for item in data["tasks"]:
            if str(item.get("id")) == task_id:
                return ScheduledTask.from_dict(item)
        raise SchedulerError(f"Задача {task_id!r} не найдена.")

    def list_tasks(
        self, status: str | None = None, group: str | None = None
    ) -> list[ScheduledTask]:
        """All tasks, optionally filtered by *status* and/or *group*."""
        with self._thread_lock, self._locked() as _:
            data = self._read()
        result = []
        for item in data["tasks"]:
            task = ScheduledTask.from_dict(item)
            if status and task.status != status:
                continue
            if group and task.group != group:
                continue
            result.append(task)
        return result

    def update_task(self, task: ScheduledTask) -> ScheduledTask:
        """Persist a new version of *task* (matched by id)."""
        with self._thread_lock, self._locked() as _:
            data = self._read()
            for i, item in enumerate(data["tasks"]):
                if str(item.get("id")) == task.id:
                    data["tasks"][i] = task.to_dict()
                    self._write(data)
                    return task
        raise SchedulerError(f"Задача {task.id!r} не найдена.")

    def set_status(self, task_id: str, status: str) -> ScheduledTask:
        """Move a task to *status* (``paused`` / ``cancelled`` / ``active``).

        ``active`` also revives a ``failed`` task: its attempt counter is
        reset and the next run is computed from the schedule anew.
        """
        if status not in (STATUS_ACTIVE, STATUS_PAUSED, STATUS_CANCELLED):
            raise SchedulerError(f"Недопустимый статус: {status!r}.")
        with self._thread_lock, self._locked() as _:
            data = self._read()
            for i, item in enumerate(data["tasks"]):
                if str(item.get("id")) == task_id:
                    task = ScheduledTask.from_dict(item)
                    if task.status == STATUS_DONE:
                        raise SchedulerError(
                            "Задача уже выполнена (once) и не меняет статус."
                        )
                    changes: dict = {"status": status}
                    if status == STATUS_ACTIVE and task.status == STATUS_FAILED:
                        changes["attempts"] = 0
                        changes["next_run"] = iso(
                            next_run_after(task.schedule)
                        )
                    updated = replace(task, **changes)
                    data["tasks"][i] = updated.to_dict()
                    self._write(data)
                    return updated
        raise SchedulerError(f"Задача {task_id!r} не найдена.")

    # -- results ---------------------------------------------------------------

    def append_result(self, result: TaskResult) -> None:
        with self._thread_lock, self._locked() as _:
            data = self._read()
            data["results"].append(result.to_dict())
            self._write(data)

    def list_results(
        self,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskResult]:
        """Results oldest→newest; filter by task, optionally tail *limit*."""
        with self._thread_lock, self._locked() as _:
            data = self._read()
        results = [
            TaskResult.from_dict(item) for item in data["results"]
        ]
        if task_id:
            results = [r for r in results if r.task_id == task_id]
        if limit is not None and limit >= 0:
            results = results[-limit:]
        return results

    # -- daemon side -------------------------------------------------------------

    def claim_due(
        self, now: datetime | None = None
    ) -> list[ScheduledTask]:
        """Atomically fetch due tasks and mark them claimed.

        Claiming inside one lock matters: if the daemon only *read* due tasks
        and advanced them after execution, a concurrent reader could see and
        re-claim the same task. The claim *reserves* the task by pushing
        ``next_run`` into the future (once-tasks: +retry delay, so a daemon
        crash mid-run means a retry, not a lost task). The definitive status
        is applied later by :meth:`finalize_run` once the outcome is known.
        """
        now = now or utc_now()
        claimed: list[ScheduledTask] = []
        with self._thread_lock, self._locked() as _:
            data = self._read()
            changed = False
            for i, item in enumerate(data["tasks"]):
                task = ScheduledTask.from_dict(item)
                if not is_due(task, now):
                    continue
                claimed.append(task)
                # Reserve without committing an outcome: only push next_run
                # into the future. Status and attempts are applied later by
                # finalize_run() — claiming must never produce a fake done.
                kind = str(task.schedule.get("kind", "")).lower()
                if kind == KIND_ONCE:
                    # A daemon crash mid-run leaves the task due again after
                    # the retry delay (a retry) instead of losing it.
                    reserve_at = now + timedelta(
                        seconds=DEFAULT_RETRY_DELAY_SECONDS
                    )
                else:
                    reserve_at = next_run_after(task.schedule, now)
                reserved = replace(task, next_run=iso(reserve_at))
                data["tasks"][i] = reserved.to_dict()
                changed = True
            if changed:
                self._write(data)
        return claimed

    def finalize_run(
        self, task_id: str, *, ok: bool, now: datetime | None = None
    ) -> ScheduledTask:
        """Apply the outcome of a claimed run: definitive status + counters.

        Called by the daemon after the action executed (outside the store
        lock). Success completes a once-task (``done``); failure schedules a
        retry with a delay and, after ``max_attempts``, marks the task
        ``failed`` — a failed task is never shown as done.
        """
        now = now or utc_now()
        with self._thread_lock, self._locked() as _:
            data = self._read()
            for i, item in enumerate(data["tasks"]):
                if str(item.get("id")) != task_id:
                    continue
                task = ScheduledTask.from_dict(item)
                finalized = advance_after_run(task, now, ok=ok)
                data["tasks"][i] = finalized.to_dict()
                self._write(data)
                return finalized
        raise SchedulerError(f"Задача {task_id!r} не найдена.")
