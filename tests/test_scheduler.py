"""Unit tests for the scheduler core (llm_bot.scheduler + scheduler_actions).

Clock injection: every schedule function takes ``now``, so no test needs
sleeps or real waiting. The store is exercised against a tmp_path JSON file,
and inter-process locking is checked with real flock on two store instances.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llm_bot.scheduler import (
    ACTION_REMINDER,
    KIND_DAILY,
    KIND_INTERVAL,
    KIND_ONCE,
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PAUSED,
    SchedulerError,
    ScheduledTask,
    TaskResult,
    JsonSchedulerStore,
    advance_after_run,
    build_digest,
    is_due,
    iso,
    next_run_after,
    parse_iso,
    replace,
)
from llm_bot.scheduler_actions import (
    ActionContext,
    run_action,
    run_llm_summary,
    run_mcp_call,
    run_reminder,
    validate_action,
)

UTC = timezone.utc


def dt(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Schedule math
# ---------------------------------------------------------------------------


class TestNextRunAfter:
    def test_once_delay(self):
        now = dt(2026, 9, 24, 12, 0, 0)
        nxt = next_run_after({"kind": KIND_ONCE, "delay_seconds": 1800}, now)
        assert nxt == now + timedelta(minutes=30)

    def test_interval(self):
        now = dt(2026, 9, 24, 12, 0, 0)
        nxt = next_run_after({"kind": KIND_INTERVAL, "every_seconds": 15}, now)
        assert nxt == now + timedelta(seconds=15)

    def test_daily_later_today(self):
        # 10:00 UTC == 20:00 Vladivostok; "at 21:00" is later the same day.
        now = dt(2026, 9, 24, 10, 0, 0)
        nxt = next_run_after({"kind": KIND_DAILY, "at": "21:00"}, now)
        expected_local = now.astimezone().replace(hour=21, minute=0)
        assert nxt == expected_local

    def test_daily_tomorrow_when_passed(self):
        # 10:00 UTC == 20:00 local; "at 08:00" already passed today.
        now = dt(2026, 9, 24, 10, 0, 0)
        nxt = next_run_after({"kind": KIND_DAILY, "at": "08:00"}, now)
        expected_local = (
            now.astimezone().replace(hour=8, minute=0) + timedelta(days=1)
        )
        assert nxt == expected_local

    def test_rejects_bad_kind(self):
        with pytest.raises(SchedulerError):
            next_run_after({"kind": "weekly"}, dt(2026, 9, 24))

    def test_rejects_bad_delay(self):
        with pytest.raises(SchedulerError):
            next_run_after({"kind": KIND_ONCE, "delay_seconds": -5})
        with pytest.raises(SchedulerError):
            next_run_after({"kind": KIND_ONCE, "delay_seconds": "soon"})

    def test_rejects_bad_daily_time(self):
        with pytest.raises(SchedulerError):
            next_run_after({"kind": KIND_DAILY, "at": "25:00"})
        with pytest.raises(SchedulerError):
            next_run_after({"kind": KIND_DAILY, "at": "soon"})


class TestAdvance:
    def _task(self, schedule: dict) -> ScheduledTask:
        return ScheduledTask(
            id="t1",
            title="Тест",
            action=ACTION_REMINDER,
            schedule=schedule,
        )

    def test_once_becomes_done(self):
        now = dt(2026, 9, 24, 12, 0, 0)
        task = self._task({"kind": KIND_ONCE, "delay_seconds": 60})
        advanced = advance_after_run(task, now)
        assert advanced.status == STATUS_DONE
        assert advanced.last_run == iso(now)
        assert advanced.next_run == ""

    def test_interval_shifts_from_now(self):
        # Catch-up policy: next run counts from the execution moment, NOT
        # from the stale next_run — a downtime backlog is not replayed.
        now = dt(2026, 9, 24, 12, 0, 0)
        task = replace(
            self._task({"kind": KIND_INTERVAL, "every_seconds": 60}),
            next_run=iso(now - timedelta(hours=5)),  # long overdue
        )
        advanced = advance_after_run(task, now)
        assert advanced.status == STATUS_ACTIVE
        assert parse_iso(advanced.next_run) == now + timedelta(seconds=60)

    def test_daily_moves_to_tomorrow(self):
        now = dt(2026, 9, 24, 12, 0, 0)
        task = self._task({"kind": KIND_DAILY, "at": "08:00"})
        advanced = advance_after_run(task, now)
        expected = (
            now.astimezone().replace(hour=8, minute=0) + timedelta(days=1)
        )
        assert parse_iso(advanced.next_run) == expected

    def test_attempts_count_consecutive_failures(self):
        """Live beta: attempts grew on EVERY run, so a healthy every-minute
        task drifted past its retry budget and logged «попытка 4/3». The
        counter must instead track consecutive failures and reset on any
        success."""
        now = dt(2026, 9, 24, 12, 0, 0)
        struggling = replace(
            self._task({"kind": KIND_ONCE, "delay_seconds": 60}),
            attempts=2,
        )
        failed = advance_after_run(struggling, now, ok=False)
        assert failed.attempts == 3
        assert failed.status == STATUS_FAILED  # 3/3 consecutive failures
        recovered = advance_after_run(failed, now, ok=True)
        assert recovered.status == STATUS_DONE
        assert recovered.attempts == 0  # reset, not 4
        periodic = replace(
            self._task({"kind": KIND_INTERVAL, "every_seconds": 60}),
            attempts=5,
        )
        healthy = advance_after_run(periodic, now, ok=True)
        assert healthy.attempts == 0
        assert healthy.status == STATUS_ACTIVE


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_add_and_get_roundtrip(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        now = dt(2026, 9, 24, 12, 0, 0)
        task = store.add_task(
            "Напоминание",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 60},
            {"text": "позвонить"},
            now=now,
        )
        loaded = store.get_task(task.id)
        assert loaded.title == "Напоминание"
        assert loaded.payload == {"text": "позвонить"}
        assert loaded.status == STATUS_ACTIVE
        assert parse_iso(loaded.next_run) == now + timedelta(seconds=60)

    def test_add_rejects_bad_input(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        with pytest.raises(SchedulerError):
            store.add_task("  ", ACTION_REMINDER, {"kind": KIND_ONCE, "delay_seconds": 1})
        with pytest.raises(SchedulerError):
            store.add_task("x", ACTION_REMINDER, {"kind": "kronos"})
        with pytest.raises(SchedulerError):
            store.add_task("x", "", {"kind": KIND_ONCE, "delay_seconds": 1})

    def test_get_unknown_raises(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        with pytest.raises(SchedulerError):
            store.get_task("nope")

    def test_results_and_filters(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "t", ACTION_REMINDER, {"kind": KIND_ONCE, "delay_seconds": 1}
        )
        store.append_result(TaskResult(task.id, iso(dt(2026, 9, 24, 12)), True, "раз"))
        store.append_result(TaskResult(task.id, iso(dt(2026, 9, 24, 13)), False, "два"))
        all_results = store.list_results()
        assert [r.summary for r in all_results] == ["раз", "два"]
        tail = store.list_results(task_id=task.id, limit=1)
        assert [r.summary for r in tail] == ["два"]

    def test_set_status_lifecycle(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "t", ACTION_REMINDER, {"kind": KIND_INTERVAL, "every_seconds": 60}
        )
        store.set_status(task.id, STATUS_PAUSED)
        assert store.get_task(task.id).status == STATUS_PAUSED
        store.set_status(task.id, STATUS_ACTIVE)
        assert store.get_task(task.id).status == STATUS_ACTIVE
        store.set_status(task.id, STATUS_CANCELLED)
        assert store.get_task(task.id).status == STATUS_CANCELLED
        with pytest.raises(SchedulerError):
            store.set_status(task.id, "playing")

    def test_claim_due_moves_exactly_once(self, tmp_path):
        """A task claimed at tick N must not be re-claimed at tick N."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        store.add_task(
            "t",
            ACTION_REMINDER,
            {"kind": KIND_INTERVAL, "every_seconds": 100},
            now=dt(2026, 9, 24, 11, 0, 0),
        )
        tick = dt(2026, 9, 24, 12, 0, 0)  # 1 hour later → due
        first = store.claim_due(tick)
        assert len(first) == 1
        store.finalize_run(first[0].id, ok=True, now=tick)
        second = store.claim_due(tick)
        assert second == []  # next_run advanced into the future
        # Pause/cancel removes the task from claiming.
        task = store.add_task(
            "u",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 1},
            now=dt(2026, 9, 24, 11, 0, 0),
        )
        store.set_status(task.id, STATUS_CANCELLED)
        assert store.claim_due(dt(2026, 9, 24, 12, 0, 0)) == []

    def test_failed_once_task_retries_then_failed(self, tmp_path):
        """Regression (live beta run): a failed once-run must NOT be marked
        done. It retries (next_run = claim + retry delay) and only after
        max_attempts becomes terminal ``failed`` — never a fake done."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "хрупкая",
            "mcp_call",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            now=dt(2026, 9, 24, 11, 59, 0),
        )
        tick = dt(2026, 9, 24, 12, 0, 0)
        claimed = store.claim_due(tick)
        assert len(claimed) == 1
        finalized = store.finalize_run(task.id, ok=False, now=tick)
        # Failure: still active, retry scheduled in the future.
        assert finalized.status == STATUS_ACTIVE
        assert finalized.attempts == 1
        assert parse_iso(finalized.next_run) > tick
        assert not is_due(store.get_task(task.id), tick + timedelta(seconds=1))
        # ...second failure still leaves one attempt in reserve.
        second_tick = dt(2026, 9, 24, 12, 5, 0)
        claimed2 = store.claim_due(second_tick)
        assert len(claimed2) == 1
        store.claim_due(second_tick)  # no double-claim within a tick
        finalized2 = store.finalize_run(task.id, ok=False, now=second_tick)
        assert finalized2.status == STATUS_ACTIVE
        assert finalized2.attempts == 2
        assert parse_iso(finalized2.next_run) > second_tick
        # ...third failure burns the last attempt → failed, no next run.
        third_tick = dt(2026, 9, 24, 12, 10, 0)
        claimed3 = store.claim_due(third_tick)
        assert len(claimed3) == 1
        finalized3 = store.finalize_run(task.id, ok=False, now=third_tick)
        assert finalized3.status == STATUS_FAILED
        assert finalized3.attempts == 3
        assert finalized3.next_run == ""
        assert store.claim_due(third_tick + timedelta(hours=1)) == []

    def test_success_finalizes_once_to_done(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "ок",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 1},
            now=dt(2026, 9, 24, 11, 59, 0),
        )
        tick = dt(2026, 9, 24, 12, 0, 0)
        store.claim_due(tick)
        finalized = store.finalize_run(task.id, ok=True, now=tick)
        assert finalized.status == STATUS_DONE
        assert finalized.attempts == 0  # consecutive-failure counter reset

    def test_revive_failed_task_resets_attempts(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "хрупкая",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 1},
            now=dt(2026, 9, 24, 11, 59, 0),
        )
        tick = dt(2026, 9, 24, 12, 0, 0)
        store.claim_due(tick)
        store.finalize_run(task.id, ok=False, now=tick)
        store.finalize_run(task.id, ok=False, now=dt(2026, 9, 24, 12, 5, 0))
        store.finalize_run(task.id, ok=False, now=dt(2026, 9, 24, 12, 10, 0))
        assert store.get_task(task.id).status == STATUS_FAILED
        revived = store.set_status(task.id, STATUS_ACTIVE)
        assert revived.attempts == 0
        assert revived.status == STATUS_ACTIVE
        assert parse_iso(revived.next_run) is not None

    def test_claim_due_is_due_helper_agrees(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "t",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 30},
            now=dt(2026, 9, 24, 11, 59, 50),
        )
        before = dt(2026, 9, 24, 12, 0, 0)
        assert not is_due(store.get_task(task.id), before)
        assert is_due(store.get_task(task.id), before + timedelta(seconds=31))

    def test_two_store_instances_share_data(self, tmp_path):
        """The MCP server and the daemon use separate instances/paths of time —
        data written by one must be visible to the other immediately."""
        panel = JsonSchedulerStore(tmp_path / "db.json")
        daemon = JsonSchedulerStore(tmp_path / "db.json")
        task = panel.add_task(
            "shared", ACTION_REMINDER, {"kind": KIND_ONCE, "delay_seconds": 1}
        )
        assert daemon.get_task(task.id).title == "shared"

    def test_corrupt_file_is_empty(self, tmp_path):
        path = tmp_path / "db.json"
        path.write_text("{ не json", encoding="utf-8")
        store = JsonSchedulerStore(path)
        assert store.list_tasks() == []


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, tasks=(), results=()):
        self._tasks = list(tasks)
        self._results = list(results)

    def list_tasks(self, status=None, group=None):
        return [
            t
            for t in self._tasks
            if (status is None or t.status == status)
            and (group is None or t.group == group)
        ]

    def list_results(self, task_id=None, limit=None):
        res = [r for r in self._results if task_id is None or r.task_id == task_id]
        return res[-limit:] if limit is not None else res


class _FakeBridge:
    def __init__(self, text="данные"):
        self.text = text
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.text


def _ctx(store, bridge=None, chat=None):
    return ActionContext(
        store=store,
        bridge_factory=(lambda server: bridge) if bridge else None,
        chat_factory=(lambda: chat) if chat else None,
    )


class TestActions:
    def test_reminder(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Будильник",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"text": "встать"},
        )
        result = run_reminder(task, _ctx(store))
        assert result.ok
        assert "НАПОМИНАНИЕ" in result.summary
        assert "встать" in result.summary

    def test_reminder_accepts_message_key(self, tmp_path):
        """Live beta: the model put the text under 'message', not 'text' —
        the reminder must still carry it instead of the bare title."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Сбор курса доллара",
            ACTION_REMINDER,
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"message": "Проверьте текущий курс доллара"},
        )
        result = run_reminder(task, _ctx(store))
        assert result.ok
        assert "Проверьте текущий курс доллара" in result.summary

    def test_mcp_call_via_fake_bridge(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Сбор",
            "mcp_call",
            {"kind": KIND_INTERVAL, "every_seconds": 60},
            {"server": "notes", "tool": "list_notes", "arguments": {"tag": "x"}},
        )
        bridge = _FakeBridge("id=1: заметка")
        result = run_mcp_call(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert "id=1" in result.summary
        assert bridge.calls == [("list_notes", {"tag": "x"})]

    def test_unknown_action_is_failed_result(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "t", "телепорт", {"kind": KIND_ONCE, "delay_seconds": 1}
        )
        result = run_action(task, _ctx(store))
        assert not result.ok
        assert "телепорт" in result.summary

    def test_coerce_tool_name_in_action_with_server_tool_payload(self, tmp_path):
        """Beta live-run regression: the model wrote action='add_note' with
        {server, tool} in the payload — route it to mcp_call, not fail."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Add reminder note",
            "add_note",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"server": "notes", "tool": "add_note", "arguments": {"text": "завести будильник"}},
        )
        bridge = _FakeBridge("Заметка сохранена: id=7")
        result = run_action(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert "id=7" in result.summary
        assert bridge.calls == [("add_note", {"text": "завести будильник"})]

    def test_coerce_qualified_server_tool_action(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Снимок",
            "notes__list_notes",
            {"kind": KIND_ONCE, "delay_seconds": 1},
        )
        bridge = _FakeBridge("пусто")
        result = run_action(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert bridge.calls == [("list_notes", {})]

    def test_coerce_qualified_action_folds_flat_payload(self, tmp_path):
        """Beta live-run regression #2: action='notes__add_note' with a flat
        payload {'text': …} — 'text' must reach the tool arguments; before
        the fix the tool got {} and failed with 'Field required: text'."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Add plumber phone note",
            "notes__add_note",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"text": "телефон сантехника 88876"},
        )
        bridge = _FakeBridge("Заметка сохранена: id=8")
        result = run_action(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert "id=8" in result.summary
        assert bridge.calls == [("add_note", {"text": "телефон сантехника 88876"})]

    def test_coerce_flat_server_tool_payload_folds_extras(self, tmp_path):
        """Flat {server, tool, text} payload: 'text' folds into arguments."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Add note",
            "add_note",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"server": "notes", "tool": "add_note", "text": "позвонить"},
        )
        bridge = _FakeBridge("Заметка сохранена: id=9")
        result = run_action(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert bridge.calls == [("add_note", {"text": "позвонить"})]

    def test_coerce_reminder_synonyms(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "t",
            "Напоминание о встрече",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"text": "встреча"},
        )
        result = run_action(task, _ctx(store))
        assert result.ok
        assert "НАПОМИНАНИЕ" in result.summary

    def test_validate_action_rejects_invented_name(self):
        """Beta live-run #3: the model invented action='collect_usd_rate'
        with an empty payload — rejected at creation, not after the daemon
        burns all retries on an unrescuable task."""
        with pytest.raises(SchedulerError) as exc:
            validate_action("collect_usd_rate", {}, known_servers=["notes"])
        assert "mcp_call" in str(exc.value)
        # Everything coerce_action can rescue still passes.
        servers = ["notes", "scheduler"]
        validate_action("mcp_call", {}, known_servers=servers)
        validate_action("reminder", {"text": "x"}, known_servers=servers)
        validate_action(
            "llm_summary", {"sources": "all"}, known_servers=servers
        )
        validate_action("notes__list_notes", {}, known_servers=servers)
        validate_action(
            "add_note",
            {"server": "notes", "tool": "add_note"},
            known_servers=servers,
        )
        validate_action(
            "Напоминание о встрече", {"text": "x"}, known_servers=servers
        )

    def test_validate_action_rejects_unknown_server(self):
        """Beta live-run #4: the model invented server='default' — the task
        was created, failed against the config and burned its retries.
        Now rejected at creation with the real server list."""
        with pytest.raises(SchedulerError) as exc:
            validate_action(
                "mcp_call",
                {"server": "default", "tool": "list_notes"},
                known_servers=["notes", "scheduler"],
            )
        assert "notes, scheduler" in str(exc.value)
        # The server__tool action form is checked too.
        with pytest.raises(SchedulerError):
            validate_action("default__list_notes", {}, known_servers=["notes"])
        # And a real server passes.
        validate_action(
            "mcp_call",
            {"server": "notes", "tool": "list_notes"},
            known_servers=["notes"],
        )

    def test_validate_action_rejects_unknown_tool(self):
        """Beta live-run #4, part 2: the model guessed a tool name that does
        not exist on the server — rejected at creation with the real tool
        list and a hint to use the short (unqualified) name. A qualified
        name that resolves to a real tool passes (runtime strips the
        prefix, see test_mcp_call_strips_qualified_tool_prefix)."""
        with pytest.raises(SchedulerError) as exc:
            validate_action(
                "mcp_call",
                {"server": "notes", "tool": "notes__delete_note"},
                known_servers=["notes"],
                known_tools=["add_note", "find_notes", "list_notes"],
            )
        assert "delete_note" in str(exc.value)
        assert "add_note, find_notes, list_notes" in str(exc.value)
        assert "без префикса" in str(exc.value)
        # The short name of a real tool passes.
        validate_action(
            "mcp_call",
            {"server": "notes", "tool": "list_notes"},
            known_servers=["notes"],
            known_tools=["list_notes"],
        )
        # A qualified name resolving to a real tool passes too.
        validate_action(
            "mcp_call",
            {"server": "notes", "tool": "notes__list_notes"},
            known_servers=["notes"],
            known_tools=["add_note", "find_notes", "list_notes"],
        )

    def test_mcp_call_strips_qualified_tool_prefix(self, tmp_path):
        """Runtime rescue: payload {server: notes, tool: notes__list_notes}
        must call the bare 'list_notes', not the qualified name."""
        store = JsonSchedulerStore(tmp_path / "db.json")
        task = store.add_task(
            "Снимок",
            "mcp_call",
            {"kind": KIND_ONCE, "delay_seconds": 1},
            {"server": "notes", "tool": "notes__list_notes"},
        )
        bridge = _FakeBridge("пусто")
        result = run_action(task, _ctx(store, bridge=bridge))
        assert result.ok
        assert bridge.calls == [("list_notes", {})]

    def test_summary_deterministic_without_llm(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        src = store.add_task(
            "Курс USD",
            "mcp_call",
            {"kind": KIND_INTERVAL, "every_seconds": 60},
            group="финансы",
        )
        store.append_result(TaskResult(src.id, iso(dt(2026, 9, 24, 12)), True, "82.5"))
        digest = store.add_task(
            "Дайджест",
            "llm_summary",
            {"kind": KIND_DAILY, "at": "09:00"},
            {"sources": "финансы"},
        )
        result = run_llm_summary(digest, _ctx(store))
        assert result.ok
        assert result.details["mode"] == "deterministic"
        assert "Курс USD" in result.summary
        assert "82.5" in result.summary

    def test_summary_with_llm(self, tmp_path):
        class _Chat:
            def chat(self, messages):
                return "Курс стабильный. Всё хорошо."

        src = ScheduledTask(id="s1", title="Источник", action="mcp_call", schedule={})
        store = _FakeStore(
            tasks=[src],
            results=[
                TaskResult("s1", iso(dt(2026, 9, 24, 12)), True, "82.5 руб.")
            ],
        )
        digest = ScheduledTask(
            id="d1",
            title="Дайджест",
            action="llm_summary",
            schedule={},
            payload={"sources": "all"},
        )
        result = run_llm_summary(
            digest,
            _ctx(store, chat=_Chat()),
        )
        assert result.ok
        assert result.summary == "Курс стабильный. Всё хорошо."
        assert result.details["mode"] == "llm"

    def test_summary_empty_sources(self, tmp_path):
        store = JsonSchedulerStore(tmp_path / "db.json")
        digest = store.add_task(
            "Дайджест",
            "llm_summary",
            {"kind": KIND_DAILY, "at": "09:00"},
            {"sources": "all"},
        )
        result = run_llm_summary(digest, _ctx(store))
        assert result.ok
        assert "нет ни одного результата" in result.summary


# ---------------------------------------------------------------------------
# Digest rendering
# ---------------------------------------------------------------------------


class TestDigest:
    def test_digest_empty(self):
        assert "не найдено" in build_digest([], [])

    def test_digest_renders_counters_and_fresh(self):
        task = ScheduledTask(
            id="t1",
            title="Курс",
            action="mcp_call",
            schedule={"kind": KIND_INTERVAL, "every_seconds": 60},
            group="финансы",
            next_run=iso(dt(2026, 9, 24, 13, 0, 0)),
        )
        results = [
            TaskResult("t1", iso(dt(2026, 9, 24, 12)), True, "82.5"),
            TaskResult("t1", iso(dt(2026, 9, 24, 13)), False, "упало"),
        ]
        text = build_digest([task], results)
        assert "Курс" in text
        assert "финансы" in text
        assert "прогонов: 2" in text
        assert "ошибок 1" in text
        assert "82.5" in text
        assert "упало" in text
