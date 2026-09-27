#!/usr/bin/env python3
"""The scheduler daemon: executes due tasks 24/7.

The MCP control panel (``scripts/scheduler_mcp_server.py``) only creates and
inspects tasks; this long-lived process is the executor. Every tick it:

1. atomically claims due tasks (next_run advanced in the same lock, so a task
   fires exactly once per tick even if several processes raced);
2. runs each task's action (``mcp_call`` / ``reminder`` / ``llm_summary``);
3. appends the outcome to the shared results journal.

A failing task produces a failed TaskResult — the loop keeps going.
Missed runs during downtime are NOT replayed retroactively (catch-up = run
once on the first tick after, then shift forward), which avoids result storms
after a laptop sleep. ``--once`` runs all due tasks and exits — handy for
smoke tests and for driving the schedule with an external cron instead.

Usage:
    python scripts/scheduler_daemon.py                  # 24/7 loop
    python scripts/scheduler_daemon.py --tick 30        # slower tick
    python scripts/scheduler_daemon.py --once           # run due, exit
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.scheduler import (  # noqa: E402
    KIND_ONCE,
    STATUS_FAILED,
    JsonSchedulerStore,
    iso,
    utc_now,
)
from llm_bot.scheduler_actions import (  # noqa: E402
    ActionContext,
    run_action,
)

log = logging.getLogger("scheduler_daemon")


def _tail(text: str, limit: int = 200) -> str:
    """One-line log cut; the full text lives in the results journal."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scheduler_daemon",
        description="Execute due scheduled tasks (the 24/7 executor).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the scheduler store (default: SCHEDULER_DB or "
        "data/scheduler.json).",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=5.0,
        help="Seconds between due-checks (default: 5).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run every due task once and exit (no loop).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write logs to this file (default: stderr only).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging."
    )
    return parser


def setup_logging(verbose: bool, log_file: str | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


class Daemon:
    """Tick loop with graceful shutdown and per-task fault isolation."""

    def __init__(self, store: JsonSchedulerStore, ctx: ActionContext) -> None:
        self.store = store
        self.ctx = ctx
        self._stop = False

    def request_stop(self, signum, frame) -> None:  # noqa: ARG002
        log.info("Остановка по сигналу %s...", signum)
        self._stop = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    # -- one pass --------------------------------------------------------------

    def run_due(self) -> int:
        """Claim, execute and finalize every due task; return how many ran."""
        claimed = self.store.claim_due(utc_now())
        for task in claimed:
            # The attempt budget belongs to once-tasks (bounded retries);
            # showing it for healthy periodic tasks produced "попытка 4/3".
            if str(task.schedule.get("kind", "")).lower() == KIND_ONCE:
                attempt_note = (
                    f", попытка {task.attempts + 1}/{task.max_attempts}"
                )
            else:
                attempt_note = ""
            log.info(
                "Запуск «%s» (%s)%s...",
                task.title, task.action, attempt_note,
            )
            try:
                result = run_action(task, self.ctx)
            except Exception as exc:  # noqa: BLE001 - fault isolation
                from llm_bot.scheduler import TaskResult

                result = TaskResult(
                    task_id=task.id,
                    run_at=iso(utc_now()),
                    ok=False,
                    summary=f"{type(exc).__name__}: {exc}",
                )
            self.store.append_result(result)
            finalized = self.store.finalize_run(task.id, ok=result.ok)
            if result.ok:
                log.info("✔ «%s»: %s", task.title, _tail(result.summary))
            elif finalized.status == STATUS_FAILED:
                log.error(
                    "✖ «%s»: неудач подряд %d/%d, задача помечена failed: %s",
                    task.title, finalized.attempts, finalized.max_attempts,
                    _tail(result.summary),
                )
            else:
                log.warning(
                    "✖ «%s»: %s (неудача %d/%d)",
                    task.title, _tail(result.summary),
                    finalized.attempts, finalized.max_attempts,
                )
        return len(claimed)

    def loop(self, tick: float) -> None:
        log.info("Демон запущен: хранилище %s, тик %.1fs", self.store.path, tick)
        self.install_signal_handlers()
        while not self._stop:
            try:
                self.run_due()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                log.exception("Ошибка тика: %s", exc)
            if self._sleep(tick):
                break
        log.info("Демон остановлен.")

    def _sleep(self, seconds: float) -> bool:
        """Sleep in small chunks; True when a stop was requested."""
        import time

        end = time.monotonic() + seconds
        while not self._stop:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.2, remaining))
        return True


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose, args.log_file)

    from llm_bot.mcp_tools import load_mcp_config
    from llm_bot.scheduler_actions import _default_bridge_factory

    store = JsonSchedulerStore(args.db)

    llm_warning_shown = {"done": False}

    def chat_factory():
        # An LLM is optional for summaries: without data/models.yaml (or on
        # any wiring error) the deterministic fallback keeps the daemon fully
        # functional offline. YamlModelStore.list() returns names SORTED
        # alphabetically, so "the first model" used to be an arbitrary pick
        # (live beta: it was a GigaChat model without credentials). Choose
        # explicitly via SCHEDULER_SUMMARY_MODEL when set.
        try:
            from llm_bot.factory import build_client
            from llm_bot.yaml_stores import YamlModelStore

            models = YamlModelStore()  # default path data/models.yaml
            names = models.list()
            if not names:
                return None
            wanted = os.getenv("SCHEDULER_SUMMARY_MODEL", "").strip()
            if wanted and wanted not in names:
                if not llm_warning_shown["done"]:
                    log.warning(
                        "SCHEDULER_SUMMARY_MODEL=%r нет в data/models.yaml "
                        "(доступны: %s); использую первую модель: %s",
                        wanted, ", ".join(names), names[0],
                    )
                    llm_warning_shown["done"] = True
                wanted = ""
            return build_client(models.get(wanted or names[0]))
        except Exception as exc:  # noqa: BLE001
            if not llm_warning_shown["done"]:
                log.warning(
                    "LLM для сводок недоступен (%s); llm_summary будет "
                    "работать в детерминированном режиме.", exc,
                )
                llm_warning_shown["done"] = True
            return None

    ctx = ActionContext(
        store=store,
        bridge_factory=_default_bridge_factory,
        chat_factory=chat_factory,
    )
    daemon = Daemon(store, ctx)

    if args.once:
        ran = daemon.run_due()
        log.info("Готово (--once): выполнено задач: %d", ran)
        return 0
    daemon.loop(args.tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
