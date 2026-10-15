# Scheduler: background tasks 24/7

Delayed and periodic execution as an MCP tool: reminders, periodic data
collection and regular summaries — stored in `data/scheduler.json`, executed
by a long-lived daemon, and **aggregated on demand** via `task_digest`.

Architecture: two independent processes over one shared JSON store —
the stateless MCP server is the "control panel" (task CRUD + results), the
daemon is the "executor" (runs due tasks 24/7). Every store mutation is
guarded by an inter-process `flock`, so the panel and the executor never
interleave.

```bash
# one-time setup: add the "scheduler" server to data/mcp.yaml
cp mcp.example.yaml data/mcp.yaml            # if not done earlier

# start the executor (this IS the 24/7 agent part)
python scripts/scheduler_daemon.py           # tick loop; Ctrl-C stops it
python scripts/scheduler_daemon.py --once    # run all due tasks and exit
                                             # (alternative: external cron)

# chat with the agent that manages the schedule
python -m llm_bot --agent assistant --mcp scheduler
```

What you can ask the agent (it translates to tool calls):

| You say | Agent calls |
|---|---|
| «напомни через 30 минут пересобрать заметки» | `schedule_task(action="reminder", delay_seconds=1800, payload={text: …})` |
| «каждый час собирай курс доллара» | `schedule_task(action="mcp_call", every_seconds=3600, payload={server: …, tool: …})` |
| «что накопилось по финансам?» | `task_digest(group="финансы")` — run counters, fresh results, next fire times |
| «присылай сводку по финансам каждое утро в 9» | `schedule_task(action="llm_summary", daily_at="09:00", payload={sources: "финансы"})` |
| «каждую минуту выдавай сводку по заметкам» | связка: `mcp_call(list_notes, every_seconds=60)` + `llm_summary(sources=…)` — сбор часто, LLM-сводка реже |

Three actions cover everything (any other action name is rejected at
creation — the model gets an immediate error listing the valid names);
new data sources are config, not code:

* `mcp_call` — call any tool of any server from `data/mcp.yaml`
  (`payload: {server, tool, arguments}`); periodic currency/episodes checks
  are just new MCP servers + a scheduled call;
* `reminder` — fires into the results journal (read it back with
  `task_digest`; nothing is pushed into a chat window);
* `llm_summary` — aggregates recent results of source tasks (all / a group /
  explicit ids) into one summary; without a configured model it degrades to a
  deterministic summary, so the daemon works fully offline. When an LLM is
  used, the daemon takes the model named by `SCHEDULER_SUMMARY_MODEL` (see
  `.env.example`), or — if unset — the **alphabetically first** model in
  `data/models.yaml` (`YamlModelStore.list()` returns sorted names, not the
  file order). That model's credentials must be available in the daemon's
  environment (`.env` is loaded from the daemon's working directory).

Schedules: `once` (delay), `interval` (every N seconds), `daily` (at HH:MM
local). Missed runs during downtime are not replayed retroactively: the first
tick after resumption runs the task once and shifts it forward.

Failure handling: every run's outcome is journaled (`ok` flag in
`get_task_results`), and a failing run never yields a fake `done`. A `once`
task retries automatically — up to 3 *consecutive* failures, 30 s apart —
then becomes terminal `failed` (visible in `list_tasks` / `task_digest`).
`interval`/`daily` tasks simply wait for their next scheduled slot after a
failure. The attempt counter tracks consecutive failures (any success resets
it to 0) and appears in the daemon log only for once-tasks; periodic tasks
run without an attempt budget. `resume_task` revives a `failed` task with a
reset counter. Daemon log lines cut long results at ~200 chars with «…» —
the full text is in the results journal.

Tests: `tests/test_scheduler.py` (core math + store, including the
failed-once-must-retry regression) and `tests/test_scheduler_mcp_server.py`
(real stdio processes, including a schedule → daemon `--once` → digest
round-trip).
