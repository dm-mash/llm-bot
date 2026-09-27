# Проверка: Планировщик и фоновые задачи (MCP + демон 24/7)

Дата: 2026-09-24 · Тесты: **383 passed** (весь набор, `pytest tests/ -q`),
из них 36 unit ([`tests/test_scheduler.py`](../tests/test_scheduler.py)) и
5 e2e ([`tests/test_scheduler_mcp_server.py`](../tests/test_scheduler_mcp_server.py)).

## Требования ТЗ → реализация

| Требование | Реализация | Проверка |
|---|---|---|
| MCP-инструмент с отложенным/периодическим выполнением | [`schedule_task`](../scripts/scheduler_mcp_server.py) с тремя расписаниями: `delay_seconds` (once), `every_seconds` (interval), `daily_at` (daily) | e2e: тест каталога и схем; demo-сценарий |
| Сохранять данные (JSON / SQLite) | [`JsonSchedulerStore`](../llm_bot/scheduler.py) — `data/scheduler.json`, атомарная запись tmp+rename, **межпроцессный** `fcntl.flock` (MCP-сервер и демон — разные процессы); SQLite — запасной вариант с тем же протоколом | тесты: roundtrip, два экземпляра стора видят одни данные, битый файл = пусто |
| Выполняться по расписанию | демон [`scripts/scheduler_daemon.py`](../scripts/scheduler_daemon.py): тик, атомарный `claim_due` (задача срабатывает ровно один раз на тик), catch-up без ретро-повторов, graceful shutdown SIGINT/SIGTERM | unit: `claim_due` дважды на одном тике; e2e: smoke реального процесса демона |
| Возвращать агрегированный результат | [`task_digest`](../scripts/scheduler_mcp_server.py) + [`build_digest`](../llm_bot/scheduler.py): счётчики ок/ошибок, свежие результаты, ближайшие запуски; срезы по группе/задаче | e2e + живой прогон (секция 5 ниже) |

## Архитектура

Два процесса над общим JSON-хранилищем — планировщик нельзя поместить внутрь
MCP-сервера, потому что [`MCPToolBridge`](../llm_bot/mcp_tools.py) порождает
stdio-сервер на каждый вызов и закрывает его: расписание умерло бы вместе с
процессом.

- **панель** `scheduler_mcp_server.py` (stateless): `schedule_task`,
  `list_tasks`, `cancel/pause/resume_task`, `get_task_results`, `task_digest`;
- **исполнитель** `scheduler_daemon.py` (24/7): действия `mcp_call` (любой
  инструмент любого сервера из `data/mcp.yaml` — новые источники данных
  добавляются конфигом, а не кодом), `reminder`, `llm_summary`
  (LLM опционален, иначе детерминированная сводка — тесты без сети).

## Живой прогон (scripts/scheduler_demo.py, temp-хранилище, без сети)

```
== 1. Каталог инструментов ==
cancel_task, get_task_results, list_tasks, pause_task, resume_task, schedule_task, task_digest

== 2. Создание задач ==
Задача создана: id=f048638febb1, «Напоминание: пересобрать заметки», действие reminder, первый запуск 2026-09-24T11:50:30+00:00
Задача создана: id=c6545b8faf7c, «Периодический снимок CRM», действие mcp_call, первый запуск 2026-09-24T12:50:29+00:00
Задача создана: id=4ed6e01d677a, «Утренний дайджест», действие llm_summary, первый запуск 2026-09-24T23:00:00+00:00
Задача без расписания отвергнута: True

== 3. Список задач ==
id=f048638febb1 [active] «Напоминание: пересобрать заметки» — reminder, однократно через 1.0 с, ...
id=c6545b8faf7c [active] «Периодический снимок CRM» — mcp_call, каждые 3600.0 с, группа: данные, ...
id=4ed6e01d677a [active] «Утренний дайджест» — llm_summary, ежедневно в 09:00, ...

== 4. Демон --once (отдельный процесс) ==
✔ «Напоминание: пересобрать заметки»: НАПОМИНАНИЕ: проверь CRM
Готово (--once): выполнено задач: 1
Задача once -> done
Результат: ok=True :: НАПОМИНАНИЕ: проверь CRM

== 5. Агрегированный результат (task_digest) ==
▪ Напоминание: пересобрать заметки [done]
  прогонов: 1 (ок 1, ошибок 0)
  ✔ 2026-09-24T11:50:31+00:00: НАПОМИНАНИЕ: проверь CRM

▪ Периодический снимок CRM [active] (данные)
  прогонов: 0 (ок 0, ошибок 0)
  следующий запуск: 2026-09-24 22:50

▪ Утренний дайджест [active]
  прогонов: 0 (ок 0, ошибок 0)
  следующий запуск: 2026-09-25 09:00
```

## Покрытие тестами (ключевое)

- расписания: once/interval сдвиг от `now`, daily (сегодня/завтра по локальной
  зоне), отбраковка плохих значений (`25:00`, отрицательная задержка, неизвестный kind);
- `advance_after_run`: once→done, interval считается **от момента исполнения**
  (задержка не ретро-повторяется), daily → завтра;
- стор: CRUD, фильтры status/group, журнал с limit, жизненный цикл
  paused/cancelled, отказ менять статус done-задачи, оживление failed-задачи
  (resume сбрасывает счётчик попыток и пересчитывает next_run);
- `claim_due` / `finalize_run`: захват только **резервирует** (next_run
  уезжает в будущее, статус не трогается), исход применяется отдельно;
  задание ровно один раз на тик; paused/cancelled не выполняются;
- ретраи: упавший once остаётся `active` с повтором через 30 с, после
  `max_attempts` (3) — терминальный `failed`, никогда фиктивный `done`;
  interval/daily после неудачи ждут следующего слота;
- «попытка 4/3» (живая бета): `attempts` считал все прогоны и у healthy
  interval-задачи уползал за бюджет. Новая семантика — счётчик *подряд
  идущих неудач*, успех сбрасывает в 0; «попытка N/M» в логе демона
  показывается только once-задачам;
- обрезка в логе: длинные результаты режутся до ~200 символов одной строкой
  с «…» (полный текст — в журнале `get_task_results` / `task_digest`).
- действия: reminder, mcp_call через фейковый мост (без процессов), сводка в
  детерминированном и LLM-режимах, пустые источники, неизвестное действие;
- coerce: плоский payload модели сворачивается в `arguments` (и для
  `server__tool`-действия, и для плоского `{server, tool, …}`), явный
  `arguments` имеет приоритет над свёрнутыми ключами;
- e2e stdio: каталог/схемы, создание→список, ошибка без расписания → isError,
  cancel→фильтры, digest; smoke демона `--once` реальным процессом:
  schedule (через MCP) → демон → результат в JSON → второй проход «0 задач».

## Регрессия живой беты: упавшая once-задача получала `done`

В живом прогоне одноразовая задача, действие которой не выполнилось,
оказалась в статусе `done`, хотя результат не записался: `claim_due`
вызывал `advance_after_run(ok=True)` в момент захвата, то есть менял
статус **до** исполнения.

Исправление — разделение захвата и финализации:

- [`claim_due`](../llm_bot/scheduler.py) только **резервирует** задачу:
  `next_run` уезжает в будущее (once — на 30 с: сбой демона между захватом
  и записью превращается в повтор, а не в потерю задачи), `status` и
  `attempts` не трогаются;
- [`finalize_run(ok=…)`](../llm_bot/scheduler.py) применяет исход после
  исполнения: успех → `done`; неудача → повтор с задержкой 30 с до
  `max_attempts` (3), затем терминальный `failed` — фиктивного `done`
  больше не бывает;
- interval/daily при неудаче просто ждут следующего слота по расписанию.

Живая проверка исправления (once `mcp_call` на несуществующий сервер,
демон `--once`, temp-хранилище):

```
INFO Запуск «bad-server» (mcp_call), попытка 1/3...
WARNING ✖ «bad-server»: ValueError: MCP-сервер 'no_such_server' не найден
        в data/mcp.yaml. Доступны: notes, scheduler (повтор 1/3)
status: active | attempts: 1 | next_run: +30s    # не done
results: [(False, "ValueError: MCP-сервер 'no_such_server' не найден…")]
```

Регрессионные тесты: `test_failed_once_task_retries_then_failed`,
`test_success_finalizes_once_to_done`, `test_revive_failed_task_resets_attempts`
([`tests/test_scheduler.py`](../tests/test_scheduler.py)).

### Регрессия #2: плоский payload терял аргументы инструмента

Живой прогон: модель создала задачу с действием `notes__add_note` и payload
`{"text": …}` (без вложенного `arguments`). [`coerce_action`](../llm_bot/scheduler_actions.py)
верно превращал действие в `mcp_call(server=notes, tool=add_note)`, но `text`
оставался на верхнем уровне, инструмент вызывался с `{}` →
`Field required: text`. Ретрай-механизм при этом отработал правильно:
задача осталась `active` (попытка 1/3, повтор через 30 с).

Исправление: при обоих путях coerce в `mcp_call` плоские ключи (все, кроме
`server`/`tool`/`arguments`) сворачиваются в `arguments`
([`_fold_arguments`](../llm_bot/scheduler_actions.py)), явные значения
`arguments` приоритетны.

Полный живой цикл после исправления: неудачная попытка → `active`, повтор
+30 с → вторая попытка свёртывает `text` → `notes__add_note: Заметка
сохранена: id=8` → `done`, `attempts: 2`.

Тесты: `test_coerce_qualified_action_folds_flat_payload`,
`test_coerce_flat_server_tool_payload_folds_extras`.

### Регрессия #3: выдуманное имя действия (`collect_usd_rate`)

Живой прогон: на фразу «каждый час собирай курс доллара» модель создала
задачу с `action='collect_usd_rate'` и пустым payload — правдоподобное
*описание* работы вместо одного из трёх имён реестра. `coerce_action`
спасти это не может (нет `server__tool`, нет `{server, tool}` в payload):
задача сожгла бы 3 ретрая и умерла в `failed`.

Исправление — валидация при создании: [`validate_action`](../llm_bot/scheduler_actions.py)
вызывается в `schedule_task` и отвергает всё, что не смог бы спасти
`coerce_action`, ошибкой с перечнем допустимых имён — модель получает
мгновенную обратную связь и в том же ходе исправляется на
`mcp_call` с `payload {server, tool, arguments}`.

Тесты: `test_validate_action_rejects_invented_name` (unit),
расширен e2e `test_schedule_task_roundtrip_and_error` (изобретённое имя →
isError с подсказкой). Живая задача `e360b4936b8c` отменена вручную
(реальный сбор курса всё равно требует своего MCP-сервера в `data/mcp.yaml`).

## Как пользоваться

```bash
python scripts/scheduler_daemon.py                    # исполнитель 24/7
python -m llm_bot --agent assistant --mcp scheduler   # чат с пультом задач
```

Диалоги: «напомни через 30 минут…» → `schedule_task(reminder, delay_seconds)`;
«каждый час собирай курс» → `schedule_task(mcp_call, every_seconds)`;
«что накопилось?» → `task_digest()`; «присылай сводку по утрам» →
`schedule_task(llm_summary, daily_at, sources)`. Сценарии типа мониторинга
курса валют или эпизодов сериалов = новый MCP-сервер в `data/mcp.yaml` +
одна задача `mcp_call` — код демона не меняется.

Регрессия живой беты #5: демон брал «первую модель из `data/models.yaml`», а
`YamlModelStore.list()` возвращает имена отсортированными — на практике это
была `gigachat-max` без учетных данных, и сводки детерминированно падали.
Модель для `llm_summary` теперь выбирается явно через `SCHEDULER_SUMMARY_MODEL`
(если задана, но неизвестна — одно предупреждение и откат на первую); без
переменной берется первая по алфавиту, при ошибке сборки LLM — детерминированный
режим.
