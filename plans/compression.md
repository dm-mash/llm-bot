# План: управление контекстом — сжатие истории (rolling summary)

## Цель

Реализовать агент, который работает со сжатием истории и экономит токены:

* хранить последние **N** сообщений «как есть»;
* остальное заменять **summary** (сворачивать блоками каждые **M** сообщений);
* хранить **summary отдельно** и подставлять его в запрос вместо полной истории;
* сравнить качество ответов и расход токенов **до/после**.

## Ключевые решения

1. **Два параметра** `CompressionSettings(keep_last=N, block_size=M)`:
   - `keep_last` (N) — сколько свежих сообщений остаются дословно (качество);
   - `block_size` (M) — когда история перевалила за M, самый старый кусок
     сворачивается в summary (частота/стоимость суммаризации).
   Только N давало бы слишком частую (и дорогую) суммаризацию почти на каждый ход.

2. **Инкрементальная суммаризация.** При сворачивании в модель уходит только
   `[старый summary] + [новый отвалившийся блок]`, а не весь диалог — сжатие
   само по себе дёшево.

3. **Summary хранится отдельно** в файле сессии (`{"summary": ..., "history": [...]}`),
   с обратной совместимостью для старых плоских JSON-списков.

4. **Сжатие выключено по умолчанию** — включается явно через `CompressionSettings`
   в `make_session(...)`, либо **автоматически из конфига агента**
   (`keep_last_messages` / `summarize_messages_threshold` в `agents.yaml`);
   все существующие тесты/поведение без этих полей не меняются.

5. **Внедрение в запрос** — summary подставляется первым системным сообщением
   (перед системным промптом агента), как устойчивый контекст.

6. **Токены считаются честно** — `context_tokens` отражает реально отправленный
   стек (summary + система + живая история), что и позволяет замерить экономию.

## Конфигурация (в `data/agents.yaml`)

```yaml
agents:
  assistant:
    model: openai-gpt4o
    system_prompt: "Ты помощник."
    keep_last_messages: 10            # N — свежих сообщений дословно
    summarize_messages_threshold: 20  # M — порог сворачивания в summary
```

Когда заданы оба поля, каждая сессия этого агента автоматически включает сжатие.
Явный параметр `compression=` в `make_session(...)` имеет приоритет над конфигом.

## Сервисное сообщение и статистика

При каждом сворачивании блока `Session` формирует [`CompressionEvent`](../llm_bot/compress.py):
сколько сообщений свёрнуто, оценённые освобождённые токены (`folded_tokens`), длину
summary, историю «до -> после». События накапливаются (`session.compression_events`,
`session.total_compressions`, `session.total_messages_folded`,
`session.last_compression_event`).

CLI печатает сервисную строку в stderr после ответа:

```
[compression] свёрнуто 4 сообщений, -22 токенов контекста, история 8->4, summary 2 симв., (всего сжатий: 3)
```

Программно можно передать `on_compress=` в `make_session(...)` и получить событие
в колбэке для своей логики/UI.

## Ограничение размера summary

Бегущий summary растёт с каждым сворачиванием и со временем может не влезть в
контекстное окно модели. Чтобы этого избежать, у сжатия есть два лимита:

* `max_summary_tokens` — жёсткий предел summary в токенах;
* `max_summary_ratio` (по умолчанию `0.3`) — страховка: summary ≤ этой доли
  контекстного окна модели, даже если токен-лимит не задан.

Эффективный бюджет = **меньшее из двух**; в символы переводится из расчёта
~4 символа на токен (наш эстиматор). Работает в две ступени:

1. **Мягкий лимит в промпте** — в `summarize_prompt` добавляется инструкция
   «изложение должно быть не длиннее ~N символов», чтобы модель чаще сама
   укладывалась в бюджет;
2. **Жёсткая обрезка в коде** — `ContextCompressor` после ответа модели
   детерминированно обрезает summary до бюджета (по границе слова) с пометкой
   «…», гарантируя, что summary никогда не превысит лимит.

Поля настраиваются в `agents.yaml` (`max_summary_tokens` / `max_summary_ratio`).
Когда контекстное окно модели неизвестно и токен-лимит не задан — лимит не
применяется (поведение как раньше).

## Изменения

| Файл | Что добавлено |
| --- | --- |
| `llm_bot/compress.py` (новый) | `CompressionSettings`, `ContextCompressor`, `summarize_prompt`, `render_block` |
| `llm_bot/stores.py` | `SessionStore.load_summary/save_full` (default-реализации); поля `keep_last_messages`/`summarize_messages_threshold`/`max_summary_tokens`/`max_summary_ratio` в `AgentConfig` + свойство `compression_settings` |
| `llm_bot/json_session_store.py` | чтение/запись compound-формата + миграция старых плоских файлов |
| `llm_bot/compress.py` | `CompressionEvent`; `compress()` возвращает свёрнутые сообщения; лимит размера: `max_chars` в промпте + жёсткая обрезка `truncate_summary` |
| `llm_bot/agent.py` | `Session`: `compression`, `summary`, `_summarize_block`, сжатие в `chat_with_details`; `Agent.compression_settings`; `summary_budget_chars` (бюджет из окна модели) |
| `llm_bot/agent.py` | накопление событий: `compression_events`, `total_compressions`, `total_messages_folded`, `last_compression_event` |
| `llm_bot/agent.py` | `Agent.build_messages(..., summary="")` — инъекция summary первым сообщением |
| `llm_bot/factory.py` | `make_session(..., compression=..., on_compress=...)` + авто-включение из конфига агента |
| `llm_bot/cli.py` | сервисное сообщение `[compression] ...` в stderr при сворачивании истории |
| `llm_bot/__init__.py` | экспорт `CompressionSettings`, `ContextCompressor` |
| `agents.example.yaml` | пример полей `keep_last_messages` / `summarize_messages_threshold` |
| `scripts/compare_compression.py` (новый) | офлайн-эксперимент: качество и токены до/после |
| `tests/test_compress.py` (новый) | юнит/интеграционные тесты сжатия |
| `tests/test_compare_compression.py` (новый) | тесты помощников скрипта сравнения |
| `tests/test_stores.py` | парсинг YAML-полей сжатия + roundtrip summary в JSON-хранилище |
| `results/compression_analysis.md` | отчёт сравнения |

## Логика `chat_with_details` (exception-safe)

```
projected = history + [user_msg]
if compressor: projected, summary = compressor.compress(projected, summary)
messages = build_messages(projected, summary=summary)
# budget checks -> raise (ничего не мутируется/не сохраняется)
commit history, summary
reply = client.chat(messages)
history.append(reply); store.save_full(summary=summary)
```

Сжатие выполняется на *копии* истории до проверки бюджета, поэтому при
`ContextOverflowError` сессия и хранилище остаются нетронутыми.

## Сравнение (результаты в `results/compression_analysis.md`)

* Без сжатия: Recall 100%, промпт-токены 13447.
* Сжатие (retention=1.0): Recall 100%, всего 8157 → **экономия 39.3%**.
* Сжатие (retention=0.6): Recall 37.5%, всего 5944 → **экономия 55.8%**.

Вывод: сжатие стабильно экономит токены (учтя собственные вызовы суммаризации);
компромисс по качеству зависит от качества суммаризатора и размера `keep_last`.

## Как воспроизвести

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/compare_compression.py
.venv/bin/python scripts/compare_compression.py --retention 0.6