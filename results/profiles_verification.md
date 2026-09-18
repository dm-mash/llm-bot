# Personalization (profiles) — verification notes

**Дата:** 2026-09-18
**Фича:** персонализация поверх модели памяти. Профиль = конфиг оркестрации
(`data/profiles.yaml`), применяемый на этапе композиции агента; память
(`data/memory/`) фича не затрагивает.

## Архитектура (кратко)

| Слой | Файл | Отвечает на вопрос |
|---|---|---|
| Транспорт | `data/models.yaml` | как достучаться до LLM |
| Роль | `data/agents.yaml` | кто агент |
| **Профиль** | `data/profiles.yaml` | как агент ведёт себя для этого пользователя/задачи |
| Память | `data/memory/` | что накоплено из диалогов |

Ключевой код:

- [`ProfileConfig`](../llm_bot/stores.py) + `ProfileStore` protocol — frozen
  dataclass по образцу `AgentConfig`;
- [`YamlProfileStore`](../llm_bot/yaml_stores.py) — загрузка из
  `data/profiles.yaml` (ключ `profiles`);
- [`apply_profile`](../llm_bot/profiles.py) — композиция
  `AgentConfig + ProfileConfig → новый AgentConfig` (директивы в system prompt,
  переопределения `max_response_words` / `temperature`);
- [`make_session(..., profile=...)`](../llm_bot/factory.py) — применение до
  создания клиента/сессии; профиль не проходит через память.

## Проверено

### 1. Автоматическое применение профиля (unit, без сети)

`tests/test_profiles.py` (24 теста, вся папка — 201 passing):

- `test_make_session_applies_profile_end_to_end` — через `httpx.MockTransport`
  проверено, что директивы профиля доходят до **реального payload** запроса
  (`"Профиль пользователя"` в теле, `temperature=0.3` из профиля);
- `test_make_session_without_profile_is_unchanged` — без `--profile` поведение
  байт-в-байт прежнее (backward compatibility);
- `test_profile_is_not_written_to_memory` — после применения профиля
  `len(memory.long) == 0`, `len(memory.working) == 0`,
  `memory.prefix_messages() == []` — профиль не попадает в память;
- `test_apply_profile_*` — исходный frozen `AgentConfig` не мутируется,
  переопределения применяются только когда заданы, компрессия сохраняется;
- `test_make_session_unknown_profile_raises_clear_error` — неизвестный профиль
  даёт `KeyError: Unknown profile 'ghost'. Available: ...`.

### 2. CLI (живой запуск)

```bash
cp profiles.example.yaml data/profiles.yaml

python -m llm_bot --list-profiles        # developer / student / child с настройками
python -m llm_bot --show-profile child   # настройки + сгенерированный prompt block
python -m llm_bot --show-profile ghost   # error: unknown profile 'ghost'. Available: ...
python -m llm_bot --agent assistant --profile developer   # персонализированный чат
```

Запуск без флага `--profile` не меняется: `python -m llm_bot --agent assistant`.

### 3. Разные профили → разные ответы

`scripts/compare_profiles.py` прогоняет ОДИН вопрос через несколько профилей
и пишет side-by-side отчёт:

```bash
python scripts/compare_profiles.py --profiles developer,student,child
python scripts/compare_profiles.py --include-none    # + базовая строка без профиля
```

Выход: `results/compare_profiles.md` (для чтения) + `.json` (для сравнений).
Ожидаемое поведение: `developer` — кратко + код на Python, `student` —
пошагово с аналогиями, `child` — простые слова, лимит 100 слов.
Скрипт требует рабочих `data/models.yaml` / `data/agents.yaml` (не в git).

## Что НЕ менялось

- Модель памяти (`ShortTermMemory` / `WorkingMemory` / `LongTermMemory`) —
  без изменений; авто-извлечение фактов работает как раньше.
- Компрессия, контекстные стратегии (`sliding` / `facts` / `branching`) —
  работают с персонализированным агентом без правок (проверено
  `test_apply_profile_keeps_compression_settings_property`).
- Существующие CLI-команды и все 177 прежних тестов — зелёные.

## Замечание по CLI

Изначальный план предполагал субкоманды (`llm-bot chat`, `llm-bot profiles
list`); после проверки реального кода исправлено на флаги в стиле проекта:
`--profile NAME`, `--list-profiles`, `--show-profile NAME`. Запуск команды
`python -m llm_bot --agent assistant` сохранён как есть.
