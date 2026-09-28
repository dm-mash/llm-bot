# Демо: оркестрация — один промпт ведёт цепочку через три MCP-сервера

Источник данных: `Yahoo Finance chart API (без ключа)`
Точек в ряде: 24 (2025-01-02 .. 2025-12-31)
Бюджет раундов: `max_rounds: 8` (ключ конфига mcp.yaml)

## 1. Детерминированная кросс-серверная цепочка через MCP (без LLM)

fetch_rates -> analyze_rates -> save_report (currency_yahoo) -> add_note (notes) -> schedule_task (scheduler)

- `currency_yahoo__fetch_rates` **OK**: `{"currency": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31", "count": 24, "items": [{"date": "2025-01-02", "rate": 113.7222}, {"date": "2025-01-17", "rate": 103.6136}, {"date": "2025-02-03", "rate": 98.7553}, …`
- `currency_yahoo__analyze_rates` **OK**: `{"currency": "USD", "period": {"from": "2025-01-02", "to": "2025-12-31"}, "count": 24, "start_rate": 113.7222, "end_rate": 79.4962, "min": {"date": "2025-07-10", "rate": 77.7971}, "max": {"date": "2025-01-02", "rate": 11…`
- `currency_yahoo__save_report` **OK**: `Отчёт сохранён: id=00d50877, «Курс USD за 2025»`
- `notes__add_note` **OK**: `Заметка сохранена: id=1, теги: валюта, отчёт`
- `scheduler__schedule_task` **OK**: `Задача создана: id=23420e19e246, «Ежедневный сбор курса USD», действие mcp_call, первый запуск 2026-09-28T23:00:00+00:00`

Заметка (числа пришли из анализа через MCP): «Курс USD за 2025: 113.7222 -> 79.4962 (-30.1%, падение).»

## 2. Агент сам ведёт всю цепочку (один промпт, реальная модель)

Промпт: «Проанализируй курс доллара за 2025 год, сохрани отчёт, добавь заметку с итогом и поставь ежедневный сбор курса на 09:00.»

```
(Waiting for tool response)
```

Вызовы агента (по порядку):


Вызовов инструментов: 0; серверов задействовано: 0 ()

## 3. Хранилища после прогона

### currency_yahoo (отчёты)

```json
[
  {
    "id": "00d50877",
    "hash": "00d50877",
    "title": "Курс USD за 2025",
    "created": "2026-09-28T18:56:34+10:00",
    "report": {
      "currency": "USD",
      "period": {
        "from": "2025-01-02",
        "to": "2025-12-31"
      },
      "count": 24,
      "start_rate": 113.7222,
      "end_rate": 79.4962,
      "min": {
        "date": "2025-07-10",
        "rate": 77.7971
      },
      "max": {
        "date": "2025-01-02",
        "rate": 113.7222
      },
      "mean_rate": 84.5923,
      "change_abs": -34.226,
      "change_pct": -30.1,
      "trend": "падение"
    }
  }
]
```

### notes (CRM)

```json
[
  {
    "id": 1,
    "text": "Курс USD за 2025: 113.7222 -> 79.4962 (-30.1%, падение).",
    "tags": [
      "валюта",
      "отчёт"
    ]
  }
]
```

### scheduler (фоновые задачи)

```json
{
  "tasks": [
    {
      "id": "23420e19e246",
      "title": "Ежедневный сбор курса USD",
      "action": "mcp_call",
      "schedule": {
        "kind": "daily",
        "at": "09:00"
      },
      "payload": {
        "server": "currency_yahoo",
        "tool": "fetch_rates",
        "arguments": {
          "currency_code": "USD",
          "date_from": "2025-01-01",
          "date_to": "2025-12-31"
        }
      },
      "group": "финансы",
      "status": "active",
      "created_at": "2026-09-28T08:56:35+00:00",
      "next_run": "2026-09-28T23:00:00+00:00",
      "last_run": "",
      "attempts": 0,
      "max_attempts": 3
    }
  ],
  "results": []
}
```
