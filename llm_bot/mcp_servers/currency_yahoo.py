"""MCP server: currency_yahoo — pipeline tools the agent chains by itself.

The chat agent composes the tools into a chain on its own: every tool's FIRST
description line states its data contract (what it consumes, what it returns
and where to pass the result), and that line is what the router advertises in
the system prompt. There is no step numbering — the named input/output
references ARE the sequence, so different tasks produce different chains:

    fetch_rates -> analyze_rates -> save_report   (analysis report)
    fetch_rates -> save_series                    (raw dump to a file)

* ``fetch_rates(currency_code, date_from, date_to)`` — daily rate series in
  rubles per currency unit (Yahoo Finance, no key) as strict JSON
  (``items``); offline tests use the ``CURRENCY_FIXTURE`` env var pointing
  at a pre-saved series file;
* ``analyze_rates(data)`` — aggregates over the series (strict JSON); the
  series goes in as a JSON string or an already-parsed object;
* ``save_report(data, title?)`` — stores the aggregates as a report
  (``CURRENCY_REPORTS_DB``, default ``data/currency_reports.json``),
  idempotent like add_note;
* ``save_series(data, title?)`` — dumps the raw series to a JSON file
  (``CURRENCY_SERIES_DIR``, default ``data/currency_series``); the file name
  comes from the currency and the period, so a re-dump overwrites the same
  file instead of piling up copies.

Speaks MCP over stdio: ONLY protocol frames go to stdout, diagnostics to
stderr. Run from the project root for a smoke check (it will wait on stdin):
    python -m llm_bot.mcp_servers.currency_yahoo
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from llm_bot.api import currency_yahoo

mcp = FastMCP("currency_yahoo")


def _error(exc: Exception) -> str:
    """Uniform tool-error text (FastMCP turns raised exceptions into isError)."""
    raise ValueError(str(exc)) from exc


@mcp.tool()
def fetch_rates(currency_code: str, date_from: str, date_to: str) -> str:
    """Получить курс валюты к рублю за период; вернёт строгий JSON с полем items — ряд точек {date, rate} (длинные периоды прорежены до ~24 точек); передайте items целиком в analyze_rates для анализа или в save_series, чтобы выгрузить ряд в файл."""
    try:
        series = currency_yahoo.fetch_rates(currency_code, date_from, date_to)
    except currency_yahoo.CurrencyError as exc:
        _error(exc)
    payload = {
        "currency": str(currency_code).strip().upper(),
        "date_from": date_from,
        "date_to": date_to,
        "count": len(series),
        "items": series,
    }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def analyze_rates(data: str | dict | list, currency_code: str = "") -> str:
    """Проанализировать ряд точек из fetch_rates: передайте его в data без изменений (строкой JSON или готовым объектом); вернёт строгий JSON агрегатов, передайте его в save_report."""
    try:
        summary = currency_yahoo.analyze_rates(data, currency=currency_code)
    except currency_yahoo.CurrencyError as exc:
        _error(exc)
    return json.dumps(summary, ensure_ascii=False)


@mcp.tool()
def save_report(data: str | dict, title: str = "") -> str:
    """Сохранить отчёт об анализе (только когда просят именно анализ: динамику, тренд, min/max; простая выгрузка/сохранение курса в файл — это save_series): передайте агрегаты из analyze_rates в data без изменений (строкой JSON или готовым объектом); вернёт id отчёта (повтор — без дубликата)."""
    try:
        report = currency_yahoo.save_report(data, title)
    except currency_yahoo.DuplicateReportError as exc:
        report = exc.existing
        action = "Такой отчёт уже сохранён, дубликат не создан"
    except currency_yahoo.CurrencyError as exc:
        _error(exc)
    else:
        action = "Отчёт сохранён"
    title_part = f", «{report.title}»" if report.title else ""
    return f"{action}: id={report.id}{title_part}"


@mcp.tool()
def save_series(data: str | dict | list, title: str = "") -> str:
    """Выгрузить курс за период в JSON-файл — скачать ряд из fetch_rates и сохранить его целиком, БЕЗ анализа (просьбы «выгрузи/сохрани курс», «выгрузи … и сохрани отчёт» без анализа — это сюда): передайте его вывод в data без изменений (строкой JSON или готовым объектом); вернёт путь к файлу."""
    try:
        saved = currency_yahoo.save_series(data, title)
    except currency_yahoo.CurrencyError as exc:
        _error(exc)
    title_part = f", «{saved['title']}»" if saved["title"] else ""
    return (
        f"Ряд сохранён в файл: {saved['path']} "
        f"(точек в ряду: {saved['count']}){title_part}"
    )


if __name__ == "__main__":
    # stdio transport is the FastMCP default; stdout is reserved for the
    # protocol, so keep every diagnostic message on stderr.
    mcp.run()
