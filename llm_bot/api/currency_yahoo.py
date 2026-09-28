"""Currency-rate API — the backend wrapped by the currency MCP server.

Fetches the daily dynamics of a currency rate against the ruble from the
Yahoo Finance chart API (one request per period, no API key; the old CBR
``XML_dynamic.asp`` endpoint is decommissioned and answers 404), turns it
into a plain JSON series, computes aggregate statistics, stores analysis
reports in a JSON file and dumps raw series to files. The MCP server
(``llm_bot.mcp_servers.currency_yahoo``) exposes these operations as tools; the
chat agent chains them by itself, passing each tool's JSON output as the
next tool's input — no step numbering, the data contracts in the tool
descriptions ARE the sequence:

    fetch_rates -> analyze_rates -> save_report   (analysis report)
    fetch_rates -> save_series                    (raw dump to a file)

Offline mode: pass ``fixture=<path>`` (or set ``CURRENCY_FIXTURE``) to read a
pre-saved series from a JSON file instead of the network — unit and e2e tests
run fully offline this way.

The reports file (``CURRENCY_REPORTS_DB``, default ``data/currency_reports.json``)
is created lazily on the first write, guarded by a thread lock, and every
write is atomic (tmp file + rename), following :mod:`llm_bot.api.notes`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

# Environment variables holding the reports database, the series dump
# directory and the fixture path.
_ENV_REPORTS_DB = "CURRENCY_REPORTS_DB"
_ENV_SERIES_DIR = "CURRENCY_SERIES_DIR"
_ENV_FIXTURE = "CURRENCY_FIXTURE"
DEFAULT_REPORTS_PATH = Path("data") / "currency_reports.json"
DEFAULT_SERIES_DIR = Path("data") / "currency_series"

# Yahoo Finance chart endpoint: daily dynamics, one request per period.
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# ISO code -> Yahoo symbol quoted in rubles per one unit of the currency.
# Any other Yahoo FX symbol (e.g. ``KZTRUB=X``) is passed through as-is.
YAHOO_SYMBOLS = {"USD": "RUB=X", "EUR": "EURRUB=X", "GBP": "GBPRUB=X"}

_lock = threading.Lock()


class CurrencyError(Exception):
    """Any failure of the currency pipeline (network, parsing, input)."""


class DuplicateReportError(Exception):
    """An identical report (same content hash) is already stored.

    Carries the existing record so the MCP tool can report its id instead of
    writing a duplicate — makes re-execution after LLM retries safe, mirroring
    :class:`llm_bot.api.notes.DuplicateNoteError`.
    """

    def __init__(self, existing: Report) -> None:
        self.existing = existing
        super().__init__(f"Отчёт уже сохранён: id={existing.id}")


@dataclass(frozen=True)
class Report:
    """One stored analysis report."""

    id: str
    title: str
    created: str
    report: dict = field(default_factory=dict)


# -- dates and codes ----------------------------------------------------------


def _parse_date(value: str, name: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` date, raising CurrencyError otherwise."""
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise CurrencyError(
            f"Неверная дата {name}={value!r}: ожидается формат ГГГГ-ММ-ДД."
        ) from exc


def _resolve_symbol(currency: str) -> str:
    """Map an ISO code (USD/EUR/GBP) to a Yahoo FX symbol; pass others through."""
    code = str(currency or "").strip().upper()
    if not code:
        raise CurrencyError("Не указана валюта (currency_code).")
    return YAHOO_SYMBOLS.get(code, code)


# -- step 1: fetch ------------------------------------------------------------


# The fetched series travels THROUGH the model into the analyze tool's
# arguments, so its JSON must stay small twice over: it must fit into one
# model reply (output-token limits) and keep the WHOLE conversation cheap
# enough for tight tokens-per-minute budgets of free LLM tiers. Longer
# periods are evenly downsampled with the first and last points kept
# (24 points over a year ~ weekly sampling — plenty for a trend report).
DEFAULT_MAX_POINTS = 24


def fetch_rates(
    currency: str,
    date_from: str,
    date_to: str,
    *,
    fixture: Path | str | None = None,
    max_points: int | None = None,
) -> list[dict]:
    """Return the rate series ``[{"date": "YYYY-MM-DD", "rate": float}, ...]``.

    Sorted ascending by date, one point per trading day. Periods longer than
    *max_points* (default :data:`DEFAULT_MAX_POINTS`) are evenly downsampled,
    keeping the first and the last point — the JSON must stay small enough
    for the agent to copy it into the next tool's arguments. With *fixture*
    (or ``CURRENCY_FIXTURE``) the series is read from a local JSON file
    instead of the network — the file format is exactly the returned list.
    """
    day_from = _parse_date(date_from, "date_from")
    day_to = _parse_date(date_to, "date_to")
    if day_from > day_to:
        raise CurrencyError(
            f"date_from ({date_from}) позже date_to ({date_to})."
        )
    limit = DEFAULT_MAX_POINTS if max_points is None else max_points

    fixture_path = Path(fixture) if fixture else _fixture_from_env()
    if fixture_path is not None:
        return _downsample(_load_fixture(fixture_path), limit)

    symbol = _resolve_symbol(currency)
    period1 = int(
        datetime(
            day_from.year, day_from.month, day_from.day, tzinfo=timezone.utc
        ).timestamp()
    )
    period2 = (
        int(
            datetime(
                day_to.year, day_to.month, day_to.day, tzinfo=timezone.utc
            ).timestamp()
        )
        + 86400  # include the last day (exclusive upper bound)
    )
    try:
        response = httpx.get(
            YAHOO_CHART_URL.format(symbol=symbol),
            params={"period1": period1, "period2": period2, "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0 (llm-bot currency-mcp)"},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CurrencyError(f"Источник курса недоступен: {exc}") from exc

    series = _parse_yahoo_chart(response.text)
    if not series:
        raise CurrencyError(
            "Источник не вернул данных за период "
            f"{date_from}..{date_to} (валюта {currency})."
        )
    return _downsample(series, limit)


def _downsample(series: list[dict], limit: int) -> list[dict]:
    """Evenly shrink *series* to *limit* points, keeping first and last."""
    total = len(series)
    if limit < 2 or total <= limit:
        return series
    step = (total - 1) / (limit - 1)
    return [series[round(i * step)] for i in range(limit)]


def _fixture_from_env() -> Path | None:
    raw = os.getenv(_ENV_FIXTURE)
    return Path(raw) if raw else None


def _load_fixture(path: Path) -> list[dict]:
    """Read a pre-saved series (same format as :func:`fetch_rates` returns)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CurrencyError(f"Fixture-файл не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CurrencyError(f"Fixture-файл повреждён: {path}: {exc}") from exc
    if not isinstance(data, list):
        raise CurrencyError("Fixture-файл должен содержать JSON-массив.")
    return data


def _parse_yahoo_chart(json_text: str) -> list[dict]:
    """Extract ``{date, rate}`` records from the Yahoo chart reply.

    The chart quotes rubles per one unit of the currency; non-trading days
    come as ``null`` closes and are skipped.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise CurrencyError(f"Некорректный JSON от источника курса: {exc}") from exc
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        return []
    stamps = result[0].get("timestamp") or []
    quote = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    records: list[dict] = []
    for stamp, close in zip(stamps, closes):
        if close is None:
            continue
        day = datetime.fromtimestamp(stamp, tz=timezone.utc).date()
        records.append(
            {"date": day.strftime("%Y-%m-%d"), "rate": round(float(close), 4)}
        )
    records.sort(key=lambda item: item["date"])
    return records


# -- step 2: analyze ----------------------------------------------------------


def extract_series(data: str | dict | list) -> list[dict]:
    """Normalize the output of ``fetch_rates`` into a plain list of records.

    Accepts a raw JSON string (the previous tool's text result), a dict with
    an ``items`` key, or an already-parsed list — the chain contract is
    «feed the previous tool's output here as-is».
    """
    if isinstance(data, str):
        text = data.strip()
        if not text:
            raise CurrencyError("Пустые данные: ожидается JSON-ряд из fetch_rates.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CurrencyError(
                "Данные не являются валидным JSON: передайте вывод fetch_rates "
                "без изменений."
            ) from exc
    if isinstance(data, dict):
        data = data.get("items", data)
    if not isinstance(data, list):
        raise CurrencyError("Ожидается JSON-ряд (массив точек с date и rate).")

    series: list[dict] = []
    for point in data:
        if not isinstance(point, dict):
            continue
        try:
            series.append(
                {
                    "date": str(point["date"]),
                    "rate": float(point["rate"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not series:
        raise CurrencyError(
            "В данных нет ни одной корректной точки (нужны поля date и rate)."
        )
    series.sort(key=lambda item: item["date"])
    return series


def analyze_rates(data: str | dict | list, *, currency: str = "") -> dict:
    """Compute aggregates over a rate series; returns a JSON-ready dict.

    Metrics: count, first/last date, start/end rate, min/max (with dates),
    mean, absolute and percent change, and a qualitative trend.
    """
    series = extract_series(data)
    rates = [point["rate"] for point in series]
    start, end = rates[0], rates[-1]
    change = round(end - start, 4)
    change_pct = round((end - start) / start * 100, 2) if start else 0.0
    if change_pct > 0:
        trend = "рост"
    elif change_pct < 0:
        trend = "падение"
    else:
        trend = "без изменений"
    min_point = min(series, key=lambda point: point["rate"])
    max_point = max(series, key=lambda point: point["rate"])
    return {
        "currency": str(currency or "").strip().upper(),
        "period": {"from": series[0]["date"], "to": series[-1]["date"]},
        "count": len(series),
        "start_rate": start,
        "end_rate": end,
        "min": {"date": min_point["date"], "rate": min_point["rate"]},
        "max": {"date": max_point["date"], "rate": max_point["rate"]},
        "mean_rate": round(sum(rates) / len(rates), 4),
        "change_abs": change,
        "change_pct": change_pct,
        "trend": trend,
    }


# -- step 3: save -------------------------------------------------------------


def default_reports_path() -> Path:
    """Return the reports database path (``CURRENCY_REPORTS_DB`` or default)."""
    raw = os.getenv(_ENV_REPORTS_DB)
    return Path(raw) if raw else DEFAULT_REPORTS_PATH


def default_series_dir() -> Path:
    """Return the directory for raw series dumps (``CURRENCY_SERIES_DIR``)."""
    raw = os.getenv(_ENV_SERIES_DIR)
    return Path(raw) if raw else DEFAULT_SERIES_DIR


def _load(path: Path) -> list[dict]:
    """Read the reports list; missing/corrupt file means empty."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _save(path: Path, reports: list[dict]) -> None:
    """Atomically write the reports list (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def _report_hash(report: dict) -> str:
    """Stable hash of the report content (idempotency key)."""
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def save_report(
    data: str | dict,
    title: str = "",
    *,
    db_path: Path | None = None,
    allow_duplicates: bool = False,
) -> Report:
    """Store an analysis report; returns it with its assigned id.

    Idempotent by default: an identical report (same content hash) is NOT
    written twice — :class:`DuplicateReportError` carries the existing record
    so retries after LLM/tool failures never duplicate data. *data* is the
    output of :func:`analyze_rates` (dict or its JSON text).
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CurrencyError(
                "Отчёт не является валидным JSON: передайте вывод analyze_rates."
            ) from exc
    if not isinstance(data, dict):
        raise CurrencyError("Ожидается JSON-объект с агрегатами из analyze_rates.")
    if not data:
        raise CurrencyError("Отчёт пустой — сохранять нечего.")

    path = db_path or default_reports_path()
    content_hash = _report_hash(data)
    clean_title = str(title or "").strip()
    with _lock:
        reports = _load(path)
        if not allow_duplicates:
            for item in reports:
                if str(item.get("hash", "")) == content_hash:
                    raise DuplicateReportError(
                        Report(
                            id=str(item.get("id", "")),
                            title=str(item.get("title", "")),
                            created=str(item.get("created", "")),
                            report=dict(item.get("report", {})),
                        )
                    )
        report_id = content_hash
        created = datetime.now().astimezone().isoformat(timespec="seconds")
        reports.append(
            {
                "id": report_id,
                "hash": content_hash,
                "title": clean_title,
                "created": created,
                "report": data,
            }
        )
        _save(path, reports)
    return Report(id=report_id, title=clean_title, created=created, report=data)


def save_series(
    data: str | dict | list,
    title: str = "",
    *,
    dir_path: Path | None = None,
) -> dict:
    """Dump a raw rate series (the output of :func:`fetch_rates`) to a file.

    Accepts the same forms the MCP tool receives: the fetch payload as its
    JSON text or an already-parsed object, or a bare list of points. The file
    name is derived from the currency and the period, so re-saving the same
    period overwrites the same file instead of piling up copies. Writes are
    atomic (tmp file + rename). Returns ``{"path", "count", "title"}``.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CurrencyError(
                "Ряд не является валидным JSON: передайте вывод fetch_rates."
            ) from exc
    if isinstance(data, dict) and "items" in data:
        payload = dict(data)
    elif isinstance(data, list):
        payload = {"items": data}
    else:
        raise CurrencyError(
            "Ожидается ряд из fetch_rates: объект с полем items или список точек."
        )
    series = extract_series(payload)
    if not series:
        raise CurrencyError("Ряд пустой — сохранять нечего.")
    payload["items"] = series
    payload["count"] = len(series)
    clean_title = str(title or "").strip()
    if clean_title:
        payload["title"] = clean_title

    code = str(payload.get("currency", "")).strip().lower() or "rates"
    date_from = str(payload.get("date_from", "")).strip() or "unknown"
    date_to = str(payload.get("date_to", "")).strip() or "unknown"
    stem = f"{code}_{date_from}_{date_to}"
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem)
    directory = Path(dir_path) if dir_path else default_series_dir()
    target = directory / f"{safe}.json"
    directory.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(target)
    return {"path": str(target), "count": len(series), "title": clean_title}
