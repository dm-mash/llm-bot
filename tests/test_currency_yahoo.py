"""Unit tests for the currency_yahoo API (backend behind the MCP server)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bot.api import currency_yahoo
from llm_bot.api.currency_yahoo import (
    CurrencyError,
    DuplicateReportError,
    analyze_rates,
    default_reports_path,
    extract_series,
    fetch_rates,
    save_report,
    save_series,
)

# Rising series with a dip: easy exact aggregates.
FIXTURE_SERIES = [
    {"date": "2025-01-01", "rate": 100.0},
    {"date": "2025-01-02", "rate": 101.0},
    {"date": "2025-01-03", "rate": 99.0},
    {"date": "2025-01-04", "rate": 102.5},
]


@pytest.fixture()
def fixture_file(tmp_path):
    """A pre-saved series file for offline fetch_rates."""
    path = tmp_path / "series.json"
    path.write_text(json.dumps(FIXTURE_SERIES), encoding="utf-8")
    return path


@pytest.fixture()
def reports_db(tmp_path, monkeypatch):
    """Point CURRENCY_REPORTS_DB at a temp file for every test."""
    path = tmp_path / "reports.json"
    monkeypatch.setenv("CURRENCY_REPORTS_DB", str(path))
    return path


@pytest.fixture()
def series_dir(tmp_path, monkeypatch):
    """Point CURRENCY_SERIES_DIR at a temp directory for every test."""
    path = tmp_path / "series_out"
    monkeypatch.setenv("CURRENCY_SERIES_DIR", str(path))
    return path


# -- fetch --------------------------------------------------------------------


def test_fetch_rates_offline_via_fixture(fixture_file):
    series = fetch_rates("USD", "2025-01-01", "2025-12-31", fixture=fixture_file)
    assert series == FIXTURE_SERIES


def test_fetch_rates_downsamples_long_series(fixture_file):
    """Long periods shrink to max_points; endpoints survive."""
    series = fetch_rates(
        "USD", "2025-01-01", "2025-12-31", fixture=fixture_file, max_points=3
    )
    assert series == [
        FIXTURE_SERIES[0],
        FIXTURE_SERIES[2],
        FIXTURE_SERIES[3],
    ]


def test_fetch_rates_validates_dates(fixture_file):
    with pytest.raises(CurrencyError, match="ГГГГ-ММ-ДД"):
        fetch_rates("USD", "01.01.2025", "2025-12-31", fixture=fixture_file)
    with pytest.raises(CurrencyError, match="позже"):
        fetch_rates("USD", "2025-12-31", "2025-01-01", fixture=fixture_file)


YAHOO_SAMPLE = {
    "chart": {
        "result": [
            {
                # 2025-01-01, 2025-01-02, 2025-01-03 (UTC); the middle day is
                # a non-trading gap (null close) and must be skipped.
                "timestamp": [1735689600, 1735776000, 1735862400],
                "indicators": {"quote": [{"close": [100.25103, None, 99.51338]}]},
            }
        ]
    }
}


def test_parse_yahoo_chart_skips_nulls_and_rounds():
    series = currency_yahoo._parse_yahoo_chart(json.dumps(YAHOO_SAMPLE))
    assert series == [
        {"date": "2025-01-01", "rate": 100.251},
        {"date": "2025-01-03", "rate": 99.5134},
    ]


def test_parse_yahoo_chart_empty_result():
    assert currency_yahoo._parse_yahoo_chart('{"chart": {"result": null}}') == []


# -- extract / analyze --------------------------------------------------------


def test_extract_series_accepts_raw_tool_output():
    payload = {"count": 1, "items": FIXTURE_SERIES}
    assert extract_series(json.dumps(payload)) == FIXTURE_SERIES
    assert extract_series(payload) == FIXTURE_SERIES
    assert extract_series(FIXTURE_SERIES) == FIXTURE_SERIES


def test_extract_series_rejects_garbage():
    with pytest.raises(CurrencyError, match="валидным JSON"):
        extract_series("не json вообще")
    with pytest.raises(CurrencyError, match="ни одной корректной точки"):
        extract_series([{"foo": 1}])


def test_analyze_rates_exact_aggregates():
    summary = analyze_rates(FIXTURE_SERIES, currency="USD")
    assert summary["count"] == 4
    assert summary["period"] == {"from": "2025-01-01", "to": "2025-01-04"}
    assert summary["start_rate"] == 100.0
    assert summary["end_rate"] == 102.5
    assert summary["min"] == {"date": "2025-01-03", "rate": 99.0}
    assert summary["max"] == {"date": "2025-01-04", "rate": 102.5}
    assert summary["mean_rate"] == 100.625
    assert summary["change_abs"] == 2.5
    assert summary["change_pct"] == 2.5
    assert summary["trend"] == "рост"


def test_analyze_rates_trends():
    falling = [
        {"date": "2025-01-01", "rate": 100.0},
        {"date": "2025-01-02", "rate": 98.0},
    ]
    assert analyze_rates(falling)["trend"] == "падение"
    flat = [
        {"date": "2025-01-01", "rate": 100.0},
        {"date": "2025-01-02", "rate": 100.0},
    ]
    assert analyze_rates(flat)["trend"] == "без изменений"


def test_analyze_rates_from_fetch_json_text(fixture_file):
    series = fetch_rates("USD", "2025-01-01", "2025-12-31", fixture=fixture_file)
    payload = {"count": len(series), "items": series}
    summary = analyze_rates(json.dumps(payload), currency="USD")
    assert summary["end_rate"] == 102.5


# -- save ---------------------------------------------------------------------


def test_default_reports_path_env_override(reports_db):
    assert default_reports_path().name == "reports.json"


def test_save_report_roundtrip(reports_db):
    summary = analyze_rates(FIXTURE_SERIES, currency="USD")
    report = save_report(summary, "Курс USD")
    assert report.id
    data = json.loads(reports_db.read_text(encoding="utf-8"))
    assert data[0]["id"] == report.id
    assert data[0]["title"] == "Курс USD"
    assert data[0]["report"]["trend"] == "рост"


def test_save_report_is_idempotent(reports_db):
    summary = analyze_rates(FIXTURE_SERIES, currency="USD")
    first = save_report(summary)
    with pytest.raises(DuplicateReportError) as exc_info:
        save_report(summary)
    assert exc_info.value.existing.id == first.id
    # Retry-safety: the duplicate path returns the existing id to re-report.
    again_id = exc_info.value.existing.id
    data = json.loads(reports_db.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["id"] == again_id


def test_save_report_allow_duplicates_opt_out(reports_db):
    summary = analyze_rates(FIXTURE_SERIES, currency="USD")
    save_report(summary)
    save_report(summary, allow_duplicates=True)
    data = json.loads(reports_db.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_save_report_validates_input(reports_db):
    with pytest.raises(CurrencyError, match="валидным JSON"):
        save_report("мусор")
    with pytest.raises(CurrencyError, match="пустой"):
        save_report({})


# -- save_series: raw dump of the fetch output ---------------------------------


FETCH_PAYLOAD = {
    "currency": "USD",
    "date_from": "2025-01-01",
    "date_to": "2025-12-31",
    "count": 4,
    "items": FIXTURE_SERIES,
}


def test_save_series_writes_fetch_payload(series_dir):
    saved = save_series(FETCH_PAYLOAD, title="Курс USD за 2025")
    assert saved["count"] == 4
    assert saved["title"] == "Курс USD за 2025"
    written = json.loads(Path(saved["path"]).read_text(encoding="utf-8"))
    assert written["currency"] == "USD"
    assert written["items"] == FIXTURE_SERIES
    assert written["title"] == "Курс USD за 2025"


def test_save_series_accepts_json_text_and_bare_list(series_dir):
    as_text = save_series(json.dumps(FETCH_PAYLOAD))
    as_list = save_series(FIXTURE_SERIES)
    assert Path(as_text["path"]).exists()
    assert as_list["count"] == 4


def test_save_series_same_period_overwrites_same_file(series_dir):
    first = save_series(FETCH_PAYLOAD)
    again = save_series(json.dumps(FETCH_PAYLOAD))
    assert first["path"] == again["path"]
    assert len(list(series_dir.iterdir())) == 1


def test_save_series_rejects_garbage(series_dir):
    with pytest.raises(CurrencyError, match="валидным JSON"):
        save_series("мусор")
    with pytest.raises(CurrencyError, match="полем items"):
        save_series({"no_items": True})
