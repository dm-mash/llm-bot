"""End-to-end tests for the currency MCP server (real stdio child process).

Each test spawns the ``llm_bot.mcp_servers.currency_yahoo`` module through the same
MCP client stack the agent uses, offline via the ``CURRENCY_FIXTURE`` env var.
The key test executes the whole pipeline fetch_rates -> analyze_rates ->
save_report, feeding each tool's text output into the next tool's arguments —
this is exactly how the agent passes data between the tools.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK is not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SERVER_MODULE = "llm_bot.mcp_servers.currency_yahoo"

SERIES = [
    {"date": "2025-01-01", "rate": 100.0},
    {"date": "2025-01-02", "rate": 101.0},
    {"date": "2025-01-03", "rate": 99.0},
    {"date": "2025-01-04", "rate": 102.5},
]

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def paths(tmp_path):
    fixture = tmp_path / "series.json"
    fixture.write_text(json.dumps(SERIES), encoding="utf-8")
    return {
        "fixture": fixture,
        "reports": tmp_path / "reports.json",
        "series_dir": tmp_path / "series_out",
    }


def _params(paths: dict) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=str(ROOT),
        env={
            **os.environ,
            "CURRENCY_FIXTURE": str(paths["fixture"]),
            "CURRENCY_REPORTS_DB": str(paths["reports"]),
            "CURRENCY_SERIES_DIR": str(paths["series_dir"]),
        },
    )


async def _with_session(paths: dict, action):
    async with stdio_client(_params(paths)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


def _text(result) -> str:
    return result.content[0].text


async def test_server_lists_tools_with_schemas(paths):
    """Registration + typed parameter schemas."""

    async def action(session: ClientSession):
        return await session.list_tools()

    catalog = await _with_session(paths, action)
    tools = {t.name: t for t in catalog.tools}
    assert {"fetch_rates", "analyze_rates", "save_report", "save_series"} <= set(tools)

    fetch = tools["fetch_rates"]
    assert set(fetch.inputSchema.get("properties", {})) >= {
        "currency_code",
        "date_from",
        "date_to",
    }
    assert set(fetch.inputSchema.get("required", [])) == {
        "currency_code",
        "date_from",
        "date_to",
    }
    assert set(tools["analyze_rates"].inputSchema.get("required", [])) == {"data"}
    assert set(tools["save_report"].inputSchema.get("required", [])) == {"data"}

    # The chain contract lives in the FIRST line of every description: that is
    # exactly what MCPRouter shows the model in the prompt block.
    assert "analyze_rates" in tools["fetch_rates"].description.splitlines()[0]
    assert "fetch_rates" in tools["analyze_rates"].description.splitlines()[0]
    assert "analyze_rates" in tools["save_report"].description.splitlines()[0]
    assert "fetch_rates" in tools["save_series"].description.splitlines()[0]


async def test_server_full_chain_passes_data_between_tools(paths, tmp_path):
    """fetch -> analyze -> save: each output feeds the next tool's input."""

    async def action(session: ClientSession):
        fetched = await session.call_tool(
            "fetch_rates",
            {"currency_code": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31"},
        )
        analyzed = await session.call_tool(
            "analyze_rates",
            {"data": _text(fetched), "currency_code": "USD"},  # raw previous output
        )
        saved = await session.call_tool(
            "save_report",
            {"data": _text(analyzed), "title": "Курс USD за 2025"},  # raw previous output
        )
        return fetched, analyzed, saved

    fetched, analyzed, saved = await _with_session(paths, action)

    # Step 1 returned the strict-JSON series.
    payload = json.loads(_text(fetched))
    assert payload["count"] == 4
    assert payload["items"][0] == {"date": "2025-01-01", "rate": 100.0}

    # Step 2 consumed the raw step-1 text and produced the aggregates.
    summary = json.loads(_text(analyzed))
    assert summary["count"] == 4
    assert summary["end_rate"] == 102.5
    assert summary["change_pct"] == 2.5
    assert summary["trend"] == "рост"

    # Step 3 stored the report built from the raw step-2 text...
    assert not saved.isError
    assert "id=" in _text(saved)

    # ...and the store really contains the expected numbers.
    stored = json.loads(paths["reports"].read_text(encoding="utf-8"))
    assert len(stored) == 1
    assert stored[0]["report"]["min"]["rate"] == 99.0
    assert stored[0]["report"]["max"]["rate"] == 102.5
    assert stored[0]["title"] == "Курс USD за 2025"


async def test_server_accepts_data_as_inline_object(paths):
    """The model may pass the previous output as a parsed object, not a string.

    Regression from the live run: the model inlined the fetch result into
    ``data`` as a JSON object and the strict ``str`` annotation rejected it,
    costing a corrective round-trip. Both tools now accept either form.
    """

    async def action(session: ClientSession):
        fetched = await session.call_tool(
            "fetch_rates",
            {"currency_code": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31"},
        )
        series_object = json.loads(_text(fetched))  # inline object, NOT a string
        analyzed = await session.call_tool(
            "analyze_rates", {"data": series_object, "currency_code": "USD"}
        )
        summary_object = json.loads(_text(analyzed))
        saved = await session.call_tool("save_report", {"data": summary_object})
        return analyzed, saved

    analyzed, saved = await _with_session(paths, action)

    assert not analyzed.isError
    summary = json.loads(_text(analyzed))
    assert summary["end_rate"] == 102.5
    assert summary["trend"] == "рост"
    assert not saved.isError
    assert "id=" in _text(saved)


async def test_server_save_report_is_idempotent(paths):
    """Re-saving the same analysis returns the existing id, no duplicate."""

    async def action(session: ClientSession):
        fetched = await session.call_tool(
            "fetch_rates",
            {"currency_code": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31"},
        )
        analyzed = await session.call_tool(
            "analyze_rates", {"data": _text(fetched)}
        )
        first = await session.call_tool("save_report", {"data": _text(analyzed)})
        again = await session.call_tool("save_report", {"data": _text(analyzed)})
        return first, again

    first, again = await _with_session(paths, action)
    first_id = _text(first).split("id=")[1].split(",")[0]
    assert "дубликат не создан" in _text(again)
    assert first_id in _text(again)
    stored = json.loads(paths["reports"].read_text(encoding="utf-8"))
    assert len(stored) == 1


async def test_server_error_result_on_bad_input(paths):
    """Invalid input surfaces as an MCP error result, not a crash."""

    async def action(session: ClientSession):
        return await session.call_tool("analyze_rates", {"data": "мусор"})

    result = await _with_session(paths, action)
    assert result.isError


async def test_server_save_series_dumps_raw_file_without_analysis(paths):
    """fetch -> save_series: the raw series lands in a file, analyze skipped.

    The partial-chain scenario: the task «получи курс и выгрузи в файл»
    needs no analyze_rates — the agent picks save_series by its contract.
    """

    async def action(session: ClientSession):
        fetched = await session.call_tool(
            "fetch_rates",
            {"currency_code": "USD", "date_from": "2025-01-01", "date_to": "2025-12-31"},
        )
        # Inline object, not a string — the form the live model actually used.
        saved = await session.call_tool(
            "save_series", {"data": json.loads(_text(fetched))}
        )
        return saved

    saved = await _with_session(paths, action)

    assert not saved.isError
    assert "Ряд сохранён в файл" in _text(saved)

    files = list(paths["series_dir"].glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["count"] == 4
    assert payload["items"][0] == {"date": "2025-01-01", "rate": 100.0}
    assert payload["items"][-1]["rate"] == 102.5
