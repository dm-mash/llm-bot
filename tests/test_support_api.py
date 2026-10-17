"""Tests for the read-only support desk API (the backend behind the MCP server)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from llm_bot.api.support import (
    MAX_TICKETS,
    NotFoundError,
    SupportError,
    default_db_path,
    get_ticket,
    get_user,
    list_tickets,
    list_users,
    page_tickets,
    render_ticket,
    render_tickets,
    render_user,
    search_tickets,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "support.example.json"


@pytest.fixture()
def desk(tmp_path, monkeypatch):
    """A copy of the shipped demo desk, pointed at by SUPPORT_DB."""
    path = tmp_path / "support.json"
    shutil.copy(EXAMPLE, path)
    monkeypatch.setenv("SUPPORT_DB", str(path))
    return path


def test_default_db_path_env_override(desk):
    assert default_db_path() == desk


def test_the_shipped_example_is_valid_json():
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(data["users"], list) and data["users"]
    assert isinstance(data["tickets"], list) and data["tickets"]


def test_get_user_returns_the_record(desk):
    user = get_user("U-1001")
    assert user.email == "anna@example.com"
    assert user.plan == "pro"


def test_an_unknown_user_is_an_error_not_an_empty_record(desk):
    # A silent empty record would let the assistant tell a customer their account
    # simply does not exist.
    with pytest.raises(NotFoundError):
        get_user("U-9999")


def test_get_ticket_carries_the_whole_conversation(desk):
    ticket = get_ticket("T-5001")
    assert ticket.user_id == "U-1001"
    assert ticket.status == "open"
    assert len(ticket.messages) == 3
    assert "сброс" in ticket.messages[-1].text.lower()


def test_an_unknown_ticket_is_an_error(desk):
    with pytest.raises(NotFoundError):
        get_ticket("T-0000")


def test_tickets_are_listed_newest_first(desk):
    tickets = list_tickets()
    stamps = [t.updated_at or t.created_at for t in tickets]
    assert stamps == sorted(stamps, reverse=True)


def test_tickets_can_be_filtered_by_user_and_status(desk):
    only_anna = list_tickets(user_id="U-1001")
    assert {t.id for t in only_anna} == {"T-5001", "T-5003"}
    open_only = list_tickets(status="OPEN")  # case-insensitive
    assert all(t.status == "open" for t in open_only)


def test_search_looks_inside_the_messages_not_only_the_subject(desk):
    # A support agent searches by what the customer wrote, not by a subject line.
    hits = search_tickets("резервные коды")
    assert [t.id for t in hits] == ["T-5002"]


def test_search_escapes_the_query_instead_of_compiling_it(desk):
    # A regex metacharacter must be matched literally, not blow up or match all.
    assert search_tickets(".") != []
    assert len(search_tickets(".")) == len(list_tickets())


def test_an_empty_search_query_is_rejected(desk):
    with pytest.raises(ValueError):
        search_tickets("   ")


def test_a_missing_desk_file_is_reported_as_a_setup_problem(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_DB", str(tmp_path / "absent.json"))
    with pytest.raises(SupportError) as excinfo:
        list_tickets()
    assert "support.example.json" in str(excinfo.value)


def test_a_corrupt_desk_file_is_not_read_as_an_empty_desk(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SUPPORT_DB", str(path))
    with pytest.raises(SupportError):
        list_tickets()


def test_renders_name_the_customer_in_the_ticket(desk):
    ticket = get_ticket("T-5001")
    text = render_ticket(ticket, get_user(ticket.user_id))
    assert "T-5001" in text
    assert "anna@example.com" in text
    assert "Переписка:" in text


def test_render_ticket_without_a_user_still_works(desk):
    assert "Клиент:" not in render_ticket(get_ticket("T-5001"))


def test_render_tickets_says_so_when_empty(desk):
    assert "Не найдено" in render_tickets([])


def test_render_user_lists_the_plan(desk):
    text = render_user(get_user("U-1002"))
    assert "business" in text
    assert "dmitry@example.com" in text


def test_the_demo_desk_covers_the_authorisation_case(desk):
    # The day-33 example question must be answerable from the shipped data.
    ticket = get_ticket("T-5001")
    assert "авторизация" in [t.lower() for t in ticket.tags]
    assert "не могу войти" in ticket.subject.lower() or "войти" in ticket.subject.lower()

# --------------------------------------------------------------------------- #
# The display cap must never turn into a lookup limit
# --------------------------------------------------------------------------- #


def _big_desk(path: Path, count: int) -> Path:
    data = {
        "users": [{"id": "U-1", "email": "a@b.c", "name": "A"}],
        "tickets": [
            {
                "id": f"T-{i:04d}",
                "user_id": "U-1",
                "subject": f"обращение {i}",
                "status": "open",
                "updated_at": f"2026-01-{i % 28 + 1:02d}",
                "tags": [],
                "messages": [],
            }
            for i in range(count)
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_ticket_older_than_the_cap_is_still_found_by_id(tmp_path):
    # A lookup capped at MAX_TICKETS would answer "не найден" about a ticket that
    # exists — the worst thing a support desk can tell a customer.
    desk = _big_desk(tmp_path / "big.json", 120)
    assert get_ticket("T-0000", path=desk).id == "T-0000"


def test_search_sees_the_whole_desk_not_only_the_first_page(tmp_path):
    desk = _big_desk(tmp_path / "big.json", 120)
    assert [t.id for t in search_tickets("обращение 0", path=desk)]


def test_page_tickets_reports_whether_it_actually_truncated(tmp_path):
    desk = _big_desk(tmp_path / "big.json", 120)
    _, truncated = page_tickets(path=desk)
    assert truncated is True
    # Exactly at the cap nothing was cut off, and saying otherwise would tell the
    # model the list is partial when it is not.
    desk2 = _big_desk(tmp_path / "exact.json", MAX_TICKETS)
    _, truncated2 = page_tickets(path=desk2)
    assert truncated2 is False


def test_the_truncation_line_appears_only_when_something_was_cut(tmp_path):
    exact = _big_desk(tmp_path / "exact.json", MAX_TICKETS)
    assert "первые" not in render_tickets(*[list_tickets(path=exact)])
    big = _big_desk(tmp_path / "big.json", 120)
    tickets, truncated = page_tickets(path=big)
    assert "первые" in render_tickets(tickets, truncated=truncated)
