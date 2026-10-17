"""Read-only support desk API — users and tickets behind the support MCP server.

A deliberately small stand-in for a CRM: the whole desk is a JSON file
(``SUPPORT_DB`` env var, default ``data/support.json``) holding two lists,
``users`` and ``tickets``. The MCP server exposes it as tools, so the support
assistant reaches a customer's data through MCP — never by importing this
module directly.

Everything here is **read-only by design**: an assistant that may only look may
not invent state, and a demo data file stays reproducible across repeated runs.
Writes belong in the system of record, not in the assistant.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable holding the support desk data file.
_ENV_SUPPORT_DB = "SUPPORT_DB"
DEFAULT_DB_PATH = Path("data") / "support.json"

# The descriptor is rendered into the model's context, so a pathologically large
# desk would crowd out the dialog. The cap applies to LISTS the assistant reads,
# never to a lookup by id: a ticket that exists must be findable by its id, or
# the assistant tells a customer their ticket does not exist.
MAX_TICKETS = 100


class SupportError(RuntimeError):
    """Raised when the support data is unavailable or a lookup is impossible."""


class NotFoundError(SupportError):
    """Raised when a requested user or ticket does not exist."""


def default_db_path() -> Path:
    """Return the support data path (``SUPPORT_DB`` or ``data/support.json``)."""
    raw = os.getenv(_ENV_SUPPORT_DB)
    return Path(raw) if raw else DEFAULT_DB_PATH


@dataclass(frozen=True)
class User:
    """One customer record."""

    id: str
    email: str
    name: str
    plan: str = "free"
    status: str = "active"
    created_at: str = ""


@dataclass(frozen=True)
class TicketMessage:
    """One message inside a ticket."""

    author: str
    text: str
    at: str = ""


@dataclass(frozen=True)
class Ticket:
    """One support ticket with its message history."""

    id: str
    user_id: str
    subject: str
    status: str
    priority: str = "normal"
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    messages: list[TicketMessage] = field(default_factory=list)


def _read_db(path: Path) -> dict:
    """Read the desk file; a missing or corrupt file is an error, not empty data.

    Returning "no tickets" for a broken file would let the assistant tell a
    customer that their ticket does not exist — the worst possible answer.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SupportError(
            f"Файл данных поддержки не найден: {path}. "
            f"Скопируйте support.example.json в data/support.json."
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupportError(f"Файл данных поддержки повреждён: {path} ({exc}).") from exc
    if not isinstance(data, dict):
        raise SupportError(f"Ожидался JSON-объект в {path}.")
    return data


def _user_from(item: dict) -> User:
    return User(
        id=str(item.get("id", "")),
        email=str(item.get("email", "")),
        name=str(item.get("name", "")),
        plan=str(item.get("plan", "free")),
        status=str(item.get("status", "active")),
        created_at=str(item.get("created_at", "")),
    )


def _ticket_from(item: dict) -> Ticket:
    messages = []
    for message in item.get("messages") or []:
        if not isinstance(message, dict):
            continue
        messages.append(
            TicketMessage(
                author=str(message.get("author", "")),
                text=str(message.get("text", "")),
                at=str(message.get("at", "")),
            )
        )
    return Ticket(
        id=str(item.get("id", "")),
        user_id=str(item.get("user_id", "")),
        subject=str(item.get("subject", "")),
        status=str(item.get("status", "")),
        priority=str(item.get("priority", "normal")),
        tags=[str(t) for t in (item.get("tags") or [])],
        created_at=str(item.get("created_at", "")),
        updated_at=str(item.get("updated_at", "")),
        messages=messages,
    )


def list_users(path: Path | None = None) -> list[User]:
    """Return every customer on the desk."""
    data = _read_db(path or default_db_path())
    return [_user_from(item) for item in data.get("users") or [] if isinstance(item, dict)]


def get_user(user_id: str, path: Path | None = None) -> User:
    """Return one customer by id.

    Raises :class:`NotFoundError` when there is no such customer: the assistant
    must hear that the id is wrong rather than reason about an empty record.
    """
    wanted = (user_id or "").strip()
    if not wanted:
        raise ValueError("Не указан id пользователя.")
    for user in list_users(path):
        if user.id == wanted:
            return user
    raise NotFoundError(f"Пользователь {wanted!r} не найден.")


def _all_tickets(path: Path | None = None) -> list[Ticket]:
    """Every ticket on the desk, newest first, with no cap.

    Lookups go through here: applying :data:`MAX_TICKETS` to a search would make
    an existing ticket unreachable by id and answer "не найден" about it.
    """
    data = _read_db(path or default_db_path())
    tickets = [
        _ticket_from(item)
        for item in data.get("tickets") or []
        if isinstance(item, dict)
    ]
    tickets.sort(key=lambda t: t.updated_at or t.created_at, reverse=True)
    return tickets


def page_tickets(
    user_id: str | None = None,
    status: str | None = None,
    path: Path | None = None,
    limit: int = MAX_TICKETS,
) -> tuple[list[Ticket], bool]:
    """Return ``(tickets, truncated)`` for the listing an assistant reads.

    *truncated* says whether *limit* actually cut something off, so the rendered
    text never claims a list is partial when it is not.
    """
    wanted_user = (user_id or "").strip()
    wanted_status = (status or "").strip().lower()
    tickets = _all_tickets(path)
    if wanted_user:
        tickets = [t for t in tickets if t.user_id == wanted_user]
    if wanted_status:
        tickets = [t for t in tickets if t.status.lower() == wanted_status]
    return tickets[:limit], len(tickets) > limit


def list_tickets(
    user_id: str | None = None,
    status: str | None = None,
    path: Path | None = None,
) -> list[Ticket]:
    """Return tickets, optionally filtered by customer and/or status.

    The most recently updated ticket comes first — a support agent asking "what
    is going on with me?" means the latest one. The result is capped at
    :data:`MAX_TICKETS`; use :func:`page_tickets` when you also need to know
    whether the cap cut anything.
    """
    return page_tickets(user_id=user_id, status=status, path=path)[0]


def get_ticket(ticket_id: str, path: Path | None = None) -> Ticket:
    """Return one ticket with its message history.

    Raises :class:`NotFoundError` when there is no such ticket. The search covers
    the whole desk, not the capped listing: an id that exists must resolve.
    """
    wanted = (ticket_id or "").strip()
    if not wanted:
        raise ValueError("Не указан id тикета.")
    for ticket in _all_tickets(path):
        if ticket.id == wanted:
            return ticket
    raise NotFoundError(f"Тикет {wanted!r} не найден.")


def search_tickets(query: str, path: Path | None = None) -> list[Ticket]:
    """Return tickets whose subject, tags or messages contain *query*.

    Case-insensitive substring search over subject, tags and every message, so
    a customer can be found by what they wrote rather than by an id. The query
    is escaped, so user input is never treated as a regular expression. Matches
    are returned newest first; the caller decides how many to render, since only
    the rendering needs a cap.
    """
    needle = (query or "").strip().lower()
    if not needle:
        raise ValueError("Пустой поисковый запрос.")
    pattern = re.compile(re.escape(needle))
    hits: list[Ticket] = []
    for ticket in _all_tickets(path):
        haystack = [ticket.subject.lower(), *[t.lower() for t in ticket.tags]]
        haystack += [m.text.lower() for m in ticket.messages]
        if any(pattern.search(part) for part in haystack):
            hits.append(ticket)
    return hits


def render_user(user: User) -> str:
    """Render one customer as text (an MCP tool's return value)."""
    return (
        f"Пользователь {user.id}: {user.name} <{user.email}>\n"
        f"Тариф: {user.plan}; статус: {user.status}; с нами с: {user.created_at}"
    )


def render_ticket(ticket: Ticket, user: User | None = None) -> str:
    """Render one ticket with its history as text.

    When *user* is given, the customer line is included too: a ticket id alone
    says nothing about who is waiting.
    """
    lines = [f"Тикет {ticket.id} — {ticket.subject}"]
    lines.append(
        f"Статус: {ticket.status}; приоритет: {ticket.priority}; "
        f"теги: {', '.join(ticket.tags) if ticket.tags else '—'}"
    )
    lines.append(
        f"Создан: {ticket.created_at or '—'}; обновлён: {ticket.updated_at or '—'}"
    )
    if user is not None:
        lines.append(f"Клиент: {user.id} ({user.name}, тариф {user.plan}, {user.status})")
    if ticket.messages:
        lines.append("Переписка:")
        for message in ticket.messages:
            stamp = f" ({message.at})" if message.at else ""
            lines.append(f"  [{message.author}]{stamp} {message.text}")
    else:
        lines.append("Переписка пуста.")
    return "\n".join(lines)


def render_tickets(
    tickets: list[Ticket], label: str = "тикетов", truncated: bool = False
) -> str:
    """Render a ticket list compactly (status, priority, subject).

    *truncated* must come from :func:`page_tickets`: the line saying the list is
    partial goes into the model's context, so it may only appear when something
    really was cut off.
    """
    if not tickets:
        return f"Не найдено ни одного {label}."
    lines = [f"Найдено {label}: {len(tickets)}"]
    for ticket in tickets:
        tags = f" [{', '.join(ticket.tags)}]" if ticket.tags else ""
        lines.append(
            f"- {ticket.id} ({ticket.status}/{ticket.priority}){tags}: {ticket.subject} "
            f"— обновлён {ticket.updated_at or '—'}"
        )
    if truncated:
        lines.append(f"(показаны первые {MAX_TICKETS}; всего больше)")
    return "\n".join(lines)