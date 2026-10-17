"""MCP server exposing the support desk (users and tickets) as tools.

The support assistant must answer a customer's question *about their own case*,
so the desk has to be reachable as tools. Four read-only tools over
:mod:`llm_bot.api.support`:

* ``get_ticket(ticket_id)``   — one ticket with its message history;
* ``get_user(user_id)``       — one customer record;
* ``list_tickets(user_id, status)`` — the customer's tickets, newest first;
* ``search_tickets(query)``   — substring search over subject, tags and text.

There is deliberately no tool that changes anything: an assistant that answers
questions about a desk must not be able to alter it, and a demo data file stays
reproducible when a demo is run twice.

The server speaks MCP over **stdio** (spawned by the client as a child
process). It must print ONLY protocol frames to stdout: any diagnostics go to
stderr. The desk file comes from ``SUPPORT_DB`` (default ``data/support.json``).

Run from the project root for a smoke check (it will wait on stdin):
    python -m llm_bot.mcp_servers.support
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from llm_bot.api import support as support_api

mcp = FastMCP("support")


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Тикет с перепиской по его id, например "T-5001".

    Возвращает статус, приоритет, теги и всю переписку. Если нужен ответ по
    конкретному обращению клиента — сначала возьмите его отсюда, а не из
    общих объяснений: у клиента могут быть свои обстоятельства (смена пароля,
    потерянные резервные коды, неподтверждённая почта).
    """
    try:
        ticket = support_api.get_ticket(ticket_id)
    except support_api.SupportError as exc:
        return f"Не удалось получить тикет: {exc}"
    user = None
    try:
        user = support_api.get_user(ticket.user_id)
    except support_api.SupportError:
        user = None
    return support_api.render_ticket(ticket, user)


@mcp.tool()
def get_user(user_id: str) -> str:
    """Клиент по его id, например "U-1001": тариф, статус, дата регистрации."""
    try:
        user = support_api.get_user(user_id)
    except support_api.SupportError as exc:
        return f"Не удалось получить клиента: {exc}"
    return support_api.render_user(user)


@mcp.tool()
def list_tickets(user_id: str | None = None, status: str | None = None) -> str:
    """Список тикетов, новые сверху; можно отфильтровать по клиенту и статусу.

    Без фильтров вернёт все тикеты стенда. С *user_id* — только тикеты этого
    клиента, что и нужно, чтобы понять историю обращения.
    """
    try:
        tickets, truncated = support_api.page_tickets(user_id=user_id, status=status)
    except support_api.SupportError as exc:
        return f"Не удалось получить тикеты: {exc}"
    return support_api.render_tickets(tickets, truncated=truncated)


@mcp.tool()
def search_tickets(query: str) -> str:
    """Найти тикеты по подстроке в теме, тегах или тексте переписки.

    Поиск без учёта регистра, например "авторизация" или "сброс пароля".
    """
    try:
        matches = support_api.search_tickets(query)
    except (support_api.SupportError, ValueError) as exc:
        return f"Не удалось найти тикеты: {exc}"
    truncated = len(matches) > support_api.MAX_TICKETS
    return support_api.render_tickets(
        matches[: support_api.MAX_TICKETS],
        label="подходящих тикетов",
        truncated=truncated,
    )


if __name__ == "__main__":
    # stdio transport is the FastMCP default; stdout is reserved for the
    # protocol, so keep every diagnostic message on stderr.
    mcp.run()