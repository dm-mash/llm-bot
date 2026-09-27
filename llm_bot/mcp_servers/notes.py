"""MCP server exposing the mock-CRM notes API as tools.

Implements the "own MCP server" task: around a small API (a JSON-file CRM)
three tools are registered with typed input schemas and text results:

* ``add_note(text, tags)``   — append a note, returns its id;
* ``list_notes(tag)``        — all notes, optionally filtered by tag;
* ``find_notes(query)``      — substring search over text and tags.

The server speaks MCP over **stdio** (spawned by the client as a child
process). It must print ONLY protocol frames to stdout: any diagnostics go to
stderr. The notes database path comes from ``NOTES_DB`` (default
``data/notes.json``) — set it in the MCP server config.

Run from the project root for a smoke check (it will wait on stdin):
    python -m llm_bot.mcp_servers.notes
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from llm_bot.api import notes as notes_api

mcp = FastMCP("notes-crm")


@mcp.tool()
def add_note(text: str, tags: list[str] | None = None) -> str:
    """Добавить заметку в CRM. Возвращает подтверждение с id заметки.

    Идемпотентно: повторное добавление той же заметки (тот же текст и теги)
    НЕ создаёт дубликат — возвращается id уже существующей записи. Это делает
    безопасным повторное выполнение пакета после сетевых сбоев и ретраев LLM.
    """
    try:
        note = notes_api.add_note(text, tags)
        action = "Заметка сохранена"
    except notes_api.DuplicateNoteError as exc:
        note = exc.existing
        action = "Заметка уже существует, дубликат не создан"
    tags_part = f", теги: {', '.join(note.tags)}" if note.tags else ""
    return f"{action}: id={note.id}{tags_part}"


@mcp.tool()
def list_notes(tag: str | None = None) -> str:
    """Список заметок, при указании *tag* — отфильтрованный по тегу."""
    return notes_api.render_notes(notes_api.list_notes(tag))


@mcp.tool()
def find_notes(query: str) -> str:
    """Найти заметки по подстроке в тексте или тегах."""
    return notes_api.render_notes(notes_api.find_notes(query))


if __name__ == "__main__":
    # stdio transport is the FastMCP default; stdout is reserved for the
    # protocol, so keep every diagnostic message on stderr.
    mcp.run()
