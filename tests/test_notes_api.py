"""Tests for the mock-CRM notes API (the backend behind the MCP server)."""

from __future__ import annotations

import json

import pytest

from llm_bot.notes_api import (
    Note,
    add_note,
    default_db_path,
    find_notes,
    list_notes,
    render_notes,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point NOTES_DB at a temp file for every test."""
    path = tmp_path / "notes.json"
    monkeypatch.setenv("NOTES_DB", str(path))
    return path


def test_default_db_path_env_override(db):
    assert default_db_path() == db


def test_add_note_assigns_sequential_ids(db):
    first = add_note("Купить хлеб", ["покупки"])
    second = add_note("Отпуск в Сочи", ["отпуск"])
    assert (first.id, second.id) == (1, 2)
    data = json.loads(db.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["text"] == "Купить хлеб"
    assert data[0]["tags"] == ["покупки"]


def test_add_note_normalizes_tags(db):
    note = add_note("text", ["A", " a ", "b"])
    assert note.tags == ["a", "b"]


def test_add_note_rejects_empty_text(db):
    with pytest.raises(ValueError):
        add_note("   ")


def test_list_notes_filters_by_tag(db):
    add_note("one", ["work"])
    add_note("two", ["home"])
    assert [n.text for n in list_notes()] == ["one", "two"]
    assert [n.text for n in list_notes("HOME")] == ["two"]
    assert list_notes("missing") == []


def test_find_notes_searches_text_and_tags(db):
    add_note("Отпуск в Сочи", ["отпуск"])
    add_note("Купить хлеб", ["покупки"])
    assert len(find_notes("сочи")) == 1
    assert len(find_notes("покупки")) == 1  # tag match
    assert find_notes("СОЧИ")[0].text == "Отпуск в Сочи"


def test_find_notes_rejects_empty_query(db):
    with pytest.raises(ValueError):
        find_notes("  ")


def test_find_notes_query_is_not_regex(db):
    add_note("a.b.c (x)", [])
    # '.' and '(' would be regex metacharacters; must match literally.
    assert len(find_notes("a.b.c (x)")) == 1
    assert find_notes("aXb") == []


def test_corrupt_db_is_treated_as_empty(db):
    db.write_text("not json", encoding="utf-8")
    assert list_notes() == []
    note = add_note("recovered")
    assert note.id == 1


def test_render_notes_empty_and_populated():
    assert render_notes([]) == "Заметок не найдено."
    text = render_notes([Note(id=3, text="hello", tags=["a", "b"])])
    assert text == "id=3 [a, b]: hello"
