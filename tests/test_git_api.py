"""Tests for the read-only git API backing the git MCP server.

Each test builds a throwaway repository in ``tmp_path`` with real ``git``
commands, so the helpers are exercised against the same binary the MCP server
shells out to — no mocking of subprocess.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from llm_bot.api import git as git_api


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "-c", "init.defaultBranch=main", "init")
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("two\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path


def test_current_branch(repo: Path):
    assert git_api.current_branch(repo) == "main"


def test_current_branch_on_detached_head_falls_back_to_hash(repo: Path):
    sha = _git(repo, "rev-parse", "--short", "HEAD").strip()
    _git(repo, "checkout", "--detach", "HEAD")
    assert git_api.current_branch(repo) == sha


def test_list_files_and_pattern(repo: Path):
    assert git_api.list_files(repo=repo) == ["a.txt", "sub/b.txt"]
    assert git_api.list_files("sub/*", repo=repo) == ["sub/b.txt"]
    assert git_api.list_files("*.md", repo=repo) == []


def test_diff_is_empty_on_clean_tree(repo: Path):
    assert git_api.diff(repo=repo).strip() == ""


def test_diff_reports_working_tree_change(repo: Path):
    (repo / "a.txt").write_text("one changed\n", encoding="utf-8")
    text = git_api.diff(repo=repo)
    assert "-one" in text
    assert "+one changed" in text


def test_diff_can_be_scoped_to_a_path(repo: Path):
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    (repo / "sub" / "b.txt").write_text("b\n", encoding="utf-8")
    text = git_api.diff(path="sub/b.txt", repo=repo)
    assert "b.txt" in text
    assert "a.txt" not in text


def test_changed_files_against_ref(repo: Path):
    first = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "new.txt").write_text("n\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add new")

    # Against the first commit only the new file differs.
    assert git_api.changed_files(ref=first, repo=repo) == [("A", "new.txt")]
    assert git_api.changed_files(ref=first, added_only=True, repo=repo) == [
        ("A", "new.txt")
    ]
    # Against HEAD the tree is clean.
    assert git_api.changed_files(ref="HEAD", repo=repo) == []


def test_changed_files_working_tree_includes_untracked(repo: Path):
    (repo / "untracked.txt").write_text("u\n", encoding="utf-8")
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")

    by_path = {path: status for status, path in git_api.changed_files(repo=repo)}
    assert by_path["untracked.txt"] == "??"
    assert by_path["a.txt"] == "M"
    assert git_api.changed_files(added_only=True, repo=repo) == [
        ("??", "untracked.txt")
    ]


def test_render_changes():
    assert "Нет изменений" in git_api.render_changes([])
    assert "новых файлов" in git_api.render_changes([], added_only=True)
    rendered = git_api.render_changes([("A", "x.py"), ("M", "y.py")])
    assert "новый" in rendered and "x.py" in rendered
    assert "изменён" in rendered and "y.py" in rendered


def test_non_repo_raises_git_error(tmp_path: Path):
    with pytest.raises(git_api.GitError):
        git_api.current_branch(tmp_path)


def test_render_files_reports_scope_and_empty():
    assert git_api.render_files([]) == "Файлы не найдены."
    assert "по маске *.md" in git_api.render_files([], "*.md")
    rendered = git_api.render_files(["a.txt", "sub/b.txt"])
    assert "Файлов: 2" in rendered
    assert "- sub/b.txt" in rendered


def test_render_diff_marks_truncation():
    assert "чистое" in git_api.render_diff("")
    assert "изменения" in git_api.render_diff("", staged=True)
    huge = "x" * (git_api.MAX_DIFF_CHARS + 10)
    rendered = git_api.render_diff(huge)
    assert "усечён" in rendered
    # The body is cut to the cap; only the marker is appended after it.
    assert rendered.count("x") == git_api.MAX_DIFF_CHARS


def test_render_diff_names_the_files_the_cut_hid():
    """A truncated diff lists only the files that sort before the cut.

    Real run: a 27k-char working-tree diff showed README.md and llm_bot/agent.py
    and the answer reported "two files changed" — of ten. The full list is
    appended after the marker so the model knows what it did not see.
    """
    huge = "x" * (git_api.MAX_DIFF_CHARS + 10)
    files = [("M", "README.md"), ("M", "llm_bot/agent.py"), ("??", "tasks/day31.md")]

    rendered = git_api.render_diff(huge, files=files)

    assert "усечён" in rendered
    assert "Файлов: 3" in rendered
    assert "tasks/day31.md" in rendered

    # Nothing appended when the diff fits: the list is not noise on every call.
    assert "tasks/day31.md" not in git_api.render_diff("small", files=files)
    # And a caller that passes no list keeps the old marker-only output.
    assert "Файлов:" not in git_api.render_diff(huge)
