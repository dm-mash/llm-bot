"""Read-only Git API — the project context wrapped by the git MCP server.

The developer assistant reaches the repository through MCP tools, never by
importing this module directly: the MCP server exposes these functions, so the
agent can learn the current branch, the file list and the working-tree diff.

Everything here is **read-only** and shells out to the ``git`` binary with an
explicit ``cwd``. A non-zero exit is surfaced as :class:`GitError` carrying
git's own stderr, so the model sees *why* a command failed instead of a generic
message.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

# Environment variable holding the repository path (default: current dir).
_ENV_GIT_REPO = "GIT_REPO"
DEFAULT_REPO = "."

# Tool results land in the model's context, so a large diff would crowd out the
# dialog. Both tools truncate with an explicit marker rather than silently
# slicing, so the model knows the output is partial.
MAX_DIFF_CHARS = 6000
MAX_FILES = 300


class GitError(RuntimeError):
    """Raised when a git command exits non-zero or git is unavailable."""


def default_repo() -> str:
    """Return the repository path (``GIT_REPO`` or the current directory)."""
    return os.getenv(_ENV_GIT_REPO) or DEFAULT_REPO


def _run_git(args: list[str], repo: str | Path) -> str:
    """Run ``git <args>`` in *repo* and return stdout, raising on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # git is not installed
        raise GitError("git не найден в PATH.") from exc
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise GitError(
            message or f"git {' '.join(args)} завершился с кодом "
            f"{proc.returncode}."
        )
    return proc.stdout


def current_branch(repo: str | Path | None = None) -> str:
    """Return the active branch name.

    On a detached HEAD ``git rev-parse --abbrev-ref HEAD`` answers ``HEAD``,
    which says nothing useful; the short commit hash is returned instead.
    """
    target = repo if repo is not None else default_repo()
    name = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], target).strip()
    if name and name != "HEAD":
        return name
    return _run_git(["rev-parse", "--short", "HEAD"], target).strip()


def list_files(
    pattern: str | None = None, repo: str | Path | None = None
) -> list[str]:
    """List tracked files, optionally filtered by an fnmatch *pattern*.

    ``git ls-files`` only reports files known to git — the right notion of "the
    project's files" for a structure question (untracked scratch files would be
    noise). The result is capped at :data:`MAX_FILES`.
    """
    target = repo if repo is not None else default_repo()
    files = [
        line.strip()
        for line in _run_git(["ls-files"], target).splitlines()
        if line.strip()
    ]
    if pattern:
        files = [f for f in files if fnmatch.fnmatch(f, pattern)]
    return files[:MAX_FILES]


def changed_files(
    ref: str | None = None,
    added_only: bool = False,
    path: str | None = None,
    repo: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Return ``(status, path)`` pairs for changed files.

    With *ref* (a branch/commit) it compares the working tree against it
    (``git diff --name-status <ref>``) — the way to answer "what did this
    branch change relative to main". Without *ref* it reports the current
    working-tree status **including untracked files** (``git status
    --porcelain --untracked-files=all``), so brand-new uncommitted files show
    up too. *added_only* keeps additions (``A``/``??``); *path* limits the
    report.

    This is deliberately separate from :func:`diff`: the compact status list
    stays complete within the context budget, while a unified diff is
    truncated and can hide files that sort after the cut.
    """
    target = repo if repo is not None else default_repo()
    if ref:
        args = ["diff", "--name-status"]
        if added_only:
            args.append("--diff-filter=A")
        args.append(ref)
        if path:
            args += ["--", path]
        entries: list[tuple[str, str]] = []
        for line in _run_git(args, target).splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                entries.append((parts[0], parts[-1]))
        return entries
    args = ["status", "--porcelain", "--untracked-files=all"]
    if path:
        args += ["--", path]
    entries = []
    for line in _run_git(args, target).splitlines():
        if len(line) < 4:
            continue
        status = line[:2].strip() or "?"
        if added_only and status not in ("A", "??"):
            continue
        entries.append((status, line[3:]))
    return entries


def diff(
    ref: str | None = None,
    staged: bool = False,
    path: str | None = None,
    repo: str | Path | None = None,
) -> str:
    """Return a unified diff (working tree by default).

    *staged* compares the index against HEAD (``--staged``); *ref* compares
    against an arbitrary commit/branch; *path* limits the diff to one path.
    """
    target = repo if repo is not None else default_repo()
    args = ["diff"]
    if staged:
        args.append("--staged")
    if ref:
        args.append(ref)
    if path:
        args += ["--", path]
    return _run_git(args, target)


def render_files(files: list[str], pattern: str | None = None) -> str:
    """Render a file list as the tool's text result."""
    if not files:
        scope = f" по маске {pattern}" if pattern else ""
        return f"Файлы{scope} не найдены."
    lines = [f"Файлов: {len(files)}"]
    lines += [f"- {f}" for f in files]
    if len(files) == MAX_FILES:
        lines.append(f"(показаны первые {MAX_FILES})")
    return "\n".join(lines)


_STATUS_LABELS = {
    "A": "новый",
    "?": "неотслеживаемый",
    "??": "неотслеживаемый",
    "M": "изменён",
    "D": "удалён",
    "R": "переименован",
    "C": "скопирован",
}


def render_changes(
    entries: list[tuple[str, str]],
    ref: str | None = None,
    added_only: bool = False,
) -> str:
    """Render a changed-file list compactly (status letter + label + path)."""
    if not entries:
        scope = " новых файлов" if added_only else " изменений"
        against = f" относительно {ref}" if ref else ""
        return f"Нет{scope}{against}."
    lines = [f"Файлов: {len(entries)}"]
    for status, path in entries:
        label = _STATUS_LABELS.get(status, status)
        lines.append(f"- {status} ({label}): {path}")
    return "\n".join(lines)


def render_diff(
    text: str,
    ref: str | None = None,
    staged: bool = False,
    files: list[tuple[str, str]] | None = None,
) -> str:
    """Render a diff as the tool's text result (truncated when huge).

    Truncation alone hides files that sort after the cut: the reader sees two
    files and has no way to know ten changed. So when the body is cut and the
    caller supplies the complete ``changed_files`` list, that list is appended
    after the marker — the model is told what it did *not* see.
    """
    if not text.strip():
        what = "проиндексированные изменения" if staged else "изменения"
        against = f" относительно {ref}" if ref else ""
        return f"Нет {what}{against} — рабочее дерево чистое."
    if len(text) > MAX_DIFF_CHARS:
        result = (
            text[:MAX_DIFF_CHARS]
            + f"\n... (diff усечён до {MAX_DIFF_CHARS} символов)"
        )
        if files:
            result += (
                "\n\nПолный список затронутых файлов (сам diff показан не весь):\n"
                + render_changes(files, ref=ref)
            )
        return result
    return text
