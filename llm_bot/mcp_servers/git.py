"""MCP server exposing the project's git context as tools.

The developer-assistant task needs the model to know WHERE it is working, so
three read-only tools are registered over the :mod:`llm_bot.api.git` backend:

* ``current_branch()``            — the active branch (or the commit hash on a
  detached HEAD);
* ``list_files(pattern)``         — tracked files, optionally filtered;
* ``diff(ref, staged, path)``     — a unified diff of the working tree.

The server speaks MCP over **stdio** (spawned by the client as a child
process). It must print ONLY protocol frames to stdout: any diagnostics go to
stderr. The repository path comes from ``GIT_REPO`` (default: the working
directory the client spawns the server in, i.e. the project root).

Run from the project root for a smoke check (it will wait on stdin):
    python -m llm_bot.mcp_servers.git
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from llm_bot.api import git as git_api

mcp = FastMCP("git")


@mcp.tool()
def current_branch() -> str:
    """Текущая git-ветка проекта (на detached HEAD — короткий хеш коммита)."""
    try:
        return git_api.current_branch()
    except git_api.GitError as exc:
        return f"Не удалось определить ветку: {exc}"


@mcp.tool()
def list_files(pattern: str | None = None) -> str:
    """Список отслеживаемых файлов проекта.

    При указании *pattern* — только совпадающие с маской (fnmatch), например
    ``llm_bot/*.py`` или ``docs/*``.
    """
    try:
        files = git_api.list_files(pattern)
    except git_api.GitError as exc:
        return f"Не удалось получить список файлов: {exc}"
    return git_api.render_files(files, pattern)


@mcp.tool()
def changed_files(
    ref: str | None = None,
    added_only: bool = False,
) -> str:
    """Список новых/изменённых/удалённых файлов — полный и компактный.

    Для вопросов «какие файлы новые / изменились» используйте ЭТОТ инструмент,
    а не diff (unified diff усекается и может не показать часть файлов).
    *ref* — ветка/коммит для сравнения (например, ref=main покажет всё новое
    относительно main); без *ref* — текущее состояние рабочего дерева, включая
    неотслеживаемые новые файлы. *added_only=True* — только новые файлы.
    """
    try:
        entries = git_api.changed_files(ref=ref, added_only=added_only)
    except git_api.GitError as exc:
        return f"Не удалось получить список изменений: {exc}"
    return git_api.render_changes(entries, ref=ref, added_only=added_only)


@mcp.tool()
def diff(ref: str | None = None, staged: bool = False, path: str | None = None) -> str:
    """Унифицированный diff рабочего дерева (может быть усечён).

    *staged* — сравнить индекс с HEAD; *ref* — сравнить с веткой/коммитом;
    *path* — ограничить одним путём. По умолчанию — незакоммиченные правки.
    Если diff усечён, после пометки об усечении приводится полный список
    затронутых файлов — самой правки в теле ответа может не быть. Чтобы просто
    перечислить изменённые/новые файлы, используйте changed_files.
    """
    try:
        text = git_api.diff(ref=ref, staged=staged, path=path)
    except git_api.GitError as exc:
        return f"Не удалось получить diff: {exc}"
    # The file list is fetched separately: a failure here (a huge status, a
    # permission edge) must not turn a good diff into an error message. It is
    # only used when the diff is truncated, to name the files the cut hid.
    try:
        files = git_api.changed_files(ref=ref, path=path)
    except git_api.GitError:
        files = None
    return git_api.render_diff(text, ref=ref, staged=staged, files=files)


if __name__ == "__main__":
    # stdio transport is the FastMCP default; stdout is reserved for the
    # protocol, so keep every diagnostic message on stderr.
    mcp.run()
