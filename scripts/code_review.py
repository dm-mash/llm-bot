#!/usr/bin/env python3
"""Review a pull request with the assistant, one changed file at a time."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMENT_MARKER = "<!-- llm-bot-code-review -->"
DEFAULT_AGENT = "reviewer"
DEFAULT_INDEX = "data/rev_emb/index_structure.json"
REVIEWABLE_EXTENSIONS = (".py", ".md")
DEFAULT_MAX_FILES = 30
DEFAULT_DIFF_TOKENS = 3000
DEFAULT_RAG_TOKENS = 1500
DEFAULT_CONTEXT_LINES = 0

# A finding line is a bullet whose prefix names the rubric. The emphasis is
# optional on purpose: models happily answer "- **БАГ:** ..." instead of
# "- БАГ: ...", and a parser that only accepts the bare form silently empties
# the whole report while the findings sit unread in the appendix.
BULLET_RE = re.compile(
    r"^\s*[-*]\s*\**\s*(?P<kind>БАГ|АРХИТЕКТУРА|РЕКОМЕНДАЦИЯ)"
    r"\s*\**\s*[:—-]\s*(?P<text>.+)$"
)

RENAME_RE = re.compile(r"^R[0-9]+\t(.*)\t(.*)$")
COPY_RE = re.compile(r"^C[0-9]+\t(.*)\t(.*)$")
STATUS_RE = re.compile(r"^([ACDMRTUXB?]+)\t(.*)$")


@dataclass(frozen=True)
class PRContext:
    number: int
    base_sha: str
    head_sha: str
    base_ref: str
    head_ref: str
    repo: str
    fork: bool
    action: str = ""


@dataclass
class ChangedFile:
    status: str
    path: str
    old_path: str = ""
    binary: bool = False
    diff: str = ""


@dataclass
class FileReview:
    file: ChangedFile
    answer: str = ""
    error: str = ""


def parse_event(path: Path) -> PRContext:
    data = json.loads(path.read_text(encoding="utf-8"))
    pr = data.get("pull_request") or {}
    base = pr.get("base") or {}
    head = pr.get("head") or {}
    head_repo = head.get("repo") or {}
    return PRContext(
        number=int(pr.get("number") or data.get("number") or 0),
        base_sha=str(base.get("sha", "")),
        head_sha=str(head.get("sha", "")),
        base_ref=str(base.get("ref", "")),
        head_ref=str(head.get("ref", "")),
        repo=str(data.get("repository", {}).get("full_name", "")),
        fork=bool(head_repo.get("fork", False)),
        action=str(data.get("action", "")),
    )


def git(*args: str, cwd: Path = ROOT) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return p.stdout


def parse_name_status(text: str) -> list[ChangedFile]:
    out: list[ChangedFile] = []
    for line in text.rstrip().splitlines():
        line = line.strip("\n\r")
        if not line:
            continue
        m = RENAME_RE.match(line)
        if m:
            out.append(ChangedFile(status="R", path=m.group(2).strip(), old_path=m.group(1).strip()))
            continue
        m = COPY_RE.match(line)
        if m:
            out.append(ChangedFile(status="C", path=m.group(2).strip(), old_path=m.group(1).strip()))
            continue
        m = STATUS_RE.match(line)
        if m:
            out.append(ChangedFile(status=m.group(1).strip(), path=m.group(2).strip()))
    return out


def parse_numstat(text: str) -> dict[str, bool]:
    res: dict[str, bool] = {}
    for line in text.rstrip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        add, rem, path = parts[0], parts[1], "\t".join(parts[2:])
        if add == "-" and rem == "-":
            res[path.strip()] = True
    return res


def select_reviewable(files: list[ChangedFile], exts: tuple[str, ...], max_files: int = DEFAULT_MAX_FILES) -> list[ChangedFile]:
    sel: list[ChangedFile] = []
    for f in files:
        if f.status.upper() == "D":
            continue
        if f.binary:
            continue
        p = f.path
        if p.startswith("data/"):
            continue
        if any(p.endswith(e) for e in exts):
            sel.append(f)
        if len(sel) >= max_files:
            break
    return sel


def mark_binary(files: list[ChangedFile], binmap: dict[str, bool]) -> None:
    for f in files:
        if binmap.get(f.path) or binmap.get(f.old_path):
            f.binary = True


def truncate_diff(diff: str, budget_tokens: int) -> tuple[str, bool]:
    if budget_tokens <= 0:
        return diff, False
    tokens = diff.count("\n") * 1 + len(diff) // 4
    if tokens <= budget_tokens:
        return diff, False
    lines = diff.splitlines(True)
    total = 0
    out = []
    for i, line in enumerate(lines):
        t = 1 + len(line) // 4
        if total + t > budget_tokens:
            return "".join(out), True
        out.append(line)
        total += t
    return "".join(out), False


def changed_symbols(diff: str) -> list[str]:
    syms: list[str] = []
    for line in diff.splitlines():
        m = re.match(r"^\+\+\+|^---|^@@", line)
        if m:
            continue
        if line.startswith("+"):
            mm = re.match(r"^\+\s*(def|class|async def)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if mm:
                syms.append(f"{mm.group(1)} {mm.group(2)}")
    return syms[:5]


def build_prompt(changed: ChangedFile, budget: int) -> str:
    diff, truncated = truncate_diff(changed.diff, budget)
    symbols = changed_symbols(changed.diff)
    head = (
        f"Файл: {changed.path} ({changed.status})\n"
        f"Изменённые символы: {', '.join(symbols) or '—'}\n"
    )
    note = "\n(дифф обрезан по бюджету — конец файла не показан)\n" if truncated else ""
    return (
        f"{head}\n"
        "Ниже diff одного файла pull request. Проверь только его.\n"
        f"{note}\n"
        "```diff\n" + diff + "\n```\n"
    )


def _safe_url(url: str) -> str:
    """Return *url* with any embedded credentials replaced, for error messages."""
    start = url.find("://")
    if start == -1:
        return url
    at = url.find("@", start + 3)
    if at == -1:
        return url
    return url[: start + 3] + "***@" + url[at + 1 :]


def _review_model(agent: str, agent_store=None, model_store=None):
    """Return ``(name, ModelConfig)`` for *agent*, raising ValueError if absent."""
    from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

    agent_store = agent_store if agent_store is not None else YamlAgentStore()
    model_store = model_store if model_store is not None else YamlModelStore()
    try:
        name = agent_store.get(agent).model
    except Exception as exc:
        raise ValueError(f"агент {agent!r} не найден в data/agents.yaml ({exc})")
    try:
        return name, model_store.get(name)
    except Exception as exc:
        raise ValueError(f"модель {name!r} не найдена в data/models.yaml ({exc})")


def check_model_config(agent: str, *, agent_store=None, model_store=None) -> str | None:
    """Return what is wrong with the reviewer's model config, or ``None``.

    An unset ``${REVIEW_BASE_URL}`` resolves to an empty string, and the client
    then fails deep inside httpx with "Request URL is missing an 'http://' or
    'https://' protocol" — which never mentions the missing variable. Checking
    the pair up front names the piece that is actually missing.
    """
    try:
        name, model = _review_model(agent, agent_store, model_store)
    except ValueError as exc:
        return str(exc)
    raw = model.base_url or ""
    base = raw.strip()
    if not base:
        return (
            f"у модели {name!r} пустой base_url. Задайте REVIEW_BASE_URL "
            f"(Secrets или Variables в настройках репозитория) либо укажите "
            f"адрес прямо в models.yaml."
        )
    # Everything past "?" or "#" leaves the path, so the endpoint the client
    # appends disappears into the query string and the provider reports a path
    # that looks unrelated — e.g. Groq answering "Unknown request URL: POST
    # /openai/v1/chat/api" for a base_url that carried a stray "?".
    for sep, label in (("?", "query string"), ("#", "fragment")):
        if sep in base:
            return (
                f"base_url модели {name!r} содержит {sep!r} ({_safe_url(base)!r}), "
                f"то есть {label}: адрес должен быть только провайдером и путём, "
                f"без ? и #. Ожидается https://api.groq.com/openai/v1."
            )
    if base != raw:
        return (
            f"base_url модели {name!r} содержит пробелы или перевод строки по "
            f"краям ({_safe_url(raw.strip())!r}) — это ломает собранный URL. "
            f"Уберите лишние пробелы."
        )
    if not base.startswith(("http://", "https://")):
        return (
            f"base_url модели {name!r} — {_safe_url(base)!r}: он должен "
            f"начинаться с http:// или https://, иначе клиент не знает, куда "
            f"идти. Ожидается адрес провайдера до /chat/completions, например "
            f"https://api.groq.com/openai/v1."
        )
    # The client appends the endpoint itself, so a full endpoint pasted into the
    # variable doubles it and the provider answers 404 "Unknown request URL".
    if "/chat/completions" in base:
        return (
            f"base_url модели {name!r} уже содержит /chat/completions "
            f"({_safe_url(base)!r}), а клиент дописывает его сам. Укажите "
            f"адрес без него: https://api.groq.com/openai/v1."
        )
    if not model.model:
        return f"у модели {name!r} не задано имя модели (поле model)."
    return None


def describe_model(agent: str, *, agent_store=None, model_store=None) -> str:
    """Return a one-line description of the endpoint the review will call.

    Printed before the first model call so a wrong ``base_url`` is visible in
    the workflow log instead of only as a provider error further down.
    """
    try:
        name, model = _review_model(agent, agent_store, model_store)
    except ValueError as exc:
        return str(exc)
    return f"{name!r} → {_safe_url(model.base_url or '')} (model {model.model!r})"


class _RequestPrinter:
    """Detail listener that logs every request URL and response status.

    The configured ``base_url`` is not proof of the URL actually requested: the
    client appends the endpoint itself, and only the request the provider
    received tells the truth. In CI that line in the log is the difference
    between "the setting is wrong" and "something between them rewrote it".
    """

    def on_request(self, details) -> None:
        print(f"[review] POST {details.url} (model {details.model})", file=sys.stderr)

    def on_response(self, details) -> None:
        print(f"[review] ← {details.status_code}", file=sys.stderr)


def run_reviews(ctx: PRContext, files: list[ChangedFile], args: argparse.Namespace) -> list[FileReview]:
    from llm_bot.factory import make_session
    from llm_bot.rag import Retriever
    from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

    model_store = YamlModelStore()
    agent_store = YamlAgentStore()
    printer = _RequestPrinter()

    retriever = None
    if args.rag_index:
        try:
            retriever = Retriever(args.rag_index, cite=False)
        except Exception:
            retriever = None

    class _NullSessionStore:
        def load(self, session_id):
            return []

        def save(self, session_id, history):
            pass

        def list(self):
            return []

    store = _NullSessionStore()
    reviews: list[FileReview] = []
    for changed in files:
        try:
            session = make_session(
                f"review-{ctx.number}-{abs(hash(changed.path))}",
                args.agent,
                model_store=model_store,
                agent_store=agent_store,
                session_store=store,
                retriever=retriever,
                detail_listener=printer,
                rag_cite=False,
                rag_strict=False,
                rag_max_context_tokens=args.rag_tokens,
                invariants=False,
            )
            q = build_prompt(changed, args.diff_tokens)
            reviews.append(FileReview(changed, answer=session.chat(q)))
        except Exception as exc:
            reviews.append(FileReview(changed, error=f"{type(exc).__name__}: {exc}"))
    return reviews


def classify(answer: str) -> dict[str, list[str]]:
    buckets = {"БАГ": [], "АРХИТЕКТУРА": [], "РЕКОМЕНДАЦИЯ": []}
    for line in answer.splitlines():
        m = BULLET_RE.match(line)
        if m:
            # The emphasis that wrapped the prefix usually wraps the finding too.
            buckets[m["kind"]].append(m["text"].strip().strip("*").strip())
    return buckets


def _says_nothing(answer: str) -> bool:
    """Whether the reply is the model's "nothing to comment" answer."""
    text = answer.strip().strip('"').strip("*").strip()
    return not text or "нечего комментировать" in text.lower()


def render_report(ctx: PRContext, reviews: list[FileReview], skipped: list[str]) -> str:
    lines: list[str] = []
    lines.append(COMMENT_MARKER)
    lines.append("## 🤖 AI code review")
    lines.append("")
    lines.append(f"Reviewed {len(reviews)} changed files. Skipped: {len(skipped)}")
    if skipped:
        lines.append("Skipped: " + ", ".join(skipped))
    lines.append("")

    buckets: dict[str, list[str]] = {"БАГ": [], "АРХИТЕКТУРА": [], "РЕКОМЕНДАЦИЯ": []}
    unparsed: list[str] = []
    for r in reviews:
        if r.error:
            buckets.setdefault("РЕКОМЕНДАЦИЯ", []).append(f"**{r.file.path}** — ошибка при ревью: {r.error}")
            continue
        c = classify(r.answer)
        found = sum(len(v) for v in c.values())
        for k in buckets:
            for t in c.get(k, []):
                if ":" in t or "(" in t:
                    buckets[k].append(t)
                else:
                    buckets[k].append(f"**{r.file.path}** — {t}")
        # A model that ignored the prefixes still has something to say. Filing it
        # under its own heading beats an empty report that claims there is
        # nothing to comment on.
        if not found and not _says_nothing(r.answer):
            unparsed.append(f"**{r.file.path}**\n\n" + r.answer.strip())

    for title in ["БАГ", "АРХИТЕКТУРА", "РЕКОМЕНДАЦИЯ"]:
        lines.append(f"### {title}")
        if not buckets[title]:
            lines.append("- Нечего комментировать")
        else:
            for t in buckets[title]:
                lines.append(f"- {t}")
        lines.append("")

    if unparsed:
        lines.append("### Прочее (ответ без разбора по рубрикам)")
        lines.append("")
        for t in unparsed:
            lines.append(t)
            lines.append("")

    lines.append("<details><summary>Полные ответы по файлам</summary>")
    lines.append("")
    for r in reviews:
        lines.append(f"#### `{r.file.path}` ({r.file.status})")
        if r.error:
            lines.append(f"Ошибка: {r.error}")
        else:
            lines.append(r.answer or "Нечего комментировать")
        lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("<sub>Generated by scripts/code_review.py</sub>")
    return "\n".join(lines)


def find_marker_comment(comments: list[dict]) -> str | None:
    for c in comments:
        body = c.get("body", "") or ""
        if COMMENT_MARKER in body:
            return str(c.get("id"))
    return None


def post_comment(ctx: PRContext, body: str, token: str, *, use_gh: bool = False) -> str | None:
    if not token or not ctx.repo or not ctx.number:
        return None
    api = f"https://api.github.com/repos/{ctx.repo}/issues/{ctx.number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "llm-bot-code-review",
    }
    if use_gh:
        try:
            import shutil
            if shutil.which("gh"):
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                    f.write(body)
                    p = f.name
                try:
                    subprocess.run(["gh", "pr", "comment", str(ctx.number), "--body-file", p], check=False)
                finally:
                    Path(p).unlink(missing_ok=True)
                return api
        except Exception:
            pass

    try:
        req = request.Request(api, headers=headers)
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cid = find_marker_comment(data)
        if cid:
            url = f"https://api.github.com/repos/{ctx.repo}/issues/comments/{cid}"
            req = request.Request(url, data=body.encode("utf-8"), headers=headers, method="PATCH")
            with request.urlopen(req, timeout=30) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            return str(j.get("html_url"))
        else:
            req = request.Request(api, data=body.encode("utf-8"), headers=headers, method="POST")
            with request.urlopen(req, timeout=30) as resp:
                j = json.loads(resp.read().decode("utf-8"))
            return str(j.get("html_url"))
    except (URLError, HTTPError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PR code review")
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--rag-index", default=DEFAULT_INDEX)
    p.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    p.add_argument("--diff-tokens", type=int, default=DEFAULT_DIFF_TOKENS)
    p.add_argument("--rag-tokens", type=int, default=DEFAULT_RAG_TOKENS)
    p.add_argument("--context-lines", type=int, default=DEFAULT_CONTEXT_LINES)
    p.add_argument("--out", default=None)
    p.add_argument("--pr", type=int, default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--head", default=None)
    p.add_argument("--repo", default=None)
    p.add_argument("--no-comment", action="store_true")
    p.add_argument("--use-gh", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    ctx = PRContext(0, "", "", "", "", "", False, "")
    ev = os.getenv("GITHUB_EVENT_PATH")
    if ev and Path(ev).exists():
        try:
            ctx = parse_event(Path(ev))
        except Exception:
            ctx = PRContext(0, "", "", "", "", "", False, "")
    if args.pr:
        ctx = PRContext(args.pr, ctx.base_sha, ctx.head_sha, args.base or ctx.base_ref, args.head or ctx.head_ref, args.repo or ctx.repo, ctx.fork, ctx.action)

    if ctx.fork:
        out = Path(args.out) if args.out else Path("data/review/fork.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f"{COMMENT_MARKER}\n## AI code review skipped\n\nFork PR — secrets unavailable."
        out.write_text(text, encoding="utf-8")
        sm = os.getenv("GITHUB_STEP_SUMMARY")
        if sm:
            with open(sm, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        return 0

    try:
        span = f"{ctx.base_sha}...{ctx.head_sha}" if ctx.base_sha and ctx.head_sha else ""
        if not span:
            span = "HEAD~1...HEAD"
        status = git("diff", "--name-status", "--find-renames", span)
        numstat = git("diff", "--numstat", span)
        files = parse_name_status(status)
        binmap = parse_numstat(numstat)
        mark_binary(files, binmap)
        reviewable = select_reviewable(files, REVIEWABLE_EXTENSIONS, args.max_files)
    except Exception as exc:
        return 1

    skipped = [f.path for f in files if f not in reviewable and f.status.upper() != "D" and not any(f.path.endswith(e) for e in REVIEWABLE_EXTENSIONS)]

    if not reviewable:
        out = Path(args.out) if args.out else Path(f"data/review/pr-{ctx.number}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f"{COMMENT_MARKER}\n## AI code review\n\nNo reviewable text changes found."
        out.write_text(text, encoding="utf-8")
        sm = os.getenv("GITHUB_STEP_SUMMARY")
        if sm:
            with open(sm, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        return 0

    for ch in reviewable:
        try:
            ch.diff = git("diff", "--no-color", f"--unified={args.context_lines}", span, "--", ch.path)
        except Exception:
            ch.diff = ""

    if args.dry_run:
        out = Path(args.out) if args.out else Path(f"data/review/pr-{ctx.number}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(build_prompt(ch, args.diff_tokens) for ch in reviewable)
        out.write_text(body, encoding="utf-8")
        return 0

    problem = check_model_config(args.agent)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    # Echo the endpoint actually in use: in CI it is the only place a wrong
    # REVIEW_BASE_URL shows up before the provider rejects it.
    print(f"[review] model {describe_model(args.agent)}", file=sys.stderr)

    reviews = run_reviews(ctx, reviewable, args)
    report = render_report(ctx, reviews, skipped)
    out = Path(args.out) if args.out else Path(f"data/review/pr-{ctx.number}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    sm = os.getenv("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    if not args.no_comment:
        post_comment(ctx, report, os.getenv("GITHUB_TOKEN", ""), use_gh=args.use_gh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
