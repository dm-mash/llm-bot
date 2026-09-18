#!/usr/bin/env python3
"""Compare personalization profiles on the SAME request.

Sends one identical question to the same agent through several profiles from
``data/profiles.yaml`` (e.g. ``developer`` / ``student`` / ``child``) and dumps
a side-by-side report to ``results/compare_profiles.md`` (+ a JSON twin for
downstream tooling).

This is the verification script for the personalization feature: it shows that
the SAME agent + SAME question yields observably different answers per profile
(style, format, depth, constraints), because the profile is composed into the
agent config (system prompt + generation overrides) before any request.

Examples:
    python scripts/compare_profiles.py
    python scripts/compare_profiles.py --profiles developer,student,child
    python scripts/compare_profiles.py --agent assistant --question "Что такое рекурсия?"
    python scripts/compare_profiles.py --out results/compare_profiles.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the project's ``llm_bot`` package importable regardless of the current
# working directory (e.g. when running ``python scripts/compare_profiles.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_bot.client import LLMError
from llm_bot.factory import make_session
from llm_bot.json_session_store import JsonSessionStore
from llm_bot.profiles import profile_prompt_block
from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore, YamlProfileStore

DEFAULT_PROFILES = ["developer", "student", "child"]
DEFAULT_QUESTION = (
    "Объясни, что такое рекурсия в программировании. Приведи пример."
)
DEFAULT_OUT_MD = "results/compare_profiles.md"
DEFAULT_OUT_JSON = "results/compare_profiles.json"


def _ask_once(
    agent_name: str,
    question: str,
    *,
    profile: str | None,
    session_prefix: str,
) -> dict:
    """Run ONE one-shot session for *profile* and return its answer + metadata.

    A fresh session id per profile keeps histories independent and memory
    untouched (the profile path never writes to memory anyway).
    """
    session = make_session(
        f"{session_prefix}-{profile or 'none'}",
        agent_name,
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(),
        profile=profile,
    )
    started = time.perf_counter()
    try:
        reply = session.chat(question)
        error = None
    except LLMError as exc:
        reply = ""
        error = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000

    usage = session.last_usage
    return {
        "profile": profile or "(none)",
        "reply": reply,
        "error": error,
        "elapsed_ms": round(elapsed_ms),
        "reply_chars": len(reply),
        "reply_words": len(reply.split()) if reply else 0,
        "context_tokens": usage.context_tokens if usage else None,
        "reply_tokens": usage.reply_tokens if usage else None,
        "system_prompt": session.agent.config.system_prompt,
        "temperature": session.agent.config.temperature,
        "max_response_words": session.agent.config.max_response_words,
    }


def _render_markdown(
    question: str,
    agent_name: str,
    results: list[dict],
    profile_store: YamlProfileStore,
) -> str:
    lines: list[str] = [
        "# Сравнение профилей персонализации",
        "",
        f"Агент: `{agent_name}` — один и тот же; меняется только профиль.",
        "",
        f"Вопрос: **{question}**",
        "",
    ]
    for res in results:
        profile = profile_store.get(res["profile"]) if res["profile"] != "(none)" else None
        lines.append(f"## Профиль: {res['profile']}")
        if profile is not None:
            block = profile_prompt_block(profile)
            lines.append("")
            lines.append("Директивы профиля (добавлены к system prompt):")
            lines.append("")
            lines.append("```text")
            lines.append(block if block else "(пустой профиль)")
            lines.append("```")
            overrides = []
            if res["temperature"] is not None:
                overrides.append(f"temperature={res['temperature']}")
            if res["max_response_words"] is not None:
                overrides.append(f"max_response_words={res['max_response_words']}")
            if overrides:
                lines.append("")
                lines.append(f"Переопределения генерации: {', '.join(overrides)}")
        lines.append("")
        lines.append(
            f"Метрики: {res['elapsed_ms']} мс, "
            f"{res['reply_words']} слов / {res['reply_chars']} символов"
            + (f", reply_tokens={res['reply_tokens']}" if res["reply_tokens"] else "")
        )
        lines.append("")
        if res["error"]:
            lines.append(f"**Ошибка:** {res['error']}")
        else:
            lines.append("**Ответ:**")
            lines.append("")
            lines.append("> " + res["reply"].replace("\n", "\n> "))
        lines.append("")
    lines.append("## Вывод")
    lines.append("")
    words = {r["profile"]: r["reply_words"] for r in results if not r["error"]}
    if words:
        shortest = min(words, key=lambda k: words[k])
        longest = max(words, key=lambda k: words[k])
        lines.append(
            f"Один и тот же вопрос через разные профили даёт разные ответы: "
            f"самый лаконичный — `{shortest}` ({words[shortest]} слов), "
            f"самый развёрнутый — `{longest}` ({words[longest]} слов). "
            f"Профиль применяется автоматически на этапе композиции конфига "
            f"агента и не записывает ничего в память."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare personalization profiles on the same question."
    )
    parser.add_argument("--agent", default="assistant",
                        help="Agent name from data/agents.yaml (default: assistant).")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES),
                        help="Comma-separated profile names (default: developer,student,child).")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="The identical question sent through every profile.")
    parser.add_argument("--out", default=DEFAULT_OUT_MD,
                        help="Markdown report path (default: results/compare_profiles.md).")
    parser.add_argument("--json-out", default=DEFAULT_OUT_JSON,
                        help="JSON report path (default: results/compare_profiles.json).")
    parser.add_argument("--include-none", action="store_true",
                        help="Also run once WITHOUT a profile as a baseline row.")
    args = parser.parse_args(argv)

    profile_names = [p.strip() for p in args.profiles.split(",") if p.strip()]
    profile_store = YamlProfileStore()
    known = set(profile_store.list())
    unknown = [p for p in profile_names if p not in known]
    if unknown:
        print(
            f"error: unknown profile(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}.",
            file=sys.stderr,
        )
        return 2

    profile_runs: list[str | None] = (
        [None, *profile_names] if args.include_none else list(profile_names)
    )

    results: list[dict] = []
    for profile in profile_runs:
        print(f"[compare] profile={profile or '(none)'} ...", file=sys.stderr)
        results.append(
            _ask_once(
                args.agent,
                args.question,
                profile=profile,
                session_prefix="profile-compare",
            )
        )

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        _render_markdown(args.question, args.agent, results, profile_store),
        encoding="utf-8",
    )
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"question": args.question, "agent": args.agent, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written: {out_md}", file=sys.stderr)
    print(f"written: {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
