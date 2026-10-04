"""Прогнать один диалог целиком и показать, где агент путает документы.

Зачем отдельный скрипт, если есть ``compare_rag.py``: тот отвечает на
самодостаточные вопросы по одному, а провал дня23 был в анафорических ходах
(«То есть это <другой вариант>?», «а сколько это стоит?»), где смысл держится
на предыдущих репликах. По одному такой ход не воспроизвести — нужен связный
диалог.

    python scripts/eval_dialogue.py --dialogue dialogue.yaml --runs 3
    python scripts/eval_dialogue.py --dialogue dialogue.yaml --no-rerank

Каждый прогон — новая сессия с пустой памятью, чтобы прошлый диалог не влиял на
следующий. Ответы печатаются целиком: проверки в YAML намеренно грубые
(regex, а не смысловой разбор), и итог всегда смотрится глазами.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_bot.factory import make_session  # noqa: E402
from llm_bot.json_session_store import JsonSessionStore  # noqa: E402
from llm_bot.memory_store import JsonMemoryStore  # noqa: E402
from llm_bot.yaml_stores import (  # noqa: E402
    YamlAgentStore,
    YamlModelStore,
)

DEFAULT_DIALOGUE = Path("dialogue.yaml")
DEFAULT_INDEX = Path("data/index/index_structure.json")


@dataclass
class Turn:
    """Один ход клиента с ожиданиями по ответу."""

    user: str
    contains: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    any_of: list[str] = field(default_factory=list)
    note: str = ""

    def judge(self, answer: str) -> tuple[bool, list[str]]:
        """Вернуть (прошёл ли, что именно не сошлось)."""
        problems: list[str] = []
        for pattern in self.contains:
            if not re.search(pattern, answer, re.IGNORECASE):
                problems.append(f"нет {pattern!r}")
        for pattern in self.absent:
            if re.search(pattern, answer, re.IGNORECASE):
                problems.append(f"лишнее {pattern!r}")
        if self.any_of and not any(
            re.search(pattern, answer, re.IGNORECASE) for pattern in self.any_of
        ):
            problems.append(f"нет ни одного из {self.any_of}")
        return not problems, problems


@dataclass
class Dialogue:
    agent: str
    turns: list[Turn]


def load_dialogue(path: Path) -> Dialogue:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    turns: list[Turn] = []
    for position, entry in enumerate(raw.get("turns") or [], start=1):
        if not isinstance(entry, dict) or not entry.get("user"):
            raise ValueError(f"{path}: ход #{position} без поля user")
        turns.append(
            Turn(
                user=str(entry["user"]),
                contains=[str(p) for p in entry.get("contains") or []],
                absent=[str(p) for p in entry.get("absent") or []],
                any_of=[str(p) for p in entry.get("any_of") or []],
                note=str(entry.get("note") or ""),
            )
        )
    if not turns:
        raise ValueError(f"{path}: нет ходов")
    return Dialogue(agent=str(raw.get("agent") or "researcher"), turns=turns)


def build_session_for(
    dialogue: Dialogue,
    index: Path,
    rerank: bool,
    run: int,
    reuse_evidence: bool = False,
):
    """Новая сессия со своей памятью: прогон не должен видеть предыдущий."""
    workdir = Path(tempfile.mkdtemp(prefix=f"dialogue-{run}-"))
    return make_session(
        f"dialogue-{run}",
        dialogue.agent,
        model_store=YamlModelStore(),
        agent_store=YamlAgentStore(),
        session_store=JsonSessionStore(workdir),
        memory_store=JsonMemoryStore(workdir / "memory.json"),
        rag_index=index,
        rag_rerank=rerank,
        rag_reuse_evidence=reuse_evidence,
    )


def run_dialogue(
    dialogue: Dialogue,
    index: Path,
    rerank: bool,
    run: int,
    reuse_evidence: bool = False,
) -> list[dict]:
    session = build_session_for(dialogue, index, rerank, run, reuse_evidence)
    rows: list[dict] = []
    for number, turn in enumerate(dialogue.turns, start=1):
        answer = session.chat(turn.user)
        passed, problems = turn.judge(answer)
        audit = session.last_citation_audit
        for cited in audit.dropped if audit else ():
            problems.append(f"цитата не подтверждена блоком: {cited}")
            passed = False
        if audit is not None and audit.uncited:
            problems.append("ответ без единой цитаты")
            passed = False
        rows.append(
            {
                "turn": number,
                "user": turn.user,
                "answer": answer,
                "passed": passed,
                "problems": problems,
                "unconfirmed": list(audit.dropped) if audit else [],
                "uncited": bool(audit.uncited) if audit else False,
                "note": turn.note,
            }
        )
    return rows


def render_report(dialogue: Dialogue, rows_per_run: list[list[dict]], rerank: bool) -> str:
    total = len(dialogue.turns) * len(rows_per_run)
    passed = sum(1 for rows in rows_per_run for row in rows if row["passed"])
    lines = [
        "# Диалоговая проверка: два похожих документа",
        "",
        f"- Агент: `{dialogue.agent}`",
        f"- Rerank: {'включён' if rerank else 'выключен'}",
        f"- Прогонов: {len(rows_per_run)}, ходов в прогоне: {len(dialogue.turns)}",
        f"- Прошло ходов: **{passed}/{total}**",
        "",
        "## Сводка по ходам",
        "",
        "| # | ход клиента | " + " | ".join(f"прогон {n}" for n in range(1, len(rows_per_run) + 1)) + " |",
        "|---|-------------|" + "|".join(["---"] * len(rows_per_run)) + "|",
    ]
    for index, turn in enumerate(dialogue.turns):
        cells = []
        for rows in rows_per_run:
            row = rows[index]
            cells.append("ок" if row["passed"] else "**нет**")
        first = turn.user.replace("|", "\\|")
        lines.append(f"| {index + 1} | {first} | " + " | ".join(cells) + " |")

    for run, rows in enumerate(rows_per_run, start=1):
        lines += ["", f"## Прогон {run}", ""]
        for row in rows:
            mark = "ок" if row["passed"] else "НЕ ПРОШЁЛ"
            lines += [
                f"### Ход {row['turn']} — {mark}",
                "",
                f"**Клиент:** {row['user']}",
                "",
                f"**Агент:** {row['answer']}",
                "",
            ]
            if row["problems"]:
                lines += [f"Расхождения: {', '.join(row['problems'])}", ""]
            if row["note"]:
                lines += [f"_{row['note']}_", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dialogue", type=Path, default=DEFAULT_DIALOGUE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--reuse-evidence",
        action="store_true",
        help="считать подтверждённым источник из прошлого хода сессии",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.index.is_file():
        print(f"индекс не найден: {args.index}", file=sys.stderr)
        return 2
    dialogue = load_dialogue(args.dialogue)
    rerank = not args.no_rerank

    rows_per_run = [
        run_dialogue(
            dialogue, args.index, rerank, run, args.reuse_evidence
        )
        for run in range(1, args.runs + 1)
    ]
    report = render_report(dialogue, rows_per_run, rerank)

    passed = sum(1 for rows in rows_per_run for row in rows if row["passed"])
    total = len(dialogue.turns) * len(rows_per_run)
    for index, turn in enumerate(dialogue.turns):
        marks = " ".join(
            "ок  " if rows[index]["passed"] else "НЕТ " for rows in rows_per_run
        )
        print(f"ход {index + 1:>2}: {marks}  {turn.user[:60]}")
    print(f"\nитого {passed}/{total}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"отчёт: {args.out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(rows_per_run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"json: {args.json_out}")
    else:
        print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())