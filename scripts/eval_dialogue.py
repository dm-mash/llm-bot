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
from llm_bot.rag import (  # noqa: E402
    NOTE_NOT_FOUND,
    NOTE_UNUSED,
    SOURCES_HEADING,
    Grounding,
    sources_note_reason,
    summarise,
)
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
    #: Reported but never decisive. For the closing check, where demanding a
    #: literal number would be testing the model's phrasing rather than whether
    #: it kept the constraint.
    also_expect: list[str] = field(default_factory=list)
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

    def missed_soft(self, answer: str) -> list[str]:
        """Which of ``also_expect`` the answer did not mention."""
        return [
            pattern
            for pattern in self.also_expect
            if not re.search(pattern, answer, re.IGNORECASE)
        ]


@dataclass
class Dialogue:
    agent: str
    turns: list[Turn]
    #: What the dialog is for, and what the user pinned along the way. These are
    #: not judged by regex during the dialog — they are held here so the closing
    #: check can ask about them and so the report can print what was at stake.
    goal: str = ""
    clarified: tuple[str, ...] = ()
    terms: dict[str, str] = field(default_factory=dict)
    #: The question asked after the last real turn. "Did not lose the goal" has no
    #: meaning without asking the bot, and asking it is the only way to separate
    #: "kept the goal" from "happened to answer the right thing this once".
    check: Turn | None = None


def _turn(entry: object, path: Path, position: int) -> Turn:
    """Build a turn from its YAML block.

    ``question`` is accepted as a synonym for ``user`` because the closing check
    block is written as a question about the dialog, and forcing it to reuse the
    name of a customer turn would read as if it were one.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: ход #{position} — не словарь")
    return Turn(
        user=str(entry.get("user") or entry.get("question") or ""),
        contains=[str(p) for p in entry.get("contains") or []],
        absent=[str(p) for p in entry.get("absent") or []],
        any_of=[str(p) for p in entry.get("any_of") or []],
        also_expect=[str(p) for p in entry.get("also_expect") or []],
        note=str(entry.get("note") or ""),
    )


def load_dialogue(path: Path) -> Dialogue:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    turns = []
    for position, entry in enumerate(raw.get("turns") or [], start=1):
        turn = _turn(entry, path, position)
        if not turn.user:
            raise ValueError(f"{path}: ход #{position} без поля user")
        turns.append(turn)
    if not turns:
        raise ValueError(f"{path}: нет ходов")
    check = None
    if raw.get("check"):
        check = _turn(raw["check"], path, 0)
        if not check.user:
            raise ValueError(f"{path}: в check нет поля question")
    return Dialogue(
        agent=str(raw.get("agent") or "researcher"),
        turns=turns,
        goal=str(raw.get("goal") or ""),
        clarified=tuple(str(c) for c in (raw.get("clarified") or [])),
        terms={str(k): str(v) for k, v in (raw.get("terms") or {}).items()},
        check=check,
    )


def build_session_for(
    dialogue: Dialogue,
    index: Path,
    rerank: bool,
    run: int,
    reuse_evidence: bool = False,
    memory_on: bool = True,
    top_k: int | None = None,
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
        rag_top_k=top_k,
        rag_reuse_evidence=reuse_evidence,
        # Both arms run the same dialogue, so the only variable left is whether
        # anything was remembered. Turning extraction off rather than the layers
        # keeps the history intact: the arm difference then comes from memory
        # alone and not from a shorter prompt.
        memory_auto_extract=memory_on,
    )


def sources_state(answer: str) -> str:
    """How the answer handled its sources: a list, or which gap it declared.

    Reading it back through the constants rather than by matching wording keeps
    the report measurable after a note is reworded, and keeps the three states
    apart — "no sources" and "sources went unused" are different failures, and a
    single counter would hide the one that points at the model.
    """
    reason = sources_note_reason(answer)
    if reason == NOTE_NOT_FOUND:
        return "не найдены"
    if reason == NOTE_UNUSED:
        return "не использованы"
    return "есть" if SOURCES_HEADING in answer else "НЕТ"


def answer_body(text: str) -> str:
    """The model's own words, without the source footer the code appends.

    Content checks must not be able to match the footer. The footer is
    assembled by this script out of chunk paths and section titles, so a
    keyword can appear there without the model ever saying it: «самовывоз»
    satisfied a check on a delivery.md chunk heading for an answer that
    contained nothing but audit markers. A degenerate answer then scored as
    having answered the question.
    """
    body, _, _footer = text.partition(SOURCES_HEADING)
    return body.strip()


def repetition_ratio(text: str) -> float:
    """Share of lines that are repeats of an earlier line.

    A model that falls into a loop repeats one sentence verbatim; the audits
    then replace it with markers, but the damage is already in the answer. This
    is what the loop looks like from outside, and it is not a subtle quality
    difference — 11 consecutive turns returned byte-identical replies.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def run_dialogue(
    dialogue: Dialogue,
    index: Path,
    rerank: bool,
    run: int,
    reuse_evidence: bool = False,
    memory_on: bool = True,
    top_k: int | None = None,
) -> list[dict]:
    session = build_session_for(
        dialogue, index, rerank, run, reuse_evidence, memory_on, top_k
    )
    rows: list[dict] = []
    plan: list[tuple[Turn, bool]] = [(t, False) for t in dialogue.turns]
    if dialogue.check is not None:
        plan.append((dialogue.check, True))
    for number, (turn, is_check) in enumerate(plan, start=1):
        answer = session.chat(turn.user)
        passed, problems = turn.judge(answer_body(answer))
        # A turn with no expectations in the YAML passes for free, which would
        # let it hide behind a degenerate reply. Both of these are about whether
        # there *is* an answer, not about what it says.
        if not (turn.contains or turn.absent or turn.any_of):
            problems.append("в сценарии нет ни одной проверки ответа")
            passed = False
        looped = repetition_ratio(answer_body(answer))
        if looped >= 0.6:
            problems.append(
                f"ответ вырожден: {int(looped * 100)}% строк повторяют друг друга"
            )
            passed = False
        state = sources_state(answer)
        if is_check:
            # The closing question is about the dialog, not about a document, so
            # it is judged only on what it recalls. Grading it for grounding
            # would fail the run for answering honestly from memory: it is
            # supposed to have no documents behind it, and says so.
            rows.append({
                "turn": number,
                "user": turn.user,
                "answer": answer,
                "passed": passed,
                "problems": problems,
                "missed_soft": turn.missed_soft(answer),
                "sources": state,
                "grounding": session.last_grounding.status.value
                if session.last_grounding else None,
                "is_check": True,
                "note": turn.note,
            })
            continue
        # Three questions, answered separately.
        #
        #   по существу — did it answer what was asked (the YAML checks)
        #   источники   — did it show a source list, or declare the gap
        #   строго      — did every claim carry a citation *and* a verbatim
        #                 quote back to a chunk (the day24 protocol)
        #
        # These were one number, and that number lied by being too harsh: it is
        # the strict verdict only, and over a long dialog the model largely
        # stops quoting after the first few turns. So a run that kept the goal,
        # cited something on every turn and answered every question correctly
        # reported 1/14 and read as a failure. Split, the headline is what the
        # dialogue actually demonstrated and the strict figure stays visible as
        # the separate, harder thing it is.
        fact_passed = passed
        fact_problems = list(problems)
        strict: list[str] = []

        audit = session.last_citation_audit
        for cited in summarise(audit.dropped) if audit else ():
            strict.append(f"цитата не подтверждена блоком: {cited}")
        if audit is not None and audit.uncited:
            strict.append("ответ без единой цитаты")
        # A turn can clear every fact check and still be a guess. The verdict is
        # the same one the runtime used to decide whether it was safe to answer,
        # so a dialogue that passes here behaves the same way in production.
        quotes = session.last_quote_audit
        verdict = session.last_grounding
        for bad in summarise(quotes.dropped) if quotes else ():
            strict.append(f"цитата не дословна в чанке: {bad}")
        if verdict is not None and verdict.status is Grounding.UNGROUNDED:
            strict.append("ответ не опирается на подтверждённые источники")
        sources_ok = state != "НЕТ"
        if not sources_ok:
            strict.append("ни источников, ни пометки об их отсутствии")
        strict_passed = sources_ok and not strict
        memory = getattr(session, "memory", None)
        rows.append(
            {
                "turn": number,
                "user": turn.user,
                "answer": answer,
                "passed": fact_passed,
                "fact_passed": fact_passed,
                "sources_ok": sources_ok,
                "strict_passed": strict_passed,
                "problems": fact_problems,
                "strict_problems": strict,
                "missed_soft": turn.missed_soft(answer),
                "unconfirmed": list(audit.dropped) if audit else [],
                "uncited": bool(audit.uncited) if audit else False,
                "sources": state,
                "grounding": verdict.status.value if verdict else None,
                "bad_quotes": list(quotes.dropped) if quotes else [],
                "memory": (
                    f"w{len(memory.working)}/l{len(memory.long)}"
                    if memory is not None else ""
                ),
                # What memory holds *now*, with the words each fact was taken
                # from. A closing question can only fail for the right reason if
                # the report shows what was in memory to answer it with.
                "memory_now": _memory_view(session),
                "is_check": False,
                "note": turn.note,
            }
        )
    return rows


def _memory_view(session) -> str:
    """Memory as ``key = value (слова пользователя: «…»)`` lines.

    The evidence is the point: a fact with no traceable wording is a fact
    nothing can be checked against, and a report that prints only counts hides
    it.
    """
    memory = getattr(session, "memory", None)
    if memory is None:
        return ""
    lines = []
    for label, layer in (("долгая", memory.long), ("рабочая", memory.working)):
        for key, entry in layer.snapshot().items():
            # Deliberately not «…»: that is the citation delimiter the RAG
            # protocol uses, and a report that reuses it makes its own provenance
            # look like the model quoting memory instead of a document.
            where = (
                f"слова пользователя: {entry.evidence}"
                if entry.evidence
                else "без доказательства"
            )
            turn = f", ход {entry.turn}" if entry.turn else ""
            lines.append(f"{label} · {key} = {entry.value} ({where}{turn})")
    return "\n".join(lines)


def render_report(
    dialogue: Dialogue,
    rows_per_run: list[list[dict]],
    rerank: bool,
    index: Path,
    name: str = "",
    memory_on: bool = True,
    top_k: int | None = None,
) -> str:
    real = [r for rows in rows_per_run for r in rows if not r["is_check"]]
    checks = [r for rows in rows_per_run for r in rows if r["is_check"]]
    passed = sum(1 for row in real if row["fact_passed"])
    total = len(real)
    with_sources = sum(1 for row in real if row.get("sources_ok"))
    strict = sum(1 for row in real if row.get("strict_passed"))
    lines = [
        f"# Диалог: {name or index.stem}",
        "",
        f"- Агент: `{dialogue.agent}`",
        f"- Индекс: `{index}`",
        f"- Rerank: {'включён' if rerank else 'выключен'}",
        f"- Chunks на ход: {top_k or 4}",
        f"- Память: {'включена' if memory_on else 'выключена'}",
        f"- Прогонов: {len(rows_per_run)}, ходов в прогоне: {len(dialogue.turns)}",
        "",
        "## Итоги",
        "",
        f"- **Отвечено по существу: {passed}/{total}** — ход ответил на заданный вопрос",
        f"- **Источники показаны: {with_sources}/{total}** — список источников либо пометка об их отсутствии",
        f"- Строгое опирание на источники: {strict}/{total} — каждое утверждение с цитатой и дословной выдержкой из чанка",
    ]
    if total and strict < passed:
        lines += [
            "",
            f"Строгий протокол валит {total - strict} из {total} ходов: в длинном диалоге "
            "модель перестаёт давать дословные цитаты после первых ходов. Это отдельная "
            "мера качества, а не отказ отвечать — поэтому она показана отдельно.",
        ]
    if dialogue.goal:
        lines += ["", "## Что диалог обязан был удержать", ""]
        lines.append(f"- **Цель:** {dialogue.goal}")
        for item in dialogue.clarified:
            lines.append(f"- Уточнено: {item}")
        for term, meaning in dialogue.terms.items():
            lines.append(f"- Термин «{term}»: {meaning}")
    if checks:
        kept = sum(1 for row in checks if row["passed"])
        mark = "**нет**" if kept < len(checks) else "ок"
        lines += [
            "",
            "## Удержание цели",
            "",
            f"- Ответ на закрывающий вопрос: {kept}/{len(checks)} ({mark})",
        ]
        for row in checks:
            if row["missed_soft"]:
                lines.append(
                    f"- не упомянуто (мягкая проверка): {', '.join(row['missed_soft'])}"
                )
    counts: dict[str, int] = {}
    for row in real:
        counts[row["sources"]] = counts.get(row["sources"], 0) + 1
    lines += [
        "",
        "## Источники на ходах",
        "",
        "| состояние | ходов |",
        "|---|---|",
    ]
    for state in ("есть", "не найдены", "не использованы", "НЕТ"):
        if counts.get(state):
            lines.append(f"| {state} | {counts[state]} |")
    lines += [
        "",
        "## Сводка по ходам",
        "",
        "| # | ход клиента | " + " | ".join(f"прогон {n}" for n in range(1, len(rows_per_run) + 1)) + " |",
        "|---|-------------|" + "|".join(["---"] * len(rows_per_run)) + "|",
    ]
    for index_number, turn in enumerate(dialogue.turns):
        cells = []
        for rows in rows_per_run:
            # A short run (a turn that raised, a filtered dialog) must not make
            # the report itself crash — a broken report hides the break.
            if index_number >= len(rows):
                continue
            row = rows[index_number]
            # The content verdict is the headline; the strict one is shown too,
            # so a turn that answered well but skipped the quote protocol is
            # legible as exactly that.
            fact = "ок" if row["fact_passed"] else "**нет**"
            if not row.get("strict_passed", True):
                fact += " · строго нет"
            cells.append(fact)
        first = turn.user.replace("|", "\\|")
        lines.append(f"| {index_number + 1} | {first} | " + " | ".join(cells) + " |")

    for run, rows in enumerate(rows_per_run, start=1):
        lines += ["", f"## Прогон {run}", ""]
        for row in rows:
            if row["is_check"]:
                mark = "ок" if row["passed"] else "НЕ ПРОШЁЛ"
            else:
                mark = "ок" if row["fact_passed"] else "НЕ ПРОШЁЛ"
                if not row.get("strict_passed", True):
                    mark += " (строго нет)"
            title = "ход" if not row["is_check"] else "закрывающий вопрос"
            extra = f" · источники: {row['sources']}"
            if row.get("memory"):
                extra += f" · память: {row['memory']}"
            lines += [
                f"### {title.capitalize()} {row['turn']} — {mark}{extra}",
                "",
                f"**Клиент:** {row['user']}",
                "",
                f"**Агент:** {row['answer']}",
                "",
            ]
            if row["problems"]:
                lines += [f"Расхождения по существу: {', '.join(row['problems'])}", ""]
            if row.get("strict_problems"):
                lines += [
                    f"Строгий протокол: {', '.join(row['strict_problems'])}",
                    "",
                ]
            if row.get("memory_now"):
                lines += ["Память на этом ходу:", "", row["memory_now"], ""]
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
        "--rag-top-k",
        type=int,
        default=None,
        help=(
            "сколько чанков на ход. 4 — по умолчанию; на этом корпусе меню "
            "напитков восьмое из семнадцати, поэтому на общих вопросах "
            "ответ физически не попадал в блок"
        ),
    )
    parser.add_argument(
        "--reuse-evidence",
        action="store_true",
        help="считать подтверждённым источник из прошлого хода сессии",
    )
    parser.add_argument(
        "--memory",
        choices=("on", "off"),
        default="on",
        help=(
            "рука сравнения: извлекать ли факты в память. Обе руки проходят "
            "один и тот же диалог и сохраняют историю, поэтому разница в "
            "закрывающем вопросе объясняется памятью"
        ),
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

    memory_on = args.memory == "on"
    rows_per_run = [
        run_dialogue(
            dialogue,
            args.index,
            rerank,
            run,
            args.reuse_evidence,
            memory_on,
            args.rag_top_k,
        )
        for run in range(1, args.runs + 1)
    ]
    report = render_report(
        dialogue,
        rows_per_run,
        rerank,
        args.index,
        args.dialogue.stem,
        memory_on,
        args.rag_top_k,
    )

    checks = [r for rows in rows_per_run for r in rows if r["is_check"]]
    real = [r for rows in rows_per_run for r in rows if not r["is_check"]]
    passed = sum(1 for row in real if row["fact_passed"])
    total = len(real)
    with_sources = sum(1 for row in real if row.get("sources_ok"))
    strict = sum(1 for row in real if row.get("strict_passed"))
    for position, turn in enumerate(dialogue.turns):
        marks = " ".join(
            "ок  " if rows[position]["fact_passed"] else "НЕТ " for rows in rows_per_run
        )
        states = " ".join(rows[position]["sources"] for rows in rows_per_run)
        print(f"ход {position + 1:>2}: {marks}  [{states}]  {turn.user[:56]}")
    if checks:
        kept = sum(1 for row in checks if row["passed"])
        print(f"\nудержание цели: {kept}/{len(checks)}")
    print(f"отвечено по существу: {passed}/{total}")
    print(f"источники показаны:   {with_sources}/{total}")
    print(f"строгое опирание:     {strict}/{total}")

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