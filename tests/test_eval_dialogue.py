"""Tests for scripts/eval_dialogue.py: parsing a dialogue and classifying sources.

All offline: nothing here builds a session or touches the network. The parts
worth pinning down are the ones that decide whether the day's own acceptance
criterion ("did not lose the goal") can be measured at all, plus the three-way
sources classification, where folding two of the states together would hide the
one that points at the model.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_dialogue", ROOT / "scripts" / "eval_dialogue.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before executing: dataclasses look up ``cls.__module__`` in
    # sys.modules while building the class, so an unregistered module breaks.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()

from llm_bot.rag import NO_SOURCES_NOTE, UNUSED_SOURCES_NOTE  # noqa: E402

DIALOGUE_YAML = """
agent: researcher
goal: подобрать гостю напиток без молока
clarified:
  - "гость не пьёт молоко"
terms:
  "строго без орехов": "и следы миндаля не подходят"
check:
  question: "Назови цель разговора."
  contains: ["молок"]
  also_expect: ["400"]
turns:
  - user: "Что есть из напитков?"
    any_of: ["эспрессо"]
  - user: "А сезонное меню есть?"
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "d.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_dialogue_carries_what_it_must_hold_onto(tmp_path: Path) -> None:
    """goal/clarified/terms are what the closing question is built to ask about;
    dropping them silently would leave the criterion unmeasurable."""
    dialogue = ev.load_dialogue(_write(tmp_path, DIALOGUE_YAML))

    assert dialogue.goal == "подобрать гостю напиток без молока"
    assert dialogue.clarified == ("гость не пьёт молоко",)
    assert dialogue.terms == {"строго без орехов": "и следы миндаля не подходят"}
    assert dialogue.check is not None
    assert dialogue.check.user == "Назови цель разговора."
    assert [t.user for t in dialogue.turns] == [
        "Что есть из напитков?",
        "А сезонное меню есть?",
    ]


def test_a_dialogue_without_a_check_still_loads(tmp_path: Path) -> None:
    """day10-style dialogues have no closing question; that must not be fatal."""
    dialogue = ev.load_dialogue(
        _write(tmp_path, "turns:\n  - user: \"вопрос\"\n")
    )

    assert dialogue.check is None
    assert dialogue.goal == ""


def test_a_turn_needs_a_question(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ev.load_dialogue(_write(tmp_path, "turns:\n  - contains: [\"x\"]\n"))


def test_the_check_question_may_be_written_as_question(tmp_path: Path) -> None:
    body = yaml.safe_load(DIALOGUE_YAML)
    body["check"]["user"] = body["check"].pop("question")
    path = _write(tmp_path, yaml.safe_dump(body, allow_unicode=True))

    assert ev.load_dialogue(path).check is not None


def test_sources_are_told_apart_in_three_states() -> None:
    """One counter for both gaps would report a stable number while the model
    quietly stopped using the evidence it was handed."""
    assert ev.sources_state("Ответ [1].\n\nИсточник:\n[1] a.md:1-2 — s") == "есть"
    assert ev.sources_state("Ответ.\n\n" + NO_SOURCES_NOTE) == "не найдены"
    assert ev.sources_state("Ответ.\n\n" + UNUSED_SOURCES_NOTE) == "не использованы"


def test_an_answer_with_neither_is_caught() -> None:
    """The reason the classification exists: this used to look like a normal
    grounded answer, because both cases produced identical text."""
    assert ev.sources_state("Просто ответ без источников.") == "НЕТ"


def test_soft_expectations_are_reported_but_never_decisive(tmp_path: Path) -> None:
    """Demanding a literal «400» would test the model's phrasing, not its memory."""
    check = ev.load_dialogue(_write(tmp_path, DIALOGUE_YAML)).check

    passed, problems = check.judge("Мы подобрали напиток без молока.")

    assert passed and not problems
    assert check.missed_soft("Мы подобрали напиток без молока.") == ["400"]
    assert check.missed_soft("Мы подобрали напиток без молока за 400 ₽.") == []

# --------------------------------------------------------------------------- #
# The shipped dialogues
# --------------------------------------------------------------------------- #


def _dialogues() -> list[tuple[str, "ev.Dialogue"]]:
    out = []
    for name in ("coffee", "certs"):
        path = ROOT / "dialogues" / f"{name}.yaml"
        assert path.is_file(), f"нет сценария: {path}"
        out.append((name, ev.load_dialogue(path)))
    return out


@pytest.mark.parametrize("name,dialogue", _dialogues())
def test_the_dialogue_is_long_enough_to_be_one(name: str, dialogue) -> None:
    """The assignment asks for 10–15 messages; a shorter fixture would pass
    without ever exercising a forgetting window."""
    assert 10 <= len(dialogue.turns) <= 15


@pytest.mark.parametrize("name,dialogue", _dialogues())
def test_what_it_must_hold_onto_was_actually_stated(name: str, dialogue) -> None:
    """Every constraint and term the closing check asks about has to be something
    the user actually said in the dialog.

    Declaring a constraint nobody stated makes the check unfalsifiable: the run
    fails for memory that was never asked to hold anything, which reads as a
    memory bug and sends the next person to the wrong module.
    """
    said = " ".join(t.user for t in dialogue.turns).lower()

    def stated(item: str) -> bool:
        words = [w for w in _words(item) if len(w) >= 4]
        return any(w in said for w in words)

    for item in dialogue.clarified:
        assert stated(item), f"{name}: уточнение не прозвучало в ходах: {item!r}"
    for term, meaning in dialogue.terms.items():
        assert stated(term) or stated(meaning), (
            f"{name}: термин не закреплён в ходах: {term!r}"
        )


@pytest.mark.parametrize("name,dialogue", _dialogues())
def test_the_closing_question_expects_the_goal(name: str, dialogue) -> None:
    assert dialogue.check is not None, f"{name}: нет закрывающего вопроса"
    assert dialogue.goal
    # contains/any_of must reference the goal itself, not only some detail: a
    # check satisfiable without naming the purpose would pass on any dialog.
    assert dialogue.check.contains or dialogue.check.any_of


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"[0-9A-Za-zа-яА-ЯёЁ]+", text.lower())


def test_the_memory_arm_is_the_only_variable_between_the_two_runs(tmp_path):
    """Both arms must share the whole dialogue, the index and the history; the
    only difference is extraction. Comparing anything else would let a shorter
    prompt or a lost turn masquerade as a memory result."""
    index = ROOT / "data" / "kb_emb" / "index_structure.json"
    assert index.is_file()
    dlg = ev.Dialogue(
        agent="researcher",
        turns=[ev.Turn("первый вопрос"), ev.Turn("второй вопрос")],
        goal="проверка",
    )

    on = ev.build_session_for(dlg, index, True, 1, False, True)
    off = ev.build_session_for(dlg, index, True, 1, False, False)

    assert on.memory_auto_extract is True
    assert off.memory_auto_extract is False
    # The history is intact in both arms, so a forgetting failure in the "on"
    # arm cannot be blamed on a turn that never arrived.
    assert len(off.history) == len(on.history) == 0


def test_the_report_is_titled_after_the_dialogue_not_the_index() -> None:
    """Both shipped dialogues run against index_structure.json, so a title built
    from the index names both of them «index_structure» and the two reports
    cannot be told apart in a results folder."""
    index = ROOT / "data" / "kb_emb" / "index_structure.json"
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")
    report = ev.render_report(dlg, [[]], True, index, "coffee")

    assert report.splitlines()[0] == "# Диалог: coffee"


def test_the_report_says_which_arm_it_was() -> None:
    """A result whose memory arm is unlabelled cannot be compared with anything;
    two files differing only in the flag would look like two identical runs."""
    index = ROOT / "data" / "kb_emb" / "index_structure.json"
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")

    assert "Память: выключена" in ev.render_report(dlg, [[]], True, index, "x", False)
    assert "Память: включена" in ev.render_report(dlg, [[]], True, index, "x", True)


def test_the_report_shows_where_each_stored_fact_came_from() -> None:
    """Counts alone cannot explain a failed closing question: "w6/l0" says a
    memory existed, not whether it held the constraint being asked about. The
    report has to show the fact together with the user's words it came from,
    because an untraceable fact is one nothing can be checked against."""
    row = {
        "turn": 1, "user": "вопрос", "answer": "ответ", "passed": True,
        "fact_passed": True, "sources_ok": True, "strict_passed": True,
        "strict_problems": [],
        "problems": [], "missed_soft": [], "unconfirmed": [], "uncited": False,
        "sources": "есть", "grounding": None, "bad_quotes": [],
        "memory": "w1/l0",
        "memory_now": "рабочая · current_constraints = без молока "
                      "(слова пользователя: Молоко он не пьёт, ход 3)",
        "is_check": False, "note": "",
    }
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")
    report = ev.render_report(dlg, [[row]], True, index_stub(), "coffee")

    assert "Память на этом ходу:" in report
    assert "current_constraints = без молока" in report
    assert "Молоко он не пьёт" in report
    assert "ход 3" in report


def index_stub():
    from pathlib import Path

    return Path("data/kb_emb/index_structure.json")


@pytest.mark.parametrize("name,dialogue", _dialogues())
def test_every_expected_pattern_is_a_valid_regex(name: str, dialogue) -> None:
    """Patterns are regexes, and a phone prefix like ``+7`` is not one.

    An uncompilable pattern does not fail the dialogue — it raises out of the
    judge mid-run and the whole report is lost, which is how a typo in a fixture
    cost a full 14-turn run.
    """
    items = [(f"ход {i}", t) for i, t in enumerate(dialogue.turns, 1)]
    if dialogue.check is not None:
        items.append(("закрывающий вопрос", dialogue.check))

    for where, turn in items:
        for field in ("contains", "any_of"):
            for pattern in getattr(turn, field) or ():
                try:
                    re.compile(pattern)
                except re.error as exc:
                    pytest.fail(f"{name}: {where}, {field}: {pattern!r} -> {exc}")


def _row(**kw):
    row = {
        "turn": 1, "user": "вопрос", "answer": "ответ", "passed": True,
        "fact_passed": True, "sources_ok": True, "strict_passed": True,
        "strict_problems": [], "problems": [], "missed_soft": [],
        "unconfirmed": [], "uncited": False, "sources": "есть",
        "grounding": None, "bad_quotes": [], "memory": "w1/l0",
        "memory_now": "", "is_check": False, "note": "",
    }
    row.update(kw)
    return row


def test_a_turn_that_answered_well_is_not_reported_as_a_failure() -> None:
    """The single verdict was the strict protocol only.

    It read 1/14 on a run that kept the goal, cited sources on every turn and
    answered correctly, because in a long dialog the model stops giving verbatim
    quotes after the first few turns. Reporting that as "the dialogue failed"
    contradicts the run's own evidence, so the three questions are counted
    separately and the content one leads.
    """
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")
    row = _row(fact_passed=True, sources_ok=True, strict_passed=False,
               strict_problems=["ответ не опирается на подтверждённые источники"])
    report = ev.render_report(dlg, [[row]], True, index_stub(), "coffee")

    assert "**Отвечено по существу: 1/1**" in report
    assert "**Источники показаны: 1/1**" in report
    assert "Строгое опирание на источники: 0/1" in report
    assert "строго нет" in report


def test_the_strict_measure_stays_visible_instead_of_being_dropped() -> None:
    """Splitting must not become hiding: the strict figure is still the hardest
    measure available and the report has to keep saying it failed."""
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")
    row = _row(fact_passed=True, strict_passed=False,
               strict_problems=["цитата не дословна в чанке: Тростниковый крем"])
    report = ev.render_report(dlg, [[row]], True, index_stub(), "coffee")

    assert "Строгий протокол: цитата не дословна в чанке" in report
    assert "перестаёт давать дословные цитаты" in report


def test_a_turn_with_no_sources_fails_the_source_measure() -> None:
    """Showing nothing at all is the case the markers exist for, so it has to
    keep failing even though the content verdict passed."""
    dlg = ev.Dialogue(agent="researcher", turns=[ev.Turn("вопрос")], goal="цель")
    row = _row(fact_passed=True, sources_ok=False, strict_passed=False,
               strict_problems=["ни источников, ни пометки об их отсутствии"])
    report = ev.render_report(dlg, [[row]], True, index_stub(), "coffee")

    assert "**Источники показаны: 0/1**" in report
    assert "ни источников, ни пометки" in report


# --------------------------------------------------------------------------- #
# Two ways a run could pass without the bot having answered
# --------------------------------------------------------------------------- #


def test_a_keyword_in_the_source_footer_cannot_satisfy_a_content_check() -> None:
    """The footer is assembled by this script from chunk paths and section
    titles, so the model never had to say the word.

    Measured on a coffee run: the answer to «А если он хочет взять с собой?» was
    95 lines of audit markers and nothing else, and the turn passed because
    «самовывоз» appeared in ``delivery.md — Доставка и самовывоз > Самовывоз``.
    Judging the whole reply string let the grader mark its own footer as an
    answer.
    """
    answer = (
        '[1] [источник не подтверждён].\n' * 5
        + 'Источник:\n'
        + '[1] delivery.md:10-15 — Доставка и самовывоз — Зёрна > Самовывоз'
    )
    turn = ev.Turn("А если он хочет взять с собой?", contains=["самовывоз"])

    assert ev.answer_body(answer).find("Источник") == -1
    # Without the footer there is nothing left to match.
    passed, _ = turn.judge(ev.answer_body(answer))
    assert passed is False
    # And the footer really does contain the word that would have passed it.
    assert turn.judge(answer)[0] is True


def test_a_reply_that_repeats_one_line_is_not_an_answer() -> None:
    """The audits replace a repeated sentence with markers, but the loop happens
    first: in one run, turns 4 through the closing question returned
    byte-identical replies, and the per-turn counts called them answers."""
    looped = "[1] [источник не подтверждён].\n" * 40
    varied = "\n".join("строка %d" % i for i in range(40))

    assert ev.repetition_ratio(looped) > 0.9
    assert ev.repetition_ratio(varied) == 0.0
    # Short replies are exempt — a two-line answer is not a loop.
    assert ev.repetition_ratio("да\nне") == 0.0


def test_a_turn_with_no_expectations_cannot_pass_for_free() -> None:
    """Five of the fourteen turns carried no ``contains``/``any_of`` at all, so
    they passed whatever the bot said, including nothing."""
    turn = ev.Turn("А есть сезонное меню?")

    assert not (turn.contains or turn.absent or turn.any_of)


@pytest.mark.parametrize("name,dialogue", _dialogues())
def test_every_turn_says_what_it_should_have_answered(name: str, dialogue) -> None:
    """A turn with no ``contains``/``any_of``/``absent`` passes whatever the bot
    says.

    Six turns in certs and five in coffee were written that way, so they could
    not fail — including the two out-of-base questions, where a refusal was
    wanted and nothing checked for one. run_dialogue now rejects them outright;
    this keeps the fixtures from drifting back.
    """
    for i, turn in enumerate(dialogue.turns, 1):
        assert turn.contains or turn.any_of or turn.absent, (
            f"{name}: ход {i} без проверок ответа — {turn.user!r}"
        )
    assert dialogue.check is not None
    assert dialogue.check.contains or dialogue.check.any_of
