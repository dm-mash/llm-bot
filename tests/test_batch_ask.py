from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "batch_ask", ROOT / "scripts" / "batch_ask.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch = _load_script()

_ORIGINAL_ASK = batch.ask


def _args(**overrides):
    parser = batch.build_parser()
    args = parser.parse_args(["--questions", "q.txt"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_a_plain_file_is_one_question_per_line() -> None:
    """The shape people actually write down: notes, ``#`` and all."""
    path = ROOT / "tests" / "_tmp_questions.txt"
    path.write_text(
        "# вопросы по документам\n"
        "\n"
        "С какой высоты прыжок?\n"
        "   \n"
        "  # закомментированный\n"
        "Где проходит урок?\n",
        encoding="utf-8",
    )
    try:
        assert batch.load_questions(path) == [
            "С какой высоты прыжок?",
            "Где проходит урок?",
        ]
    finally:
        path.unlink()


def test_a_yaml_list_is_read_whether_items_are_strings_or_objects() -> None:
    """The repo's own ``questions.yaml`` is a list of objects, so both shapes
    have to work or the file has to be rewritten by hand."""
    path = ROOT / "tests" / "_tmp_questions.yaml"
    path.write_text(
        "questions:\n"
        "  - \"Первый вопрос?\"\n"
        "  - text: \"Второй вопрос?\"\n"
        "  - question: \"Третий вопрос?\"\n",
        encoding="utf-8",
    )
    try:
        assert batch.load_questions(path) == [
            "Первый вопрос?",
            "Второй вопрос?",
            "Третий вопрос?",
        ]
    finally:
        path.unlink()


def test_a_missing_or_empty_file_is_reported_not_guessed_at() -> None:
    """A silently empty question list would look like a run where the bot had
    nothing to say about anything."""
    try:
        batch.load_questions(ROOT / "tests" / "_tmp_absent.txt")
    except FileNotFoundError as exc:
        assert "не найден" in str(exc)
    else:
        raise AssertionError("ожидался FileNotFoundError")

    path = ROOT / "tests" / "_tmp_empty.txt"
    path.write_text("# только комментарий\n", encoding="utf-8")
    try:
        batch.load_questions(path)
    except ValueError as exc:
        assert "не найдено" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")
    finally:
        path.unlink()


def test_the_command_is_the_one_a_person_would_have_typed() -> None:
    """The whole point of the script is not to become a second implementation of
    the CLI, so what it builds has to be exactly the documented invocation."""
    args = _args(
        agent="researcher",
        rag_index=Path("data/emb/index_structure.json"),
        rerank=True,
        strict=True,
        no_cite=False,
        top_k=None,
        model=None,
    )

    command = batch.build_command(args, "Вопрос?")

    assert command[1:] == [
        "-m",
        "llm_bot",
        "--agent",
        "researcher",
        "--rag",
        "--rag-index",
        "data/emb/index_structure.json",
        "--rag-rerank",
        "--rag-strict",
        "Вопрос?",
    ]


def test_no_index_means_no_rag_flags_at_all() -> None:
    """Without an index there is no block, and passing ``--rag-rerank`` anyway
    would be a command the CLI rejects."""
    args = _args(
        agent="researcher", rag_index=None, rerank=True, strict=False, no_cite=False
    )

    assert "--rag" not in batch.build_command(args, "Вопрос?")
    assert "--rag-rerank" not in batch.build_command(args, "Вопрос?")


def test_strict_and_no_cite_are_refused_before_spending_a_model_load() -> None:
    """The CLI refuses this pair too, and finding out after loading the reranker
    costs half a minute for nothing."""
    assert (
        batch.main(
            [
                "--questions",
                str(ROOT / "tests" / "test_batch_ask.py"),
                "--rag-strict",
                "--rag-no-cite",
            ]
        )
        == 2
    )

def test_only_failures_that_can_heal_are_retried() -> None:
    """A rate limit is worth another go; a bad index is not.

    The CLI already retries a provider request four times, and a batch asks the
    same provider the same question back to back, which is what trips the limit
    in the first place. Retrying everything instead would turn a wrong index
    into several minutes of waiting for the same error.
    """
    for text in ("429 Rate limit exceeded", "raw_status_code\":429", "request timed out"):
        assert batch._transient(text), text
    for text in ("index not found: data/nope.json", "ValueError: bad dimension", ""):
        assert not batch._transient(text), text


def test_the_retry_loop_gives_up_after_the_configured_number_of_tries() -> None:
    calls: list[list[str]] = []

    def fake(command, timeout):
        calls.append(command)
        return "", "429 Rate limit exceeded", True

    batch.ask = fake
    try:
        _, _, broke = batch.ask_with_retries(["q"], None, 2)
    finally:
        batch.ask = _ORIGINAL_ASK

    assert broke is True
    assert len(calls) == 3


def test_the_retry_loop_stops_as_soon_as_the_answer_arrives() -> None:
    calls: list[list[str]] = []

    def fake(command, timeout):
        calls.append(command)
        if len(calls) == 1:
            return "", "429", True
        return "Ответ.", "", False

    batch.ask = fake
    try:
        answer, _, broke = batch.ask_with_retries(["q"], None, 2)
    finally:
        batch.ask = _ORIGINAL_ASK

    assert (answer, broke, len(calls)) == ("Ответ.", False, 2)


def test_each_question_gets_a_new_session_even_when_models_are_shared() -> None:
    """The only thing worth reusing between questions is the pair of models.

    A shared session would carry history, memory and the follow-up document bias
    from one question into the next, which is a different measurement entirely —
    so the retriever is shared and the session is thrown away.
    """
    built: list[str] = []

    class _FakeSession:
        def __init__(self, session_id):
            built.append(session_id)

        def chat(self, question):
            return f"ответ на «{question}»"

    class _Store:
        def __init__(self, *args, **kwargs):
            pass

    runner = batch.WarmRunner.__new__(batch.WarmRunner)
    runner._make_session = lambda session_id, *a, **kw: _FakeSession(session_id)
    runner._session_store = _Store
    runner._memory_store = _Store
    runner._model_store = _Store()
    runner._agent_store = _Store()
    runner._agent = "researcher"
    runner._retriever = None
    runner._no_cite = False
    runner._strict = False

    assert runner.ask("первый", 1) == "ответ на «первый»"
    assert runner.ask("второй", 2) == "ответ на «второй»"
    assert built == ["batch-1", "batch-2"]


def test_a_block_leads_with_its_question_and_ends_with_its_own_gap() -> None:
    """The saved file used to contain answers with no trace of which question
    each belonged to, and the separation between them was printed outside the
    block, so it landed wherever the two streams interleaved."""
    answer = (
        "С высоты 800 м [1] «прыжки проводятся с высоты 800 м».\n\n"
        "Источники (фрагменты из ответа):\n"
        "[1] Прыжок с парашютом.pdf:19-23 — page 1 · chunk c1\n"
        "[2] Прыжок в тандеме.pdf:11-19 — page 1 · chunk c2"
    )

    block = batch.render_block(1, 40, "С какой высоты прыжок?", answer)
    lines = block.splitlines()

    assert lines[0].startswith("─")
    assert lines[1] == "1/40  С какой высоты прыжок?"
    assert "Источники:" in block
    assert "    [1] Прыжок с парашютом.pdf:19-23" in block
    # Ends with a newline of its own: printing the block then supplies the blank
    # line, so blocks never touch however they are joined or streamed.
    assert block.endswith("\n")
    assert not block.endswith("\n\n")


def test_the_number_in_the_heading_does_not_look_like_a_citation() -> None:
    """``[1/40]`` and ``[1]`` were the same shape, which is most of what made the
    old output hard to scan."""
    block = batch.render_block(7, 40, "Вопрос?", "Ответ.")

    assert "[7/40]" not in block
    assert block.splitlines()[1].startswith("7/40")


def test_styling_never_touches_the_answers_own_text() -> None:
    """The ``[1]`` in the reply is the model's, and it has to keep matching the
    ``[1]`` in the source list. Restyling it would break the only correspondence
    that makes either of them checkable."""
    answer = (
        "Высота 800 м [1] «фраза».\n\n"
        "Источник:\n"
        "[1] kb/a.pdf:1-5 — s · chunk c1"
    )

    block = batch.render_block(1, 1, "Вопрос?", answer)

    assert "Высота 800 м [1] «фраза»." in block
    assert "    [1] kb/a.pdf:1-5" in block


def test_an_answer_with_no_sources_keeps_its_own_last_line() -> None:
    """An earlier version trimmed "the last line if it did not end in a full stop",
    which ate a closing quotation mark and part of the sentence with it."""
    answer = "Ответ, который заканчивается цитатой «вот так»"

    body, entries = batch.split_sources(answer)

    assert body == answer
    assert entries == []


def test_the_source_heading_says_singular_or_plural_by_the_count() -> None:
    """One entry under «Источники» read like a list that lost its neighbours."""

    def entries(count: int) -> str:
        head = "Источник:\n" if count == 1 else "Источники (фрагменты из ответа):\n"
        return "Ответ.\n\n" + head + "".join(f"[{n}] kb/f{n}.md\n" for n in range(1, count + 1))

    assert "  Источник:" in batch.render_block(1, 1, "Вопрос?", entries(1))
    assert "  Источники:" in batch.render_block(1, 2, "Вопрос?", entries(2))
