"""Спросить бота список вопросов из файла и показать только ответы с источниками.

Зачем отдельный скрипт, если ``python -m llm_bot`` и так отвечает на один
вопрос: при десяти вопросах копировать команду десять раз — это десять
копипастов одного и того же длинного вызова, и в восьмой раз аргумент
опечатается.

    python scripts/batch_ask.py --questions q.txt \\
        --rag-index data/emb/index_structure.json

Вопросы в файле — по одному в строке, ``#`` начинает комментарий, пустые
строки игнорируются. Поддерживаются также ``.yaml``/``.json`` со списком
вопросов, так что уже подготовленные файлы вопросов переиспользуются без правки.

    python scripts/batch_ask.py --questions questions.yaml \\
        --rag-index data/emb/index_structure.json --rag-strict --rag-rerank

Фильтрация вывода здесь не нужна, и это не совпадение: CLI уже разводит потоки.
Ответ уходит в stdout, вся диагностика — ``[tokens]``, ``[memory]``, ``[rag]``,
``Loading weights`` и баннер сессии — в stderr. Скрипт читает stdout и
отбрасывает stderr целиком. Отсюда важное следствие: **ошибка модели тоже
уходит в stderr** и в вывод не попадёт, поэтому непустой stderr показывается
предупреждением, а не теряется.

Каждый вопрос — отдельный процесс, как если бы команда была вбита руками.
Замерено на этом корпусе: из ~40 секунд на вопрос около 35 уходит на загрузку
двух моделей с диска, а сама работа занимает меньше секунды.

    импорт llm_bot                 0.4с
    загрузка модели реранкера     16.4с
    первый запрос (эмбеддер)      19.0с
    второй запрос                 0.6с

Поэтому есть ``--reuse-models``: модель эмбеддингов и реранкер грузятся один раз
на весь прогон, а сессия — новая на каждый вопрос. Результат тот же, а сорок
вопросов укладываются в полторы минуты вместо получаса. Ретривер здесь
stateless — единственное, что в нём меняется между вызовами, это ленивая
загрузка самой модели эмбеддинга, а не состояние от вопроса, — так что общий
ретривер не может утечь из одного вопроса в другой.

По умолчанию оставлен подпроцессный режим: он гарантированно совпадает с
ручным запуском, а отличия ``--reuse-models`` стоит понимать, а не угадывать.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from llm_bot.rag import DEFAULT_TOP_K, SOURCES_HEADING  # noqa: E402
from llm_bot.rerank import DEFAULT_RERANK_CANDIDATES  # noqa: E402

QUESTION_HELP = (
    "Файл с вопросами: .txt/.md — по одному в строке, # начинает комментарий; "
    ".yaml/.json — список вопросов (строки или объекты с ключом text/question)."
)


def load_questions(path: Path) -> list[str]:
    """Read the questions, whichever of the three shapes the file happens to use.

    Plain lines and YAML lists both end up here because both are things people
    actually have lying around: a scratch file of questions, or the
    ``questions.yaml`` a previous run already wrote. Rejecting either would only
    mean converting it by hand for no reason.
    """
    if not path.is_file():
        raise FileNotFoundError(f"файл с вопросами не найден: {path}")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if isinstance(raw, dict):
            raw = raw.get("questions", [])
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("questions", [])
    else:
        questions = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not questions:
            raise ValueError(f"{path}: вопросов не найдено")
        return questions

    if not isinstance(raw, list):
        raise ValueError(f"{path}: ожидался список вопросов")
    questions: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("question") or "").strip()
        else:
            text = str(item).strip()
        if text:
            questions.append(text)
    if not questions:
        raise ValueError(f"{path}: вопросов не найдено")
    return questions


class WarmRunner:
    """Asks questions in this process, paying the model load once.

    The expensive part of a batch is not the questions. Measured on a 40-question
    run: ~16s to load the cross-encoder, ~19s more for the embedding model on its
    first query, and 0.6s for every query after that. A separate process per
    question pays all of it 40 times over.

    Sharing the retriever is safe and the session is still rebuilt per question,
    so the only thing reused is the pair of immutable models. :class:`Retriever`
    holds nothing derived from a query — its single lazily-assigned attribute is
    the embedder itself — so one retriever gives the same answer N times as N
    retrievers would.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        from llm_bot.factory import make_session
        from llm_bot.json_session_store import JsonSessionStore
        from llm_bot.memory_store import JsonMemoryStore
        from llm_bot.rag import Retriever
        from llm_bot.rerank import Reranker
        from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

        self._make_session = make_session
        self._session_store = JsonSessionStore
        self._memory_store = JsonMemoryStore
        self._model_store = YamlModelStore()
        self._agent_store = YamlAgentStore()
        self._agent = args.agent
        self._retriever = None
        if args.rag_index:
            reranker = Reranker() if args.rerank else None
            self._retriever = Retriever(
                args.rag_index,
                # The CLI substitutes the default here rather than letting the
                # retriever do it, so ``None`` has to be resolved the same way.
                top_k=args.top_k if args.top_k is not None else DEFAULT_TOP_K,
                reranker=reranker,
                candidate_k=DEFAULT_RERANK_CANDIDATES,
                cite=not args.no_cite,
            )

    def ask(self, question: str, number: int) -> str:
        """One question, one throwaway session — only the models are reused."""
        workdir = Path(tempfile.mkdtemp(prefix=f"batch-{number}-"))
        session = self._make_session(
            f"batch-{number}",
            self._agent,
            model_store=self._model_store,
            agent_store=self._agent_store,
            session_store=self._session_store(str(workdir)),
            memory_store=self._memory_store(),
            retriever=self._retriever,
            rag_cite=not self._no_cite,
            rag_strict=self._strict,
        )
        return session.chat(question)

    def bind(self, args: argparse.Namespace) -> "WarmRunner":
        self._no_cite = args.no_cite
        self._strict = args.strict
        return self


#: Width of the rule between answers. Fixed rather than terminal-aware: the
#: output is often read in a file after the fact, and a rule that depended on
#: where it was printed would be ragged in the file.
RULE_WIDTH = 68


def split_sources(answer: str) -> tuple[str, str]:
    """``(body, sources)`` of a finished answer.

    The source list is our own output and already separated by a blank line, so
    the reply body is everything above it. Kept apart here rather than by
    pattern-matching the citation markers, which belong to the model and would
    make the styling depend on the answer's own text.
    """
    body, marker, sources = answer.partition(SOURCES_HEADING)
    if not marker:
        # Always a list, so a caller never has to know whether the answer had a
        # source list to learn that it does not.
        return answer.strip(), []
    # Everything above the list is the answer, verbatim. No guessing at where the
    # prose ends: an earlier attempt trimmed "the last line if it does not end in
    # a full stop", which silently ate a closing quotation mark and with it part
    # of the sentence.
    entries = [line for line in sources.splitlines()[1:] if line.strip()]
    return body.rstrip(), entries


def render_block(number: int, total: int, question: str, answer: str) -> str:
    """One question and its answer, laid out to be read rather than parsed.

    Three things were wrong with printing the replies bare. The progress marker
    and the citations looked alike — ``[1/40]`` and ``[1]`` are the same shape —
    and nothing said which question a block belonged to, so a saved file was a
    wall of text with sources in it. The question now heads its own block, the
    rule keeps blocks apart, and the sources are indented under their own
    heading.

    The answer's own text is left untouched. Its ``[1]`` markers are the model's
    and they have to keep matching the numbers in the source list; restyling
    them would break the one correspondence that makes either checkable.
    """
    lines = ["─" * RULE_WIDTH, f"{number}/{total}  {question}", ""]
    body, entries = split_sources(answer)
    lines.append(body)
    if entries:
        heading = SOURCES_HEADING + ("и:" if len(entries) > 1 else ":")
        lines += ["", f"  {heading}", *("    " + line for line in entries)]
    # The trailing gap belongs to the block, not around it. Printed separately it
    # lands wherever the stream interleaving puts it, and the rule for the next
    # question ends up glued to the last source line. One newline here is not a
    # blank line: ``print`` of ``"x\n"`` is what puts the next block a line away.
    return "\n".join(lines) + "\n"


def build_command(args: argparse.Namespace, question: str) -> list[str]:
    """The exact command a person would have typed, as a list."""
    command = [
        sys.executable,
        "-m",
        "llm_bot",
        "--agent",
        args.agent,
    ]
    if args.rag_index:
        command += ["--rag", "--rag-index", str(args.rag_index)]
        if args.rerank:
            command.append("--rag-rerank")
        if args.top_k is not None:
            command += ["--top-k", str(args.top_k)]
        if args.strict:
            command.append("--rag-strict")
        if args.no_cite:
            command.append("--rag-no-cite")
    if args.model:
        command += ["--model", args.model]
    return [*command, question]


def _transient(stderr: str) -> bool:
    """Whether the failure is worth another go.

    A batch asks the same provider the same question back to back, which is
    exactly what trips a rate limit: measured on a 40-question run, one question
    came back ``429 Rate limit exceeded`` after the client had already retried
    four times. Treating that as fatal loses a question that would have answered
    a moment later, so a limit and a timeout are retried and anything else is
    left to fail visibly — retrying a bad index would only cost minutes.
    """
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in ("429", "rate limit", "rate_limit", "timeout", "timed out", "connection")
    )


def ask_with_retries(
    command: list[str], timeout: float | None, retries: int
) -> tuple[str, str, bool]:
    """Run one question, retrying only the failures that can plausibly heal."""
    stderr = ""
    for attempt in range(retries + 1):
        answer, stderr, broke = ask(command, timeout)
        if answer and not broke:
            return answer, stderr, False
        if attempt == retries or not _transient(stderr):
            return answer, stderr, True
        wait = 5.0 * (attempt + 1)
        print(
            f"        повтор через {wait:.0f}с ({stderr.splitlines()[0][:70] if stderr else 'без ответа'})",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(wait)
    return "", stderr, True


def ask(command: list[str], timeout: float | None) -> tuple[str, str, bool]:
    """Run one question. Returns ``(stdout, stderr, failed)``.

    A non-empty stderr is not an error — that is where the token accounting goes
    on every run — so failure is decided by the exit code and by whether anything
    reached stdout at all. A provider error prints to stderr and exits non-zero,
    and silently dropping it would leave a hole in the output that looks like a
    refusal rather than like a crash.
    """
    try:
        done = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"таймаут {timeout}с", True
    return done.stdout.strip(), done.stderr.strip(), done.returncode != 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Спросить бота список вопросов из файла, показав только ответы.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Пример:\n"
            "  python scripts/batch_ask.py --questions q.txt \\\n"
            "      --rag-index data/emb/index_structure.json\n\n"
            "Диагностика CLI уходит в stderr и здесь отбрасывается; чтобы увидеть\n"
            "её целиком, запустите с --debug."
        ),
    )
    parser.add_argument("--questions", required=True, type=Path, help=QUESTION_HELP)
    parser.add_argument("--agent", default="researcher", help="Агент из agents.yaml.")
    parser.add_argument(
        "--rag-index",
        type=Path,
        default=None,
        help="Путь к index_structure.json. Без него вопросы уйдут без RAG.",
    )
    parser.add_argument("--model", default=None, help="Модель из models.yaml.")
    parser.add_argument("--top-k", type=int, default=None, help="Сколько чанков в блок.")
    parser.add_argument(
        "--rag-rerank",
        dest="rerank",
        action="store_true",
        default=True,
        help="Переранживать выдачу кросс-энкодером (по умолчанию включено).",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Без переранжирования: заметно быстрее, качество ниже.",
    )
    parser.add_argument(
        "--rag-strict",
        dest="strict",
        action="store_true",
        default=False,
        help="Заменять неподкреплённый ответ отказом.",
    )
    parser.add_argument(
        "--rag-no-cite",
        dest="no_cite",
        action="store_true",
        default=False,
        help="Убрать цитаты из ответа (несовместимо с --rag-strict).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Сохранить ответы в файл.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Таймаут на один вопрос, секунд (0 — не ограничивать).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Остановиться на первом вопросе, где CLI вернул ненулевой код.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help=(
            "Сколько раз повторить вопрос при лимите провайдера или таймауте "
            "(по умолчанию 2). Ошибки, которые не пройдут повтором, "
            "показываются и не останавливают прогон."
        ),
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help=(
            "Грузить модели эмбеддингов и реранкера один раз на весь прогон "
            "вместо перезагрузки на каждый вопрос. Сессия всё равно новая на "
            "каждый вопрос. Ускоряет 40 вопросов с ~27 минут до полутора."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Показать stderr каждого запуска: команды, токены, память.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать команды, не выполняя их.",
    )
    parser.add_argument(
        "--no-sep",
        dest="sep",
        action="store_false",
        default=True,
        help="В режиме --plain не печатать пустую строку между ответами.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Показывать на stderr, какой вопрос сейчас считается и сколько он "
            "занял. По умолчанию тихо: в stdout идут только ответы, а в stderr — "
            "итоговая строка. Молчание полезно с --reuse-models, где вопрос "
            "занимает секунды; с подпроцессным режимом его включают руками."
        ),
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Только ответы, без разделителей, заголовков и вопросов. Для "
            "перенаправления в другой инструмент, а не для чтения."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.strict and args.no_cite:
        print(
            "error: --rag-strict несовместим с --rag-no-cite: без цитат проверять нечего.",
            file=sys.stderr,
        )
        return 2

    try:
        questions = load_questions(args.questions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    timeout = args.timeout or None
    answers: list[str] = []
    failed = 0
    started = time.perf_counter()

    warm = WarmRunner(args).bind(args) if args.reuse_models else None
    shown = 0
    for number, question in enumerate(questions, start=1):
        command = build_command(args, question)
        if args.dry_run and not warm:
            print(" ".join(command))
            continue
        if args.progress and len(questions) > 1:
            print(
                f"вопрос {number}/{len(questions)}…",
                file=sys.stderr,
                flush=True,
            )

        asked = time.perf_counter()
        if warm is not None:
            answer, stderr, broke, retries = "", "", False, args.retries
            for attempt in range(retries + 1):
                try:
                    answer = warm.ask(question, number).strip()
                    stderr, broke = "", False
                except Exception as exc:  # noqa: BLE001 - one question, not the batch
                    stderr, broke = f"{type(exc).__name__}: {exc}", True
                if answer and not broke:
                    break
                if attempt == retries or not _transient(stderr):
                    break
                wait = 5.0 * (attempt + 1)
                print(
                    f"        повтор через {wait:.0f}с ({stderr[:70]})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        else:
            answer, stderr, broke = ask_with_retries(command, timeout, args.retries)
        took = time.perf_counter() - asked

        # Printed as soon as it arrives. Buffering the whole run and printing it
        # at the end looked like a hang: at roughly 25 seconds a question, forty
        # questions is a quarter of an hour of silence followed by a wall of text.
        if broke or not answer:
            failed += 1
            print(f"error: вопрос не отвечен ({question})", file=sys.stderr, flush=True)
            if stderr:
                print(stderr, file=sys.stderr)
            if args.fail_fast:
                break
            block = f"── не отвечен: {question}"
        else:
            if args.debug and stderr:
                print(stderr, file=sys.stderr)
            block = (
                answer
                if args.plain
                else render_block(number, len(questions), question, answer)
            )
        print(block, flush=True)
        answers.append(block)
        shown += 1 if not broke else 0
        if args.progress and len(questions) > 1:
            print(f"  {took:.0f}с", file=sys.stderr, flush=True)

    if args.dry_run:
        return 0

    if args.out:
        # A styled block carries its own trailing gap, so it joins with a single
        # newline; bare answers need the gap added back, which is what --no-sep
        # turns off.
        joiner = "\n" if not args.plain else ("\n\n" if args.sep else "\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(joiner.join(answers).rstrip() + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(
        f"[batch] {shown}/{len(questions)} ответов, {elapsed:.0f}с",
        file=sys.stderr,
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())