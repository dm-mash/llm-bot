# Retrieval: the local index and RAG

## Local document index (chunking + embeddings)

`scripts/index_documents.py` builds a local JSON index over a directory of
documents: it chunks them two ways, embeds the chunks with a local CPU model
(`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, ~470 MB download, 128-token
window) and reports which chunking strategy retrieves better.

```bash
# index the markdown of this repo (first run downloads the model)
python scripts/index_documents.py --input_dir . --extensions .md

# no network: use only the local HuggingFace cache
python scripts/index_documents.py --input_dir . --extensions .md --local_only

# code as well (.py files are chunked by top-level def/class), custom output
python scripts/index_documents.py --input_dir . --extensions .md,.py \
    --index_dir data/emb --out results/index_documents_report.md

# add hand-written questions to the auto-generated benchmark
python scripts/index_documents.py --input_dir . --extensions .md \
    --eval index_queries.example.yaml
```

### Two chunking strategies

|                    | `fixed_size`                                            | `structure`                                                    |
| ------------------ | ------------------------------------------------------- | -------------------------------------------------------------- |
| Boundaries         | sliding windows of `--chunk_size` tokens, snapped to word edges | Markdown headings, top-level `def`/`class`, PDF pages, paragraph runs |
| Size control       | exact: `--chunk_size` tokens, `--chunk_overlap` reused  | `--structure_max_tokens` budget (default: the model window minus `[CLS]`/`[SEP]`); an oversized section is packed on paragraphs and bisected on line boundaries |
| `section` metadata | empty                                                   | the section path (`"A > B"`), so a hit points at a real section |
| Weakness           | cuts sections in half — a heading query matches a fragment | short sections stay short (`--structure_min_chars`); a one-line section (a Markdown table row) cannot be split at all |

Sizes are counted in **tokens of the embedding model**, not characters: a
1200-character Russian chunk is ~320 MiniLM tokens and everything past 256 is
silently dropped by the model. That is what the report's
`доля чанков длиннее окна модели` row measures, and why the original
"1000 characters per chunk" default was wrong.

### Output

- `index_<strategy>.json` — chunks, L2-normalized embeddings, metadata;
- `comparison.json` — the same numbers for both strategies, machine-readable;
- a markdown report: corpus summary, metadata sample, size/coverage/duplicate
  statistics and the retrieval benchmark.

Metadata per chunk: `chunk_id`, `source`, `title`, `section`, `section_title`,
`document_id`, `chunk_position`, `start_line`, `end_line`, `char_start`,
`char_end`, `char_count`, `token_count`, `content_hash`, `chunking_strategy`.
The `char_*` offsets are exact spans of the source file, so a hit can be
verified by span overlap and a chunk can be re-read straight from disk.

### Asking the index by hand

```bash
# one-off: build the index and ask it in the same run
python scripts/index_documents.py --input_dir . --extensions .md \
    --query "GigaChat OAuth2 token" --top_k 3

# afterwards: ask the stored index, no re-chunking and no re-embedding
python scripts/index_documents.py --reuse --strategy structure --top_k 3 \
    --query "rolling summary сжатие истории" --local_only
```

```
[structure] GigaChat OAuth2 token
  0.626  README.md:162-183  llm-bot > Configuration > Example: GigaChat (Sber)
       ### Example: GigaChat (Sber)
  0.486  plans/details.md:85-89  ... > Changes > 4. `llm_bot/gigachat.py`
       ### 4. `llm_bot/gigachat.py`
```

`--query` is repeatable, `--top_k` sets how many hits are printed, `--strategy`
picks the index to search. `--reuse` takes the model from the index itself (a
`--model` that contradicts it is a hard error — vectors of two models are not
comparable) and refuses to run without `--query`.

The two strategies differ exactly where it matters: for the same question
`fixed_size` returns the neighbouring "Ollama server" fragment, `structure`
returns the whole GigaChat section. Every hit carries `file:line`, so verify it
with `sed -n '162,183p' README.md`.

#### Which model, and why not E5

The default is `paraphrase-multilingual-MiniLM-L12-v2` because it was measured
against `all-MiniLM-L6-v2` on this corpus at an equal chunk budget
(`--chunk_size 120`, both strategies, reproduce with `--model all-MiniLM-L6-v2`).
On the 7 hand-written Russian questions the multilingual model answered 43% at
rank 1 (`structure`), the English-only MiniLM 0% — while the 117
heading-derived queries of the full benchmark reported 72% vs 49% and hid the
gap almost completely. Model choice mattered far more for real questions than
the average suggested, which is the argument for keeping `--eval` questions in
the benchmark.

`intfloat/multilingual-e5-*` retrieves better still, but it is only trained to
work with `query: ` / `passage: ` prefixes. This indexer encodes passages and
questions through one path, so an E5 model here would either be fed the wrong
prefix (silently worse numbers) or need prefix plumbing in every call site. Not
worth it for a local index; keep it in mind if precision becomes the goal.

### Benchmark

Every section heading becomes a query whose gold answer is that section's exact
character span; a retrieved chunk counts as a hit when it overlaps the span.
Reported as `recall@1/3/5` and `MRR@10`. `--eval` adds domain questions on top
(see `index_queries.example.yaml`); an unknown `source` or an ambiguous
`section` is a hard error, so the file cannot silently rot.

```python
import json, math

index = json.load(open("data/emb/index_structure.json", encoding="utf-8"))
query = index["chunks"][0]          # in production: embed a real question
ranked = sorted(
    ((math.fsum(a * b for a, b in zip(row["embedding"], query["embedding"])),
      row["metadata"]) for row in index["chunks"]),
    key=lambda pair: pair[0],   # key обязателен: при равных score сравниваются
    reverse=True,               # dict-ы, и сортировка падает с TypeError
)
for score, meta in ranked[:3]:
    print(f"{score:.3f} {meta['source']}:{meta['start_line']}-{meta['end_line']}"
          f"  {meta['section']}")
```

Vectors are normalized, so a dot product *is* cosine similarity.

Useful flags: `--index_dir` (default `data/emb`, git-ignored), `--out`
(empty string disables the report), `--strategy both|fixed_size|structure`,
`--unit tokens|chars`, `--structure_min_chars`, `--structure_max_chars`
(secondary cap, off by default), `--query`/`--top_k`/`--reuse`, `--model`,
`--batch_size`, `--max_docs`, `--dedup`, `--local_only`. `.pdf` is supported with the optional `pypdf`
dependency; vendored trees (`.venv`, `node_modules`, `data/`, `results/`, …)
are never indexed.

`--pdf_max_pages N` keeps only the first `N` pages of each PDF (`0` = all). It
exists because a one-page service description repeats its terms verbatim on
pages 2–4: in a measured 13-file corpus those pages were 80% of the volume and
0.90–1.00 similar to each other, while page 1 held the only unique facts. Truncating
quietly would make "the fact is not in the corpus" and "the fact was never
indexed" look identical, so the report states how many pages were left out.
Line numbers in a PDF citation (`file.pdf:12-38`) are positions in the text
extracted by pypdf, not in the original file; the chunk metadata carries
`page 1` alongside them.

Tests: `tests/test_index_documents.py` (offline, no model download).

## RAG: ответы по локальному индексу

The index above is a library. `llm_bot/rag.py` makes it answer questions:
`Retriever` searches a prebuilt `index_*.json` with cosine similarity, formats
the best chunks as a `file:line`-traceable context block, and hands it to the
model as an extra per-request prefix.

```bash
# answer from the repo index (build it first, see above)
python -m llm_bot --agent researcher --rag \
    --rag-index data/emb/index_structure.json \
    "Что такое RAG и как он устроен в этом проекте?"
```

`--rag` works with **any** agent — `--agent researcher` above is a ready-made
low-temperature agent from `data/agents.yaml`, not a requirement. The grounding
rule is carried by the block itself, identically for every agent.

| flag | default | meaning |
| ---- | ------- | ------- |
| `--rag` | off | enable retrieval for this run; without it the bot behaves exactly as before |
| `--rag-index` | `data/emb/index_structure.json` | which index to search; the embedding model is taken from the index itself, and a `--model` contradicting it is a hard error |
| `--rag-top-k` | `4` | how many chunks reach the prompt |
| `--rag-rerank` | off | re-score a wider shortlist with a cross-encoder first; extra model load, zero prompt tokens |
| `--rag-rerank-candidates` | `50` | shortlist handed to the reranker; irrelevant without `--rag-rerank` |
| `--rag-rerank-min-score` | unset | optional threshold; see below for why it is off by default |
| `--rag-max-tokens` | unset | hard cap on the context block; unset means "derive it from the context window" |

`--rag-rerank` needs `--rag`; on its own it exits with an error rather than
quietly doing nothing, so "the reranker found nothing" never gets confused with
"the reranker never ran".

### Why the second stage exists

Dense search compares one vector per chunk, so a chunk that is on-topic for a
whole document outranks the chunk that states the fact. On a measured
near-duplicate corpus the right chunk sat at rank 11 for a question about one
service's height and weight limits — present, close, outside `top_k`. Measured
over the 34 answerable questions:

| | retrieval@1 | retrieval@4 | fact coverage | PASS |
| --- | --- | --- | --- | --- |
| dense only | 13/34 | 22/34 | .662 | 23/40 (.575) |
| + cross-encoder | 20/34 | **32/34** | **.971** | **31/40 (.775)** |

The retrieval columns are unaffected by later work; the `PASS` column was
re-measured once answers also had to carry a source and a verbatim quote, and it
fell from 36/40 to 9/40. That fall was the point, not a regression: the earlier
`PASS` rewarded naming a line range, which says nothing about wording, so 36/40
counted answers whose facts nobody had checked. With the stricter rule the
source list and a confirmed citation were in every answer and a quote came back
verbatim in only **5 of 34** — the entire cost of the rule was paid by answers
that were probably right.

That also made the source list worth trusting less than it looked. It was built
from every chunk sent, so a four-chunk block whose answer cited two printed four
sources. Fixed below; the numbers here predate the fix only in that column.

Asking better fixed most of that. See
[Asking for the phrase](#asking-for-the-phrase-made-the-quotes-appear) — one
worked example in the protocol took verbatim quotes from 5/34 to 26/34 and `PASS`
from 9/40 to 31/40, and the reranker now shows the margin it was added for.

Two defaults come from measurements. The passage handed to the cross-encoder is
prefixed with `source — section`, which is worth three questions out of 34 on
its own: the corpus is full of near-duplicate documents, and the filename is
what tells them apart, not the body text.

The shortlist is **50** wide. The 34 questions above live on a 17-chunk corpus,
so a 20-chunk shortlist already held every chunk it had (recall@20 was 34/34
against 22/34 at rank 4) — width never bound there. It binds on a larger index,
and 20 was too small: on the **742-chunk repository index** the gold chunk for
«как устроен планировщик задач в проекте?» sat at dense rank **32**, so the
reranker never saw it and the answer came from unrelated task-planning pages. At
50 that question is re-scored to rank 1, and the repo question set goes from
2/5 to 4/5 retrieved. On the 17-chunk corpus the shortlist is capped by the
corpus either way, so nothing changes there and no latency is added where 20
already covered everything.

### Four chunks is too few for a question about the whole catalogue

`--rag-top-k` defaults to 4. On the coffee corpus the drinks menu is chunk 8 of
17 for the question «привет, какой кофе есть?» — 0.390 against 0.457 for the
document's own header, so the ranking barely distinguishes them. At `top_k=4` the
menu is cut, the model gets syrups and milk rules, and answers «полного списка
напитков нет» — which is true of the block it was handed and false of the corpus.
`--rag-top-k 12` returns the whole menu, correctly, on the first try.

The reranker does not fix this and on this class of question makes it worse: the
menu stays 8th while syrups, milk and the allergy page rise above it. The measured
0.647 → 0.941 gain comes from fact-shaped questions; on «что вообще есть» it
loses. The word "напиток" appears in the syrup and milk *prose* and only as a
table header in the menu, and the cross-encoder scores on exactly that.

This is a general shape, not a tuning accident: a question about the catalogue
matches no particular fragment well, so it sits in the middle of a vector ranking
that is trying to rank fragments.

### Why there is no threshold

There is `--rag-rerank-min-score`, and it is off by default because measuring it
said so. The two score distributions overlap: the lowest-scoring correct chunk
sits at **-4.0**, while an unanswerable question about a product that is
genuinely absent from the corpus peaks at **+0.7** — above ten correct answers,
because the document set really is about that product. Replaying one run's
shortlist at several thresholds, reading the scores back off the saved run:

| threshold | answerable kept | traps still fed | chunks per question |
| --- | --- | --- | --- |
| off | 32/34 | 6/6 | 4.0 |
| -4 | 30/34 | 4/6 | 3.2 |
| -2 | 26/34 | 4/6 | 1.9 |
| 0 | 23/34 | 2/6 | 1.1 |
| +2 | 17/34 | 0/6 | 0.5 |

The filter removes correct answers long before it removes misleading ones. It
is available, it is measured, and it is not the default.

The embedding model (~470 MB) is downloaded on the first `--rag` run and cached
afterwards; the runtime then loads it from the cache and never asks the Hub,
so an offline bot starts without the connect-timeout stall. The reranker
(`--rag-rerank`) is loaded the same way. The `--local_only` flag of
`scripts/index_documents.py` is a separate knob for the indexer.

The block is placed **after** the MCP/tool context and **before** the strategy
prefix, so tool output stays closest to the question and the retrieval block
reads as part of the same evidence. Its budget is `min(--rag-max-tokens, 25% of
the agent's context window)`. Chunks are added worst-score-first and dropped
again if the next one would overflow the budget; a chunk is only included if at
least one whole word of it fits, so the model never sees a fragment cut mid
sentence.

Grounding is *asked for* by instruction: the block tells the model to answer
**only** from the context, to cite the chunk's number after every fact, to quote
the phrase it took from that chunk, and to say it did not find an answer rather
than invent one. That rule lives in the
block (`_RAG_PROTOCOL` in `llm_bot/rag.py`) and nowhere else — not in
`data/invariants.yaml`, and not in any agent's `system_prompt`. Two reasons:

- an invariant would be global, so it would force the agent to refuse everything
  in `--no-rag` mode, when there is no context to ground in;
- a copy inside an agent would be a third source of truth. It already drifted
  once, which is why the agent configs no longer restate the rule.

### Asking a list of questions

One question is `python -m llm_bot --agent … --rag --rag-index … "вопрос"`. Ten
of them is ten copies of the same long invocation, and the eighth is where an
argument gets mistyped. `scripts/batch_ask.py` runs the list for you and prints
only the answers:

```
python scripts/batch_ask.py --questions q.txt \
    --rag-index data/emb/index_structure.json --rag-strict --rag-rerank
```

The questions file is one per line, `#` starts a comment, blank lines are
skipped. `.yaml` and `.json` lists work too, so a `questions.yaml` from an
earlier run is reused as it is.

Each answer gets its own block, headed by the question and divided by a rule:

```
────────────────────────────────────────────────────────────────────
7/40  С какой высоты выполняется прыжок с парашютом?

С высоты 800 м [1] «прыжки проводятся с высоты 800 м».

  Источники:
    [1] Прыжок с парашютом.pdf:19-23 — page 1 · chunk structure-006-002
```

The question heads the block because a saved file of answers with no trace of
which question each belongs to is not readable afterwards. The answer's own text
is never restyled — its `[1]` has to keep matching the `[1]` below it, and that
correspondence is the only thing either of them is checkable against. `--plain`
prints the bare replies instead, for piping into something else.

stderr is quiet by default: the answers, and one summary line at the end. A
per-question marker on stderr was tried and removed — the block already carries
its number, so the line only repeated what the next block was about to say, less
clearly and sooner. `--progress` brings it back, which is worth having in the
subprocess mode, where a question takes forty seconds and there is otherwise
nothing to look at.

Each answer is printed as soon as it arrives. Buffering the run and printing it
at the end was tried first and looked like a hang: at the rate below, forty
questions is a quarter of an hour of silence followed by a wall of text. `--out`
writes the same blocks to a file, which is the form to read after the fact.

There is no output filtering to get wrong, and that is a property of the CLI
rather than luck: the answer goes to stdout, and every diagnostic — token
accounting, memory, the `[rag]` chunk list, `Loading weights`, the session
banner — goes to stderr. The script reads stdout and drops stderr. One
consequence is worth knowing: a provider failure prints to stderr and exits
non-zero, so it would not show up in the answers at all. The script therefore
treats a non-empty answer with a non-zero exit as a failure, prints the stderr
for that one question, carries on, and reports the count at the end.

Each question is its own process, exactly as if the command were typed by hand.
That is deliberate — same code, same flags, no second way of building a session
that would drift from the CLI — and it is also why a batch is slow. Measured on a
40-question run, of roughly 40 seconds per question:

| | |
|---|---|
| import `llm_bot` | 0.4s |
| load the cross-encoder | 16.4s |
| first query (embedding model loads here) | 19.0s |
| every query after that | 0.6s |

Thirty-five seconds of forty is disk. `--reuse-models` loads both models once for
the whole run and still builds a **new session per question**, so only the two
immutable models are shared and nothing can leak from one question into the next:
40 questions go from about 27 minutes to 3m17s. Verified by diffing both modes —
they differ by the same two lines as two runs of one mode differ from each other,
which is sampling noise, not behaviour.

A batch also asks one provider the same question back to back, which is what
trips a rate limit: one question in forty came back `429` after the client had
already retried four times. `--retries` (2 by default) repeats those, with a
pause, and leaves anything that would not heal to fail visibly instead of
burning minutes on a bad index.

### Three things every answer is checked for

A line range says which chunk a fact came from. It says nothing about how the
fact was worded, and that gap is where answers go wrong: the model can cite a
real chunk, cite the right lines, and still state the number wrong, with every
existing check passing. So the answer is checked three ways, and none of the
three needs a second model:

| | what it means | how it is verified |
|---|---|---|
| **a source** | the answer ends with the list of chunks it actually leaned on | `render_sources_footer` appends it in code — it is not asked for, so it cannot be forgotten |
| **a confirmed citation** | every `[N]` names a chunk that was really sent | `audit_citations(..., handles=…)` resolves the number to the hit it was printed above |
| **a verbatim quote** | the phrase in «» or backticks is contained, character for character, in the chunk it is attributed to | `audit_quotes` is a substring test against that chunk's text |

A claim counts as grounded only when all three hold. Anything else is
`Grounding.UNGROUNDED`, and `Session.last_grounding` says which check failed.

Measured on 10 questions against the repository's own documents, and then on a
harder near-duplicate corpus of 34 answerable ones:

| | source list | citation confirmed | quote verbatim | grounded |
|---|---|---|---|---|
| repo + kb, 8 answerable | 7/8 | 7/8 | 7/8 | 7/8 |
| near-duplicate corpus, dense, 34 | 31/34 | 31/34 | 26/34 | 24/34 |
| near-duplicate corpus, + reranker | 34/34 | 34/34 | 31/34 | 29/34 |

Two things about that list are worth stating outright, because both were wrong
at first.

It lists the chunks the answer **cited**, not the chunks retrieval returned. A
four-chunk block where the model cited two does not have four sources, it has
two, and printing the rest claims provenance the reply never had — the extra
lines being exactly the ones a reader cannot tell apart from real support. The
printed numbers are the chunk's own, never renumbered: a `[3]` in the text and
`[3]` in the list must be the same chunk, even when the answer never said `[1]`.

An answer that cited nothing gets no list. A refusal has no evidence, and a
source list under «I did not find it» states the opposite of what was said.

One consequence is not visible in the output and is worth knowing about. A
follow-up turn is kept on the document the dialog is already about, by reading
the citations out of the visible history. A handle names a position in a block
the reader never saw, so `[2]` is not a document — the file name comes from this
same list, which prints the location for exactly that handle. And the list is
excluded from that reading, because otherwise the reply's own subject came back
as the bare string `"2"` and biased the next search towards a filename that does
not exist.

The first two columns are near-perfect because code produces them. The third took
three rewrites of the protocol to get off the floor, and it is the column the
mechanism exists for — see below.

### Asking for the phrase made the quotes appear

An attempt to fix the long-dialogue case by repeating the protocol as a system
message immediately before each question, next to the invariant reminder, is
recorded here because it did not work: on one 14-turn dialogue the strict column
rose from 1 to 6 while every single turn degenerated, and on the other it went
from 0 to 2. Whether the extra system message provoked the looping is not
established — the loop had already been observed without it, at one run per
arm. See "the model repeats itself" below.

The third check sat at **5 of 34** for a long time, which looked like a limit of
the corpus rather than a limit of the prompt. It was the prompt. Asking for a
phrase in the abstract produced a verbatim quote in 15 of 32 answers; the same
protocol with one worked example produced one in 24 of 32.

Four wordings, same 10 questions, 4 runs each:

| protocol | quote verbatim | grounded |
|---|---|---|
| abstract request, no example | 15/32 (.47) | 13/32 (.41) |
| abstract, short version | 23/32 (.72) | 14/32 (.44) |
| placeholder example `«фраза из фрагмента» [1]` | 17/24 (.71) | 15/24 (.62) |
| **concrete example `срок — 6 месяцев [1] «срок — 6 месяцев»`** | **21/24 (.88)** | **19/24 (.79)** |

Three things are worth keeping out of that table, and one is a warning.

The example has to be *concrete*. A placeholder that names the shape without
content did score better than nothing, and nowhere near as well as a real
example — and this repo has the scar to prove why. An earlier placeholder of
exactly that shape, `[файл:строки]`, was copied into a saved run as
`[файл: kofeinya_zerna.md:14]`: a citation to a file that name, whose range
pointed at nothing.

The example has to be *neutral*, since this repository carries no customer's
documents. That is why it is a deadline and not the jump height that first
seemed like the obvious illustration. Checked for leakage rather than assumed:
24 answers, zero occurrences of the example's text.

The example's *order* matters and is not symmetric. `«фраза» [1]` grounded
15/24, the same words as `[1] «фраза»` grounded 19/24. The audit accepts a
citation on either side of a quote, so this is compliance, not a blind spot.

And `grounded` is not `quote verbatim`: the short protocol quoted almost as often
as the concrete one (23/32 against 21/24) but grounded far less (14/32 against
19/24), because quoting while still citing something the block never sent is
worse than not quoting. A protocol that makes the model quote is not thereby a
protocol that makes it right.

The check reports what it could *not* verify too. `QuoteAudit.unchecked` holds
quoted fragments that no citation claimed — usually the customer's own wording,
like «Тростниковый крем». Those are neither passes nor failures, and reporting
them is what keeps a check that skips part of its input from looking like a check
that passes everything.

### Strict mode: answer, or admit nothing

`--rag-strict` replaces an ungrounded answer with a fixed refusal:

```
--rag --rag-strict
```

It is **off by default**, because it is a behaviour switch rather than a check:
it throws away a possibly-correct answer and replaces it with «I cannot answer
from this context. Please clarify the question». On a corpus where a fifth of
answers carry a verbatim quote, turning it on means most questions get no answer
at all — correct, and useless. It is worth it where a wrong price is worse than
no price.

It cannot be combined with `--rag-no-cite`, which removes the very citations and
quotes there would be nothing to verify; that combination is refused at startup
rather than producing a bot that only ever declines.

### Two ways to run it: with sources, without

`--rag` asks the model to tag every fact with the chunk's number and to quote
the phrase it took, then audits both afterwards. When the sources are noise for
the current task, `--rag-no-cite` answers from the same retrieved context without
them:

```
--rag --rag-no-cite
```

| | `--rag` | `--rag --rag-no-cite` |
|---|---|---|
| Retrieval, reranking, token budget | on | on |
| "Answer only from this context" | on | on |
| No-blending rule | on | on |
| "Say you did not find it" | on | on |
| "Tag every fact with `[N]` and quote the phrase" | on | **off** |
| Source list appended by code | on | **off** |
| Citation and quote audits | on | **off** |
| Grounding verdict | on | **off**, and `checked=False` |

Nothing was deleted: the citation format lives in `_RAG_PROTOCOL`, the version
without it in `_RAG_PROTOCOL_NO_CITATIONS`, and `Retriever(cite=False)` picks
between them. Drop the flag to get sources back.

With citations off there is nothing to verify, so the verdict says `checked=False`
rather than blaming the answer — an unjudged answer is never counted as a good
one, which is what keeps the two modes comparable.

### Every answer shows sources, or says why there are none

A list of sources on some turns and nothing on others is the worst of both: the
reader cannot tell an answer without evidence from an answer that hid it. So a
response always ends in one of three states:

| | how it ends | when |
|---|---|---|
| sources used | `Источники (фрагменты из ответа):` + the cited chunks | the answer cites something |
| none found | `Документы не найдены — ответ не подтверждён.` | retrieval ran and returned nothing |
| none used | `Документы не использованы — ответ не подтверждён.` | chunks were sent, none of them cited |

The two notes are separate states because they point at different failures:
"found nothing" is a retrieval gap, "used nothing" is the model ignoring what it
was handed. A single counter would hide the second one.

A refusal gets the note too. It declines in its own words, but the reader still
has to know whether documents were in front of it — that is the difference
between "found nothing" and "refused to answer". The note claims no support, so
unlike a source list it cannot contradict a refusal the way `Источник [1]` under
"не нашёл" would.

With `--rag-no-cite` there is nothing to audit, so no note is added: the flag
means "do not show sources", not "show a note about not showing sources". Same
for `--rag` absent — RAG off is not a claim about documents.

### Which corpus to demonstrate this on

`data/kb_emb` holds **two** documents. That is enough to show that retrieval and
citations work, and not enough to show what they are for: there is nothing to
confuse one document with another, so the hardest part of the mechanism cannot
appear. It also means the block is always a mix of the same two files, and no
amount of tuning changes that — a per-document cap on a two-document corpus has
nothing to diversify and can only truncate. Measured: a cap of 3 kept 13 of 14
expected answers against 14 of 14 with no cap, and the one it lost was a
synthesis question, the same class as the catalogue question `top_k=12` exists to
serve.

`data/certs_emb` holds seven near-identical gift certificates: two of them differ
mainly by whether there are two participants, and one document is a parachuting
certificate whose marketing text sits next to a flight certificate. That is where
the mechanism either earns its place or does not — and it is where the bugs were
found. Demonstrate on certs.

Two retrieval hypotheses were measured on the coffee corpus and both rejected:
cutting at the steepest drop in the score list (it does not exist — the ranking is
flat, and the cuts landed at 15–16 of 17), and capping chunks per document (there
are two documents, so it only truncates).

### Running a whole dialogue

`scripts/eval_dialogue.py` plays a scripted conversation turn by turn against a
local index and grades every turn. A dialogue is YAML in `dialogues/`:

```yaml
agent: researcher
goal: "подобрать десерт без молока и орехов"
clarified:
  - "гость не пьёт молоко"
terms:
  "строго без орехов": "следы миндаля тоже не подходят"
turns:
  - user: "Есть веганские десерты без молока и без орехов?"
    contains: ["молок", "орех"]     # required
  - user: "Молоко он не пьёт."
    any_of: ["молок"]               # at least one
    note: "факт, который должен долежать до конца"
check:
  user: "Напомни, что он не пьёт и что посоветовали."
  contains: ["молок"]               # the closing question
```

```
.venv/bin/python scripts/eval_dialogue.py --dialogue dialogues/coffee.yaml \
    --index data/kb_emb/index_structure.json --runs 1
```

The report separates the closing check from ordinary turns: the check is graded
on what it recalls, with no grounding requirement, because it is a question about
the conversation rather than about a document. Every other turn must carry
sources or a note.

It also grades three different things, because "did it work" is not one question:

| measure | what it asks |
|---|---|
| **Отвечено по существу** | did the turn answer the question that was asked |
| **Источники показаны** | did it show a source list, or declare that there was none |
| Строгое опирание на источники | did every claim carry a citation *and* a verbatim quote from a chunk |

The first is the headline. The third is the strict protocol above, and it is
much the harshest: over a long dialog the model largely stops quoting after the
first few turns, so the strict figure is low even when the run answered
everything correctly. Counting it as *the* verdict made a dialogue that kept
its goal, cited something on every turn and answered every question correctly
report 1/14 — a number that contradicted the rest of its own report. Split, each
turn says what it did and did not do, and the per-turn table marks
`ок · строго нет` for the answers that skipped the quote protocol.

Two things are checked so the headline cannot be satisfied without an answer.

The source footer is **excluded** from the content checks. The footer is
assembled by the script out of chunk paths and section titles, so a keyword can
appear there without the model having said it — in one coffee run the answer to
«А если он хочет взять с собой?» was 95 lines of audit markers and nothing else,
and it passed because `самовывоз` appears in
`delivery.md — Доставка и самовывоз > Самовывоз`. A grader that accepts its own
output is not a grader.

A reply that **repeats itself** fails. Over these dialogs the model falls into a
loop on longer runs — in one coffee run, turns 4 through the closing question
came back byte-identical, 2946 characters each, five unique lines between them,
all of them audit markers. The audits are what made it invisible: they replaced
the repeated sentence with placeholders, so the report showed a reply that
looked damaged rather than one that had never contained an answer. Content
checks also have to be non-empty — a turn with no `contains`/`any_of`/`absent`
passes whatever the bot says, and both shipped fixtures had such turns,
including the two questions that are deliberately not in the corpus.

The point of the closing question is that it is the only place where forgetting
shows up, so `clarified` and `terms` are things the dialog must have actually
established. A test enforces this — a constraint nobody stated would make the
check unfalsifiable, and the run would then fail for memory that was never asked
to hold anything.

### Comparing with and without memory

```
--memory on     # default: extract facts into working / long-term
--memory off    # same dialogue, same history, no extraction
```

Both arms run the same dialogue and keep the history, so a difference in the
closing check is attributable to memory alone rather than to a shorter prompt or
a lost turn. The report prints which arm it was.

Measured on the two shipped dialogues, one run each (`data/results/day25_*.md`):

| dialogue | memory | answered | sources shown | strictly grounded | goal kept |
|---|---|---|---|---|---|
| coffee | off | 14/14 | 14/14 | 7/14 | yes |
| coffee | on | 14/14 | 14/14 | 1/14 | yes |
| certs | off | 13/14 | 14/14 | 1/14 | yes |
| certs | on | 10/14 | 14/14 | 3/14 | no |

Two of these are solid and one is not yet shown.

**Solid.** Sources appear on every turn of every run — a list, or a note saying
there were none. That is the requirement that cannot be met halfway, and it is
met 56/56. The dialogs answer the question that was asked on 10–14 turns out of
14.

**Not yet shown.** That memory is *why* the closing question is answered. An
earlier pair of runs looked like a clean 0/1 → 1/1 on both dialogues, and it was
not: regenerating all four flipped `certs`, and both memory-off runs pass the
check. A single run per arm cannot separate the effect of memory from ordinary
variation, and this dialogue is long enough that one run tells you very little.
Saying otherwise would be reading a coincidence as a result. To claim it, run each
arm several times and compare rates.

The strict column is the one that is genuinely poor — 1–7 of 14 — and it is the
same on both arms, so whatever causes it is not about memory.

A fact only reaches memory on the strength of words the user actually said. The
classifier must quote them, and a quote that is not found in the user's message
is dropped with a reason recorded in the report, rather than silently trusted:

```
[memory] факт отброшен (long_term:allergies): доказательства нет в словах пользователя
```

That check is not decoration. Before it existed, a turn as small as
«привет, как дела?» left «аллергия на орехи» and «две недели» in the guest's
long-term profile, because the classifier had inferred them from nothing, and
the profile is injected into every later turn.

### A handle means nothing outside the turn it was printed in

Handle numbers are assigned per retrieval, so the same number names a different
document from one turn to the next. The model was reusing numbers from earlier
answers without re-reading the block, and that is the one observed way this
system attributed a real fact to a real but wrong chunk:

- turn 1 — `[4]` was `Чудеса на виражах.pdf`, and «участник не пилотирует» was
  indeed from that document;
- turn 2 — same `[4]`, now `Прыжок с парашютом.pdf`, and the answer still carried
  the plane facts, now citing a parachuting document.

The audit cannot see this. The handle resolves, so nothing is flagged, and there
is no quote to compare against the chunk the number now points at. Detecting the
attribution without a quote would mean matching the answer's wording to a chunk
lexically, which is the number-proxy problem this repo already rejected once.

So the protocol states the rule the model was never told: numbers belong to this
block, change from question to question, re-read before citing. A detector was
written off rather than built — handles legitimately remap on nearly every turn,
so "this number meant something else last turn" is also true of correct answers,
and a warning that fires on good turns is worse than none.

On a long dialog the model sometimes stops answering and starts repeating one
short fragment — a citation marker with a short quote — until it runs out. The
audits make it worse: each repetition is replaced by a marker, so the reply the
user sees is thirty copies of `[1] [источник не подтверждён]».`

The part that turned a bad turn into a ruined run is what happens next. That text
goes into the history, so the next turn opens with a page of identical lines as
the most recent thing in context — the exact pattern the model is already in.
One 14-turn coffee dialog held **402** such markers, and from turn 3 onward every
reply was the same 31 markers. The model was not being stubborn; it was being
shown its own pattern back.

So a reply whose lines are 80% one repeated line, over at least twelve lines, is
refused outright rather than displayed. Two reasons, and the second is the point:
a page of repeated markers is not an answer, and refusing leaves a short honest
sentence in the history instead of feeding the loop forward. The threshold is
tight on purpose — a twenty-row table, a list of similar items and a caveat
repeated twice all pass.

This is a guard, not a fix. It stops the damage from compounding; it does not stop
the model from looping, and it does not produce quotes. What it does buy is that
a run stays measurable: every number afterwards comes from a reply that exists.

### Invariant ids are not citations

The prompt renders every invariant as `- [STACK-1] (kind) statement`, and models
often work their checklist into the reply:

```
Проверяю запрос против инвариантов:
- [RULE-LANG]: Вопрос на русском → отвечаю на русском ✓
```

`[STACK-1]` names a rule, not a document. The audit used to count those brackets
as source citations, could not find them in the retrieved block, and replaced
five valid ids with `[источник не подтверждён]` while logging five warnings on an
answer whose facts were all correct. `audit_citations(..., ignore=…)` now takes
the ids from the registry, so they pass through untouched and stay out of both
`kept` and `dropped`. A reply citing nothing but invariants still counts as
uncited, which is what it is.

What code *does* enforce is the citation. `audit_citations` checks every
marker in the reply against the block that was actually sent — a chunk number
resolved through the handles the block printed, or a file name with its line
range — and replaces anything unsupported with
`[источник не подтверждён]`. This is not a formality. On a near-duplicate corpus
**12 of 15** citations to one document came back with a single digit
changed from the name on disk: the model rewrites what it copies, and nothing
noticed. Unsupported citations are marked rather
than deleted, because a wrong source in brackets reads as a verified one. Over a
10-turn dialogue run three times (27/30 correct by content) the audit also caught
12 turns that cited a document the block never contained — the model answering
from its own earlier replies in history rather than from the evidence sent this
turn. `Session.citation_audits` exposes the per-turn result.

That 27/30 is the number this section used to end on, and it is worth being
precise about what it measured: **content**, not grounding. Once the same
dialogue is also required to carry a source and a verbatim quote, **6 of 30
turns pass**. The gap is not twenty-four turns of wrong answers — it is turns
whose facts were right and whose wording nobody checked. A bot trusted with a
price cannot tell the difference, so the evaluation scores the difference rather
than the content.

Two details of the check are deliberate. A citation may name the document by
its id alone (`[ID-250-439-931-337]`) — the model does that — and that is
accepted **only if that id belongs to exactly one document in play**; if two
share it, the id identifies nothing and the citation stays unsupported. And
a factual turn that cites nothing at all is not silently clean: `uncited` is set,
because "no citation" is the case the protocol exists for and there is nothing to
replace in the text.

Whether a citation from an *earlier* turn of the same dialog counts is a policy
call, not a technical one, and both are implemented: `audit_citations(also_backed=…)`
accepts ranges this dialog already verified, and `make_session(rag_reuse_evidence=True)`
turns it on. Measured over the same 10 turns x 3 runs it is 15/30 strict versus
17/30 with reuse, and it does not touch the real defect — the rewritten number
survives either way. It is off by default because the block is rebuilt per turn
on purpose.

### Two rules for documents that look alike

The corpus is 13 one-page descriptions of unrelated services, and two of them
describe the same offering: they share 15 of their 18 indexed lines and differ
in three — the title, one word of the service line, and one number (20 minutes
against 90). Without help the model answers a question about one from the
other's chunk and cites the first, so two mechanisms were added:

- **No blending** (`_RAG_NO_BLENDING`, part of the block): facts belong to one
  document, take them only from the block with the same file name, and when the
  customer has not said which one they mean, show the difference or ask.
- **Subject continuity** (`Session._rag_subject`): the documents already cited
  in this dialog keep a slot in the block. A follow-up like «а он долго
  действует?» names no product at all, and since every document repeats the
  same wording, the customer's own document fell out of the top-4 entirely and
  the answer came from an unrelated business whose document *does* state a
  flat validity period.

Neither of these is a win on a dialog benchmark. Over the same 10 turns x 3 runs
the scores are 28/30, 27/30, 26/30 and 23/30 for no blending, plus subject
continuity, plus two further prompt rules — all inside the run-to-run noise, and
every added rule made it worse.

A third mechanism was built and then removed: the indexer computes, for each
document, the lines whose **numbers** occur nowhere in its nearest near-duplicate
(«занятие длится 20 минут» against «занятие длится 90 минут»), and the
retriever prints them above the chunk as `ОТЛИЧАЕТСЯ ОТ …`. The selection rule
cannot be wrong — a number is either in the sibling or it is not — and word-level
alternatives were tried first and rejected for exactly that reason (token
similarity rates a marketing slogan «это настоящий экстрим!» at 0.07 and a
factual clause «длительность 20 минут» at 0.10, so it cannot tell a fact from
a slogan). It still measured **worse: 15/30 with the card, 16/30 without**, and
the reason is structural rather than a tuning problem. A card can only be printed
for the chunk that contains its lines, so on a follow-up that resolves a pronoun
— «То есть в моём документе это <другой вариант>?» — where the retrieved chunk
is the marketing one and the card lives in a different chunk —
there is no card at all, which is the one case it was built for. And it prints the
sibling's file name directly above the text, which is precisely what the
no-blending rule tells the model not to do. Turn 3 went 1/3 → 0/3.

What is left failing, over 10 turns x 3 runs:

| turn | correct | why |
| ---- | --- | --- |
| 3 | 1/3 | answers about the other document when asked about *«моём»* |
| 5 | 0/3 | «рекомендуем использовать в течение 6 месяцев» becomes «срок действия — 6 месяцев» |
| 9 | 0/3 | cites the other product's file for «оба в одном городе» |

Turn 5 is a claim-strength failure with the right evidence sitting in the prompt,
and prompt wording has been tried four times without moving it — see
`data/certs_eval/dialogue.yaml` for the measurements and the raw answers.

`researcher` therefore carries only a persona and `temperature: 0.1`. Nothing
about grounding depends on it.

Two properties are deliberate:

- **The block is not persisted.** It is rebuilt per request from the current
  index, so editing a document is visible immediately and the history never
  accumulates stale quotes.
- **Retrieval never breaks the bot.** A missing or corrupt index, a model that
  cannot be loaded, a query the embedder chokes on — all of it is logged to
  stderr and the turn proceeds without context. `--rag` is an enhancement, not
  a new failure mode.

`Session` exposes what actually happened: `rag_events` (all queries) and
`last_rag_event` (the last one) with hits, scores, dropped chunks and the
token accounting. The CLI prints the last event under the answer.

`knowledge_base/` holds a second, deliberately boring corpus (a coffee shop
menu, delivery terms) to check that grounding generalizes outside the repo's own
prose. It is added to the indexer's `DEFAULT_EXCLUDES`, so building the repo
index stays at 12 documents and does not silently mix the two corpora.

A third corpus lives entirely under `data/`, git-ignored: a set of scanned
one-page service descriptions with near-duplicate bodies. Nothing about it is
committed — not the documents, not the control questions. The lessons below
are measured there, and the numbers are quoted without the corpus.

Its reports go to `data/results/`, which the `data/` rule already covers, so
they need no entry of their own in `.gitignore`. `results/` stays tracked and
holds only the repo and coffee-shop reports — point `--out` at `data/results/`
for this corpus.

One tuning note generalises past it. Chunking with `--structure_max_tokens 128`
split two senses of one word — the imperative «сделать это самостоятельно?» (line
4) against the parenthetical «(самостоятельно)» in a venue line (line 11) — into
*different* chunks. Both the cross-encoder and the model then matched the wrong
sense and the bot invented a rule about solo participation that no document
states. At 192 both senses share one chunk and the model answers from the real
sentence. Re-ranked
retrieval went 32/34 → 33/34; the block grew from 396 to 588 tokens.

`scripts/compare_rag.py` gained a third arm, `rag_rerank`, and runs it *next to*
the plain `rag` arm rather than instead of it, so the before/after lives in one
run and one set of questions:

```bash
python scripts/compare_rag.py --questions "$QUESTIONS" \
    --model leanstral --corpus "$CORPUS" --rerank --top-k 4 \
    --out "$OUT"
```

Every case also stores the re-ranked shortlist with its scores
(`RagEvent.scored`), so the threshold sweep is pure arithmetic over a saved run
— `--from_json` rebuilds the whole report, sweep included, without touching the
provider at all.

Tests: `tests/test_rag.py`, `tests/test_rerank.py`, `tests/test_compare_rag.py`
(offline).
