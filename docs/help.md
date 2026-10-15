# Developer assistant: `/help` over the project

`/help <вопрос>` (or `/помощь`) in the interactive chat answers questions about
the project itself, from two sources attached to the session at launch:

- **documentation via RAG** — the repository corpus is every Markdown file at
  the root (`--rag --rag-index data/emb/index_structure.json`, built from `.` by
  [`scripts/index_documents.py`](../scripts/index_documents.py)), which covers
  [`README.md`](../README.md) and the planning notes in [`plans/`](../plans/);
- **git context via MCP** — the project's own git server
  ([`llm_bot/mcp_servers/git.py`](../llm_bot/mcp_servers/git.py), backend
  [`llm_bot/api/git.py`](../llm_bot/api/git.py)), four read-only tools:
  `current_branch`, `list_files(pattern)`, `changed_files(ref, added_only)`
  (a complete, compact status list — including untracked files) and
  `diff(ref, staged, path)` (unified diff — when it is truncated, the complete
  changed-file list is appended after the marker, so the cut cannot hide files
  that sort past it).

```bash
python -m llm_bot --agent researcher --rag --rag-rerank \
    --rag-index data/emb/index_structure.json --mcp git

> /help как устроен проект и какая сейчас ветка?
```

`--rag-rerank` matters here: the repo index has 742 chunks and the dense
shortlist alone misses some gold chunks that the cross-encoder recovers (see
[the second stage](#why-the-second-stage-exists)); the default 50-candidate
shortlist is what makes the reranker effective on an index this size.

Unlike an ordinary message, `/help` is a *project mode*: it marks the turn with
a transient directive ("это вопрос о проекте", never stored in the history) and
**refuses to answer** when no project context is attached — otherwise a
plausible answer from memory would hide the fact that the model never saw the
project. It also obeys the task pause gate. A bare `/help` prints the usage
hint.
