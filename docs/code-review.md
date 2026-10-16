# Automated code review

This pipeline runs an AI code reviewer on each pull request (opened/synchronize/reopened). It builds a small docs+code index, reviews each changed file one by one, and posts a single markdown comment with three sections: bugs, architectural problems, recommendations.

## How it works

- **Trigger**: GitHub Actions (`.github/workflows/code-review.yml`), `pull_request` events. Draft PRs are skipped. PRs from forks are skipped (no secrets available).
- **Diff**: taken from git as `base.sha...head.sha` (merge-base diff). Unified context configurable via `--context-lines` (default 0).
- **Index**: `data/rev_emb/index_structure.json` built from `.md,.py` using structure strategy.
- **Review**: Per-file pass with shared retriever, fresh session per file. `rag_cite=False`, `rag_strict=False`, `invariants=False`.
- **Output**: Single comment updated on synchronize (marker-based), appended to `$GITHUB_STEP_SUMMARY`, report uploaded as artifact.

## How to try it in action

1. **Prepare the config (once)**
   - `cp models.example.yaml data/models.yaml`
   - `cp agents.example.yaml data/agents.yaml`
   Make sure the `reviewer` block exists in `data/models.yaml`. For local runs you can use a local model (e.g. `qwen-local-gpu` or `qwen-local-cpu`) or configure a cloud one (Groq/Mistral/OpenRouter).

2. **Build the index (once, reusable)**
   - `.venv/bin/python scripts/index_documents.py --input_dir . --extensions .md,.py --strategy structure --index_dir data/rev_emb --out ""`
   This creates `data/rev_emb/index_structure.json` (the RAG context for the review).

3. **Run the review locally (no PR comment, dry run)**
   - `.venv/bin/python scripts/code_review.py --base <base> --head <head> --agent reviewer --rag-index data/rev_emb/index_structure.json --dry-run --out /tmp/review.md`
   Examples:
   - Changes relative to previous commit: `--base HEAD~1 --head HEAD`
   - Diff between branches: `--base main --head day31`
   `--dry-run` does not call the model or post a comment; it only generates the prompts/report into the specified file.

4. **Check the result**
   - `cat /tmp/review.md` — contains a summary in three sections (Баги / Архитектурные проблемы / Рекомендации) plus collapsible full answers per file.

5. **Run it for real (with the model)**
   - Remove `--dry-run`: `.venv/bin/python scripts/code_review.py --base HEAD~1 --head HEAD --agent reviewer --rag-index data/rev_emb/index_structure.json --out /tmp/review.md`
   This requires a working `reviewer` configuration in `data/models.yaml` (keys in `.env` or environment).

6. **Try on a real PR (GitHub Actions)**
   - Create a branch off the current one, commit changes, and open a PR in this repository (not a fork). The workflow `.github/workflows/code-review.yml` runs automatically on `opened/synchronize/reopened`.
   - Add repository secrets: `REVIEW_API_KEY` (required), optionally `REVIEW_BASE_URL`. These are mapped into the `reviewer` block in `models.example.yaml`. The comment is created/updated on each push (marker-based `<!-- llm-bot-code-review -->`).
   - Fork PRs are skipped (no access to secrets); a short note appears in the step summary.

## Running locally (short)

```bash
cp models.example.yaml data/models.yaml
cp agents.example.yaml data/agents.yaml
.venv/bin/python scripts/index_documents.py --input_dir . --extensions .md,.py \
  --strategy structure --index_dir data/rev_emb --out ""
.venv/bin/python scripts/code_review.py --base HEAD~1 --head HEAD \
  --agent reviewer --rag-index data/rev_emb/index_structure.json --dry-run --out /tmp/review.md
```
