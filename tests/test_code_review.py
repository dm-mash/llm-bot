#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import code_review as mod


def test_parse_name_status_simple():
    text = "M\tllm_bot/agent.py\nA\tllm_bot/new.py\nD\told.py\n"
    files = mod.parse_name_status(text)
    assert [f.path for f in files] == ["llm_bot/agent.py", "llm_bot/new.py", "old.py"]


def test_truncate_diff_boundaries():
    diff = "diff --git a/x b/x\n" + "line\n" * 5
    res, trunc = mod.truncate_diff(diff, 100)
    assert trunc is False
    assert res == diff


def test_classify_bullets():
    ans = """- БАГ: something broken
- АРХИТЕКТУРА: design issue
- РЕКОМЕНДАЦИЯ: do this
"""
    c = mod.classify(ans)
    assert c["БАГ"] == ["something broken"]
    assert c["АРХИТЕКТУРА"] == ["design issue"]
    assert c["РЕКОМЕНДАЦИЯ"] == ["do this"]


def test_render_report_has_marker():
    from scripts.code_review import ChangedFile, FileReview, PRContext, render_report
    ctx = PRContext(1, "a", "b", "main", "feat", "o/r", False)
    reviews = [FileReview(ChangedFile("M", "x.py"), answer="- РЕКОМЕНДАЦИЯ: ok")]
    rpt = render_report(ctx, reviews, [])
    assert mod.COMMENT_MARKER in rpt


def _stores(tmp_path: Path, models: str, agents: str):
    from llm_bot.yaml_stores import YamlAgentStore, YamlModelStore

    (tmp_path / "models.yaml").write_text(models, encoding="utf-8")
    (tmp_path / "agents.yaml").write_text(agents, encoding="utf-8")
    return YamlAgentStore(str(tmp_path / "agents.yaml")), YamlModelStore(str(tmp_path / "models.yaml"))


AGENTS_YAML = "agents:\n  reviewer:\n    model: reviewer\n"
MODELS_OK = (
    "models:\n"
    "  reviewer:\n"
    "    provider: openai\n"
    "    base_url: ${REVIEW_BASE_URL}\n"
    "    api_key: ${REVIEW_API_KEY}\n"
    "    model: openai/gpt-oss-120b\n"
)


def test_a_configured_reviewer_passes_the_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_BASE_URL", "https://api.groq.com/openai/v1")
    agents, models = _stores(tmp_path, MODELS_OK, AGENTS_YAML)
    assert mod.check_model_config("reviewer", agent_store=agents, model_store=models) is None


def test_an_unset_base_url_is_reported_by_name(tmp_path, monkeypatch):
    # The failure this guards against: ${REVIEW_BASE_URL} resolving to "" and the
    # client dying inside httpx without ever mentioning the missing variable.
    monkeypatch.delenv("REVIEW_BASE_URL", raising=False)
    agents, models = _stores(tmp_path, MODELS_OK, AGENTS_YAML)
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "REVIEW_BASE_URL" in problem


def test_an_unknown_agent_is_reported(tmp_path):
    agents, models = _stores(tmp_path, MODELS_OK, AGENTS_YAML)
    problem = mod.check_model_config("nope", agent_store=agents, model_store=models)
    assert problem is not None
    assert "nope" in problem


def test_a_base_url_without_a_scheme_is_named(tmp_path):
    agents, models = _stores(
        tmp_path,
        MODELS_OK.replace("${REVIEW_BASE_URL}", "api.groq.com/openai/v1"),
        AGENTS_YAML,
    )
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "http://" in problem


def test_a_base_url_that_already_has_the_endpoint_is_refused(tmp_path):
    # The client appends /chat/completions itself, so a full endpoint pasted into
    # the variable is doubled and the provider answers 404 "Unknown request URL".
    agents, models = _stores(
        tmp_path,
        MODELS_OK.replace(
            "${REVIEW_BASE_URL}", "https://api.groq.com/openai/v1/chat/completions"
        ),
        AGENTS_YAML,
    )
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "/chat/completions" in problem


def test_credentials_in_a_url_are_not_echoed(tmp_path):
    agents, models = _stores(
        tmp_path,
        MODELS_OK.replace(
            "${REVIEW_BASE_URL}",
            "https://user:secret@host.invalid/chat/completions",
        ),
        AGENTS_YAML,
    )
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "secret" not in problem
    assert "***@" in problem


def test_a_query_string_in_a_base_url_is_refused(tmp_path):
    # Everything past "?" leaves the path, so the endpoint the client appends
    # vanishes into the query string and the provider reports an unrelated path.
    agents, models = _stores(
        tmp_path,
        MODELS_OK.replace(
            "${REVIEW_BASE_URL}", "https://api.groq.com/openai/v1/chat/api?x=1"
        ),
        AGENTS_YAML,
    )
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "query string" in problem


def test_whitespace_from_the_variable_is_refused(tmp_path, monkeypatch):
    # The YAML scalar is stripped by the parser, so junk can only enter through
    # the substituted ${REVIEW_BASE_URL} — a copy-paste that kept a newline.
    monkeypatch.setenv("REVIEW_BASE_URL", "https://api.groq.com/openai/v1\n")
    agents, models = _stores(tmp_path, MODELS_OK, AGENTS_YAML)
    problem = mod.check_model_config("reviewer", agent_store=agents, model_store=models)
    assert problem is not None
    assert "пробелы" in problem


def test_describe_model_shows_the_endpoint_in_use(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_BASE_URL", "https://api.groq.com/openai/v1")
    agents, models = _stores(tmp_path, MODELS_OK, AGENTS_YAML)
    line = mod.describe_model("reviewer", agent_store=agents, model_store=models)
    assert "https://api.groq.com/openai/v1" in line
    assert "openai/gpt-oss-120b" in line


def test_emphasised_rubric_prefixes_are_parsed():
    # A live run answered "- **БАГ:** ..." and the bare-form parser emptied all
    # three sections while the findings sat unread in the appendix.
    ans = (
        "- **БАГ:** `x.py:10` — падение при пустом ответе\n"
        "- **АРХИТЕКТУРА:** модуль отвечает за две задачи\n"
        "- **РЕКОМЕНДАЦИЯ:** разнести по двум модулям\n"
    )
    c = mod.classify(ans)
    assert c["БАГ"] == ["`x.py:10` — падение при пустом ответе"]
    assert c["АРХИТЕКТУРА"] == ["модуль отвечает за две задачи"]
    assert c["РЕКОМЕНДАЦИЯ"] == ["разнести по двум модулям"]


def test_an_unparsed_answer_is_kept_in_its_own_section(tmp_path):
    from scripts.code_review import ChangedFile, FileReview, PRContext, render_report

    ctx = PRContext(1, "a", "b", "main", "feat", "o/r", False)
    reviews = [FileReview(ChangedFile("M", "x.py"), answer="Просто текст без рубрик.")]
    rpt = render_report(ctx, reviews, [])
    assert "Прочее" in rpt
    assert "Просто текст без рубрик." in rpt


def test_a_nothing_to_say_reply_does_not_create_a_leftover():
    from scripts.code_review import ChangedFile, FileReview, PRContext, render_report

    ctx = PRContext(1, "a", "b", "main", "feat", "o/r", False)
    reviews = [FileReview(ChangedFile("M", "x.py"), answer="Нечего комментировать")]
    rpt = render_report(ctx, reviews, [])
    assert "Прочее" not in rpt
