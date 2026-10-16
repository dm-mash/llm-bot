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
