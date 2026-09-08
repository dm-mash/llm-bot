"""Unit tests for the pure helpers in scripts/compare_models.py (no network).

The script is importable without side effects (its CLI only runs under
``__main__``), so we can test the scoring and cost helpers directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/compare_models.py importable as a module. The script is not a
# real package, so we add its folder to sys.path; Pylance can't statically
# resolve it, hence the type-ignore. At runtime this works because the module's
# CLI only runs under ``__main__``.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import compare_models as cm  # noqa: E402  # type: ignore[import-not-found]


def test_score_numbers_are_language_agnostic():
    ok, summary = cm._score(
        "На ферме: Chickens: 16, Cows: 6", "6 коров, 16 кур"
    )
    assert ok is True
    assert "числа совпадают" in summary


def test_score_missing_number_fails():
    ok, summary = cm._score("На ферме 20 голов", "6 коров, 16 кур")
    assert ok is False
    assert "нет чисел" in summary


def test_score_word_overlap_without_numbers():
    ok, _ = cm._score("fizzbuzz sequence is correct", "FizzBuzz")
    assert ok is True


def test_score_no_expected_is_manual():
    ok, summary = cm._score("anything", None)
    assert ok is True
    assert "вручную" in summary


def test_score_strict_exact_match_trims_whitespace():
    expected = "1,2,Fizz,4,Buzz"
    ok, summary = cm._score("  1,2,Fizz,4,Buzz\n", expected, strict=True)
    assert ok is True
    assert "точно совпадает" in summary


def test_score_strict_rejects_different_order_of_same_numbers():
    # Same number set, different order/wording -> strict must reject.
    ok, summary = cm._score("16 кур и 6 коров", "6 коров, 16 кур", strict=True)
    assert ok is False
    assert "дословно" in summary


def test_score_strict_rejects_when_one_token_off():
    expected = "1,2,Fizz,4,Buzz,6"
    ok, summary = cm._score("1,2,Fizz,4,Buzz,5", expected, strict=True)
    assert ok is False
    assert "дословно" in summary


def test_score_strict_is_case_sensitive():
    ok, _ = cm._score("fizzbuzz", "FizzBuzz", strict=True)
    assert ok is False


def test_score_strict_without_expected_is_manual():
    ok, summary = cm._score("anything", None, strict=True)
    assert ok is True
    assert "вручную" in summary


def test_price_lookup_builtin_and_override():
    assert cm._price_for("gpt-4o-mini", {}) == (0.15, 0.60)
    overrides = {"my-model": (1.0, 2.0)}
    assert cm._price_for("my-model", overrides) == (1.0, 2.0)
    # Unknown models cost 0.
    assert cm._price_for("openai/gpt-oss-20b", overrides) == (0.0, 0.0)


def test_estimate_cost_zero_for_free_model():
    assert cm._estimate_cost({"prompt_tokens": 100, "completion_tokens": 50}, (0.0, 0.0)) == 0.0


def test_estimate_cost_calculates_usd():
    # 1M prompt @ $2.50 => 100k tokens cost $0.25; 1M completion @ $10 => 10k cost $0.10.
    usage = {"prompt_tokens": 100_000, "completion_tokens": 10_000}
    cost = cm._estimate_cost(usage, (2.50, 10.00))
    assert abs(cost - 0.35) < 1e-9


def test_finalize_cost_fills_results():
    result = cm.ModelResult(
        model="gpt-4o-mini",
        response="x",
        elapsed_ms=100.0,
        first_byte_ms=90.0,
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        correct=True,
        score_summary="ok",
    )
    cm._finalize_cost([result], {})
    assert abs(result.cost_usd - (0.15 + 0.30)) < 1e-9


def test_parse_prices_overrides_defaults():
    prices = cm._parse_prices(["gpt-4o-mini=1,2"])
    assert prices["gpt-4o-mini"] == (1.0, 2.0)
    # Unrelated defaults survive.
    assert prices["gpt-4o"] == (2.50, 10.00)