import io

import pytest

from analysis import build_context, compute_summary
from cleaning import load_and_clean

CSV = """date,category,type,amount,description
2026-01-05,Salary,income,1000,pay
2026-01-10,Rent,expense,400,rent
2026-02-05,Salary,income,1000,pay
2026-02-12,Food,expense,100,food
2026-02-20,Food,expense,50,food
"""


@pytest.fixture
def summary():
    df, _ = load_and_clean(io.StringIO(CSV))
    return compute_summary(df)


def test_totals(summary):
    assert summary["income"] == 2000
    assert summary["expenses"] == 550
    assert summary["profit"] == 1450


def test_monthly_profit(summary):
    assert summary["monthly"].loc["2026-01", "profit"] == 600
    assert summary["monthly"].loc["2026-02", "profit"] == 850


def test_expenses_by_category_sorted_desc(summary):
    assert summary["by_category"].index.tolist() == ["Rent", "Food"]
    assert summary["by_category"]["Food"] == 150


def test_context_contains_key_numbers(summary):
    text = build_context(summary)
    assert "Total profit: 1450.00" in text and "Rent" in text
