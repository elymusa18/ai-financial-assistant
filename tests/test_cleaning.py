import io

import pytest

from cleaning import UNKNOWN, load_and_clean

HEADER = "date,category,type,amount,description\n"


def csv(body: str) -> io.StringIO:
    return io.StringIO(HEADER + body.strip())


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing columns"):
        load_and_clean(io.StringIO("date,amount\n2026-01-01,10"))


def test_removes_duplicates_and_invalid_rows():
    df, report = load_and_clean(csv(
        "2026-01-05,Food,expense,10,a\n"
        "2026-01-05,Food,expense,10,a\n"      # duplicate
        "not a date,Food,expense,10,b\n"      # bad date
        "2026-01-06,Food,expense,,c\n"        # missing amount
        "2026-01-07,Food,refund,5,d"          # unknown type
    ))
    assert report["rows_loaded"] == 5
    assert report["duplicates_removed"] == 1
    assert report["invalid_rows_removed"] == 3
    assert report["rows_final"] == len(df) == 1


def test_comma_decimal_is_parsed():
    df, _ = load_and_clean(csv('2026-01-05,Food,expense,"45,50",x'))
    assert df["amount"].iloc[0] == pytest.approx(45.5)


def test_category_is_normalised_and_blank_becomes_unknown():
    df, _ = load_and_clean(csv(
        "2026-01-05, groceries ,expense,10,a\n"
        "2026-01-06,,expense,20,b"
    ))
    assert df["category"].tolist() == ["Groceries", UNKNOWN]


def test_bulgarian_types_are_mapped():
    df, _ = load_and_clean(csv(
        "2026-01-05,Заплата,приход,100,a\n"
        "2026-01-06,Храна,разход,20,b"
    ))
    assert df["type"].tolist() == ["income", "expense"]


def test_description_column_is_optional():
    df, _ = load_and_clean(io.StringIO("date,category,type,amount\n2026-01-05,Food,expense,10"))
    assert df["description"].tolist() == [""]


def test_month_column_added():
    df, _ = load_and_clean(csv("2026-03-05,Food,expense,10,a"))
    assert df["month"].iloc[0] == "2026-03"
