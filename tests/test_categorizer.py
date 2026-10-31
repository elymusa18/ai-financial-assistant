import io
from pathlib import Path

import pytest

from categorizer import KNNCategorizer, categorize_dataframe, evaluate_categorizer
from cleaning import UNKNOWN, load_and_clean
from embeddings import HashingEmbedder

DESCRIPTIONS = ["Netflix subscription", "Spotify premium subscription", "Lidl groceries",
                "Kaufland weekly groceries", "Electricity bill", "Water bill"]
LABELS = ["Subscriptions", "Subscriptions", "Groceries", "Groceries", "Utilities", "Utilities"]


@pytest.fixture
def model():
    return KNNCategorizer(HashingEmbedder()).fit(DESCRIPTIONS, LABELS)


def test_predicts_closest_category(model):
    labels = [label for label, _ in model.predict(["Netflix monthly payment", "Billa groceries"])]
    assert labels == ["Subscriptions", "Groceries"]


def test_unrelated_text_stays_unknown(model):
    (label, confidence), = model.predict(["zzzz qqqq"])
    assert label == UNKNOWN and confidence < 0.25


def test_empty_description_is_unknown(model):
    assert model.predict(["   "]) == [(UNKNOWN, 0.0)]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNCategorizer(HashingEmbedder()).predict(["x"])


def test_fit_validates_input():
    with pytest.raises(ValueError):
        KNNCategorizer(HashingEmbedder()).fit(["a"], [])


CSV = """date,category,type,amount,description
2026-01-01,Subscriptions,expense,10,Netflix subscription
2026-01-02,Subscriptions,expense,8,Spotify premium subscription
2026-01-03,Groceries,expense,40,Lidl groceries
2026-01-04,Groceries,expense,35,Kaufland weekly groceries
2026-01-05,,expense,10,Netflix monthly payment
2026-01-06,,expense,12,
2026-01-07,,expense,15,zzzz qqqq
"""


def test_categorize_dataframe_only_fills_unknown_rows():
    df, _ = load_and_clean(io.StringIO(CSV))
    out, report = categorize_dataframe(df, HashingEmbedder())

    assert report == {"labeled_examples": 4, "to_categorize": 2, "categorized": 1, "left_unknown": 1}
    predicted = out[out["category_predicted"]]
    assert predicted["description"].tolist() == ["Netflix monthly payment"]
    assert predicted["category"].tolist() == ["Subscriptions"]
    # originally labelled rows untouched, no-description row untouched
    assert out.loc[out["description"] == "Lidl groceries", "category"].iloc[0] == "Groceries"
    assert out.loc[out["description"] == "", "category"].iloc[0] == UNKNOWN


def test_categorize_dataframe_without_labels_is_noop():
    df, _ = load_and_clean(io.StringIO("date,category,type,amount,description\n2026-01-01,,expense,5,Coffee"))
    out, report = categorize_dataframe(df, HashingEmbedder())
    assert report["categorized"] == 0 and out["category"].iloc[0] == UNKNOWN


def test_evaluate_on_sample_data_is_precise_and_abstains_when_unsure():
    # The offline embedder is lexical: it cannot match merchants it never saw, so it
    # should abstain ("Unknown") rather than guess. Confident predictions must be right.
    df, _ = load_and_clean(Path(__file__).parent.parent / "sample_data.csv")
    result = evaluate_categorizer(df, HashingEmbedder())
    assert result["n_train"] + result["n_test"] == (df["category"] != UNKNOWN).sum()
    assert result["accuracy_when_confident"] >= 0.9
    assert result["coverage"] >= 0.5


def test_evaluate_needs_enough_labels():
    df, _ = load_and_clean(io.StringIO(CSV))
    with pytest.raises(ValueError):
        evaluate_categorizer(df.iloc[:3], HashingEmbedder())
