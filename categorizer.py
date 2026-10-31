"""Automatic transaction categorisation with embeddings + k-nearest neighbours.

Transactions that already have a category are the training examples. For rows
with an unknown category we embed the description, find the most similar labelled
descriptions and vote. Low-confidence predictions stay "Unknown" for human review.
"""
import numpy as np
import pandas as pd

from cleaning import UNKNOWN

DEFAULT_K = 3
DEFAULT_THRESHOLD = 0.25


class KNNCategorizer:
    def __init__(self, embedder, k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD):
        self.embedder = embedder
        self.k = k
        self.threshold = threshold
        self._vectors = None
        self._labels: list[str] = []

    def fit(self, descriptions: list[str], labels: list[str]) -> "KNNCategorizer":
        if not descriptions or len(descriptions) != len(labels):
            raise ValueError("descriptions and labels must be non-empty and equal in length")
        self._vectors = self.embedder.embed(list(descriptions))
        self._labels = list(labels)
        return self

    def predict(self, descriptions: list[str]) -> list[tuple[str, float]]:
        """Return (label, confidence) per description; confidence = best cosine similarity."""
        if self._vectors is None:
            raise RuntimeError("Call fit() before predict()")
        if not descriptions:
            return []
        sims = self.embedder.embed(list(descriptions)) @ self._vectors.T
        results = []
        for desc, row in zip(descriptions, sims):
            if not desc.strip():
                results.append((UNKNOWN, 0.0))
                continue
            votes: dict[str, float] = {}
            best: dict[str, float] = {}
            for idx in np.argsort(row)[::-1][: self.k]:
                label, sim = self._labels[idx], float(row[idx])
                votes[label] = votes.get(label, 0.0) + max(sim, 0.0)
                best[label] = max(best.get(label, 0.0), sim)
            winner = max(votes, key=votes.get)
            confidence = best[winner]
            results.append((winner if confidence >= self.threshold else UNKNOWN, confidence))
        return results


def categorize_dataframe(df: pd.DataFrame, embedder, k: int = DEFAULT_K,
                         threshold: float = DEFAULT_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """Fill in 'Unknown' categories. Adds `category_predicted` and `category_confidence`."""
    out = df.copy()
    out["category_predicted"] = False
    out["category_confidence"] = np.nan

    has_desc = out["description"].str.strip() != ""
    labeled = out[(out["category"] != UNKNOWN) & has_desc]
    todo = out[(out["category"] == UNKNOWN) & has_desc]
    report = {"labeled_examples": len(labeled), "to_categorize": len(todo),
              "categorized": 0, "left_unknown": len(todo)}
    if labeled.empty or todo.empty:
        return out, report

    model = KNNCategorizer(embedder, k, threshold).fit(
        labeled["description"].tolist(), labeled["category"].tolist())
    for idx, (label, confidence) in zip(todo.index, model.predict(todo["description"].tolist())):
        out.at[idx, "category_confidence"] = confidence
        if label != UNKNOWN:
            out.at[idx, "category"] = label
            out.at[idx, "category_predicted"] = True

    report["categorized"] = int(out["category_predicted"].sum())
    report["left_unknown"] = report["to_categorize"] - report["categorized"]
    return out, report


def evaluate_categorizer(df: pd.DataFrame, embedder, test_size: float = 0.3, seed: int = 42,
                         k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Hold-out evaluation on the already-labelled rows (reproducible via `seed`)."""
    has_desc = df["description"].str.strip() != ""
    labeled = df[(df["category"] != UNKNOWN) & has_desc]
    if len(labeled) < 4:
        raise ValueError("Need at least 4 labelled transactions with descriptions")

    order = np.random.default_rng(seed).permutation(len(labeled))
    n_test = max(1, int(len(labeled) * test_size))
    test, train = labeled.iloc[order[:n_test]], labeled.iloc[order[n_test:]]

    model = KNNCategorizer(embedder, k, threshold).fit(
        train["description"].tolist(), train["category"].tolist())
    predicted = [label for label, _ in model.predict(test["description"].tolist())]
    truth = test["category"].tolist()

    correct = sum(p == t for p, t in zip(predicted, truth))
    answered = [(p, t) for p, t in zip(predicted, truth) if p != UNKNOWN]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": correct / len(test),  # "Unknown" counts as wrong
        "coverage": len(answered) / len(test),
        "accuracy_when_confident": (sum(p == t for p, t in answered) / len(answered)) if answered else 0.0,
    }
