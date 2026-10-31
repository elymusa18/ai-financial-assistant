"""Retrieval-augmented generation over individual transactions.

Each transaction becomes a short text document, embedded once. At question time
we embed the question, retrieve the most similar transactions and give them to the
LLM next to the aggregated summary.
"""
import numpy as np
import pandas as pd


def transaction_to_text(row) -> str:
    return (f"{row['date']:%Y-%m-%d} | {row['type']} | {row['category']} | "
            f"{row['amount']:.2f} | {row['description']}")


class TransactionIndex:
    def __init__(self, df: pd.DataFrame, embedder):
        self.embedder = embedder
        self.documents = [transaction_to_text(row) for _, row in df.iterrows()]
        self._vectors = embedder.embed(self.documents) if self.documents else None

    def search(self, query: str, k: int = 8) -> list[str]:
        if self._vectors is None or not query.strip():
            return []
        query_vector = self.embedder.embed([query])[0]
        scores = self._vectors @ query_vector
        return [self.documents[i] for i in np.argsort(scores)[::-1][:k]]
