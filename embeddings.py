"""Text embeddings: an offline hashing embedder and an OpenAI embedder.

Both expose `embed(texts) -> np.ndarray` of L2-normalised row vectors, so cosine
similarity is just a dot product. The offline embedder needs no API key, which
keeps tests and demos free and deterministic.
"""
import re
import zlib

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class HashingEmbedder:
    """Bag of words + character trigrams hashed into a fixed-size vector."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        words = re.findall(r"\w+", text.lower())
        feats = list(words)
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                out[i, zlib.crc32(feat.encode("utf-8")) % self.dim] += 1.0
        return _normalize(out)


class OpenAIEmbedder:
    """Semantic embeddings from the OpenAI API (batched)."""

    def __init__(self, client, model: str = "text-embedding-3-small",
                 batch_size: int = 100):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + self.batch_size]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return _normalize(np.array(vectors, dtype=np.float32))
