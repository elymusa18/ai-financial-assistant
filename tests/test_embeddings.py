from types import SimpleNamespace

import numpy as np
import pytest

from embeddings import HashingEmbedder, OpenAIEmbedder


def test_vectors_are_unit_length():
    vectors = HashingEmbedder().embed(["Netflix subscription", "Electricity bill"])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_embedding_is_deterministic_across_instances():
    a = HashingEmbedder().embed(["Lidl groceries"])
    b = HashingEmbedder().embed(["Lidl groceries"])
    assert np.array_equal(a, b)


def test_similar_texts_score_higher_than_unrelated():
    v = HashingEmbedder().embed(["Netflix subscription", "Netflix monthly payment", "Electricity bill"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_empty_text_does_not_crash():
    vector = HashingEmbedder().embed([""])
    assert vector.shape == (1, 512) and not np.isnan(vector).any()


class FakeEmbeddingsClient:
    def __init__(self):
        self.calls = []
        self.embeddings = self

    def create(self, model, input):
        self.calls.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])


def test_openai_embedder_batches_and_normalises():
    client = FakeEmbeddingsClient()
    vectors = OpenAIEmbedder(client, batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert [len(c) for c in client.calls] == [2, 2, 1]
    assert vectors.shape == (5, 2)
    assert vectors[0] == pytest.approx([0.6, 0.8])
