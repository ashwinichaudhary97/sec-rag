import numpy as np
import pytest

from sec_rag.retrieval import build_dense_index, dense_search


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = {
            "first": [1.0, 0.0],
            "second": [0.0, 1.0],
            "third": [-1.0, 0.0],
            "question": [1.0, 0.0],
        }

        return np.array(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def collection():
    chunks = [
        {"chunk_id": "C1", "text": "first"},
        {"chunk_id": "C2", "text": "second"},
        {"chunk_id": "C3", "text": "third"},
    ]

    model = FakeModel()
    index = build_dense_index(chunks, model)

    return chunks, model, index


def test_dense_ranking(collection):
    chunks, model, index = collection

    results = dense_search("question", model, index, chunks, k=3)

    assert [result["chunk_id"] for result in results] == ["C1", "C2", "C3"]

    assert [result["score"] for result in results] == pytest.approx([1.0, 0.0, -1.0])


def test_retrieval_depth(collection):
    chunks, model, index = collection

    assert len(dense_search("question", model, index, chunks, k=2)) == 2

    assert len(dense_search("question", model, index, chunks, k=20)) == 3


def test_invalid_retrieval_inputs(collection):
    chunks, model, index = collection

    with pytest.raises(ValueError):
        dense_search("", model, index, chunks)

    with pytest.raises(ValueError):
        dense_search("question", model, index, chunks, k=0)

    with pytest.raises(ValueError):
        dense_search("question", model, index, chunks[:1])


def test_empty_index_build():
    with pytest.raises(ValueError):
        build_dense_index([], FakeModel())
