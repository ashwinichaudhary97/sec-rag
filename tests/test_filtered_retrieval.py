import faiss
import numpy as np
import pytest

from sec_rag.filtered_retrieval import (
    filtered_bm25_search,
    filtered_dense_search,
    filtered_hybrid_search,
    load_embeddings,
)
from sec_rag.hybrid import build_bm25, hybrid_search


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype=np.float32,
        )


@pytest.fixture
def retrieval_data():
    chunks = [
        {
            "chunk_id": "C1",
            "ticker": "AAPL",
            "filing_year": 2025,
            "text": "Apple iPhone sales",
        },
        {
            "chunk_id": "C2",
            "ticker": "MSFT",
            "filing_year": 2025,
            "text": "Microsoft cloud revenue",
        },
        {
            "chunk_id": "C3",
            "ticker": "MSFT",
            "filing_year": 2023,
            "text": "Microsoft software products",
        },
    ]

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    return chunks, vectors, index, build_bm25(chunks)


def test_load_embeddings(retrieval_data):
    chunks, vectors, index, _ = retrieval_data

    embeddings = load_embeddings(index, chunks)

    np.testing.assert_array_equal(embeddings, vectors)


def test_filtered_dense_search(retrieval_data):
    chunks, vectors, _, _ = retrieval_data

    results = filtered_dense_search(
        "revenue",
        FakeModel(),
        vectors,
        chunks,
        [1, 2],
        k=1,
    )

    assert [result["chunk_id"] for result in results] == ["C2"]
    assert results[0]["score"] == pytest.approx(0.6)


def test_filtered_bm25_search(retrieval_data):
    chunks, _, _, bm25 = retrieval_data

    results = filtered_bm25_search(
        "iPhone cloud",
        bm25,
        chunks,
        [1, 2],
    )

    assert [result["chunk_id"] for result in results] == ["C2"]
    assert results[0]["score"] == pytest.approx(bm25.get_scores(["iphone", "cloud"])[1])


def test_filtered_hybrid_search(retrieval_data):
    chunks, vectors, index, bm25 = retrieval_data

    results = filtered_hybrid_search(
        "Cloud revenue in Microsoft's FY2025 10-K",
        FakeModel(),
        index,
        vectors,
        bm25,
        chunks,
    )

    assert [result["chunk_id"] for result in results] == ["C2"]


def test_no_eligible_chunks(retrieval_data):
    chunks, vectors, index, bm25 = retrieval_data

    results = filtered_hybrid_search(
        "Apple's FY2023 10-K",
        FakeModel(),
        index,
        vectors,
        bm25,
        chunks,
    )

    assert results == []


def test_unrestricted_search(retrieval_data):
    chunks, vectors, index, bm25 = retrieval_data
    model = FakeModel()

    expected = hybrid_search(
        "cloud revenue",
        model,
        index,
        bm25,
        chunks,
    )

    actual = filtered_hybrid_search(
        "cloud revenue",
        model,
        index,
        vectors,
        bm25,
        chunks,
    )

    assert actual == expected
