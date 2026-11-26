import pytest

from sec_rag.hybrid import (
    bm25_search,
    build_bm25,
    reciprocal_rank_fusion,
    tokenize,
)


def test_financial_tokenization():
    assert tokenize("FY2025 10-K $12,286 million 15.5% ASC 842") == [
        "fy2025",
        "10-k",
        "12,286",
        "million",
        "15.5%",
        "asc",
        "842",
    ]


def test_bm25_retrieval():
    chunks = [
        {"chunk_id": "C1", "text": "Apple iPhone sales"},
        {"chunk_id": "C2", "text": "Microsoft cloud revenue"},
        {"chunk_id": "C3", "text": "Amazon advertising revenue"},
    ]

    bm25 = build_bm25(chunks)
    results = bm25_search("iPhone", bm25, chunks)

    assert [result["chunk_id"] for result in results] == ["C1"]
    assert results[0]["score"] > 0


def test_bm25_no_match():
    chunks = [
        {"chunk_id": "C1", "text": "Apple revenue"},
        {"chunk_id": "C2", "text": "Microsoft cloud"},
    ]

    bm25 = build_bm25(chunks)

    assert bm25_search("unmatched", bm25, chunks) == []


def test_rrf():
    dense = [
        {"chunk_id": "C1", "score": 0.9},
        {"chunk_id": "C2", "score": 0.8},
    ]

    lexical = [
        {"chunk_id": "C3", "score": 12.0},
        {"chunk_id": "C2", "score": 8.0},
    ]

    results = reciprocal_rank_fusion([dense, lexical])

    assert [result["chunk_id"] for result in results] == ["C2", "C1", "C3"]

    assert results[0]["score"] == pytest.approx(2 / 62)


def test_rrf_cutoff():
    ranking = [
        {"chunk_id": "C1"},
        {"chunk_id": "C2"},
    ]

    results = reciprocal_rank_fusion([ranking], k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "C1"
