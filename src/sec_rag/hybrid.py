import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from sec_rag.retrieval import dense_search

TOKEN_PATTERN = re.compile(
    r"\d+(?:,\d{3})+(?:\.\d+)?%?"
    r"|\d+(?:\.\d+)?%"
    r"|[a-z0-9]+(?:[.-][a-z0-9]+)*"
)


def tokenize(text: str) -> list[str]:
    """Tokenize financial text for lexical retrieval."""
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    if not chunks:
        raise ValueError("Cannot index an empty chunk collection")

    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def bm25_search(
    question: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    k: int = 20,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question must not be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if bm25.corpus_size != len(chunks):
        raise ValueError("BM25 and chunk count mismatch")

    tokens = tokenize(question)

    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    positions = np.argsort(-scores, kind="stable")

    return [
        {
            **chunks[int(position)],
            "score": float(scores[position]),
        }
        for position in positions[:k]
        if scores[position] > 0
    ]


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 20,
    rrf_constant: int = 60,
) -> list[dict]:
    """Combine ranked retrieval results using RRF."""
    if k <= 0 or rrf_constant <= 0:
        raise ValueError("Ranking parameters must be positive")

    scores = defaultdict(float)
    chunks = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["chunk_id"]
            chunks[chunk_id] = chunk
            scores[chunk_id] += 1 / (rrf_constant + rank)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )

    return [
        {
            **chunks[chunk_id],
            "score": scores[chunk_id],
        }
        for chunk_id in ranked_ids[:k]
    ]


def hybrid_search(
    question,
    model,
    index,
    bm25,
    chunks,
    k=20,
    rrf_constant=60,
):
    dense_results = dense_search(question, model, index, chunks, k=k)

    lexical_results = bm25_search(question, bm25, chunks, k=k)

    return reciprocal_rank_fusion(
        [dense_results, lexical_results],
        k=k,
        rrf_constant=rrf_constant,
    )
