import numpy as np

from sec_rag.hybrid import (
    hybrid_search,
    reciprocal_rank_fusion,
    tokenize,
)
from sec_rag.metadata import eligible_positions, extract_constraints


def load_embeddings(index, chunks: list[dict]) -> np.ndarray:
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count mismatch")

    return index.reconstruct_n(0, index.ntotal)


def filtered_dense_search(
    question,
    model,
    embeddings,
    chunks,
    positions,
    k=20,
):
    if k <= 0:
        raise ValueError("k must be positive")

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding and chunk count mismatch")

    if not positions:
        return []

    query = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    eligible = np.asarray(positions, dtype=np.intp)
    scores = embeddings[eligible] @ query

    ranking = np.argsort(-scores, kind="stable")[:k]

    return [
        {
            **chunks[int(eligible[position])],
            "score": float(scores[position]),
        }
        for position in ranking
    ]


def filtered_bm25_search(
    question,
    bm25,
    chunks,
    positions,
    k=20,
):
    if k <= 0:
        raise ValueError("k must be positive")

    if bm25.corpus_size != len(chunks):
        raise ValueError("BM25 and chunk count mismatch")

    tokens = tokenize(question)

    if not positions or not tokens:
        return []

    scores = np.asarray(bm25.get_batch_scores(tokens, positions))

    ranking = np.argsort(-scores, kind="stable")[:k]

    return [
        {
            **chunks[positions[position]],
            "score": float(scores[position]),
        }
        for position in ranking
        if scores[position] > 0
    ]


def filtered_hybrid_search(
    question,
    model,
    index,
    embeddings,
    bm25,
    chunks,
    k=20,
    rrf_constant=60,
):
    if not question.strip():
        raise ValueError("Question must not be empty")

    if k <= 0 or rrf_constant <= 0:
        raise ValueError("Ranking parameters must be positive")

    constraints = extract_constraints(question)

    if not constraints["tickers"] and not constraints["filing_years"]:
        return hybrid_search(
            question,
            model,
            index,
            bm25,
            chunks,
            k=k,
            rrf_constant=rrf_constant,
        )

    positions = eligible_positions(chunks, constraints)

    if not positions:
        return []

    dense_results = filtered_dense_search(
        question,
        model,
        embeddings,
        chunks,
        positions,
        k=k,
    )

    lexical_results = filtered_bm25_search(
        question,
        bm25,
        chunks,
        positions,
        k=k,
    )

    return reciprocal_rank_fusion(
        [dense_results, lexical_results],
        k=k,
        rrf_constant=rrf_constant,
    )
