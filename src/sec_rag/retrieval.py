import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_dense_index(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> faiss.IndexFlatIP:
    """Build an exact cosine-similarity index."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk collection")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings are not L2-normalized")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def dense_search(
    question: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    k: int = 20,
) -> list[dict]:
    """Retrieve chunks ranked by cosine similarity."""
    if not question.strip():
        raise ValueError("Question must not be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count mismatch")

    if not chunks:
        return []

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embedding = np.ascontiguousarray(embedding, dtype=np.float32)

    scores, positions = index.search(embedding, min(k, len(chunks)))

    return [
        {
            **chunks[int(position)],
            "score": float(score),
        }
        for score, position in zip(scores[0], positions[0])
    ]
