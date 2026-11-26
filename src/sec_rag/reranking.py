import numpy as np
from sentence_transformers import CrossEncoder


def rerank(
    question: str,
    candidates: list[dict],
    model: CrossEncoder,
    k: int = 10,
    batch_size: int = 8,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question must not be empty")

    if k <= 0 or batch_size <= 0:
        raise ValueError("k and batch_size must be positive")

    if not candidates:
        return []

    texts = [chunk["text"] for chunk in candidates]
    questions = [question] * len(texts)

    encoded = model.tokenizer(
        questions,
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    lengths = [len(ids) for ids in encoded["input_ids"]]

    if max(lengths) > model.max_seq_length:
        raise ValueError(
            f"Question-chunk pair exceeds cross-encoder limit: "
            f"{max(lengths)} > {model.max_seq_length}"
        )

    pairs = list(zip(questions, texts))

    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    )

    if scores.shape != (len(candidates),):
        raise ValueError("Unexpected cross-encoder score shape")

    if not np.isfinite(scores).all():
        raise ValueError("Cross-encoder returned non-finite scores")

    ranking = np.argsort(-scores, kind="stable")[:k]

    return [
        {
            **candidates[int(position)],
            "retrieval_score": candidates[int(position)]["score"],
            "score": float(scores[position]),
        }
        for position in ranking
    ]
