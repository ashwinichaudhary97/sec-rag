import numpy as np
import pytest

from sec_rag.reranking import rerank


class FakeCrossEncoder:
    max_seq_length = 512

    def __init__(self, scores, pair_length=20):
        self.scores = scores
        self.pair_length = pair_length

    def tokenizer(self, questions, texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[0] * self.pair_length for _ in texts]}

    def predict(self, pairs, **kwargs):
        return np.asarray(self.scores)


@pytest.fixture
def candidates():
    return [
        {
            "chunk_id": "C1",
            "text": "First passage",
            "score": 0.03,
            "source_spans": [{"start": 0, "end": 13}],
        },
        {
            "chunk_id": "C2",
            "text": "Second passage",
            "score": 0.02,
            "source_spans": [{"start": 20, "end": 34}],
        },
        {
            "chunk_id": "C3",
            "text": "Third passage",
            "score": 0.01,
            "source_spans": [{"start": 40, "end": 53}],
        },
    ]


def test_reranking_and_cutoff(candidates):
    model = FakeCrossEncoder([-2.0, 5.0, 1.0])

    results = rerank("Question", candidates, model, k=2)

    assert [item["chunk_id"] for item in results] == ["C2", "C3"]
    assert [item["score"] for item in results] == [5.0, 1.0]


def test_metadata_preserved(candidates):
    model = FakeCrossEncoder([1.0, 5.0, 0.0])

    result = rerank("Question", candidates, model, k=1)[0]

    assert result["source_spans"] == candidates[1]["source_spans"]
    assert result["retrieval_score"] == 0.02
    assert candidates[1]["score"] == 0.02


def test_equal_scores_preserve_order(candidates):
    model = FakeCrossEncoder([1.0, 1.0, 1.0])

    results = rerank("Question", candidates, model)

    assert [item["chunk_id"] for item in results] == ["C1", "C2", "C3"]


def test_empty_candidates():
    assert rerank("Question", [], FakeCrossEncoder([])) == []


def test_oversized_pair_rejected(candidates):
    model = FakeCrossEncoder([1.0, 2.0, 3.0], pair_length=513)

    with pytest.raises(ValueError, match="exceeds cross-encoder limit"):
        rerank("Question", candidates, model)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 2.0],
        [1.0, float("nan"), 3.0],
        [1.0, float("inf"), 3.0],
    ],
)
def test_invalid_scores_rejected(candidates, scores):
    with pytest.raises(ValueError):
        rerank("Question", candidates, FakeCrossEncoder(scores))


@pytest.mark.parametrize(
    "question,k,batch_size",
    [
        ("", 10, 8),
        ("Question", 0, 8),
        ("Question", 10, 0),
    ],
)
def test_invalid_arguments(candidates, question, k, batch_size):
    with pytest.raises(ValueError):
        rerank(
            question,
            candidates,
            FakeCrossEncoder([]),
            k=k,
            batch_size=batch_size,
        )
