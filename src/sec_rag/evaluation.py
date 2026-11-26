from collections.abc import Sequence

from sklearn.metrics import ndcg_score


def covers_location(chunk: dict, location: dict) -> bool:
    """Check full evidence containment within a single chunk."""
    if chunk["accession"] != location["accession"]:
        return False

    position = location["start"]
    end = location["end"]

    for span in sorted(chunk["source_spans"], key=lambda span: span["start"]):
        if span["end"] <= position:
            continue

        if span["start"] > position:
            return False

        position = max(position, span["end"])

        if position >= end:
            return True

    return False


def covered_units(chunk: dict, gold_evidence: Sequence[dict]) -> set[str]:
    """Return evidence units fully supported by this chunk."""
    return {
        unit["unit_id"]
        for unit in gold_evidence
        if any(covers_location(chunk, location) for location in unit["locations"])
    }


def evidence_recall_at_k(
    retrieved: Sequence[dict],
    gold_evidence: Sequence[dict],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not gold_evidence:
        raise ValueError("Gold evidence must not be empty")

    covered = set()

    for chunk in retrieved[:k]:
        covered.update(covered_units(chunk, gold_evidence))

    return len(covered) / len(gold_evidence)


def ndcg_at_k(
    retrieved: Sequence[dict],
    indexed_chunks: Sequence[dict],
    gold_evidence: Sequence[dict],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not gold_evidence:
        raise ValueError("Gold evidence must not be empty")

    relevance = [
        int(bool(covered_units(chunk, gold_evidence))) for chunk in indexed_chunks
    ]

    relevant_count = sum(relevance)

    if not retrieved or relevant_count == 0:
        return 0.0

    ranked_relevance = [
        int(bool(covered_units(chunk, gold_evidence))) for chunk in retrieved[:k]
    ]

    remaining_relevant = relevant_count - sum(ranked_relevance)

    if remaining_relevant < 0:
        raise ValueError("Inconsistent retrieved and indexed relevance")

    ranked_relevance.extend([0] * (k - len(ranked_relevance)))
    ranked_relevance.extend([1] * remaining_relevant)

    if len(ranked_relevance) == 1:
        return float(ranked_relevance[0])

    scores = list(range(len(ranked_relevance), 0, -1))

    return float(
        ndcg_score(
            [ranked_relevance],
            [scores],
            k=k,
            ignore_ties=True,
        )
    )
