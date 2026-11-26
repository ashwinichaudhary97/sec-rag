from math import log2

import pytest

from sec_rag.evaluation import (
    covered_units,
    evidence_recall_at_k,
    ndcg_at_k,
)


def chunk(accession, start, end):
    return {
        "accession": accession,
        "source_spans": [{"start": start, "end": end}],
    }


def location(accession, start, end):
    return {
        "accession": accession,
        "start": start,
        "end": end,
    }


def multi_span_chunk(accession, spans):
    return {
        "accession": accession,
        "source_spans": [{"start": start, "end": end} for start, end in spans],
    }


@pytest.fixture
def gold():
    return [
        {
            "unit_id": "G1",
            "locations": [
                location("filing-a", 10, 20),
                location("filing-b", 30, 40),
            ],
        },
        {
            "unit_id": "G2",
            "locations": [
                location("filing-a", 50, 60),
            ],
        },
    ]


def test_evidence_matching(gold):
    assert covered_units(chunk("filing-a", 10, 20), gold) == {"G1"}
    assert covered_units(chunk("filing-a", 15, 25), gold) == set()
    assert covered_units(chunk("filing-c", 10, 20), gold) == set()
    assert covered_units(chunk("filing-a", 0, 70), gold) == {"G1", "G2"}


def test_alternative_locations(gold):
    retrieved = [
        chunk("filing-a", 0, 25),
        chunk("filing-b", 25, 45),
    ]

    assert evidence_recall_at_k(retrieved, gold, 5) == 0.5


def test_recall_cutoffs(gold):
    retrieved = [
        chunk("filing-a", 0, 25),
        *[chunk("filing-c", i, i + 10) for i in range(4)],
        chunk("filing-a", 45, 65),
    ]

    assert evidence_recall_at_k(retrieved, gold, 5) == 0.5
    assert evidence_recall_at_k(retrieved, gold, 10) == 1.0


def test_perfect_ndcg(gold):
    indexed = [
        chunk("filing-a", 0, 25),
        chunk("filing-a", 45, 65),
    ]

    assert ndcg_at_k(indexed, indexed, gold, 5) == pytest.approx(1.0)


def test_ndcg_ranking(gold):
    relevant = chunk("filing-a", 0, 25)
    irrelevant = chunk("filing-c", 0, 25)
    indexed = [irrelevant, relevant]

    assert ndcg_at_k(indexed, indexed, gold, 5) == pytest.approx(1 / log2(3))


def test_ndcg_full_index(gold):
    indexed = [
        chunk("filing-a", 0, 25),
        chunk("filing-a", 45, 65),
    ]

    expected = 1 / (1 + 1 / log2(3))

    assert ndcg_at_k(indexed[:1], indexed, gold, 5) == pytest.approx(expected)


def test_empty_retrieval(gold):
    indexed = [chunk("filing-a", 0, 25)]

    assert evidence_recall_at_k([], gold, 5) == 0.0
    assert ndcg_at_k([], indexed, gold, 5) == 0.0


def test_no_relevant_chunks(gold):
    indexed = [chunk("filing-c", 0, 25)]

    assert ndcg_at_k(indexed, indexed, gold, 5) == 0.0


def test_unanswerable_question():
    with pytest.raises(ValueError):
        evidence_recall_at_k([], [], 5)

    with pytest.raises(ValueError):
        ndcg_at_k([], [], [], 5)


def test_contiguous_spans(gold):
    candidate = multi_span_chunk("filing-a", [(5, 15), (15, 25)])

    assert covered_units(candidate, gold) == {"G1"}


def test_overlapping_spans(gold):
    candidate = multi_span_chunk("filing-a", [(14, 25), (5, 16)])

    assert covered_units(candidate, gold) == {"G1"}


def test_discontinuous_spans(gold):
    candidate = multi_span_chunk("filing-a", [(5, 14), (15, 25), (45, 65)])

    assert covered_units(candidate, gold) == {"G2"}


def test_evidence_cannot_span_multiple_chunks(gold):
    retrieved = [
        chunk("filing-a", 5, 15),
        chunk("filing-a", 15, 25),
    ]

    assert evidence_recall_at_k(retrieved, gold, 5) == 0.0


def test_single_result_ndcg(gold):
    indexed = [chunk("filing-a", 0, 25)]

    assert ndcg_at_k(indexed, indexed, gold, 1) == 1.0
