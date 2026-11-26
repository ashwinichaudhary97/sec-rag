# SEC 10-K RAG

A retrieval-augmented generation project focused on SEC 10-K filings.

## Development setup

Requires Python 3.11, uv, Git and Gitleaks.

    uv sync --locked
    uv run pre-commit install

## Development checks

    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q

## SEC filing corpus

The corpus contains 15 original 10-K filings from Apple, Microsoft,
NVIDIA, Alphabet and Amazon for fiscal years 2023-2025.

### Build the corpus

Run from the repository root using PowerShell:

```powershell
$env:SEC_USER_AGENT = Read-Host "Enter your full name and contact email"
uv run --locked python scripts/build_corpus.py
```

Raw HTML is downloaded from SEC EDGAR and parsed using sec2md 0.1.25.

- `data/filings.json`: selected filing records
- `data/raw/`: original HTML filings
- `data/corpus/`: normalized text, elements and section spans
- `data/manifest.json`: source URLs, metadata and SHA-256 hashes

Raw and normalized files are excluded from Git.

### Parsing limitations

Section extraction is incomplete for some elements. Unassigned
content is retained without inventing section labels. Repeated
running headers are removed when recovering missing element offsets.

## Evaluation dataset

The evaluation dataset contains 100 questions based on corpus v1.

| Category | Development | Heldout |
|---|---:|---:|
| Single-hop | 14 | 7 |
| Exact-term | 14 | 6 |
| Multi-hop | 14 | 5 |
| Adversarial | 12 | 5 |
| Table lookup | 16 | 7 |
| Total | 70 | 30 |

- `data/eval/dev.jsonl`: development questions
- `data/eval/heldout.jsonl`: heldout questions

The development and heldout splits contain 8 and 4 unanswerable
questions respectively.

### Annotation format

Example from the development dataset:

```json
{
  "question_id": "Q002",
  "question": "How many full-time and part-time employees did Amazon have at the end of 2024?",
  "category": "single_hop",
  "adversarial_subtype": null,
  "answerable": true,
  "reference_answer": "Approximately 1,556,000.",
  "gold_evidence": [
    {
      "unit_id": "G1",
      "locations": [
        {
          "accession": "0001018724-25-000004",
          "start": 14240,
          "end": 14335,
          "text": "As of December 31, 2024, we employed approximately 1,556,000 full-time and part-time employees."
        }
      ]
    }
  ],
  "expected_constraints": {
    "tickers": [
      "AMZN"
    ],
    "filing_years": []
  },
  "annotation_note": ""
}
```

Evidence offsets refer to the normalized filing text, with an inclusive
start and exclusive end. All evidence units are required, but alternative
locations within a unit are interchangeable.

Unanswerable questions have a null reference answer and empty gold
evidence. Empty metadata constraint lists indicate no restriction.
Filing years refer to the fiscal year covered by the requested filing,
rather than the reporting period of an individual financial figure.

Questions were prepared with LLM assistance and manually verified
in two separate review passes within a week. Reference answers and
supporting evidence were checked against the original SEC filings.
Both splits were frozen after structural validation and overlap checks.

## Retrieval evaluation

Retrieval quality is measured using Evidence Recall@5, Evidence
Recall@10, nDCG@5 and nDCG@10.

Evidence recall measures the fraction of required gold evidence
units retrieved. A chunk matches an evidence location when its
accession matches and its source span fully contains the gold span.
Alternative locations for the same evidence unit count only once.

nDCG uses binary chunk relevance and scikit-learn's ndcg_score,
with the ideal ranking based on relevance across the complete index.

Retrieval quality metrics apply to answerable questions.

## Dense retrieval

The retrieval pipeline splits filings into text and table chunks
while preserving their original source offsets. Chunks are limited
to 224 embedding tokens, with table headers repeated where needed.

The baseline uses all-MiniLM-L6-v2 embeddings and FAISS IndexFlatIP
for cosine similarity search.

### Run retrieval evaluation

```bash
uv run --locked python scripts/build_index.py
uv run --locked python scripts/evaluate_retrieval.py
```

Settings are maintained in `configs/baseline.yaml`.

### Baseline results

| Metric | Development set |
|---|---:|
| Evidence Recall@5 | 45.16% |
| Evidence Recall@10 | 66.94% |
| nDCG@5 | 0.3184 |
| nDCG@10 | 0.3884 |
| p95 retrieval latency | 41.02 ms |

The index contains 7,807 chunks from 15 filings. Retrieval quality
is evaluated on 62 answerable questions, with latency measured
across all 70 development questions.

Detailed results are stored in `results/E00/`.

## Hybrid retrieval

BM25 complements dense retrieval by matching financial terms and
identifiers directly. Reciprocal Rank Fusion combines the two
rankings without requiring comparable similarity scores.

The hybrid retriever reuses the existing dense index and chunks.

Run the hybrid evaluation with:

    uv run --locked python scripts/evaluate_retrieval.py --config configs/hybrid.yaml

| Metric | Dense (E00) | Hybrid (E01) |
|---|---:|---:|
| Evidence Recall@5 | 45.16% | 67.74% |
| Evidence Recall@10 | 66.94% | 80.65% |
| nDCG@5 | 0.3184 | 0.5102 |
| nDCG@10 | 0.3884 | 0.5630 |
| p95 retrieval latency | 41.02 ms | 87.31 ms |

Hybrid retrieval improved evidence coverage and ranking at the
cost of additional retrieval latency.

## Metadata filtering

Company aliases and explicit filing-year references are extracted
using rule-based patterns. Financial reporting periods are not
treated as filing-year constraints.

Both dense and BM25 retrieval are restricted to eligible chunks
before candidate ranking.

Run the filtered hybrid evaluation with:

    uv run --locked python scripts/evaluate_retrieval.py --config configs/filtered_hybrid.yaml

| Metric | Hybrid (E01) | Filtered hybrid (E02) |
|---|---:|---:|
| Evidence Recall@5 | 67.74% | 77.42% |
| Evidence Recall@10 | 80.65% | 83.87% |
| nDCG@5 | 0.5102 | 0.6446 |
| nDCG@10 | 0.5630 | 0.6736 |
| p95 retrieval latency | 87.31 ms | 31.79 ms |

Metadata filtering improved retrieval quality while reducing
p95 retrieval latency by 63.59% on the development dataset.
Constraint extraction matched the annotations for 67 of 70 questions.
