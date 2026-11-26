# Project Log

## Development environment

- OS: Windows
- Shell: PowerShell 5.1
- CPU: Intel Core i5-10210U @ 1.60 GHz
- CPU cores: 4 physical, 8 logical
- RAM: 7.83 GB
- GPU: Intel UHD Graphics
- Python: 3.11.9
- uv: 0.13.3

## Development log

### 2026-11-25 - Project initialization

- Initialized repository with uv and src package layout.
- Configured Ruff, pytest and pre-commit.
- Configured Gitleaks 8.30.1 for staged secret scanning.
- Added GitHub Actions workflow.
- Added environment variable template.
- Verified dependency lockfile.
- Local linting, formatting, smoke test and secret scanning passed.
- Remote CI execution not yet verified.

## Step 1 - Corpus ingestion

- Step 0 remote CI verified successfully.
- Corpus version: v1
- Companies: AAPL, MSFT, NVDA, GOOGL, AMZN
- Fiscal years: 2023-2025
- Filings: 15 original 10-Ks
- Parser: sec2md 0.1.25
- Source: SEC EDGAR
- All filing identities validated.
- All 15 filings parsed with strict element-offset validation.
- Five representative financial tables inspected for structural quality.
- Missing offsets caused by running headers recovered through exact matching.
- Unassigned section elements retained.
- Cached corpus rebuild produced an identical manifest.
- Ruff, six automated tests and dependency lock checks passed.

Limitations: incomplete section assignments and sampled rather than
exhaustive financial content validation.

Corpus v1 frozen using the hashes in data/manifest.json.

## Step 2 - Evaluation dataset

- Dataset version: v1
- Corpus version: v1
- Prepared 10 pilot questions before completing the development split.
- Questions prepared with LLM assistance and manually verified in
  two separate passes within a week.
- Reference answers and supporting evidence checked against SEC filings.
- Heldout questions prepared separately from the development set.
- Checked development and heldout questions for overlap.
- Verified all 158 evidence locations against normalized filing text.
- Question uniqueness, schema, metadata constraints and corpus integrity
  checks passed.
- Both datasets frozen after structural validation.

### Dataset hashes (SHA-256)

Development:
09977be282f26ade4595b0444424780093f0afb84cf719b1972c59198ea00238

Heldout:
e04b947d06a37a4a128f681aad636cb878028214142f636ed1969b5838a99129

## Step 3 - Retrieval evaluation metrics

- Implemented evidence matching and Evidence Recall@K.
- Used scikit-learn 1.9.1 for nDCG@K.
- Selected @10 as the primary cutoff and @5 as the secondary cutoff.
- Tested accession matching, span containment, alternative evidence
  locations, recall cutoffs and nDCG normalization.
- All 15 unit tests passed, including 9 retrieval evaluation tests.

## Step 4 - Dense retrieval (E00)

Parent commit: `a7d52d1`

Corpus and evaluation dataset: v1

Configuration: `configs/baseline.yaml`

### Implementation decisions

Used structure-aware chunking to retain financial table context
and trace retrieved evidence back to the original filings.
Multiple source spans support repeated table headers without
incorrectly treating omitted content as retrieved evidence.

Corpus validation found no missing content within parsed elements.
Two uncovered characters outside elements belonged to a standalone
page number.

After correcting span handling and table boundaries, 72 of 76
development evidence units were available within individual chunks.
The remaining four were retained as a baseline limitation.

### Observations

Embedding and indexing took 444.8 seconds on CPU, producing
an 11.44 MiB FAISS index.

Table lookup was the weakest retrieval category. Although 87.50%
of its evidence units were available in the index, Recall@10
was only 37.50%, indicating a retrieval limitation beyond chunking.

The retrieval smoke test returned similar Microsoft disclosures
from multiple filing years, highlighting the limitations of
unrestricted semantic search.

### Verification

Corpus hashes, source spans, token limits, embedding normalization
and saved index dimensions were verified. All 30 unit tests and
the real-index retrieval smoke test passed.

Development retrieval results are recorded in `results/E00/`.
The heldout dataset was not used.

## Step 5 - Hybrid retrieval (E01)

Parent commit: `08355e3`

Parent experiment: E00

Configuration: `configs/hybrid.yaml`

Added BM25 with financial-text tokenization and combined lexical
and dense rankings using Reciprocal Rank Fusion. The existing
chunks and dense index were reused to isolate the retrieval change.

Table lookup showed a substantial improvement, with Recall@5
increasing from 6.25% to 43.75%, although it remained the weakest
category. Hybrid Recall@5 exceeded dense Recall@10.

No question category regressed in retrieval quality. The additional
latency was considered acceptable given the improvement in evidence
coverage and ranking.

Decision: retain hybrid retrieval.

BM25 compatibility was verified, all 35 unit tests passed, and
development retrieval evaluation completed successfully.

Detailed results are stored in `results/E01/`. The corpus and
evaluation datasets remain unchanged.
