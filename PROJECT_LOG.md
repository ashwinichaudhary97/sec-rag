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
