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
