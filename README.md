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
