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
