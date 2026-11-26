import re

COMPANY_ALIASES = {
    "AAPL": ("AAPL", "Apple", "Apple Inc"),
    "MSFT": ("MSFT", "Microsoft", "Microsoft Corporation"),
    "NVDA": ("NVDA", "NVIDIA", "NVIDIA Corporation"),
    "GOOGL": ("GOOGL", "Alphabet", "Google", "GOOG"),
    "AMZN": ("AMZN", "Amazon", "Amazon.com"),
}

COMPANY_PATTERNS = {
    ticker: re.compile(
        r"\b(?:"
        + "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
        + r")\b",
        re.IGNORECASE,
    )
    for ticker, aliases in COMPANY_ALIASES.items()
}

YEAR = r"(?:19|20)\d{2}"

FILING_YEAR_PATTERNS = (
    re.compile(
        rf"\b(?:FY\s*|fiscal\s+year\s+)?({YEAR})"
        r"\s+(?:Form\s+)?(?:10-K\b|annual\s+report\b|filing\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:10-K|annual\s+report|filing)"
        rf"\s+(?:for\s+)?(?:FY\s*|fiscal\s+year\s+)?({YEAR})\b",
        re.IGNORECASE,
    ),
)


def extract_constraints(question: str) -> dict:
    tickers = sorted(
        ticker
        for ticker, pattern in COMPANY_PATTERNS.items()
        if pattern.search(question)
    )

    filing_years = sorted(
        {
            int(match.group(1))
            for pattern in FILING_YEAR_PATTERNS
            for match in pattern.finditer(question)
        }
    )

    return {
        "tickers": tickers,
        "filing_years": filing_years,
    }


def eligible_positions(
    chunks: list[dict],
    constraints: dict,
) -> list[int]:
    tickers = set(constraints["tickers"])
    years = set(constraints["filing_years"])

    return [
        position
        for position, chunk in enumerate(chunks)
        if (not tickers or chunk["ticker"] in tickers)
        and (not years or chunk["filing_year"] in years)
    ]
