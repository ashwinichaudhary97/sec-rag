import pytest

from sec_rag.metadata import (
    eligible_positions,
    extract_constraints,
)


@pytest.mark.parametrize(
    ("question", "tickers", "years"),
    [
        ("Microsoft's FY2025 10-K", ["MSFT"], [2025]),
        ("Apple's 2024 annual report", ["AAPL"], [2024]),
        ("NVIDIA's 10-K for fiscal year 2023", ["NVDA"], [2023]),
        ("Google's revenue in 2024", ["GOOGL"], []),
        ("Amazon.com revenue in FY2025", ["AMZN"], []),
        ("Compare Apple and NVIDIA", ["AAPL", "NVDA"], []),
        ("Compare GOOG and Alphabet", ["GOOGL"], []),
        ("What did msft disclose?", ["MSFT"], []),
        ("What is pineapple production?", [], []),
        ("What are the major risk factors?", [], []),
        (
            "Microsoft's 2023 10-K and 2025 10-K",
            ["MSFT"],
            [2023, 2025],
        ),
    ],
)
def test_extract_constraints(question, tickers, years):
    assert extract_constraints(question) == {
        "tickers": tickers,
        "filing_years": years,
    }


@pytest.fixture
def chunks():
    return [
        {"ticker": "AAPL", "filing_year": 2023},
        {"ticker": "AAPL", "filing_year": 2025},
        {"ticker": "MSFT", "filing_year": 2025},
        {"ticker": "NVDA", "filing_year": 2024},
    ]


@pytest.mark.parametrize(
    ("tickers", "years", "expected"),
    [
        ([], [], [0, 1, 2, 3]),
        (["AAPL"], [], [0, 1]),
        ([], [2025], [1, 2]),
        (["AAPL"], [2025], [1]),
        (["AAPL", "MSFT"], [2025], [1, 2]),
        (["NVDA"], [2023], []),
    ],
)
def test_eligible_positions(chunks, tickers, years, expected):
    constraints = {
        "tickers": tickers,
        "filing_years": years,
    }

    assert eligible_positions(chunks, constraints) == expected
