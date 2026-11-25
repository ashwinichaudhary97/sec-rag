import pytest

from sec_rag.ingestion import fetch
from sec_rag.parsing import normalize_element


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("PART I\n\nItem 1", ""),
        ("PART II\n\nItem 8\n\nRevenue", "Revenue"),
        ("PART I\n\nItem 1B, 2, 3, 4", ""),
        ("Revenue increased by 10%.", "Revenue increased by 10%."),
    ],
)
def test_normalize_element(content, expected):
    assert normalize_element(content) == expected


def test_fetch_requires_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        fetch("https://www.sec.gov")
