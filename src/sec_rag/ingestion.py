import os
import time
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url: str) -> bytes:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise ValueError("SEC_USER_AGENT is not set.")

    request = Request(url, headers={"User-Agent": user_agent})

    time.sleep(1)
    with urlopen(request, timeout=60) as response:
        return response.read()


def download_filing(url: str, destination: Path) -> Path:
    content = fetch(url)

    if not content.strip():
        raise ValueError(f"Empty response from {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def download_corpus(records: list[dict], raw_dir: Path) -> None:
    for record in records:
        accession = record["Accession"]
        cik = {
            "AAPL": 320193,
            "MSFT": 789019,
            "NVDA": 1045810,
            "GOOGL": 1652044,
            "AMZN": 1018724,
        }[record["Ticker"]]
        document = record["Document"]

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik}/{accession.replace('-', '')}/{document}"
        )

        destination = raw_dir / document

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Cached: {document}")
            continue

        download_filing(url, destination)
        print(f"Downloaded: {document}")
