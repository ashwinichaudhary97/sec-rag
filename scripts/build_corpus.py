import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from sec_rag.ingestion import download_corpus
from sec_rag.parsing import parse_filing

CIKS = {
    "AAPL": 320193,
    "MSFT": 789019,
    "NVDA": 1045810,
    "GOOGL": 1652044,
    "AMZN": 1018724,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = json.loads(Path("data/filings.json").read_text(encoding="utf-8"))

    raw_dir = Path("data/raw")
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    download_corpus(records, raw_dir)

    manifest = []

    for record in records:
        raw = raw_dir / record["Document"]
        parsed = corpus_dir / f"{raw.stem}.json"

        fiscal_year = int(record["PeriodEnd"][:4])
        accession = record["Accession"]
        cik = CIKS[record["Ticker"]]

        result = parse_filing(raw)
        result["metadata"] = {
            **record,
            "FiscalYear": fiscal_year,
            "FilingType": "10-K",
        }

        parsed.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest.append(
            {
                "ticker": record["Ticker"],
                "company": record["Company"],
                "fiscal_year": fiscal_year,
                "period_end": record["PeriodEnd"],
                "filing_date": record["Filed"],
                "filing_type": "10-K",
                "accession": accession,
                "source_url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik}/{accession.replace('-', '')}/{raw.name}"
                ),
                "raw_path": raw.as_posix(),
                "parsed_path": parsed.as_posix(),
                "raw_sha256": sha256(raw),
                "parsed_sha256": sha256(parsed),
                "parser_version": version("sec2md"),
            }
        )

        print(f"Parsed: {raw.name}")

    Path("data/manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Corpus built: {len(manifest)} filings")


if __name__ == "__main__":
    main()
