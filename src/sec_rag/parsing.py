import re
from pathlib import Path

from sec2md import Parser, extract_sections

RUNNING_HEADER = re.compile(r"^PART [IVX]+\n\nItem [0-9A-Z, ]+(?:\n\n|$)")


def normalize_element(content: str) -> str:
    return RUNNING_HEADER.sub("", content).strip()


def parse_filing(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    pages = Parser(html).get_pages(include_images=False)

    text = "\n\n".join(page.content for page in pages)
    elements = []
    element_lookup = {}
    skipped_headers = set()
    offset = 0

    for page in pages:
        for element in page.elements or []:
            content = element.content
            start = element.content_start_offset
            end = element.content_end_offset

            if start is None or end is None:
                content = normalize_element(content)

                if not content:
                    skipped_headers.add(element.id)
                    continue

                if page.content.count(content) != 1:
                    raise ValueError(f"Cannot recover offset: {element.id}")

                start = page.content.index(content)
                end = start + len(content)

            start += offset
            end += offset

            if text[start:end] != content:
                raise ValueError(f"Offset mismatch: {element.id}")

            record = {
                "id": element.id,
                "kind": element.kind,
                "page": page.number,
                "start": start,
                "end": end,
            }

            elements.append(record)
            element_lookup[element.id] = record

        offset += len(page.content) + 2

    sections = extract_sections(pages, filing_type="10-K")
    section_spans = []

    for section in sections:
        for page in section.pages:
            for element in page.elements or []:
                if element.id in skipped_headers:
                    continue

                original = element_lookup[element.id]
                content = text[original["start"] : original["end"]]

                if content != element.content and content != normalize_element(
                    element.content
                ):
                    raise ValueError(f"Section element modified: {element.id}")

                section_spans.append(
                    {
                        "item": section.item,
                        "title": section.item_title,
                        "element_id": element.id,
                        "page": original["page"],
                        "start": original["start"],
                        "end": original["end"],
                    }
                )

    return {
        "text": text,
        "elements": elements,
        "sections": section_spans,
    }
