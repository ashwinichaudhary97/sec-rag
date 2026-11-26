import re
from collections.abc import Iterator

TARGET_TOKENS = 224


def token_count(text, tokenizer):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    )


def trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def normalize_spans(text, spans):
    """Preserve original whitespace between adjacent source spans."""
    merged = []

    for start, end in sorted(spans):
        start, end = trim_span(text, start, end)

        if start >= end:
            continue

        if merged:
            previous_start, previous_end = merged[-1]

            if start <= previous_end or not text[previous_end:start].strip():
                merged[-1] = (
                    previous_start,
                    max(previous_end, end),
                )
                continue

        merged.append((start, end))

    return merged


def render_spans(text, spans):
    return "\n".join(text[start:end] for start, end in normalize_spans(text, spans))


def fits(text, spans, tokenizer, limit):
    return token_count(render_spans(text, spans), tokenizer) <= limit


def split_tokens(text, start, end, tokenizer, limit):
    """Split oversized text without losing original character offsets."""
    while start < end:
        if token_count(text[start:end], tokenizer) <= limit:
            yield (start, end)
            return

        encoded = tokenizer(
            text[start:end],
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        offsets = encoded["offset_mapping"]
        budget = limit - tokenizer.num_special_tokens_to_add(pair=False)

        stop = start + offsets[min(budget, len(offsets)) - 1][1]

        while token_count(text[start:stop], tokenizer) > limit:
            stop -= 1

        if stop <= start:
            raise ValueError("Unable to split oversized text")

        yield (start, stop)
        start = stop


def split_text(text, start, end, tokenizer, limit):
    """Prefer sentence boundaries before token-level splitting."""
    boundaries = [
        start + match.end()
        for match in re.finditer(r"(?<=[.!?])\s+|\n\n+", text[start:end])
    ]
    boundaries.append(end)

    current = start

    for boundary in boundaries:
        if boundary <= current:
            continue

        if token_count(text[current:boundary], tokenizer) <= limit:
            yield from ()
            continue

        previous = max(
            (point for point in boundaries if current < point < boundary),
            default=current,
        )

        if previous > current:
            yield (current, previous)
            current = previous

        if token_count(text[current:boundary], tokenizer) > limit:
            yield from split_tokens(text, current, boundary, tokenizer, limit)
            current = boundary

    if current < end:
        yield (current, end)


def split_single_table(text, start, end, tokenizer, limit):
    """Split Markdown tables into row groups with repeated headers."""
    lines = []
    position = start

    for line in text[start:end].splitlines(keepends=True):
        lines.append((position, position + len(line)))
        position += len(line)

    separator = next(
        (
            index
            for index, (left, right) in enumerate(lines)
            if re.fullmatch(
                r"\s*\|(?:\s*:?-+:?\s*\|)+\s*",
                text[left:right],
            )
        ),
        None,
    )

    if separator is None:
        for span in split_text(text, start, end, tokenizer, limit):
            yield [span]
        return

    header = (start, lines[separator][1])

    if not fits(text, [header], tokenizer, limit):
        for span in split_text(text, start, end, tokenizer, limit):
            yield [span]
        return

    group = [header]

    for left, right in lines[separator + 1 :]:
        candidate = [*group, (left, right)]

        if fits(text, candidate, tokenizer, limit):
            group = candidate
            continue

        if len(group) > 1:
            yield group

        group = [header]

        if fits(text, [header, (left, right)], tokenizer, limit):
            group.append((left, right))
            continue

        # Reserve space for the repeated header when splitting a long row.
        budget = limit - token_count(text[header[0] : header[1]], tokenizer)

        if budget <= 8:
            yield [header]

            for span in split_tokens(text, left, right, tokenizer, limit):
                yield [span]
            continue

        for span in split_tokens(text, left, right, tokenizer, budget):
            if fits(text, [header, span], tokenizer, limit):
                yield [header, span]
            else:
                yield [span]

    if len(group) > 1:
        yield group
    elif separator == len(lines) - 1:
        yield [header]


def split_table(text, start, end, tokenizer, limit):
    """Split separate Markdown tables independently."""
    lines = []
    position = start

    for line in text[start:end].splitlines(keepends=True):
        lines.append((position, position + len(line)))
        position += len(line)

    table_starts = []

    for index in range(1, len(lines)):
        left, right = lines[index]
        previous_left, previous_right = lines[index - 1]

        previous = text[previous_left:previous_right].strip()

        if (
            previous.startswith("|")
            and previous.endswith("|")
            and re.fullmatch(
                r"\s*\|(?:\s*:?-+:?\s*\|)+\s*",
                text[left:right],
            )
        ):
            table_starts.append(index - 1)

    if len(table_starts) <= 1:
        yield from split_single_table(text, start, end, tokenizer, limit)
        return

    cursor = start

    for index, table_start in enumerate(table_starts):
        next_start = (
            table_starts[index + 1] if index + 1 < len(table_starts) else len(lines)
        )

        table_end = table_start + 2

        while table_end < next_start:
            left, right = lines[table_end]
            line = text[left:right].strip()

            if not (line.startswith("|") and line.endswith("|")):
                break

            table_end += 1

        block_end = lines[table_end - 1][1]

        if fits(text, [(cursor, block_end)], tokenizer, limit):
            yield [(cursor, block_end)]
        else:
            yield from split_single_table(text, cursor, block_end, tokenizer, limit)

        cursor = block_end

    if text[cursor:end].strip():
        for span in split_text(text, cursor, end, tokenizer, limit):
            yield [span]


def chunk_document(
    document: dict,
    tokenizer,
    limit: int = TARGET_TOKENS,
) -> Iterator[dict]:
    """Create structure-aware chunks with original source spans."""
    text = document["text"]
    metadata = document["metadata"]

    sections = {
        section["element_id"]: section["item"] for section in document["sections"]
    }

    chunk_number = 0
    pending = []
    pending_section = None

    def make_chunk(spans, section, kind):
        nonlocal chunk_number

        spans = normalize_spans(text, spans)

        content = render_spans(text, spans)
        count = token_count(content, tokenizer)

        if not content or count > limit:
            raise ValueError("Invalid chunk content or token count")

        chunk_number += 1

        return {
            "chunk_id": f"{metadata['Accession']}:{chunk_number:05d}",
            "accession": metadata["Accession"],
            "ticker": metadata["Ticker"],
            "company": metadata["Company"],
            "filing_year": metadata["FiscalYear"],
            "period_end": metadata["PeriodEnd"],
            "filing_type": metadata["FilingType"],
            "section": section,
            "kind": kind,
            "text": content,
            "source_spans": [{"start": start, "end": end} for start, end in spans],
            "token_count": count,
        }

    for element in document["elements"]:
        start, end = trim_span(text, element["start"], element["end"])

        if start == end:
            continue

        section = sections.get(element["id"])
        span = (start, end)

        if element["kind"] == "table":
            if pending:
                yield make_chunk(pending, pending_section, "text")
                pending = []

            if fits(text, [span], tokenizer, limit):
                yield make_chunk([span], section, "table")
            else:
                for group in split_table(text, start, end, tokenizer, limit):
                    yield make_chunk(group, section, "table")

            continue

        if pending and (
            section != pending_section
            or not fits(text, [*pending, span], tokenizer, limit)
        ):
            yield make_chunk(pending, pending_section, "text")
            pending = []

        if fits(text, [span], tokenizer, limit):
            pending.append(span)
            pending_section = section
        else:
            for part in split_text(text, start, end, tokenizer, limit):
                yield make_chunk([part], section, "text")

    if pending:
        yield make_chunk(pending, pending_section, "text")
