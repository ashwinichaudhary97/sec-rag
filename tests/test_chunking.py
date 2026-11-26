import re

from sec_rag.chunking import (
    normalize_spans,
    render_spans,
    split_table,
    split_text,
    token_count,
)


class SimpleTokenizer:
    def encode(self, text, add_special_tokens=True, **kwargs):
        count = len(re.findall(r"\w+|[^\w\s]", text))
        return list(range(count + (2 if add_special_tokens else 0)))

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {
            "offset_mapping": [
                match.span() for match in re.finditer(r"\w+|[^\w\s]", text)
            ]
        }


TOKENIZER = SimpleTokenizer()


def assert_coverage(text, groups, limit):
    covered = bytearray(len(text))

    for group in groups:
        content = render_spans(text, group)

        assert content
        assert token_count(content, TOKENIZER) <= limit

        for start, end in normalize_spans(text, group):
            assert 0 <= start < end <= len(text)
            covered[start:end] = b"\x01" * (end - start)

    assert all(char.isspace() or covered[index] for index, char in enumerate(text))


def test_adjacent_spans_preserve_original_whitespace():
    text = "First row\n\nSecond row"
    spans = [(0, 9), (11, len(text))]

    assert normalize_spans(text, spans) == [(0, len(text))]
    assert render_spans(text, spans) == text


def test_discontinuous_spans_remain_separate():
    text = "Header\nOmitted row\nSelected row"
    spans = [(0, 6), (text.index("Selected"), len(text))]

    assert normalize_spans(text, spans) == spans
    assert render_spans(text, spans) == "Header\nSelected row"


def test_oversized_text_preserves_content():
    text = " ".join(["A sentence with several words."] * 30)

    groups = [[span] for span in split_text(text, 0, len(text), TOKENIZER, 30)]

    assert len(groups) > 1
    assert_coverage(text, groups, 30)


def test_table_repeats_header():
    header = "| Metric | Amount |"
    text = (
        header
        + "\n| --- | --- |\n"
        + "\n".join(f"| Product {i} | 100 |" for i in range(20))
    )

    groups = list(split_table(text, 0, len(text), TOKENIZER, 40))

    assert len(groups) > 1
    assert_coverage(text, groups, 40)

    for group in groups:
        assert header in render_spans(text, group)


def test_separate_tables_preserve_correct_headers():
    text = (
        "| Metric | 2024 |\n"
        "| --- | --- |\n"
        + "\n".join(f"| Old product {i} | 100 |" for i in range(8))
        + "\n\n| Metric | 2025 |\n"
        "| --- | --- |\n" + "\n".join(f"| New product {i} | 200 |" for i in range(8))
    )

    groups = list(split_table(text, 0, len(text), TOKENIZER, 40))

    assert_coverage(text, groups, 40)

    for group in groups:
        content = render_spans(text, group)

        if "New product" in content:
            assert "| Metric | 2025 |" in content
            assert "| Metric | 2024 |" not in content


def test_header_only_table_preserves_content():
    text = "| Metric | Amount |\n| --- | --- |"

    groups = list(split_table(text, 0, len(text), TOKENIZER, 40))

    assert_coverage(text, groups, 40)
