from typing import Any

import fitz


def _look_like_heading(text: str, max_len: int = 60) -> bool:
    """Heuristic: short, and doesn't end in sentence-ending punctuation -
    real sentences end in '.', '!', '?', or ':'; titles usually don't."""
    return len(text) <= max_len and not text.rstrip().endswith((".", "!", "?", ":"))


def merge_headings(paragraphs: list[str]) -> list[str]:
    """Merge a heading-like block into the paragraph immediately following
    it, so heading + its body become one retrievable unit instead of two
    - the heading alone carries almost no information on its own."""
    merged = []
    i = 0
    while i < len(paragraphs):
        current = paragraphs[i]
        if _look_like_heading(current) and i + 1 < len(paragraphs):
            merged.append(f"{current}\n{paragraphs[i + 1]}")
            i += 2
        else:
            merged.append(current)
            i += 1
    return merged


def extract_text(pdf_path: str) -> str:
    doc: Any = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_paragraphs(pdf_path: str) -> list[str]:
    """Extract text as paragraph-level blocks using PyMuPDF's own layout
    detection, instead of one flattened string sliced by character count."""
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if text:
                paragraphs.append(text)
    doc.close()
    return merge_headings(paragraphs)
