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


def extract_paragraphs(
    pdf_path: str,
    table_bboxes: list[list[tuple]] | None = None,
) -> list[str]:
    """Extract text as paragraph-level blocks using PyMuPDF's own layout
    detection, instead of one flattened string sliced by character count."""
    doc: Any = fitz.open(pdf_path)
    paragraphs = []
    for page_num, page in enumerate(doc):
        page_tables = table_bboxes[page_num] if table_bboxes else []
        for block in page.get_text("blocks"):
            bbox, text = block[:4], block[4].strip()
            if not text:
                continue
            if any(_overlap_fraction(bbox, t) > 0.5 for t in page_tables):
                continue
            paragraphs.append(text)
    doc.close()
    return merge_headings(paragraphs)


def _overlap_fraction(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    return inter / area_a if area_a > 0 else 0.0
