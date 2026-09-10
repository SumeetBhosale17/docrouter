import re
from typing import Any

import fitz

from docrouter.ocr import ocr_page_text

TEXT_LAYER_MIN_CHARS = 20

# Private Use Area. A PDF whose fonts ship no usable ToUnicode CMap extracts
# its glyphs as these, and both PyMuPDF and pdfplumber return them verbatim.
_PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")
_PUA_BETWEEN_DIGITS_RE = re.compile(
    r"(?<=[0-9])[\ue000-\uf8ff\U000f0000-\U000ffffd](?=[0-9])"
)


def normalize_pua(text: str) -> str:
    """Make private-use glyphs harmless.

    attention.pdf carries 56 of them, and they fuse into neighbouring words
    to make tokens no query can match: "0.9" indexes as "0<U+E004>9".

    The codes are font-relative and genuinely ambiguous - U+E004 is a period
    in "0.9" but an ellipsis in "(x1, ..., xn)", and U+E000 is a period in
    "d^-0.5" but a summation sign in "q.k = SUM q_i k_i" - so there is no
    honest per-character mapping to recover. Only one case is unambiguous: a
    private-use glyph flanked by digits is a decimal point. That is restored;
    everything else becomes a space, which invents no character that was not
    there while still letting the surrounding words tokenize cleanly."""
    return _PUA_RE.sub(" ", _PUA_BETWEEN_DIGITS_RE.sub(".", text))


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


def _split_ocr_text(text: str) -> list[str]:
    """Tesseract inserts blank line between segmented blocks -
    mirrors the paragraph boundaries get_text('blocks') gives natively.
    Use it the same way, instead of treating the whole page as one blob."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def extract_paragraphs(
    pdf_path: str,
    exclude_bboxes: list[list[tuple]] | None = None,
) -> list[str]:
    """Extract text as paragraph-level blocks using PyMuPDF's own layout
    detection, instead of one flattened string sliced by character count."""
    doc: Any = fitz.open(pdf_path)
    paragraphs = []
    for page_num, page in enumerate(doc):
        page_exclude = exclude_bboxes[page_num] if exclude_bboxes else []
        text = page.get_text()
        if len(text.strip()) > TEXT_LAYER_MIN_CHARS:
            for block in page.get_text("blocks"):
                bbox, block_text = block[:4], normalize_pua(block[4]).strip()
                if not block_text or any(
                    _overlap_fraction(bbox, t) > 0.5 for t in page_exclude
                ):
                    continue
                paragraphs.append(block_text)
        else:
            ocr_text = normalize_pua(ocr_page_text(pdf_path, page_num)).strip()
            if ocr_text:
                paragraphs.extend(_split_ocr_text(ocr_text))
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
