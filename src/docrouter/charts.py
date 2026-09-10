import re
from functools import lru_cache

import fitz

RASTER_CLUSTER_PADDING = 45.0
VECTOR_CLUSTER_PADDING = 20.0


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Loaded on first use, not at import - importing this module should not
    require a HuggingFace round-trip on a cold cache."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def _union(rects: list[fitz.Rect]) -> fitz.Rect:
    """Numeric union. fitz's |, |= and include_rect all silently IGNORE a
    zero-area operand, and chart axes and tick marks are exactly that - a
    fitz union quietly crops every axis off the figure it is bounding."""
    return fitz.Rect(
        min(r.x0 for r in rects),
        min(r.y0 for r in rects),
        max(r.x1 for r in rects),
        max(r.y1 for r in rects),
    )


def _as_tuple(rect: fitz.Rect) -> tuple[float, float, float, float]:
    """fitz.Rect is iterable at runtime but its stubs do not declare
    __iter__, so tuple(rect) does not type-check."""
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _touches(a: fitz.Rect, b: fitz.Rect, padding: float) -> bool:
    """Proximity test that works on degenerate rects, which Rect.intersects()
    reports as touching nothing."""
    return (
        a.x0 - padding <= b.x1
        and b.x0 <= a.x1 + padding
        and a.y0 - padding <= b.y1
        and b.y0 <= a.y1 + padding
    )


def cluster_rects(
    rects: list[fitz.Rect], padding: float
) -> list[tuple[fitz.Rect, list[int]]]:
    """Single-linkage clustering by proximity, via union-find. Returns each
    cluster's bounding rect alongside the indices that formed it, so callers
    can inspect the members without re-deriving membership geometrically."""
    n = len(rects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = find(i), find(j)
            if ri != rj and _touches(rects[i], rects[j], padding):
                parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [(_union([rects[i] for i in idxs]), idxs) for idxs in groups.values()]


def get_page_raster_bboxes(
    page: fitz.Page,
    padding: float = RASTER_CLUSTER_PADDING,
    min_side: float = 8.0,
    min_area: float = 400.0,
    max_page_frac: float = 0.8,
) -> list[tuple[float, float, float, float]]:
    """Raster figure regions on one page.

    get_images() lists the same xref once per resource-table reference, so
    identical rects come back repeatedly (16 rects / 12 unique on page 13 of
    attention.pdf) - dedupe first. Complex figures are then embedded as many
    fragments, so cluster them: without this, page 14 yields a 2.4x94pt sliver
    that costs an API call and pollutes retrieval with a description of
    nothing. A region covering most of the page is a page scan or background,
    not a figure, so it is dropped and left to the OCR path."""
    page_area = page.rect.get_area()
    seen: set[tuple[float, ...]] = set()
    rects: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            key = tuple(round(v, 2) for v in _as_tuple(rect))
            if key in seen:
                continue
            seen.add(key)
            rects.append(fitz.Rect(rect) & page.rect)

    result = []
    for rect, _ in cluster_rects(rects, padding):
        if rect.width < min_side or rect.height < min_side:
            continue
        if rect.get_area() < min_area:
            continue
        if page_area and rect.get_area() > max_page_frac * page_area:
            continue
        result.append(_as_tuple(rect))
    return result


def _is_chromatic(fill, tol: float = 0.03) -> bool:
    try:
        return max(fill) - min(fill) > tol
    except TypeError:  # grayscale colorspaces hand back a bare float
        return False


def _candidate_drawings(
    page: fitz.Page, max_page_frac: float = 0.5
) -> list[tuple[fitz.Rect, dict]]:
    """Drawings that could belong to a figure. Page borders and background
    fills span the sheet and would chain every unrelated mark into one
    cluster, so they are dropped before clustering."""
    page_area = page.rect.get_area()
    result = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"]) & page.rect
        if rect.is_infinite or (rect.width <= 0 and rect.height <= 0):
            continue
        if page_area and rect.get_area() > max_page_frac * page_area:
            continue
        result.append((rect, drawing))
    return result


def get_page_vector_bboxes(
    page: fitz.Page,
    padding: float = VECTOR_CLUSTER_PADDING,
    min_page_frac: float = 0.01,
    max_page_frac: float = 0.6,
    max_text_frac: float = 0.35,
    min_items: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Vector figure regions on one page.

    Unioning every drawing on the page produced a bbox covering the whole
    sheet, and because these regions are also the text-exclusion mask, that
    deleted the page's entire text layer (charts.pdf: 100 paragraphs -> 0).
    Clustering keeps figures separate, and the guards below make the failure
    mode a missed figure rather than a silently emptied page: reject a
    cluster that spans most of the sheet, and reject one that swallows more
    than max_text_frac of the page's text blocks - that is a page, not a
    chart."""
    page_area = page.rect.get_area()
    if not page_area:
        return []
    candidates = _candidate_drawings(page)
    if not candidates:
        return []

    blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if b[4].strip()]
    result = []
    for rect, idxs in cluster_rects([r for r, _ in candidates], padding):
        frac = rect.get_area() / page_area
        if frac < min_page_frac or frac > max_page_frac:
            continue

        members = [candidates[i][1] for i in idxs]
        chart_like = (
            max((len(m["items"]) for m in members), default=0) >= min_items
            or len(members) >= 4
            or any(_is_chromatic(m["fill"]) for m in members if m.get("fill"))
        )
        if not chart_like:
            continue

        if blocks:
            swallowed = sum(
                1
                for b in blocks
                if b.get_area() and (b & rect).get_area() > 0.5 * b.get_area()
            )
            if swallowed > max_text_frac * len(blocks):
                continue

        padded = fitz.Rect(rect) + (-padding, -padding, padding, padding)
        result.append(_as_tuple(padded & page.rect))
    return result


def get_figure_bboxes(
    pdf_path: str,
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """(raster_per_page, vector_per_page) from a single document open."""
    doc = fitz.open(pdf_path)
    try:
        raster = [get_page_raster_bboxes(page) for page in doc]
        vector = [get_page_vector_bboxes(page) for page in doc]
    finally:
        doc.close()
    return raster, vector


def get_raster_bboxes(pdf_path: str) -> list[list[tuple]]:
    return get_figure_bboxes(pdf_path)[0]


def get_vector_chart_bboxes(pdf_path: str) -> list[list[tuple]]:
    return get_figure_bboxes(pdf_path)[1]


def render_page_region(page: fitz.Page, bbox: tuple, dpi: int = 150) -> bytes:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(bbox))
    return pix.tobytes("png")


def render_region(pdf_path: str, page_num: int, bbox: tuple, dpi: int = 150) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        return render_page_region(doc[page_num], bbox, dpi=dpi)
    finally:
        doc.close()


def _split_by_token_limit(
    text: str, max_tokens: int = 254, overlap_tokens: int = 20
) -> list[str]:
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    encoding = _get_tokenizer()(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoding["offset_mapping"]
    if len(offsets) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == len(offsets):
            break
        start = end - overlap_tokens
    return chunks


def split_chart_description(description: str, source_label: str = "") -> list[str]:
    """Gemini organizes multi-panel chart descriptions with numbered markdown
    headers - split on those instead of embedding one oversized blob that
    risks silent truncation."""
    header_parts = re.split(r"(?=^\*\*.*?:\*\*)", description, flags=re.MULTILINE)
    header_parts = [p.strip() for p in header_parts if p.strip()]
    result = []
    for part in header_parts:
        pieces = _split_by_token_limit(part)
        if source_label:
            pieces = [f"[{source_label}] {p}" for p in pieces]
        result.extend(pieces)
    return result
