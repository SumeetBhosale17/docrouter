from typing import Any

import fitz


def has_text_layer(pdf_path: str, char_threshold: int = 20) -> list[bool]:
    """Return one bool per page: True if the page has real extractable
    text layer, False if it looks like a scanned/image-only page."""
    doc: Any = fitz.open(pdf_path)
    results: list[bool] = []
    for page in doc:
        text = page.get_text()
        results.append(len(text.strip()) > char_threshold)
    doc.close()
    return results


def has_visual_content(pdf_path: str) -> list[bool]:
    """Return one bool per page: True if the page contains an embedded
    raster image (photo, or a chart exported as a bitmap)."""
    doc: Any = fitz.open(pdf_path)
    results: list[bool] = [len(page.get_images(full=True)) > 0 for page in doc]
    doc.close()
    return results


def has_drawing(pdf_path: str, min_items: int = 4) -> list[bool]:
    """Return one bool per page: True if the page contains vector graphics
    (line, curves, rectangles) beyond a trivial few - likely a chart or
    diagram drawn as vectors rather than embedded as a raster image."""
    doc = fitz.open(pdf_path)
    results: list[bool] = [len(page.get_drawings()) > min_items for page in doc]
    doc.close()
    return results


def drawing_diagnostics(pdf_path: str) -> None:
    doc: Any = fitz.open(pdf_path)
    for _i, page in enumerate(doc):
        for d in page.get_drawings():
            print(f"type={d['type']!r} fill={d.get('fill')} n_items={len(d['items'])}")
    doc.close()


def is_chromatic(rgb: tuple[float, float, float], tol: float = 0.03) -> bool:
    r, g, b = rgb
    return max(r, g, b) - min(r, g, b) > tol


def has_chart_like_drawing(pdf_path: str, min_items: int = 2) -> list[bool]:
    doc: Any = fitz.open(pdf_path)
    results = []
    for page in doc:
        drawings = page.get_drawings()
        max_items = max((len(d["items"]) for d in drawings), default=0)
        has_color_fill = any(
            d.get("fill") is not None and is_chromatic(d["fill"]) for d in drawings
        )
        results.append(max_items >= min_items or has_color_fill)
    doc.close()
    return results
