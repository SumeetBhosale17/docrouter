import fitz


def get_raster_bboxes(pdf_path: str) -> list[list[tuple[float, float, float, float]]]:
    """One list of raster image bounding boxes per page"""
    doc = fitz.open(pdf_path)
    result = []
    for page in doc:
        bboxes = []
        for img in page.get_images(full=True):
            xref = img[0]
            bboxes.extend(tuple(r) for r in page.get_image_rects(xref))
        result.append(bboxes)
    doc.close()
    return result


def get_vector_chart_bbox(
    page: fitz.Page, padding: float = 20.0
) -> tuple[float, float, float, float] | None:
    """Union bbox of a page's chart-like vector drawings"""
    drawings = page.get_drawings()
    if not drawings:
        return None
    max_items = max(len(d["items"]) for d in drawings)
    has_color_fill = any(
        d.get("fill") and max(d["fill"]) - min(d["fill"]) > 0.03 for d in drawings
    )
    if max_items < 2 and not has_color_fill:
        return None
    rects = [d["rect"] for d in drawings]
    x0 = min(r.x0 for r in rects) - padding
    y0 = min(r.y0 for r in rects) - padding
    x1 = max(r.x1 for r in rects) + padding
    y1 = max(r.y1 for r in rects) + padding
    return tuple(fitz.Rect(x0, y0, x1, y1) & page.rect)


def get_vector_chart_bboxes(
    pdf_path: str,
) -> list[list[tuple[float, float, float, float]]]:
    doc = fitz.open(pdf_path)
    result = []
    for page in doc:
        bbox = get_vector_chart_bbox(page)
        result.append([bbox] if bbox is not None else [])
    doc.close()
    return result


def render_region(pdf_path: str, page_num: int, bbox: tuple, dpi: int = 150) -> bytes:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(bbox))
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
