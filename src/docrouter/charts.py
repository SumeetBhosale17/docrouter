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


def get_vector_chart_bbox(page: fitz.Page) -> tuple[float, float, float, float] | None:
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
    return (
        min(r.x0 for r in rects),
        min(r.y0 for r in rects),
        max(r.x1 for r in rects),
        max(r.y1 for r in rects),
    )


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


def render_region(
    pdf_path: str,
    page_num: int,
    bbox: tuple,
    dpi: int = 150,
    padding: float = 20.0,
) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    x0, y0, x1, y1 = bbox
    padded = (
        fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding) & page.rect
    )
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=padded)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
