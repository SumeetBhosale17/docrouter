import pdfplumber


def has_table(pdf_path: str) -> list[bool]:
    """Return one bool per page: True if pdfplumber's table finder
    detects at least one table on that page."""
    with pdfplumber.open(pdf_path) as pdf:
        return [len(page.find_tables()) > 0 for page in pdf.pages]


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned_rows: list[list[str]] = [
        [cell if cell is not None else "" for cell in row] for row in rows
    ]

    if not cleaned_rows:
        return ""

    header, *body = cleaned_rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _is_real_table(rows: list[list[str | None]]) -> bool:
    """Rejects spurious table detections: pdfplumber's ruling-line
    detector fires on dense grid-aligned chart labels as often as on
    real tables. Genuine tables have substantial per-cell text;
    misfires are mostly-empty grids holding bare short numbers."""
    cells = [c for row in rows for c in row if c]
    total_cells = sum(len(row) for row in rows)
    if not cells or not total_cells:
        return False
    fill_ratio = len(cells) / total_cells
    avg_len = sum(len(c) for c in cells) / len(cells)
    alpha_ratio = sum(1 for c in cells if any(ch.isalpha() for ch in c)) / len(cells)
    return fill_ratio >= 0.3 and avg_len >= 4 and alpha_ratio >= 0.2


def find_real_tables(
    pdf_path: str,
) -> tuple[list[list[tuple[float, float, float, float]]], list[str]]:
    """(bboxes_per_page, markdown_chunks) from ONE pass over the document.

    Detection and emission must not be separate passes: find_tables() fires on
    grid-aligned chart labels as often as on real tables, and _is_real_table
    rejects those. When only the emission side filtered, every rejected region
    was still cut out of the text layer and never re-added - 40 regions
    excluded, 1 emitted, on charts.pdf. Sharing one pass makes the region
    removed from the text exactly the region re-added as a table."""
    bboxes: list[list[tuple[float, float, float, float]]] = []
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_bboxes = []
            for table in page.find_tables():
                rows = table.extract()
                if not rows or not _is_real_table(rows):
                    continue
                page_bboxes.append(tuple(table.bbox))
                chunks.append(_rows_to_markdown(rows))
            bboxes.append(page_bboxes)
    return bboxes, chunks


def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    return find_real_tables(pdf_path)[1]


def get_table_bboxes(pdf_path: str) -> list[list[tuple[float, float, float, float]]]:
    return find_real_tables(pdf_path)[0]
