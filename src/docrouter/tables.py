import pdfplumber


def has_table(pdf_path: str) -> list[bool]:
    """Return one boll per page: True if pdfplumber's table finder
    detects atleast one table on that page."""
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
    """Rejects spurious table detections: pdfplumper's ruling-line
    detectors fires on dense grid-aligned chart labels as often as on
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


def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    """One markdown-formatted table string per detected table in the
    document."""
    result: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if table and _is_real_table(table):
                    result.append(_rows_to_markdown(table))
    return result


def get_table_bboxes(pdf_path: str) -> list[list[tuple[float, float, float, float]]]:
    """One list of table bounding boxes per page."""
    with pdfplumber.open(pdf_path) as pdf:
        return [[t.bbox for t in page.find_tables()] for page in pdf.pages]
