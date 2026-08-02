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


def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    """One markdown-formatted table string per detected table in the
    document."""
    result: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if table:
                    result.append(_rows_to_markdown(table))
    return result


def get_table_bboxes(pdf_path: str) -> list[list[tuple[float, float, float, float]]]:
    """One list of table bounding boxes per page."""
    with pdfplumber.open(pdf_path) as pdf:
        return [[t.bbox for t in page.find_tables()] for page in pdf.pages]
