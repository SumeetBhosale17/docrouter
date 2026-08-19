import fitz


def merge_pdfs(sources: list[str], output: str) -> None:
    out = fitz.open()
    for src in sources:
        with fitz.open(src) as doc:
            out.insert_pdf(doc)
    out.save(output)
    out.close()


if __name__ == "__main__":
    merge_pdfs(
        [
            "test_pdfs/clean_text_distinct_sections.pdf",
            "test_pdfs/clean_text_1.pdf",
            "test_pdfs/bitmap_chart_2.pdf",
            "test_pdfs/vector_chart_1.pdf",
            "test_pdfs/scanned_clean_text_distinct_sections.pdf",
        ],
        "test_pdfs/multi_page_mixed.pdf",
    )
