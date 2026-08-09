import fitz


def make_scanned_test_pdf(source_pdf: str, output_pdf: str, dpi: int = 150) -> None:
    src = fitz.open(source_pdf)
    page = src[0]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")

    out = fitz.open()
    new_page = out.new_page(width=page.rect.width, height=page.rect.height)
    new_page.insert_image(new_page.rect, stream=img_bytes)
    out.save(output_pdf)
    out.close()
    src.close()


if __name__ == "__main__":
    make_scanned_test_pdf(
        "test_pdfs/clean_text_distinct_sections.pdf",
        "test_pdfs/scanned_clean_text_distinct_sections.pdf",
    )
