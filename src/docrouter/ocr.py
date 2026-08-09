import io

import fitz
import pytesseract
from PIL import Image


def ocr_page_text(pdf_path: str, page_num: int, dpi: int = 300) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return pytesseract.image_to_string(img)
