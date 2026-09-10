import fitz

doc = fitz.open("test_pdfs/attention.pdf")
page = doc[12]  # page 13, 0-indexed
for img in page.get_images(full=True):
    xref = img[0]
    for rect in page.get_image_rects(xref):
        print(
            f"bbox=({rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}) "
            f"size={rect.width:.1f}x{rect.height:.1f}"
        )
doc.close()
