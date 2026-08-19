from docrouter.cli import build_chunks_for_pdf
from docrouter.index import build_index, retrieve

pdf_path = "test_pdfs/multi_page_mixed.pdf"
all_chunks = build_chunks_for_pdf(pdf_path)
index, model = build_index(all_chunks)

question = "In the raster bitmap bar chart, which category has the highest value?"
retrieved = retrieve(question, all_chunks, index, model, k=5)
for score, chunk in retrieved:
    print(f"[{score:.3f}] {chunk[:150]!r}")
