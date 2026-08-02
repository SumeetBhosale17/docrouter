# from itertools import chain
from pathlib import Path

from docrouter.charts import get_raster_bboxes, get_vector_chart_bboxes, render_region
from docrouter.describe import describe_chart
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve

folder_path = Path("test_pdfs")

pdf_files = folder_path.glob("vector*.pdf")


for pdf_file in pdf_files:
    print(f"PDF: {str(pdf_file)}")

    bboxes = get_vector_chart_bboxes(str(pdf_file))
    chart_chunks = []

    for page_num, page_bboxes in enumerate(bboxes):
        if page_bboxes is None:
            continue
        for bbox in page_bboxes:
            img_bytes = render_region(str(pdf_file), page_num, bbox)
            chart_chunks.append(describe_chart(img_bytes))

        index, model = build_index(chart_chunks)
        q = "Which category has the highest value, and what is it?"
        retrieved = retrieve(q, chart_chunks, index, model, k=1)
        print("Answer:", generate_answer(q, retrieved))

pdf_files = folder_path.glob("bitmap*.pdf")

for pdf_file in pdf_files:
    print(f"PDF: {pdf_file}")
    bboxes = get_raster_bboxes(str(pdf_file))
    chart_chunks = []

    for page_num, page_bboxes in enumerate(bboxes):
        for bbox in page_bboxes:
            img_bytes = render_region(str(pdf_file), page_num, bbox)
            chart_chunks.append(describe_chart(img_bytes))

    index, model = build_index(chart_chunks)
    q = "Which category has the highest value, and what is it?"
    retrieved = retrieve(q, chart_chunks, index, model, k=1)
    print("Answer:", generate_answer(q, retrieved))
