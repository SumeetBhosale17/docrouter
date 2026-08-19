from pathlib import Path

from docrouter.charts import get_raster_bboxes, get_vector_chart_bboxes, render_region
from docrouter.clip_index import build_clip_index, retrieve_by_image
from docrouter.describe import describe_chart_cached
from docrouter.fusion import reciprocal_rank_fusion
from docrouter.index import build_index, retrieve

files = [
    # "test_pdfs/bitmap_chart_1.pdf",
    "test_pdfs/bitmap_chart_2.pdf",
    # "test_pdfs/vector_chart_1.pdf",
    # "test_pdfs/vector_chart_2.pdf",
]

image_bytes_list, descriptions, labels = [], [], []
for pdf_path in files:
    raster, vector = get_raster_bboxes(pdf_path), get_vector_chart_bboxes(pdf_path)
    for page_num, (r, v) in enumerate(zip(raster, vector, strict=True)):
        for bbox in r + v:
            img_bytes = render_region(pdf_path, page_num, bbox)
            image_bytes_list.append(img_bytes)
            descriptions.append(describe_chart_cached(img_bytes))
            labels.append(Path(pdf_path).name)

print("Loaded charts: ", labels)

clip_index, clip_model = build_clip_index(image_bytes_list)

print("\nCLIP-only: 'a bar chart'")

for score, label in retrieve_by_image(
    "a bar chart", labels, clip_index, clip_model, k=4
):
    print(f"    [{score:.3f}] {label}")

question = "Which category has the highest value, and what is it?"
text_index, text_model = build_index(descriptions)
text_results = retrieve(question, descriptions, text_index, text_model, k=1)
clip_results = retrieve_by_image(question, descriptions, clip_index, clip_model, k=1)
fused = reciprocal_rank_fusion(text_results, clip_results)

# print(f"\nFused results for: {question!r}")
# for score, desc in fused[:4]:
#     print(f"    [{score:.4f}] {desc[:80]!r}")

print(fused[0])
