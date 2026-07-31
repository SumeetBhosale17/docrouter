from docrouter.chunking import chunk_paragraphs
from docrouter.extract import extract_paragraphs
from docrouter.index import build_index, retrieve

text = extract_paragraphs("test_pdfs/clean_text_distinct_sections.pdf")
chunks = chunk_paragraphs(text)
index, model = build_index(chunks)

questions = [
    "What is Water Cycle?",
    "How is paper produced?",
    "Role of Sleep and Recovery",
]

print("Len of chunks: ", len(chunks))

for q in questions:
    print(f"\nQ: {q}")
    for score, c in retrieve(q, chunks, index, model, k=3):
        print(f"  [{score:.3f}]", c[:200].replace("\n", " "), "...")
