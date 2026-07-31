from docrouter.chunking import chunk_paragraphs
from docrouter.extract import extract_paragraphs
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve

text = extract_paragraphs("test_pdfs/zoro_fictional_story.pdf")
chunks = chunk_paragraphs(text)
index, model = build_index(chunks)

questions = [
    "What is Zoro's Profession?",
    "HWhat and where did Zoro record?",
    "What is Zoro's favorite game?",
]

print("Len of chunks: ", len(chunks))

for q in questions:
    print(f"\nQ: {q}")
    for score, c in retrieve(q, chunks, index, model, k=3):
        print(f"  [{score:.3f}]", c[:200].replace("\n", " "), "...")

    print("Answer: ", generate_answer(q, retrieve(q, chunks, index, model, k=3)))
