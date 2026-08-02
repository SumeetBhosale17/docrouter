from docrouter.chunking import chunk_paragraphs
from docrouter.extract import extract_paragraphs
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve
from docrouter.tables import extract_tables_as_markdown, get_table_bboxes

pdf_path = "test_pdfs/clean_text_1.pdf"

table_bboxes = get_table_bboxes(pdf_path)
paragraphs = extract_paragraphs(pdf_path, table_bboxes=table_bboxes)
tables = extract_tables_as_markdown(pdf_path)

chunks = chunk_paragraphs(paragraphs) + tables
index, model = build_index(chunks)

questions = [
    "What value is associated with Beta in the table?",
    "What value is associated with Gamma in the table?",
]

for q in questions:
    print(f"\nQ: {q}")
    retrieved = retrieve(q, chunks, index, model, k=3)
    for score, c in retrieved:
        print(f"    [{score:.3f}]", repr(c[:80]))
    print("\nAnswer: ", generate_answer(q, retrieved))
