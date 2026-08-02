from pathlib import Path

from docrouter.extract import extract_paragraphs
from docrouter.tables import extract_tables_as_markdown, get_table_bboxes, has_table

if __name__ == "__main__":
    folder_path = Path("test_pdfs")
    pdf_files = folder_path.glob("**/*.pdf")

    for pdf_file in pdf_files:
        print(f"PDF name: {str(pdf_file)}")
        print(has_table(str(pdf_file)))
        print(extract_tables_as_markdown(str(pdf_file)))

        print("Paragraphs:")
        paragraphs = extract_paragraphs(str(pdf_file), get_table_bboxes(str(pdf_file)))
        for p in paragraphs:
            print(repr(p))

        print("\n\n")
