import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import sys
from pathlib import Path

from docrouter.charts import (
    get_raster_bboxes,
    get_vector_chart_bboxes,
    render_region,
    split_chart_description,
)
from docrouter.chunking import chunk_paragraphs
from docrouter.describe import describe_chart_cached
from docrouter.extract import extract_paragraphs
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve
from docrouter.tables import extract_tables_as_markdown, get_table_bboxes


def build_chunks_for_pdf(pdf_path: str) -> list[str]:
    """Run every applicable extractor and merge results - not a branch on
    'document type', independent detectors run unconditionally, same
    design as Phase 1. A page can contribute text, a table, AND a chart
    chunk all at once."""

    from docrouter.detect import has_text_layer

    text_layer_flags = has_text_layer(pdf_path)
    table_bboxes = get_table_bboxes(pdf_path)
    raster_bboxes = get_raster_bboxes(pdf_path)
    vector_bboxes = get_vector_chart_bboxes(pdf_path)

    for i, has_text in enumerate(text_layer_flags):
        if not has_text:
            raster_bboxes[i] = []
            vector_bboxes[i] = []
            table_bboxes[i] = []

    exclude_bboxes = [
        t + r + v
        for t, r, v in zip(table_bboxes, raster_bboxes, vector_bboxes, strict=True)
    ]

    paragraphs = extract_paragraphs(pdf_path, exclude_bboxes=exclude_bboxes)
    text_chunks = chunk_paragraphs(paragraphs)
    table_chunks = extract_tables_as_markdown(pdf_path)

    chart_chunks = []
    for page_num, (raster, vector) in enumerate(
        zip(raster_bboxes, vector_bboxes, strict=True)
    ):
        for bbox in raster + vector:
            img_bytes = render_region(pdf_path, page_num, bbox)
            description = describe_chart_cached(img_bytes)
            chart_chunks.extend(split_chart_description(description))

    return text_chunks + table_chunks + chart_chunks


def main() -> None:
    parser = argparse.ArgumentParser(prog="docrouter")
    parser.add_argument("path", help="PDF file or folder of the PDFs")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    parser.add_argument(
        "--dump-chunks", action="store_true", help="Print all indexed chunks and exit"
    )
    args = parser.parse_args()

    target = Path(args.path)
    pdf_paths = [target] if target.is_file() else sorted(target.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found at {target}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")
        all_chunks.extend(build_chunks_for_pdf(str(pdf_path)))

    print(f"Indexed {len(all_chunks)} chunks from {len(pdf_paths)} file(s).")
    index, model = build_index(all_chunks)

    if args.dump_chunks:
        for i, c in enumerate(all_chunks):
            print(f"--- chunk {i} ---\n{c}\n")
        return

    if args.question:
        retrieved = retrieve(args.question, all_chunks, index, model, k=3)
        print(generate_answer(args.question, retrieved))
        return

    print("Enter a question (or 'exit' to quit)")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        retrieved = retrieve(query, all_chunks, index, model, k=3)
        print(generate_answer(query, retrieved))


if __name__ == "__main__":
    main()
