import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import logging
import sys
from pathlib import Path

import fitz

from docrouter.charts import (
    get_figure_bboxes,
    render_page_region,
    split_chart_description,
)
from docrouter.chunking import chunk_paragraphs
from docrouter.describe import describe_chart_cached
from docrouter.extract import extract_paragraphs
from docrouter.generate import generate_answer
from docrouter.index import build_index, retrieve
from docrouter.tables import find_real_tables


def _describe_figures(
    pdf_path: str,
    raster_bboxes: list[list[tuple]],
    vector_bboxes: list[list[tuple]],
) -> list[str]:
    """One document open for every region, instead of one open per region."""
    chunks: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page_num, (raster, vector) in enumerate(
            zip(raster_bboxes, vector_bboxes, strict=True)
        ):
            page = doc[page_num]
            for kind, boxes in (("raster image", raster), ("vector graphic", vector)):
                for bbox in boxes:
                    img_bytes = render_page_region(page, bbox)
                    description = describe_chart_cached(img_bytes)
                    label = f"{kind}, page {page_num + 1}"
                    chunks.extend(split_chart_description(description, label))
    finally:
        doc.close()
    return chunks


def build_chunks_for_pdf(pdf_path: str) -> list[str]:
    """Run every applicable extractor and merge results - not a branch on
    'document type', independent detectors run unconditionally, same design
    as Phase 1. A page can contribute text, a table, AND a chart chunk all at
    once.

    Figure regions are also the text-exclusion mask, so a detector that
    over-reaches does not just waste an API call, it deletes text. The
    detectors are tuned to fail toward missing a figure rather than emptying
    a page, and scanned pages need no special case: a full-page raster is
    filtered as a page scan, and OCR ignores the exclusion mask anyway."""
    raster_bboxes, vector_bboxes = get_figure_bboxes(pdf_path)
    table_bboxes, table_chunks = find_real_tables(pdf_path)

    exclude_bboxes = [
        list(t) + list(r) + list(v)
        for t, r, v in zip(table_bboxes, raster_bboxes, vector_bboxes, strict=True)
    ]

    paragraphs = extract_paragraphs(pdf_path, exclude_bboxes=exclude_bboxes)
    text_chunks = chunk_paragraphs(paragraphs)
    chart_chunks = _describe_figures(pdf_path, raster_bboxes, vector_bboxes)

    return text_chunks + table_chunks + chart_chunks


def main() -> None:
    # Root stays at WARNING: setting it to INFO turns on every library's
    # INFO stream too, and huggingface_hub and httpx bury the output.
    logging.basicConfig(level=logging.WARNING, format="    %(message)s")
    logging.getLogger("docrouter").setLevel(logging.INFO)

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

    if not all_chunks:
        print(
            f"No extractable content found in {len(pdf_paths)} file(s).",
            file=sys.stderr,
        )
        sys.exit(1)

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
