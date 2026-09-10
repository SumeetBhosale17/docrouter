# docrouter

A layout-aware multimodal RAG system for PDF understanding and retrieval.

docrouter indexes a PDF the way a reader sees it: prose, tables, and figures
are extracted by separate detectors, and figures are described by a vision
model so a chart's content becomes searchable as text. Ask a question and it
retrieves across all three, then answers from the retrieved context only.

## Why the extractors run unconditionally

There is no "document type" branch. Every detector runs on every page, and a
single page can contribute a text chunk, a table chunk, and a figure chunk at
once. A paper page carrying a figure, its caption, and body text is the
normal case, not an exception.

The detectors share one constraint that shapes the whole design: **the regions
they find are also the mask that removes text from the text layer.** A
detector that over-reaches does not merely waste an API call - it deletes
content from the index, silently, with no error anywhere. Every threshold in
`charts.py` and `tables.py` is therefore tuned to fail toward *missing a
figure* rather than *emptying a page*.

## Pipeline

```
PDF
 ├─ detect.has_text_layer ──── no text? ──► ocr.ocr_page_text (Tesseract)
 │
 ├─ charts.get_figure_bboxes ─► raster + vector figure regions
 │        │                     (deduped, clustered, size- and
 │        │                      density-filtered)
 │        └──► render_page_region ──► describe.describe_chart_cached
 │                                     (Gemini vision, disk-cached)
 │                                     └─► split_chart_description
 │
 ├─ tables.find_real_tables ──► markdown tables + their bboxes
 │                              (one pass: what is cut out is what
 │                               comes back)
 │
 └─ extract.extract_paragraphs(exclude_bboxes=…)
          └─► chunking.chunk_paragraphs ──► text chunks

  all chunks ──► index.build_index            (FAISS + MiniLM embeddings)
             └─► lexical.build_lexical_index  (BM25)

  question ──► index.retrieve_hybrid ──► generate.generate_answer (Gemini)
```

## Install

Requires Python 3.12+, and Tesseract on the system for scanned PDFs.

```bash
# Arch
sudo pacman -S tesseract tesseract-data-eng
# Debian/Ubuntu
sudo apt install tesseract-ocr

uv sync
echo "GEMINI_API_KEY=your-key" > .env
```

## Use

```bash
# interactive
uv run docrouter test_pdfs/attention.pdf

# one-shot
uv run docrouter test_pdfs/attention.pdf -q "How many layers are in the encoder?"

# a folder of PDFs
uv run docrouter test_pdfs/

# inspect what was indexed, without asking anything
uv run docrouter test_pdfs/attention.pdf --dump-chunks
```

Browser UI:

```bash
uv run streamlit run app.py
```

## Modules

| Module | Responsibility |
| --- | --- |
| `detect.py` | Per-page predicates: text layer, raster images, vector drawings |
| `ocr.py` | Tesseract fallback for pages with no text layer |
| `extract.py` | Paragraph blocks, private-use-area repair, exclusion masking |
| `chunking.py` | Paragraph coalescing and the token budget every chunk must fit |
| `tables.py` | Table detection, filtering, and markdown rendering |
| `charts.py` | Figure region detection, clustering, rendering, description splitting |
| `describe.py` | Gemini vision descriptions, cached on disk by image + prompt |
| `index.py` | FAISS dense index, hybrid retrieval |
| `lexical.py` | BM25 index - the arm that finds numbers and identifiers |
| `fusion.py` | Reciprocal rank fusion (used for CLIP↔text, not the text path) |
| `clip_index.py` | CLIP image index for retrieving figures by their pixels |
| `generate.py` | Prompt construction, model fallback chain, retry policy |
| `cli.py` | Orchestration and the command-line entry point |

## Retrieval

`retrieve_hybrid` returns the dense arm's results, then appends lexical hits
the dense arm missed. It deliberately does **not** use reciprocal rank
fusion: RRF weights both arms equally, and measured on this corpus that let a
weaker lexical arm displace good dense hits - it rescued one query and pushed
three others down. Backfilling appends instead of reordering, so the dense
ranking is preserved exactly and the lexical arm can only add.

Results carry the arm that found them, because cosine similarity and BM25 are
unrelated scales. **Compare scores only within one source.**

## Evaluation

`evals/` is a runnable harness over a golden set of questions, scoring
recall@k and mean reciprocal rank. Retrieval metrics run offline against the
chart-description cache, so a sweep is free; only `--answers` spends API
calls. Both arms come from one code path - `backfill=0` disables the lexical
arm - so the comparison cannot drift from what ships.

```bash
uv run python -m evals.run              # retrieval only - offline, free
uv run python -m evals.run --answers    # also generates and scores answers
```

Cases known to fail carry their diagnosis and are scored separately rather
than deleted. Deleting one destroys the only record that the limitation
exists; scoring it inside recall lets a real regression hide behind a
limitation already understood.

## Results

### Extraction

| | before | after |
| --- | --- | --- |
| `charts.pdf` text chunks | 0 | 19 |
| `attention.pdf` text retention | 94% | 99% |
| `attention.pdf` chunks under 60 chars | 167 of 380 | 2 of 184 |
| `attention.pdf` vision API calls | 16 | 8 |
| Largest chunk vs the 256-token embedding limit | 262 (truncated) | 254 |
| Table regions excluded vs re-emitted | 40 excluded, 1 emitted | equal, asserted |

`charts.pdf` losing all 100 of its paragraphs was one bug: a vector-figure
bbox that unioned every drawing on the page grew to page size, and that bbox
is the text-exclusion mask.

### Retrieval

Recall@k and MRR over the golden set, ablating both the chunking fix and the
lexical arm:

| Chunking | Arm | recall@k | MRR |
| --- | --- | --- | --- |
| one chunk per paragraph | dense | 83% | 0.792 |
| one chunk per paragraph | hybrid | 92% | 0.812 |
| coalesced (current) | dense | **100%** | **0.903** |
| coalesced (current) | hybrid | **100%** | **0.903** |

Answer accuracy over the same set is **100%** (12 scored cases, `--answers`),
with the known-failing case excluded and reported separately.

Two things worth reading off that table.

**Chunking dominated.** 167 of 380 chunks were under 60 characters - page
numbers, author lines, fragments of split display math. Short chunks are not
merely useless: they score misleadingly high, because there is little content
in them to dilute a match. A bare `Figure 1: The Transformer - model
architecture.` outranked every paragraph explaining the architecture, then
gave the generator nothing to answer from. Coalescing paragraphs forward
moved recall 83% → 100% on its own.

**Hybrid retrieval no longer earns its place on this corpus.** It was
justified by one query whose answer chunk ranked 16th - but that ranking was
mostly an artifact of the tiny-chunk problem. Once chunking was fixed, dense
alone reaches 100% and the lexical arm adds nothing measurable. It is kept as
insurance for corpora unlike this one, not as a demonstrated win.

## Known limitations

**Vocabulary mismatch on conceptual queries.** For *"What problem does the
Transformer architecture aim to solve?"*, the chunk that states the problem
ranks 72/184 dense and 85/138 BM25. No practical `k` reaches it. The chunk
reads *"Recurrent models typically factor computation… This inherently
sequential nature precludes parallelization"* and never uses the words
"Transformer" or "problem", so neither surface matching nor independently
embedded vectors bridge the gap. Adding more retrievers of the same kind
cannot fix this; it needs a cross-encoder reranker, query expansion, or HyDE.
The case is kept in the eval set as a measured failure.

**pdfplumber collapses some ruled tables to a single column.** Both tables in
`attention.pdf` extract that way, so their content is indexed as prose rather
than markdown. `_is_real_table` rejects single-column and header-only
detections, which keeps the text rather than emitting broken markdown, but
the relational structure is lost.

**Private-use glyphs are only partly recoverable.** A PDF whose fonts lack a
usable ToUnicode CMap extracts to U+E000–U+F8FF; `attention.pdf` carries 56
such characters, which made `0.9` index as `0<U+E004>9`. The codes are
font-relative and genuinely ambiguous - U+E004 is a period in `0.9` and an
ellipsis in `(x1, …, xn)` - so `extract.normalize_pua` restores only the
unambiguous digit-flanked case and blanks the rest rather than inventing
characters.

**Figures on scanned pages are skipped.** A full-page raster is filtered as a
page scan and left to OCR, so a genuine chart inside a scanned report is
never described.

**No unit tests.** `evals/` covers retrieval end-to-end; it does not cover the
extraction invariants that produced most of the bugs above.

**Answers are scored by substring match**, which is a weak proxy. It cannot
tell a correct answer from one quoting the right number in a wrong claim, and
it fails correct answers that paraphrase - a needle of `Multi-Head Attention`
scored "a multi-head self-attention mechanism" as wrong. Needles are written
against the wording the document actually uses, but an LLM judge would be a
better instrument.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run pre-commit install     # hooks run ruff, format, and pyright
```
