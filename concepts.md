# concepts.md

Running notes on concepts encountered while building this project.
Format: What / Why / How / When / Gotchas. Kept for personal reference
and interview prep — updated as the project progresses.

---

## Text-layer detection (`has_text_layer`)

**What:** a per-page check for whether a PDF page has real, extractable
text, as opposed to being a scanned/image-only page.

**Why:** this is the first branch point in the document-type router —
it decides whether a page needs OCR at all. Running OCR on pages that
already have a text layer wastes time and can introduce OCR errors into
text that was already perfect.

**How:** `page.get_text()` (PyMuPDF) reads the text-drawing operators
in the PDF's content stream. A native/digital PDF has real text objects
with positions; a scanned page is just one embedded raster image
(an XObject) with no text operators at all, so `get_text()` returns an
empty or near-empty string regardless of what's visually on the page.
A small threshold (`> 20` chars, not `> 0`) guards against pages with a
thin garbage text layer from a prior bad OCR pass.

**When:** run per page at ingestion time, not once per document — a
single PDF can mix native and scanned pages.

**Gotchas:** not yet tested against a real mixed-page document (all
test files so far were cleanly one type or the other) — worth doing
before trusting this in production.

---

## Raster visual detection (`has_visual_content`)

**What:** detects embedded raster images (photos, or charts exported as
bitmaps) on a page.

**Why:** text-layer detection alone says nothing about whether a page
*also* has visual content that needs separate handling (a VLM
description, not OCR or plain text extraction). This is a second,
independent axis from text-layer presence — not a sub-case of it. A
page can be (and often is) `True` on both at once.

**How:** `page.get_images(full=True)` lists the raster image XObjects
embedded in the page's content stream.

**When:** used as one half of the final "does this page have visual
content" flag — ORed with `has_chart_like_drawing` below.

**Gotchas:** completely blind to vector-drawn charts — confirmed
empirically: both `vector_chart_1.pdf` and `vector_chart_2.pdf` came
back `False` here despite clearly containing charts. That gap is
exactly why the next detector exists.

---

## Vector chart detection (`has_chart_like_drawing`)

**What:** flags whether a page's vector graphics (not raster images)
look like a chart — as opposed to a table's gridlines or a decorative
header band.

**Why:** some charting output is vector graphics, not embedded bitmaps,
so `has_visual_content` misses it entirely. Needed a second, different
signal.

**How (arrived at after two failed hypotheses — see the process entry
below):** a page counts as chart-like if either —
1. some single vector path has more than one connected segment
   (`len(items) > 1`) — a plotted line is drawn as one continuous
   multi-segment path, while a table's gridlines are each a separate
   one-segment path, or
2. some path has a fill that is *chromatic* (R≠G≠B, e.g. `(1,0,0)`)
   rather than neutral gray — decorative/structural fills (table
   header shading) are almost always neutral gray; chart elements
   (bars, markers) use color because it's encoding information.

**When:** cheap, so it's worth running on every page before any
expensive OCR/VLM call. Tuned deliberately to be permissive rather than
precise — see the cost-asymmetry note below.

**Gotchas:** two earlier hypotheses were tried and falsified against
real test files before this one held:
- *Raw path count* — failed: a small table's border also generates
  several separate paths, indistinguishable by count alone from a
  chart's gridlines/axes.
- *Max item-count per single path* — failed on the bar chart: each bar
  is one simple filled path (`n_items=1`), structurally identical in
  complexity to a table's simple stroke paths. This signal only worked
  for the *line* chart, not the general case.

Also: false positives (flagging a plain table as chart-like) cost one
extra API call downstream. False negatives (missing a real chart)
silently lose that chart's content from retrieval entirely. Those costs
aren't symmetric, so the heuristic is deliberately biased toward
over-flagging rather than chasing perfect precision.

---

## Empirical iteration for heuristic design (process, not a specific function)

**What:** the general loop used to arrive at `has_chart_like_drawing` —
propose a heuristic from structural reasoning, test it against a small
deliberately adversarial set of real files, let failures redirect the
next hypothesis, repeat until it holds.

**Why:** a heuristic that sounds right by reasoning about it in the
abstract ("more vector paths = more complex = probably a chart") can be
flatly wrong once tested — a bar chart and a table's gridlines are
structurally identical in path count and per-path complexity; only
fill color separated them, and that wasn't obvious until path-count and
complexity were both tried and failed.

**How:** built a 6-file test corpus up front — two bitmap charts, two
vector charts, two clean-text documents — specifically to try to break
each hypothesis, not just to confirm it works. Each round: state the
hypothesis and *why* it should work structurally, run it against all
six files, if any file breaks it, figure out specifically why before
proposing the next one (don't just retune a threshold on the same weak
signal).

**When:** reach for this loop any time a heuristic is trying to encode
a fuzzy, semantic distinction ("does this look like a chart") using
cheap structural signals rather than a trained model. It generalizes
past this project.

**Gotchas:** passing on N test files means "not yet falsified by these
files," not "correct in general." A grayscale chart, or a table with a
strongly colored branded header, would likely fool the current
heuristic in one direction or the other — known, accepted limitation
given the cost asymmetry above, not something chased to perfection here.

---

<!-- Add new entries below this line as the project progresses -->
