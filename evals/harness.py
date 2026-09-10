"""Retrieval and answer evaluation over the golden set.

Retrieval metrics run offline against the chart-description cache, so a full
sweep costs nothing and is fast enough to run on every change. Answer metrics
call Gemini and are opt-in.
"""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from docrouter.cli import build_chunks_for_pdf
from docrouter.index import build_index, retrieve, retrieve_hybrid
from docrouter.lexical import build_lexical_index
from evals.dataset import CASES, Case

PDF_DIR = Path("test_pdfs")


@dataclass
class CaseResult:
    case: Case
    hit: bool
    rank: int | None  # 1-based rank of the first chunk carrying a needle
    n_context: int
    answer: str | None = None
    answer_hit: bool | None = None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class Report:
    label: str
    results: list[CaseResult]
    elapsed: float

    def _scored(self) -> list[CaseResult]:
        """Known-failing cases are reported separately so a real regression
        elsewhere is not masked by a limitation already understood."""
        return [r for r in self.results if not r.case.expected_fail]

    @property
    def recall(self) -> float:
        s = self._scored()
        return sum(r.hit for r in s) / len(s) if s else 0.0

    @property
    def mrr(self) -> float:
        s = self._scored()
        return sum(r.reciprocal_rank for r in s) / len(s) if s else 0.0

    @property
    def known_fail_recall(self) -> float:
        k = [r for r in self.results if r.case.expected_fail]
        return sum(r.hit for r in k) / len(k) if k else 0.0

    @property
    def answer_accuracy(self) -> float | None:
        scored = [r for r in self._scored() if r.answer_hit is not None]
        return (sum(r.answer_hit for r in scored) / len(scored)) if scored else None


def _matches(text: str, needles: Iterable[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _load(docs: Iterable[str]) -> dict[str, tuple]:
    """Build every index once and share it across the cases for that doc."""
    built = {}
    for doc in sorted(set(docs)):
        chunks = build_chunks_for_pdf(str(PDF_DIR / doc))
        index, model = build_index(chunks)
        built[doc] = (chunks, index, model, build_lexical_index(chunks))
    return built


def run(
    label: str,
    *,
    k: int = 3,
    backfill: int = 2,
    with_answers: bool = False,
    cases: tuple[Case, ...] = CASES,
) -> Report:
    """backfill=0 disables the lexical arm, which is how the before/after
    comparison is produced without maintaining two code paths."""
    started = time.perf_counter()
    built = _load(c.doc for c in cases)
    results: list[CaseResult] = []

    for case in cases:
        chunks, index, model, lexical = built[case.doc]
        if backfill > 0:
            got = [
                (it.score, it.text)
                for it in retrieve_hybrid(
                    case.question, chunks, index, model, lexical, k=k, backfill=backfill
                )
            ]
        else:
            got = retrieve(case.question, chunks, index, model, k=k)

        rank = next(
            (i for i, (_, c) in enumerate(got, 1) if _matches(c, case.needles)), None
        )
        result = CaseResult(case, rank is not None, rank, len(got))

        if with_answers:
            from docrouter.generate import generate_answer

            result.answer = generate_answer(case.question, got).strip()
            result.answer_hit = _matches(result.answer, case.answer_targets)
        results.append(result)

    return Report(label, results, time.perf_counter() - started)


def format_report(report: Report) -> str:
    lines = [
        f"{report.label}  ({report.elapsed:.1f}s)",
        f"  recall@k        {report.recall:.0%}",
        f"  MRR             {report.mrr:.3f}",
    ]
    if report.answer_accuracy is not None:
        lines.append(f"  answer accuracy {report.answer_accuracy:.0%}")
    known = [r for r in report.results if r.case.expected_fail]
    if known:
        lines.append(
            f"  known-fail      {report.known_fail_recall:.0%} "
            f"of {len(known)} case(s) (excluded from the scores above)"
        )
    return "\n".join(lines)


def format_table(reports: list[Report]) -> str:
    """One row per case, one column per report - the before/after view."""
    width = max(len(c.question) for c in CASES)
    head = f"{'question':{width}s}" + "".join(f"{r.label:>12s}" for r in reports)
    rows = [head, "-" * len(head)]
    # "#2 A" = retrieved at rank 2, answer correct. "a!" = answer wrong.
    for i, case in enumerate(CASES):
        cells = ""
        for rep in reports:
            res = rep.results[i]
            cell = f"#{res.rank}" if res.hit else "miss"
            if res.answer_hit is not None:
                cell += " A" if res.answer_hit else " a!"
            cells += f"{cell:>12s}"
        flag = "  (known fail)" if case.expected_fail else ""
        rows.append(f"{case.question:{width}s}{cells}{flag}")
    return "\n".join(rows)
