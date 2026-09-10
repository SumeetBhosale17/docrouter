"""Lexical (BM25) retrieval, as a rescue arm for the dense index.

Dense embeddings rank tabular and numeric text poorly: on attention.pdf the
only chunk holding Table 4's parsing scores ranks 16 of 380 for "What F1
score did the Transformer reach on WSJ section 23 parsing?", while a literal
term match identifies it uniquely. Numbers and identifiers are exactly what
an embedding trained on natural language compresses away.
"""

import math
import re
from collections import Counter

# Keeps decimals ("91.3") and version-like runs intact rather than splitting
# them into meaningless single digits - the numbers are the point here.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over the same chunk list the dense index holds."""

    def __init__(self, chunks: list[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        docs = [tokenize(c) for c in chunks]
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.term_freqs = [Counter(d) for d in docs]
        doc_freq: Counter[str] = Counter()
        for d in docs:
            doc_freq.update(set(d))
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in doc_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        out = [0.0] * self.n
        if not self.avgdl:
            return out
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.term_freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


def build_lexical_index(chunks: list[str]) -> BM25Index:
    if not chunks:
        raise ValueError("Cannot build a lexical index from zero chunks.")
    return BM25Index(chunks)


def retrieve_lexical(
    query: str, chunks: list[str], index: BM25Index, k: int = 3
) -> list[tuple[float, str]]:
    scored = [(s, c) for s, c in zip(index.scores(query), chunks, strict=True) if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
