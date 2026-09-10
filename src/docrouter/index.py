import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from docrouter.lexical import BM25Index, retrieve_lexical

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(chunks: list[str]) -> tuple[faiss.Index, SentenceTransformer]:
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks.")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    return index, model


def retrieve(
    query: str,
    chunks: list[str],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = 3,
) -> list[tuple[float, str]]:
    k = min(k, index.ntotal)
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype="float32"), k)
    return list(zip(scores[0].tolist(), [chunks[i] for i in idxs[0]], strict=True))


def retrieve_hybrid(
    query: str,
    chunks: list[str],
    index: faiss.Index,
    model: SentenceTransformer,
    lexical_index: BM25Index,
    k: int = 3,
    backfill: int = 2,
) -> list[tuple[float, str]]:
    """Dense results, then lexical hits the dense arm missed.

    Deliberately not reciprocal rank fusion. RRF weights both arms equally,
    and measured on attention.pdf that let a weak lexical arm displace good
    dense hits - it rescued the Table 4 query but pushed three others down,
    a net loss. Backfilling appends instead of reordering, so the dense
    ranking is preserved exactly and the lexical arm can only add context
    the generator would not otherwise have seen.

    Scores from the two arms are not comparable (cosine similarity vs BM25),
    so treat the returned score as a within-arm figure only.
    """
    dense = retrieve(query, chunks, index, model, k=k)
    if backfill <= 0:
        return dense
    seen = {chunk for _, chunk in dense}
    extra = [
        (score, chunk)
        for score, chunk in retrieve_lexical(
            query, chunks, lexical_index, k=k + backfill
        )
        if chunk not in seen
    ]
    return dense + extra[:backfill]
