import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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
