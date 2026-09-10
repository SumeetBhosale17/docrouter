import io

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

CLIP_MODEL_NAME = "clip-ViT-B-32"


def build_clip_index(
    image_bytes_list: list[bytes],
) -> tuple[faiss.Index, SentenceTransformer]:
    if not image_bytes_list:
        raise ValueError("Cannot build a CLIP index from zero images.")
    model = SentenceTransformer(CLIP_MODEL_NAME)
    images = [Image.open(io.BytesIO(b)) for b in image_bytes_list]
    embeddings = model.encode(images, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))
    return index, model


def retrieve_by_image(
    query: str,
    payloads: list[str],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = 3,
) -> list[tuple[float, str]]:
    k = min(k, index.ntotal)
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype="float32"), k)
    return list(zip(scores[0].tolist(), [payloads[i] for i in idxs[0]], strict=True))
