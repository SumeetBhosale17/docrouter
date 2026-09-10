from functools import lru_cache

# The sentence-transformers model this feeds truncates at 256 tokens, so a
# chunk measured only in characters can be silently cut before it is embedded
# - dense, math-heavy text runs near 3 characters per token, far off the
# ~4 that prose averages.
EMBED_TOKEN_LIMIT = 254


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Loaded on first use, not at import - importing this module should not
    require a HuggingFace round-trip on a cold cache."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text: str) -> int:
    return len(
        _get_tokenizer()(text, add_special_tokens=False, verbose=False)["input_ids"]
    )


def split_by_token_limit(
    text: str, max_tokens: int = EMBED_TOKEN_LIMIT, overlap_tokens: int = 20
) -> list[str]:
    """Split on token offsets, so the pieces are bounded in the unit the
    embedding model actually measures."""
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    # verbose=False: the tokenizer warns that the sequence exceeds the model's
    # 512-token input whenever it is handed a long text. Only the offsets are
    # used here, never the ids, so nothing is at risk - and splitting on those
    # offsets is the very thing that keeps every emitted chunk under the limit.
    encoding = _get_tokenizer()(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = encoding["offset_mapping"]
    if len(offsets) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == len(offsets):
            break
        start = end - overlap_tokens
    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            while end < len(text) and not text[end].isspace():
                end += 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks


def chunk_paragraphs(
    paragraphs: list[str],
    max_chars: int = 800,
    overlap: int = 100,
    min_chars: int = 120,
) -> list[str]:
    """Pack consecutive paragraphs into chunks of at least min_chars, falling
    back to the character slicer for any single paragraph over max_chars.

    One chunk per paragraph put 167 of attention.pdf's 380 chunks under 60
    characters - page numbers, author and email lines, and the fragments that
    display math breaks into. They are not merely useless: a short chunk
    scores misleadingly high against a short query, because there is little
    else in it to dilute the match. "2\nFigure 1: The Transformer - model
    architecture." outranked every paragraph that actually explains the
    architecture, then contributed nothing for the generator to answer from.

    Coalescing forward also keeps a caption or heading attached to the text
    that follows it, which is where its meaning lives."""
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            chunks.extend(chunk_text(para, chunk_size=max_chars, overlap=overlap))
            continue
        candidate = f"{buffer}\n{para}" if buffer else para
        if buffer and len(candidate) > max_chars:
            flush()
            candidate = para
        buffer = candidate
        if len(buffer) >= min_chars:
            flush()
    flush()
    # Character budgets only approximate the token budget that matters, so
    # enforce the real one as a final pass.
    bounded: list[str] = []
    for c in chunks:
        bounded.extend(split_by_token_limit(c))
    return bounded
