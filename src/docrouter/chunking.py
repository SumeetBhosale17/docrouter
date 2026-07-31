def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def chunk_paragraphs(
    paragraphs: list[str], max_chars: int = 800, overlap: int = 100
) -> list[str]:
    """One chunk per paragraph, unless a paragraph itself exceeds max_chars
    - then fall back to the character slicer for that paragraph only."""
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks.extend(chunk_text(para, chunk_size=max_chars, overlap=overlap))
    return chunks
