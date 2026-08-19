def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[float, str]], k: int = 60
) -> list[tuple[float, str]]:
    """Combine rankings from sources whose raw similarity scores aren't
    comparable - text-embeddings cosine similarity and CLIP cosine
    similarity live in different, unrelated spaces, so merging by raw
    score would let whichever space happens to produce bigger numbers
    dominate arbitrarily. RRF uses only each result's RANK POSITION
    within its own list, sidestepping the comparability problem
    entirely - a standard IR technique, not something specific to this
    project."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (_, text) in enumerate(ranked):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
    return sorted(
        ((score, text) for text, score in scores.items()),
        key=lambda x: x[0],
        reverse=True,
    )
