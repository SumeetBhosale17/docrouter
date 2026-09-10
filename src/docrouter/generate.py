import time
from collections.abc import Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

# Verified against models.list(); gemini-3-flash, gemini-2.0-flash and
# gemini-1.5-flash are NOT served and 404 on every call.
FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
)

# 429 lands here too: a rate limit is transient, and treating it as fatal
# burned through the whole chain with no pause between attempts.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def generate_with_fallback(
    contents,
    config: types.GenerateContentConfig | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Try each model in FALLBACK_MODELS, backing off on transient failures.

    One implementation for both call sites: the two hand-written copies of
    this loop had already diverged, and the chart-description copy re-raised
    on the first model's last retry instead of falling through - so the
    fallback list past index 0 was unreachable on exactly the failure it
    existed to survive."""
    client = get_client()
    last_error: Exception | None = None

    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                if response.text is None:
                    raise ValueError(f"{model_name} generated no text response.")
                return response.text
            except errors.APIError as e:
                last_error = e
                if e.code in RETRYABLE_STATUS and attempt < max_retries - 1:
                    time.sleep(base_delay * 2**attempt)
                    continue
                break  # unusable model, or retries spent - try the next one
            except ValueError as e:
                last_error = e
                break

    raise RuntimeError(
        f"All {len(FALLBACK_MODELS)} fallback models failed. Last error: {last_error}"
    ) from last_error


def build_prompt(query: str, retrieved: Sequence[Sequence]) -> str:
    # Indexed rather than unpacked so this takes both the (score, text) pairs
    # retrieve() returns and the (score, text, source) triples from
    # retrieve_hybrid().
    context = "\n\n---\n\n".join(item[1] for item in retrieved)
    return (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so explicitly "
        "instead of guessing. "
        "If part of the question's premise doesn't apply to this context "
        "(for example, it asks about categories but the data has none), "
        "say so - AND still report any concrete numeric value, data "
        "points, or trends from the context that answers what's actually "
        "being asked. Do not stop at correcting the premise without also "
        "giving the answerable part.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )


def generate_answer(query: str, retrieved: Sequence[Sequence]) -> str:
    return generate_with_fallback(
        build_prompt(query, retrieved),
        config=types.GenerateContentConfig(temperature=0.1),
    )
