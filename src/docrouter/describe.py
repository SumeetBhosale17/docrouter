import hashlib
import logging
from pathlib import Path

from google.genai import types

from docrouter.generate import generate_with_fallback

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/chart_descriptions")

CHART_PROMPT = (
    "This image is a chart or figure extracted from a document. "
    "Describe it precisely: chart type, axis labels, key values or "
    "trend, and any notable data points. Be factual - do not "
    "speculate beyond what's visibly shown."
)


def describe_chart(image_bytes: bytes) -> str:
    return generate_with_fallback(
        [
            CHART_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ]
    )


def describe_chart_cached(image_bytes: bytes) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The prompt is part of the key: a cache keyed on the image alone keeps
    # serving answers to a question you have since rewritten.
    key = hashlib.sha256(image_bytes + CHART_PROMPT.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        logger.info("chart description cache hit")
        return cache_file.read_text()
    logger.info("chart description cache miss - calling Gemini")
    description = describe_chart(image_bytes)
    cache_file.write_text(description)
    return description
