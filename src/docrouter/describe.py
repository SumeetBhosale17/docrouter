from google.genai import types

from docrouter.generate import MODEL_NAME, get_client

CHART_PROMPT = (
    "This image is a chart or figure extracted from a document. "
    "Describe it precisely: chart type, axis labels, key values or "
    "trend, and any notable data points. Be factual - do not "
    "speculate beyond what's visibly shown."
)


def describe_chart(image_bytes: bytes) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            CHART_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
    )
    if response.text is None:
        raise ValueError("Model generated no text response.")
    return response.text
