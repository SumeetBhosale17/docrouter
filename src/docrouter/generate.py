from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-flash-latest"
# MODEL_NAME = "gemini-1.5-flash"

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def build_prompt(query: str, retrieved: list[tuple[float, str]]) -> str:
    context = "\n\n---\n\n".join(chunk for _, chunk in retrieved)
    return (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so explicitily "
        "instead of guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )


def generate_answer(query: str, retrieved: list[tuple[float, str]]) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME, contents=build_prompt(query, retrieved)
    )
    if response.text is None:
        raise ValueError("Model generated no text response.")
    return response.text
