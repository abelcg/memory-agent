import os

from google import genai

_client: genai.Client | None = None


def configure_embeddings(api_key: str):
    global _client
    _client = genai.Client(api_key=api_key)


def get_embedding(text: str) -> list[float]:
    result = _client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values
