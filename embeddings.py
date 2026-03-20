from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    result = _model.encode([text], normalize_embeddings=True)
    return result[0].tolist()
