import math
from google import genai
from google.genai import types

client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 100  # API limit per request


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS,
            ),
        )
        vectors.extend(_normalize(e.values) for e in result.embeddings)
    return vectors


def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONS,
        ),
    )
    return _normalize(result.embeddings[0].values)