import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # use your existing env variable name
EMBED_MODEL = "gemini-embedding-001"
DIM = 768

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(chunks), 100):  # API limit: 100 items per request
        batch = chunks[i:i + 100]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIM,
            ),
        )
        all_embeddings.extend(e.values for e in result.embeddings)
    return all_embeddings

def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIM,
        ),
    )
    return result.embeddings[0].values