# step 5 : storing embeddings

import faiss
import numpy as np
from fastapi import HTTPException


# in-memory storage, resets when server restarts
_indexes = {}
_chunks = {}

def store_chunks(paper_id: str, chunks: list[str], embeddings: list[list[float]]):
    if not chunks or not embeddings:
        raise ValueError("Cannot store empty chunks or embeddings.")

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    _indexes[paper_id] = index
    _chunks[paper_id] = chunks

def search_chunks(paper_id: str, query_embeddings: list[float], k: int = 5):
    if paper_id not in _indexes:
        raise HTTPException(status_code=404, detail=f"No paper found with id '{paper_id}'. Upload it first.")

    index = _indexes[paper_id]
    chunks = _chunks[paper_id]
    k = min(k, len(chunks))

    D, I = index.search(np.array([query_embeddings]).astype("float32"), k)
    result_indices = list(I[0])

    # Always include the first chunk — title/author metadata usually lives here
    if 0 not in result_indices:
        result_indices = [0] + result_indices[:-1]

    return [chunks[i] for i in result_indices if i != -1]