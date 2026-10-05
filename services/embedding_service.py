# Step -4 turn text chunks into vectors (embeddings).

from sentence_transformers import SentenceTransformer
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks:list[str]):
    embeddings=embedding_model.encode(chunks)
    return embeddings.tolist()  # convert numpy array -> plain list

def embed_query(text: str) -> list[float]:
    embedding = embedding_model.encode([text])[0]
    return embedding.tolist()  # convert numpy array -> plain list