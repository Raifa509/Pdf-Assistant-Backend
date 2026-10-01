from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from services.embedding_service import embed_query
from services.gemini_service import generate_answer
from services.vector_store_service import search_chunks

class QueryRequest(BaseModel):
    paper_id:str
    question:str
    top_k:int=10

router = APIRouter()

@router.post("/query")
async def query_paper(request: QueryRequest):
    # Embed the question
    query_embedding = embed_query(request.question)
    # Retrieve the most relevant chunks 
    relevant_chunks = search_chunks(
        paper_id=request.paper_id,
        query_embeddings=query_embedding,
        k=request.top_k
    )
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant content found for this question.")
    
    # Generate the answer
    answer = generate_answer(request.question, relevant_chunks)

    return {
        "paper_id": request.paper_id,
        "question": request.question,
        "answer": answer,
        "source_chunks": relevant_chunks
    }

