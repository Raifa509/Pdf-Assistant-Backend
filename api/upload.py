import logging
import os
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile
from services.chunk_service import split_text_into_chunks
from services.embedding_service import embed_chunks
from services.pdf_service import extract_text_from_pdf
from services.vector_store_service import store_chunks


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # check file type
    logger.info("Received file: %s, content_type=%s", file.filename, file.content_type)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")


    # Create temporary filename
    file_name = f"{uuid.uuid4()}.pdf"

    try:
        # Save uploaded PDF temporarily
        with open(file_name, "wb") as f:
            f.write(await file.read())

        # Extract text
        text = extract_text_from_pdf(file_name)
        logger.info("Extracted %d characters from %s", len(text), file.filename)
        logger.debug("First 500 characters: %s", text[:500])

        # Guard against scanned/image-only PDFs with no extractable text
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF. It may be a scanned or image-only document."
            )

        # Split text into chunks
        chunks = split_text_into_chunks(text)
        logger.info("Split into %d chunks", len(chunks))
        for index, chunk in enumerate(chunks):
            logger.debug("Chunk %d: %d characters", index + 1, len(chunk))

        embeddings = embed_chunks(chunks)
        logger.info("Generated %d embeddings (dim=%d)", len(embeddings), len(embeddings[0]))

        paper_id = str(uuid.uuid4())
        store_chunks(paper_id, chunks, embeddings)
        logger.info("Stored paper_id=%s", paper_id)

        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "paper_id": paper_id,
            "total_chunks": len(chunks)
        }

    finally:
        if os.path.exists(file_name):
            os.remove(file_name)