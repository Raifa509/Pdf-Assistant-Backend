from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from api import query, upload
import logging

logging.basicConfig(
    level=logging.INFO,  # use logging.DEBUG while developing to see everything
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
app=FastAPI(
    title="Research Paper Assistant API",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://pdf-assistant-frontend-d3ktu04b6-raifas-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload.router,prefix="/api")
app.include_router(query.router,prefix="/api")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Research Paper Assistant API is running"}