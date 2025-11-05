import os
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional

from main import run_ingestion, qa_pipeline
from utils import create_or_load_vector_store, semantic_search

PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="AI Hybrid Workforce API",
    description="Backend for document ingestion, upload, semantic search, and Q&A.",
    version="1.0.1"
)


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5


class SearchResult(BaseModel):
    page_content: str
    metadata: Optional[dict] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
async def root():
    return {"message": "AI Hybrid Workforce Backend is running."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Automatically trigger ingestion in background
        if background_tasks:
            background_tasks.add_task(run_ingestion)

        return {
            "filename": file.filename,
            "message": "File uploaded successfully and ingestion started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")



@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_ingestion)
    return {"status": "Ingestion started in background."}


@app.post("/search", response_model=SearchResponse)
async def search_documents(req: SearchRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    vectordb = create_or_load_vector_store(PERSIST_DIR)
    results = semantic_search(vectordb, req.query, k=req.k)
    if not results:
        return {"query": req.query, "results": []}
    formatted_results = []
    for r in results:
        page_content = getattr(r, "page_content", None)
        metadata = getattr(r, "metadata", {})
        score = getattr(r, "score", None)
        formatted_results.append(
            SearchResult(page_content=page_content, metadata=metadata, score=score)
        )
    return {"query": req.query, "results": formatted_results}


@app.post("/qa", response_model=QAResponse)
async def question_answer(req: QARequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = qa_pipeline(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running QA: {str(e)}")
    return {"question": req.question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
