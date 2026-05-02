from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.graph import ask
from pipeline.ingestion import ingest_file
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Multi-Agent RAG API",
    description="RAG pipeline with LangGraph agents powered by Mistral",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    summary: str
    chunks_found: int


class IngestRequest(BaseModel):
    file_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(request.question)
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            summary=result["summary"],
            chunks_found=len(result["chunks"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest(request: IngestRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    try:
        n = ingest_file(request.file_path)
        return {"file": request.file_path, "chunks_stored": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))