"""FastAPI endpoints for ATLAS AI."""
from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .core import AtlasRAG
from atlas.agent.routes import router as agent_router
import logging





logger = logging.getLogger(__name__)


app = FastAPI(title="ATLAS AI", version="0.1.0")
app.include_router(agent_router)

@lru_cache(maxsize=1)
def engine() -> AtlasRAG:
    return AtlasRAG()

class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", **engine().stats()}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file")
    content = await file.read(20 * 1024 * 1024 + 1)
    try:
        return engine().ingest_pdf(file.filename, content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/ask")
def ask(body: Question):
    try:
        return engine().answer(body.question, body.top_k)
    except Exception as exc:
        logger.exception("ATLAS AI answer generation failed")
        raise HTTPException(
            status_code=503,
            detail="Answer generation failed. Check server logs and provider configuration."
        ) from exc