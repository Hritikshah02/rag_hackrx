"""
RAG API for ImprovedSemanticChunker - FastAPI implementation
Follows the /api/v1/hackrx/run endpoint and Bearer token authentication as specified.
"""

import os
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
from improved_semantic_chunker import ImprovedSemanticChunker
import logging

# Constants
API_PREFIX = "/api/v1"
ENDPOINT = "/hackrx/run"
BEARER_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Auth dependency
class BearerAuth(HTTPBearer):
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials.scheme.lower() != "bearer" or credentials.credentials != BEARER_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials

auth_scheme = BearerAuth()

# Request/Response Schemas
class RAGRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RAGResponse(BaseModel):
    answers: List[str]

# Initialize FastAPI
app = FastAPI(title="RAG API", version="1.0", docs_url="/docs", redoc_url="/redoc")

# Chunker singleton
chunker = None
@app.on_event("startup")
def load_chunker():
    global chunker
    if chunker is None:
        logging.info("Loading ImprovedSemanticChunker...")
        chunker = ImprovedSemanticChunker()
        logging.info("ImprovedSemanticChunker loaded.")

@app.post(f"{API_PREFIX}{ENDPOINT}", response_model=RAGResponse)
async def run_rag(request: RAGRequest, credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """
    Accepts a document URL and a list of questions, returns a list of answers.
    """
    if chunker is None:
        raise HTTPException(status_code=503, detail="RAG system not ready.")
    try:
        payload = {"documents": str(request.documents), "questions": request.questions}
        result = chunker.process_payload(payload)
        answers = [a.get("answer", "") for a in result.get("results", [])]
        return {"answers": answers}
    except Exception as e:
        logging.exception("Error processing RAG request")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
