"""
FastAPI Webhook for HackRX Competition
Provides /hackrx/run endpoint for document question answering
"""

import os
import tempfile
import requests
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import uvicorn

# Import your existing components
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_reasoner import LLMReasoner
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="HackRX Document Query API",
    description="API for answering questions about policy documents",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Expected Bearer token (from HackRX requirements)
EXPECTED_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl  # PDF URL
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Initialize components
config = Config()
doc_processor = DocumentProcessor()
vector_store = VectorStore()
llm_reasoner = LLMReasoner()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def download_pdf(url: str) -> str:
    """Download PDF from URL and return local file path"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download PDF: {str(e)}"
        )

def process_document_and_questions(pdf_path: str, questions: List[str]) -> List[str]:
    """Process document and answer questions"""
    try:
        # Process document into chunks
        print(f"📄 Processing document: {pdf_path}")
        chunks = doc_processor.process_document(pdf_path, "policy.pdf")
        print(f"✅ Created {len(chunks)} chunks")
        
        # Create vector store
        print("🔄 Creating vector store...")
        vector_store.create_vector_store(chunks)
        print("✅ Vector store created")
        
        # Answer each question
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"❓ Processing question {i}/{len(questions)}: {question[:50]}...")
            
            # Search for relevant chunks
            search_results = vector_store.semantic_search(question, top_k=8)
            
            # Generate answer using LLM
            response = llm_reasoner.generate_response(question, search_results)
            
            # Extract the justification as the answer
            answer = response.get('justification', 'No answer found')
            answers.append(answer)
            
            print(f"✅ Answer {i}: {answer[:100]}...")
        
        return answers
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "HackRX Document Query API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test components
        config.validate_config()
        return {
            "status": "healthy",
            "components": {
                "config": "ok",
                "document_processor": "ok",
                "vector_store": "ok",
                "llm_reasoner": "ok"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Main HackRX endpoint for document question answering
    
    Accepts:
    - documents: URL to PDF document
    - questions: List of questions to answer
    
    Returns:
    - answers: List of answers corresponding to questions
    """
    print(f"🚀 Received request with {len(request.questions)} questions")
    print(f"📄 Document URL: {request.documents}")
    
    pdf_path = None
    try:
        # Download PDF
        print("⬇️ Downloading PDF...")
        pdf_path = download_pdf(str(request.documents))
        print(f"✅ PDF downloaded to: {pdf_path}")
        
        # Process document and answer questions
        answers = process_document_and_questions(pdf_path, request.questions)
        
        print(f"✅ Successfully processed {len(answers)} answers")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
                print(f"🗑️ Cleaned up temporary file: {pdf_path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temporary file: {e}")

# Alternative endpoint for testing (same functionality)
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def hackrx_run_v1(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Alternative endpoint with /api/v1 prefix"""
    return await hackrx_run(request, token)

if __name__ == "__main__":
    # For local development
    print("🚀 Starting HackRX Document Query API...")
    print(f"📋 Expected Bearer Token: {EXPECTED_TOKEN}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "webhook_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
