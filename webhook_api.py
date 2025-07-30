"""
FastAPI Webhook for HackRX Competition
Provides /hackrx/run endpoint for document question answering
"""

import os
import tempfile
import requests
import time
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
from query_logger import rag_logger

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

def process_document_and_questions(pdf_path: str, questions: List[str], request_id: str) -> List[str]:
    """Process document and answer questions with comprehensive logging"""
    
    try:
        # Process document into chunks with enhanced processing
        print(f"📄 Processing document: {pdf_path}")
        doc_start_time = time.time()
        
        # Clear cache for fresh processing
        doc_processor.clear_cache()
        
        # Process document with semantic chunking
        chunks = doc_processor.process_document(pdf_path, "policy.pdf")
        doc_processing_time = time.time() - doc_start_time
        
        print(f"✅ Created {len(chunks)} semantic chunks in {doc_processing_time:.2f}s")
        
        # Log chunk quality statistics
        avg_chunk_size = sum(len(chunk['content']) for chunk in chunks) / len(chunks) if chunks else 0
        avg_tokens = sum(chunk.get('token_count', 0) for chunk in chunks) / len(chunks) if chunks else 0
        print(f"📊 Chunk stats: Avg size={avg_chunk_size:.0f} chars, Avg tokens={avg_tokens:.0f}")
        
        # Log document processing
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        rag_logger.log_document_processing(request_id, {
            "path": pdf_path,
            "num_chunks": len(chunks),
            "total_tokens": total_tokens,
            "processing_time": doc_processing_time
        })
        
        # Validate chunks before creating vector store
        print(f"🔍 Validating {len(chunks)} chunks...")
        valid_chunks = []
        for chunk in chunks:
            # Only include chunks with meaningful content
            if (len(chunk['content'].strip()) >= 50 and  # Minimum length
                chunk.get('token_count', 0) >= 10 and     # Minimum tokens
                not chunk['content'].strip().startswith('---')):  # Skip page separators
                valid_chunks.append(chunk)
        
        print(f"✅ Using {len(valid_chunks)} valid chunks (filtered from {len(chunks)})")
        
        # Create vector store with validated chunks
        print("🔄 Creating vector store...")
        vector_store.create_vector_store(valid_chunks)
        print("✅ Vector store created")
        
        # Answer each question
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"❓ Processing question {i}/{len(questions)}: {question}")
            
            # Search for relevant chunks with improved accuracy
            search_start_time = time.time()
            # Use more chunks initially for better accuracy, then filter
            search_results = vector_store.semantic_search(question, top_k=8)
            search_time = time.time() - search_start_time
            
            # Filter for high-quality chunks only
            high_quality_results = []
            for result in search_results:
                # Only include chunks with good relevance scores
                if result['score'] >= 0.3:  # Minimum relevance threshold
                    high_quality_results.append(result)
            
            # Use top 5 high-quality chunks for better context
            search_results = high_quality_results[:5] if high_quality_results else search_results[:3]
            
            # Log chunk retrieval with enhanced details
            rag_logger.log_chunk_retrieval(
                request_id, i, question, search_results,
                {
                    "top_k_requested": 8,
                    "top_k_returned": len(search_results),
                    "search_time_seconds": search_time,
                    "min_score": min([r['score'] for r in search_results]) if search_results else 0,
                    "max_score": max([r['score'] for r in search_results]) if search_results else 0,
                    "avg_score": sum([r['score'] for r in search_results]) / len(search_results) if search_results else 0
                }
            )
            
            # Generate response using LLM with validation
            if not search_results:
                print(f"⚠️ No relevant chunks found for question {i}")
                answers.append("No relevant information found in the document.")
                continue
            
            print(f"🔍 Found {len(search_results)} relevant chunks (scores: {[f'{r['score']:.3f}' for r in search_results[:3]]})") 
            
            llm_start_time = time.time()
            response = llm_reasoner.generate_response(
                question, search_results, request_id, i
            )
            llm_time = time.time() - llm_start_time
            
            # Extract and validate answer
            answer = response.get('justification', 'Unable to find relevant information')
            
            # Ensure answer is not empty or generic
            if not answer or answer.strip() in ['Unable to find relevant information', 'No justification provided']:
                # Try to extract from the best chunk directly
                if search_results:
                    best_chunk = search_results[0]['content'][:300]
                    answer = f"Based on the document: {best_chunk}..."
            
            answers.append(answer)
            print(f"✅ Answer {i}: {answer[:100]}...")
            
            # Log final answer
            rag_logger.log_final_answer(request_id, i, question, answer, {
                "llm_processing_time": llm_time,
                "confidence": response.get('confidence', 0),
                "decision": response.get('decision', 'UNKNOWN')
            })
        
        return answers
        
    except Exception as e:
        rag_logger.log_error(request_id, "document_processing", str(e), {
            "pdf_path": pdf_path,
            "num_questions": len(questions)
        })
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
    start_time = time.time()
    
    # Start request logging
    request_id = rag_logger.start_request(str(request.documents), request.questions)
    
    print(f"🚀 Received request {request_id} with {len(request.questions)} questions")
    print(f"📄 Document URL: {request.documents}")
    
    pdf_path = None
    try:
        # Download PDF
        print("⬇️ Downloading PDF...")
        pdf_path = download_pdf(str(request.documents))
        print(f"✅ PDF downloaded to: {pdf_path}")
        
        # Process document and answer questions
        answers = process_document_and_questions(pdf_path, request.questions, request_id)
        
        total_time = time.time() - start_time
        rag_logger.complete_request(request_id, True, total_time, answers)
        
        print(f"✅ Successfully processed {len(answers)} answers in {total_time:.2f}s")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        total_time = time.time() - start_time
        rag_logger.complete_request(request_id, False, total_time, [], "HTTP Exception")
        raise
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Internal server error: {str(e)}"
        rag_logger.complete_request(request_id, False, total_time, [], error_msg)
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
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
