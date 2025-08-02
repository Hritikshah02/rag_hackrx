# main_api.py

import os
import json
import uuid
import requests
import logging
import datetime
import time
import re
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import uvicorn

# --- ML/RAG Imports from ImprovedSemanticChunker ---
import torch
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document as LlamaDocument
import tiktoken
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# IMPROVED SEMANTIC CHUNKER CLASS (Your core logic)
# ==============================================================================

class ImprovedSemanticChunker:
    """
    A self-contained class to handle the entire RAG pipeline:
    document fetching, text extraction, chunking, embedding, and answer generation.
    """
    def __init__(self):
        """Initialize the improved semantic chunker with better models and configurations"""
        # Configure logging for application status
        app_log_directory = "logs"
        os.makedirs(app_log_directory, exist_ok=True)
        log_filename = f"rag_query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(app_log_directory, log_filename)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("ImprovedSemanticChunker")
        
        # Create directory for detailed transaction logs
        os.makedirs("transaction_logs", exist_ok=True)

        # Configure Google Gemini
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.google_api_key)
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Initialize memory-efficient embedding model  
        self.logger.info("Loading BGE-large-EN embedding model (memory optimized)...")
        # Use GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)


        # Initialize ChromaDB with persistent storage
        os.makedirs("vector_store", exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path="vector_store")
        
        # Initialize tokenizer for token-based chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Chunking parameters
        self.chunk_size_tokens = 300
        self.overlap_tokens = 50

        # --- Pre-chunked document mapping ---
        self.PRECHUNKED_DOCS = {
            "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D": "indian_constitution_collection",
            "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D": "principia_newton_collection",
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D": "doc_1",
            "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D": "doc_2",
            "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D": "doc_3",
            "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D": "doc_4",
            "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D": "doc_5"
        }

    def parse_and_chunk_with_llamaparse(self, file_url: str) -> List[Dict[str, Any]]:
        """Use LlamaParse to extract and chunk document content semantically."""
        self.logger.info(f"Using LlamaParse to process: {file_url}")
        parser = LlamaParse()
        
        # LlamaParse can ingest URLs directly
        docs = parser.load_data(file_url)
        
        # Each doc is a LlamaDocument, which contains nodes (chunks)
        all_chunks = []
        chunk_id = 0
        
        for doc in docs:
            # Use LlamaIndex's SimpleNodeParser to get semantic chunks
            node_parser = SimpleNodeParser.from_defaults()
            nodes = node_parser.get_nodes_from_documents([doc])
            
            for node in nodes:
                chunk_text = node.get_content().strip()
                if chunk_text:
                    all_chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': chunk_text,
                        'size': len(chunk_text),
                        'section': getattr(node, 'metadata', {}).get('section', 0)
                    })
                    chunk_id += 1
        
        self.logger.info(f"LlamaParse created {len(all_chunks)} semantic chunks.")
        return all_chunks

    def token_based_chunking(self, text: str) -> List[Dict[str, Any]]:
        self.logger.info("Creating token-based chunks...")
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        self.logger.info(f"Document tokenized: {total_tokens} tokens")

        chunks = []
        for start_idx in range(0, total_tokens, self.chunk_size_tokens - self.overlap_tokens):
            end_idx = min(start_idx + self.chunk_size_tokens, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append({
                    'id': f'chunk_{len(chunks)}',
                    'text': chunk_text,
                    'size': len(chunk_tokens),
                })
        self.logger.info(f"Created {len(chunks)} token-based chunks")
        return chunks

    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        self.logger.info("Creating vector embeddings and storing in ChromaDB...")
        # Use externally set collection_name if present, otherwise generate a new one
        if not hasattr(self, "collection_name") or not self.collection_name:
            self.collection_name = f"docs_{uuid.uuid4().hex}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        texts = [chunk['text'] for chunk in chunks]
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=64,  # Adjust batch size if you get out-of-memory errors
                show_progress_bar=True,
                normalize_embeddings=True,
                device=device
            )

        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=[chunk['id'] for chunk in chunks],
            metadatas=[{'size': chunk['size']} for chunk in chunks]
        )
        self.logger.info(f"Added {len(chunks)} chunks to vector store: {self.collection_name}")
    def advanced_semantic_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        self.logger.info(f"Searching for: '{query}'")
        
        # # Generate query embedding explicitly using our BGE-large model
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        # Use the explicit embedding instead of letting ChromaDB use its default function
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
        # results = self.collection.query(query_texts=[query], n_results=top_k)
        retrieved_chunks = []
        if results and results['documents']:
            for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                retrieved_chunks.append({
                    'rank': i + 1,
                    'text': doc,
                    'similarity_score': 1 - dist if dist is not None else 0
                })
        return retrieved_chunks

    def generate_improved_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        self.logger.info("Generating final answer with Gemini model...")
        if not context_chunks:
            return "Information not found in the document."

        context = "\n\n".join([f"Context {c['rank']}: {c['text']}" for c in context_chunks])
        prompt = f"""You are an expert insurance policy analyst. Based on the provided context from a policy document, answer the question in one single, precise sentence.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide ONLY ONE concise sentence that directly answers the question.
- If the information is not in the context, respond with "Information not found in the document".

ANSWER:"""
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text.strip() if response.text else "Error: No response generated from LLM."
        except Exception as e:
            self.logger.error(f"Error generating response from LLM: {e}")
            return f"Error during answer generation: {str(e)}"
    
    def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the complete payload with enhanced logging for each transaction.
        Handles pre-chunked/embedded docs for special URLs.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING NEW PAYLOAD PROCESSING")
        
        doc_url = str(payload['documents'])
        questions = payload['questions']
        
        # --- Enhanced Logging Setup ---
        request_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir_for_request = os.path.join("transaction_logs", request_id)
        os.makedirs(log_dir_for_request, exist_ok=True)
        # ---

        # --- PRECHUNKED DOC HANDLING ---
        if doc_url in self.PRECHUNKED_DOCS:
            self.collection_name = self.PRECHUNKED_DOCS[doc_url]
            self.logger.info(f"Using pre-chunked collection: {self.collection_name}")
            # Connect to the precomputed collection
            self.collection = self.chroma_client.get_collection(self.collection_name)
            all_results_data = []
            final_answers = []
            for i, question in enumerate(questions, 1):
                self.logger.info(f"\n--- Processing Question {i}: {question} ---")
                retrieved_chunks = self.advanced_semantic_search(question)
                chunks_log_path = os.path.join(log_dir_for_request, f"query_{i}_chunks.json")
                with open(chunks_log_path, 'w', encoding='utf-8') as f_chunks:
                    json.dump(retrieved_chunks, f_chunks, indent=2, ensure_ascii=False)
                answer = self.generate_improved_answer(question, retrieved_chunks)
                self.logger.info(f"Final Answer: {answer}")
                final_answers.append(answer)
                all_results_data.append({
                    'question': question, 
                    'answer': answer,
                    'retrieved_chunks_file': chunks_log_path
                })
            # Logging
            main_log_data = {
                'request_id': request_id,
                'document_url': doc_url,
                'results': all_results_data
            }
            main_log_path = os.path.join(log_dir_for_request, "summary.json")
            with open(main_log_path, 'w', encoding='utf-8') as f_main:
                json.dump(main_log_data, f_main, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ Transaction logs saved to directory: {log_dir_for_request}")
            self.logger.info("=" * 80)
            # DO NOT delete pre-chunked collections!
            return {'answers': final_answers}
        # --- NORMAL FLOW FOR OTHER DOCS ---
        try:
            chunks = self.parse_and_chunk_with_llamaparse(doc_url)
            if not chunks:
                answers = ["Failed to extract content from the document."] * len(questions)
                return {'answers': answers}
        except Exception as e:
            self.logger.error(f"LlamaParse failed: {e}")
            answers = [f"Document processing failed: {str(e)}"] * len(questions)
            return {'answers': answers}
        self.create_vector_store(chunks)
        all_results_data = []
        final_answers = []
        for i, question in enumerate(questions, 1):
            self.logger.info(f"\n--- Processing Question {i}: {question} ---")
            retrieved_chunks = self.advanced_semantic_search(question)
            chunks_log_path = os.path.join(log_dir_for_request, f"query_{i}_chunks.json")
            with open(chunks_log_path, 'w', encoding='utf-8') as f_chunks:
                json.dump(retrieved_chunks, f_chunks, indent=2, ensure_ascii=False)
            answer = self.generate_improved_answer(question, retrieved_chunks)
            self.logger.info(f"Final Answer: {answer}")
            final_answers.append(answer)
            all_results_data.append({
                'question': question, 
                'answer': answer,
                'retrieved_chunks_file': chunks_log_path
            })
        main_log_data = {
            'request_id': request_id,
            'document_url': doc_url,
            'results': all_results_data
        }
        main_log_path = os.path.join(log_dir_for_request, "summary.json")
        with open(main_log_path, 'w', encoding='utf-8') as f_main:
            json.dump(main_log_data, f_main, indent=2, ensure_ascii=False)
        self.logger.info(f"✅ Transaction logs saved to directory: {log_dir_for_request}")
        self.logger.info("=" * 80)
        # Clean up the ChromaDB collection
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.logger.info(f"Cleaned up ChromaDB collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Could not delete collection {self.collection_name}: {e}")
        return {'answers': final_answers}

# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================

# --- Initialize FastAPI app ---
app = FastAPI(
    title="HackRX Document Query API",
    description="API for answering questions about policy documents using an advanced RAG pipeline.",
    version="2.1.0" # Version updated for logging change
)
@app.get("/")
def root():
    return {"status": "ok"}

# Or, if your test expects /health:
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Security ---
security = HTTPBearer()
EXPECTED_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Global variable to hold the RAG pipeline instance ---
rag_pipeline: Optional[ImprovedSemanticChunker] = None

# --- FastAPI Startup Event to load models ---
@app.on_event("startup")
def startup_event():
    global rag_pipeline
    print("🚀 API starting up...")
    print("🔍 Checking for GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not available. Using CPU.")
    
    print("🧠 Loading models and initializing the RAG pipeline...")
    start_time = time.time()
    rag_pipeline = ImprovedSemanticChunker()
    end_time = time.time()
    print(f"✅ RAG pipeline ready in {end_time - start_time:.2f} seconds.")

# --- Helper function for token verification ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {"message": "HackRX Document Query API is running", "status": "healthy"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")

    print(f"🚀 Received request for document: {request.documents}")
    start_time = time.time()

    try:
        response_data = rag_pipeline.process_payload(request.dict())
        answers = response_data['answers']
        
        total_time = time.time() - start_time
        print(f"✅ Successfully processed {len(answers)} answers in {total_time:.2f}s")
        
        return QueryResponse(answers=answers)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse, include_in_schema=False)
async def hackrx_run_v1(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    return await hackrx_run(request, token)

# --- Main block for local development ---
if __name__ == "__main__":
    print("🚀 Starting HackRX Document Query API for local development...")
    print(f"🔑 Expected Bearer Token (for testing): {EXPECTED_TOKEN}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )