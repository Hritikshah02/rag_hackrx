"""
Pure Python FastAPI Webhook for HackRX Competition
100% pure Python - no Rust dependencies, guaranteed to deploy
"""

import os
import tempfile
import requests
import json
import re
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import uvicorn

# Only pure Python imports
import PyPDF2
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRX Document Query API",
    description="Pure Python API for answering questions about policy documents",
    version="1.0.0"
)

# Security
security = HTTPBearer()
EXPECTED_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Request/Response models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

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
    """Download PDF from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download PDF: {str(e)}"
        )

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting PDF text: {str(e)}"
        )

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\|\\]', '', text)
    return text.strip()

def create_chunks(text: str, chunk_size: int = 2000) -> List[str]:
    """Create text chunks"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """Enhanced keyword-based chunk selection"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # Score chunks based on keyword overlap and relevance
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        
        # Calculate overlap score
        overlap = len(question_words.intersection(chunk_words))
        
        # Boost for important insurance terms
        boost = 0
        important_terms = {
            'grace period': 3, 'waiting period': 3, 'coverage': 2, 'benefit': 2, 
            'premium': 2, 'maternity': 3, 'dental': 2, 'surgery': 2, 'hospital': 2, 
            'ayush': 3, 'ncd': 3, 'deductible': 2, 'copay': 2, 'claim': 2,
            'policy': 1, 'insurance': 1, 'medical': 1, 'health': 1, 'treatment': 2,
            'days': 2, 'months': 2, 'years': 2, 'amount': 2, 'limit': 2,
            'excluded': 3, 'included': 2, 'covered': 2, 'eligible': 2
        }
        
        for term, weight in important_terms.items():
            if term in question_lower and term in chunk_lower:
                boost += weight
        
        # Boost for numbers (often important in insurance)
        if re.search(r'\d+', question_lower) and re.search(r'\d+', chunk_lower):
            boost += 1
        
        score = overlap + boost
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API directly via HTTP"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return "Error: Google API key not configured"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        return "Error: No response generated"
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {str(e)}"
    except Exception as e:
        return f"Error processing Gemini response: {str(e)}"

def answer_question_with_gemini(question: str, context_chunks: List[str]) -> str:
    """Use Gemini to answer question based on context"""
    # Combine context
    context = "\n\n".join(context_chunks)
    
    # Create prompt
    prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided policy document context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer directly and precisely
- Quote exact numbers, periods, and conditions from the policy
- For yes/no questions, start with "Yes" or "No" followed by conditions
- If information is not in the context, say "Information not available in the provided document"
- Be concise but complete
- Use exact policy language when possible

ANSWER:"""

    return call_gemini_api(prompt)

def process_document_and_questions(pdf_path: str, questions: List[str]) -> List[str]:
    """Process document and answer questions"""
    try:
        print(f"📄 Extracting text from PDF...")
        text = extract_pdf_text(pdf_path)
        
        print(f"🔄 Cleaning and chunking text...")
        clean_text_content = clean_text(text)
        chunks = create_chunks(clean_text_content)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"❓ Processing question {i}/{len(questions)}")
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question, chunks)
            
            # Generate answer
            answer = answer_question_with_gemini(question, relevant_chunks)
            answers.append(answer)
            
            print(f"✅ Answer {i}: {answer[:100]}...")
        
        return answers
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check"""
    return {"message": "HackRX Document Query API is running", "status": "healthy", "version": "pure-python"}

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "version": "pure-python", "dependencies": "zero-rust"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Main HackRX endpoint"""
    print(f"🚀 Received request with {len(request.questions)} questions")
    
    pdf_path = None
    try:
        # Download PDF
        pdf_path = download_pdf(str(request.documents))
        
        # Process and answer
        answers = process_document_and_questions(pdf_path, request.questions)
        
        print(f"✅ Successfully processed {len(answers)} answers")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass

if __name__ == "__main__":
    print("🚀 Starting Pure Python HackRX API...")
    uvicorn.run("webhook_pure:app", host="0.0.0.0", port=8000, reload=True)
