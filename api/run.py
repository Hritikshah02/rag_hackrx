"""
Vercel serverless function for HackRX webhook
Path: /api/run_fixed
"""

import os
import json
import tempfile
import requests
import re
from typing import List
import PyPDF2
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        return
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Check authorization
            auth_header = self.headers.get('authorization', '')
            expected_token = "Bearer 0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"
            
            if auth_header != expected_token:
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'detail': 'Invalid authentication token'}).encode())
                return
            
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            documents_url = data.get('documents')
            questions = data.get('questions', [])
            
            if not documents_url or not questions:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'detail': 'Missing documents or questions'}).encode())
                return
            
            # Process the request
            answers = process_document_and_questions(documents_url, questions)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'answers': answers}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'detail': f'Internal server error: {str(e)}'}).encode())

def download_pdf(url: str) -> str:
    """Download PDF from URL"""
    response = requests.get(url, timeout=30, verify=False)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    text = re.sub(r'\s+', ' ', text)
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
    
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        
        overlap = len(question_words.intersection(chunk_words))
        
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
        
        if re.search(r'\d+', question_lower) and re.search(r'\d+', chunk_lower):
            boost += 1
        
        score = overlap + boost
        if score > 0:
            scored_chunks.append((score, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API directly via HTTP"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return "Error: Google API key not configured"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
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
        
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def answer_question_with_gemini(question: str, context_chunks: List[str]) -> str:
    """Use Gemini to answer question based on context"""
    context = "\n\n".join(context_chunks)
    
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

ANSWER:"""

    return call_gemini_api(prompt)

def process_document_and_questions(documents_url: str, questions: List[str]) -> List[str]:
    """Process document and answer questions"""
    pdf_path = None
    try:
        # Download PDF
        pdf_path = download_pdf(documents_url)
        
        # Extract text
        text = extract_pdf_text(pdf_path)
        clean_text_content = clean_text(text)
        chunks = create_chunks(clean_text_content)
        
        answers = []
        for question in questions:
            relevant_chunks = find_relevant_chunks(question, chunks)
            answer = answer_question_with_gemini(question, relevant_chunks)
            answers.append(answer)
        
        return answers
        
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass
