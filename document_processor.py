import os
import re
import email
from email import policy
from typing import List, Dict, Any
from pathlib import Path

import PyPDF2
from docx import Document
import tiktoken

from config import Config

class DocumentProcessor:
    """Handles processing of various document types into chunks"""
    
    def __init__(self):
        self.config = Config()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks with metadata
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            
        Returns:
            List of document chunks with metadata
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_extension in ['.docx', '.doc']:
            text = self._extract_docx_text(file_path)
        elif file_extension == '.txt':
            text = self._extract_txt_text(file_path)
        elif file_extension == '.eml':
            text = self._extract_email_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Clean and preprocess text
        text = self._clean_text(text)
        
        # Create chunks
        chunks = self._create_chunks(text, filename)
        
        return chunks
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
        
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = ""
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text += row_text + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting Word document text: {str(e)}")
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error extracting text file: {str(e)}")
    
    def _extract_email_text(self, file_path: str) -> str:
        """Extract text from email file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file)
            
            text = ""
            
            # Extract headers
            text += f"From: {msg.get('From', 'Unknown')}\n"
            text += f"To: {msg.get('To', 'Unknown')}\n"
            text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            text += f"Date: {msg.get('Date', 'Unknown')}\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            text += body.decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True)
                if body:
                    text += body.decode('utf-8', errors='ignore')
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting email text: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\|\\]', '', text)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def _create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create chunks from text with overlap"""
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.config.CHUNK_SIZE and current_chunk:
                chunks.append(self._create_chunk_metadata(
                    current_chunk.strip(), 
                    source, 
                    chunk_id
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.encoding.encode(current_chunk))
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(
                current_chunk.strip(), 
                source, 
                chunk_id
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity"""
        tokens = self.encoding.encode(text)
        overlap_tokens = min(self.config.CHUNK_OVERLAP, len(tokens))
        
        if overlap_tokens > 0:
            overlap_token_ids = tokens[-overlap_tokens:]
            return self.encoding.decode(overlap_token_ids)
        
        return ""
    
    def _create_chunk_metadata(self, content: str, source: str, chunk_id: int) -> Dict[str, Any]:
        """Create chunk with metadata"""
        return {
            'content': content,
            'source': source,
            'chunk_id': chunk_id,
            'token_count': len(self.encoding.encode(content)),
            'char_count': len(content)
        }
