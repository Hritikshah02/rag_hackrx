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
        """Clean and preprocess text with better normalization for semantic search"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common variations and abbreviations
        text = self._normalize_text_variations(text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\|\\]', '', text)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Convert to lowercase for better semantic matching (preserve original structure)
        # We'll store both original and normalized versions
        return text.strip()
    
    def _normalize_text_variations(self, text: str) -> str:
        """Normalize common text variations for better semantic matching"""
        # Common insurance/medical term normalizations
        normalizations = {
            r'\bpre-existing\b': 'preexisting',
            r'\bpre existing\b': 'preexisting', 
            r'\bco-payment\b': 'copayment',
            r'\bco payment\b': 'copayment',
            r'\bdeductible\b': 'deductible',
            r'\bdeductable\b': 'deductible',
            r'\bbenefits?\b': 'benefit',
            r'\bcoverages?\b': 'coverage',
            r'\bpolicies\b': 'policy',
            r'\bpremiums?\b': 'premium',
            r'\bclaims?\b': 'claim',
            r'\bprocedures?\b': 'procedure',
            r'\btreatments?\b': 'treatment',
            r'\bhospitalizations?\b': 'hospitalization',
            r'\bsurgeries\b': 'surgery',
            r'\bsurgical\b': 'surgery',
            r'\bmedical\b': 'medical',
            r'\bdental\b': 'dental',
            r'\bmaternity\b': 'maternity',
            r'\bwaiting period\b': 'waiting period',
            r'\bgrace period\b': 'grace period',
            r'\bno claim discount\b': 'no claim discount',
            r'\bNCD\b': 'no claim discount',
            r'\bAYUSH\b': 'ayush',
            r'\bICU\b': 'icu intensive care',
            r'\bOPD\b': 'opd outpatient',
            r'\bIPD\b': 'ipd inpatient'
        }
        
        # Apply normalizations (case insensitive)
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create chunks from text with improved overlap and semantic boundaries"""
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        sentence_buffer = []  # Buffer to handle semantic boundaries
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.encoding.encode(sentence))
            sentence_buffer.append(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.config.CHUNK_SIZE and current_chunk:
                # Create chunk from current content
                chunk_content = current_chunk.strip()
                chunks.append(self._create_chunk_metadata(
                    chunk_content, 
                    source, 
                    chunk_id
                ))
                
                # Start new chunk with better overlap strategy
                overlap_sentences = self._get_overlap_sentences(sentence_buffer, sentences, i)
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_tokens = len(self.encoding.encode(current_chunk))
                chunk_id += 1
                sentence_buffer = overlap_sentences + [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(
                current_chunk.strip(), 
                source, 
                chunk_id
            ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentence_buffer: List[str], all_sentences: List[str], current_index: int) -> List[str]:
        """Get overlap sentences that maintain semantic context"""
        # Take last few sentences as overlap, but respect semantic boundaries
        overlap_target_tokens = self.config.CHUNK_OVERLAP
        overlap_sentences = []
        total_tokens = 0
        
        # Start from the end of current buffer and work backwards
        for sentence in reversed(sentence_buffer[-5:]):  # Look at last 5 sentences max
            sentence_tokens = len(self.encoding.encode(sentence))
            if total_tokens + sentence_tokens <= overlap_target_tokens:
                overlap_sentences.insert(0, sentence)
                total_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
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
        """Create chunk with enhanced metadata for better search"""
        # Create normalized version for embedding
        normalized_content = self._create_normalized_content(content)
        
        return {
            'content': content,  # Original content for display
            'normalized_content': normalized_content,  # Normalized for embedding
            'source': source,
            'chunk_id': chunk_id,
            'token_count': len(self.encoding.encode(content)),
            'char_count': len(content),
            'keywords': self._extract_keywords(content)
        }
    
    def _create_normalized_content(self, content: str) -> str:
        """Create normalized version of content for better embedding"""
        # Convert to lowercase for embedding
        normalized = content.lower()
        
        # Additional normalizations for embedding
        normalized = re.sub(r'\b(\d+)\s*(months?|years?|days?)\b', r'\1 \2', normalized)
        normalized = re.sub(r'\b(\d+)\s*%\b', r'\1 percent', normalized)
        normalized = re.sub(r'₹\s*(\d+)', r'rupees \1', normalized)
        
        return normalized
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content for enhanced search"""
        # Define important terms that should be preserved
        important_terms = [
            'grace period', 'waiting period', 'pre-existing', 'coverage', 'benefit',
            'premium', 'claim', 'deductible', 'copayment', 'maternity', 'dental',
            'surgery', 'hospitalization', 'outpatient', 'inpatient', 'ayush',
            'no claim discount', 'ncd', 'icu', 'room rent', 'sub-limit'
        ]
        
        keywords = []
        content_lower = content.lower()
        
        for term in important_terms:
            if term in content_lower:
                keywords.append(term)
        
        # Extract numbers with units (periods, amounts, percentages)
        number_patterns = [
            r'\d+\s*(months?|years?|days?)',
            r'\d+\s*%',
            r'₹\s*\d+[,\d]*',
            r'\d+\s*(lakh|crore)'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, content_lower)
            keywords.extend(matches)
        
        return list(set(keywords))  # Remove duplicates
