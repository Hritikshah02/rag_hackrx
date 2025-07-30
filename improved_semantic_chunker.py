import os
import json
import uuid
import requests
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from io import BytesIO
import re
import google.generativeai as genai
import tiktoken

# Configure logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = f"rag_query_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_directory, log_filename)

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ImprovedSemanticChunker")

from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

class ImprovedSemanticChunker:
    def __init__(self):
        """Initialize the improved semantic chunker with better models and configurations"""
        # Configure Google Gemini - using the full model instead of lite
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.google_api_key)
        # Using the full Gemini model for better reasoning
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Initialize state-of-the-art embedding model
        print("Loading BGE-Large-EN embedding model...")
        # Using BGE-Large-EN for superior semantic understanding
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Initialize re-ranking model for improved relevance
        print("Loading cross-encoder for re-ranking...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-large')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = "improved_documents_bge"
        
        # Create a ChromaDB compatible embedding function with BGE model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")
        
        # Initialize tokenizer for token-based chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Chunking parameters
        self.chunk_size_tokens = 400  # Target chunk size in tokens
        self.overlap_tokens = 50      # Overlap size in tokens
        
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        print(f"Downloading document from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content with better formatting"""
        print("Extracting text from PDF...")
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            # Better text cleaning
            page_text = re.sub(r'\n+', '\n', page_text)
            page_text = re.sub(r'\s+', ' ', page_text)
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
    
    def token_based_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks based on token count with specified overlap"""
        print(f"Creating token-based chunks with {self.overlap_tokens} token overlap...")
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size_tokens, total_tokens)
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk text
            chunk_text = chunk_text.strip()
            
            if len(chunk_text) > 50:  # Only add non-trivial chunks
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'size': len(chunk_tokens),
                    'token_start': start_idx,
                    'token_end': end_idx,
                    'section': chunk_id // 10  # Group chunks into sections
                })
                chunk_id += 1
            
            # Move start index forward, accounting for overlap
            start_idx = end_idx - self.overlap_tokens
            
            # Prevent infinite loop
            if start_idx >= end_idx:
                break
        
        print(f"Created {len(chunks)} token-based chunks (avg {total_tokens // len(chunks) if chunks else 0} tokens per chunk)")
        return chunks
    
    def intelligent_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create intelligent chunks based on document structure"""
        print("Creating intelligent semantic chunks...")
        
        # Clean text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split by common section markers in insurance documents
        section_patterns = [
            r'(?i)\b(?:section|clause|article|part|chapter)\s+\d+',
            r'(?i)\b(?:definitions?|terms?|conditions?|benefits?|exclusions?|limitations?)\b',
            r'(?i)\b(?:waiting period|grace period|premium|coverage|claim|deductible)\b',
            r'\d+\.\s+[A-Z]',  # Numbered sections
            r'[A-Z][a-z]+\s*:',  # Title-like patterns
        ]
        
        chunks = []
        chunk_id = 0
        
        # First, try to split by major sections
        sections = re.split(r'(?i)(?:section|clause|article)\s+\d+', text)
        
        for section_idx, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            # Further split large sections into smaller semantic chunks
            if len(section) > 2000:
                # Split by sentences but keep related content together
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section)
                
                current_chunk = ""
                current_size = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # If adding this sentence would make chunk too large, save current chunk
                    if current_size + len(sentence) > 1500 and current_chunk:
                        chunks.append({
                            'id': f'chunk_{chunk_id}',
                            'text': current_chunk.strip(),
                            'size': current_size,
                            'section': section_idx
                        })
                        
                        # Start new chunk with increased overlap (30-40% overlap)
                        overlap_sentences = sentences[-4:] if len(sentences) > 4 else sentences
                        current_chunk = ' '.join(overlap_sentences) + ' ' + sentence
                        current_size = len(current_chunk)
                        chunk_id += 1
                    else:
                        current_chunk += ' ' + sentence
                        current_size += len(sentence)
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append({
                        'id': f'chunk_{chunk_id}',
                        'text': current_chunk.strip(),
                        'size': current_size,
                        'section': section_idx
                    })
                    chunk_id += 1
            else:
                # Small section, add as single chunk
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': section.strip(),
                    'size': len(section),
                    'section': section_idx
                })
                chunk_id += 1
        
        print(f"Created {len(chunks)} intelligent semantic chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Create vector store from chunks with metadata"""
        print("Creating vector embeddings with metadata...")
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        
        # Create new collection with explicit embedding function to handle dimension issues
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # Generate embeddings and add to collection
        texts = [chunk['text'] for chunk in chunks]
        # Let ChromaDB handle the embeddings using our embedding function
        
        self.collection.add(
            documents=texts,
            ids=[chunk['id'] for chunk in chunks],
            metadatas=[{
                'size': chunk['size'], 
                'section': chunk.get('section', 0)
            } for chunk in chunks]
        )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def advanced_semantic_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Perform advanced semantic search with re-ranking"""
        print(f"\nSearching for: '{query}'")
        print("-" * 80)
        
        # First stage: Retrieve more candidates for re-ranking
        initial_k = min(top_k * 3, 20)  # Retrieve 3x more candidates for re-ranking
        
        # Semantic search with embeddings
        results = self.collection.query(
            query_texts=[query],
            n_results=initial_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Get initial candidates from vector search
        candidates = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for doc, distance, metadata in zip(
                results['documents'][0], 
                results['distances'][0], 
                results['metadatas'][0]
            ):
                candidates.append({
                    'text': doc,
                    'vector_score': 1 - distance,
                    'metadata': metadata
                })
        
        # Second stage: Re-rank using cross-encoder
        if candidates:
            print(f"Re-ranking {len(candidates)} candidates...")
            
            # Prepare query-document pairs for re-ranking
            query_doc_pairs = [(query, candidate['text']) for candidate in candidates]
            
            # Get re-ranking scores
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Combine with original candidates and sort by re-ranking score
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
            
            # Sort by re-ranking score (higher is better)
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Take top_k results
            candidates = candidates[:top_k]
        
        # Format final results
        retrieved_chunks = []
        for i, candidate in enumerate(candidates):
            chunk_info = {
                'rank': i + 1,
                'text': candidate['text'],
                'similarity_score': candidate['vector_score'],
                'rerank_score': candidate['rerank_score'],
                'metadata': candidate['metadata']
            }
            retrieved_chunks.append(chunk_info)
            
            # Display full retrieved chunk with both scores
            print(f"RETRIEVED CHUNK {i+1} (Vector: {chunk_info['similarity_score']:.3f}, Rerank: {chunk_info['rerank_score']:.3f}):")
            print(f"{candidate['text']}")
            print("-" * 80)
        
        return retrieved_chunks
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Common insurance terms to prioritize
        important_terms = [
            'grace period', 'waiting period', 'premium', 'coverage', 'benefit',
            'deductible', 'copayment', 'exclusion', 'limitation', 'claim',
            'maternity', 'pre-existing', 'cataract', 'AYUSH', 'hospital',
            'room rent', 'ICU', 'organ donor', 'NCD', 'health check'
        ]
        
        keywords = []
        query_lower = query.lower()
        
        for term in important_terms:
            if term in query_lower:
                keywords.append(term)
        
        # Also extract numbers and specific terms
        numbers = re.findall(r'\d+', query)
        keywords.extend(numbers)
        
        return keywords
    
    def generate_improved_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using improved prompting strategy"""
        print("\nGenerating answer with improved Gemini model...")
        
        # Prepare context with more structure
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(f"Context {i+1} (Relevance: {chunk['similarity_score']:.3f}):\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Simplified prompt with clear instructions
        prompt = f"""You are an expert insurance policy analyst. Based on the following context from an insurance policy document, answer the question in a single precise sentence.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide ONLY ONE concise sentence that directly answers the question
- Include specific numbers, percentages, time periods, and conditions from the document
- Use exact terminology from the policy document
- Start with Yes/No if applicable, followed by key details
- If information is not in the context, respond with "Information not found in the document"

ANSWER:"""
        
        generation_config = {
            'temperature': 0.1,  # Lower temperature for more consistent answers
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 256,  # Reduced to ensure we get a response
        }
        
        try:
            response = self.llm_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response and hasattr(response, 'text') and response.text.strip():
                return response.text.strip()
            else:
                return "Error: No response generated from LLM. Please try again with a different question."
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)[:100]}...".strip()
    
    def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process the complete payload with improved methods"""
        print("=" * 80)
        print("IMPROVED SEMANTIC CHUNKER - PROCESSING PAYLOAD")
        print("=" * 80)
        
        logger.info("Starting processing of new payload")
        
        # Download and process document
        doc_url = payload['documents']
        questions = payload['questions']
        
        logger.info(f"Document URL: {doc_url}")
        logger.info(f"Number of questions: {len(questions)}")
        
        # Download document
        doc_content = self.download_document(doc_url)
        
        # Extract text
        text = self.extract_text_from_pdf(doc_content)
        print(f"Extracted {len(text)} characters from document")
        logger.info(f"Extracted {len(text)} characters from document")
        
        # Create token-based chunks with overlap
        chunks = self.token_based_chunking(text)
        logger.info(f"Created {len(chunks)} token-based chunks with {self.overlap_tokens} token overlap")
        
        # Log sample chunks
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Sample chunk {i+1}: {chunk['text'][:100]}...")
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Process each question
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"QUESTION {i}: {question}")
            print('='*80)
            
            logger.info(f"Processing question {i}: {question}")
            
            # Advanced semantic search
            retrieved_chunks = self.advanced_semantic_search(question)
            
            # Log retrieved chunks
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for question {i}")
            for j, chunk in enumerate(retrieved_chunks):
                logger.info(f"Chunk {j+1} (Score: {chunk['similarity_score']:.3f}): {chunk['text'][:150]}...")
            
            # Generate improved answer
            answer = self.generate_improved_answer(question, retrieved_chunks)
            
            print(f"\n{'*'*25} FINAL ANSWER {'*'*25}")
            print(f"{answer}")
            print('*'*70)
            
            # Log the answer
            logger.info(f"Final answer for question {i}: {answer}")
            
            # Flush output to ensure it's displayed immediately
            import sys
            sys.stdout.flush()
            
            results.append({
                'question': question,
                'retrieved_chunks': retrieved_chunks,
                'answer': answer
            })
        
        # Log summary
        logger.info("Processing complete")
        logger.info(f"Document URL: {doc_url}")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Total questions processed: {len(results)}")
        
        # Save all results to a JSON file for reference
        results_file = os.path.join(log_directory, f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'document_url': doc_url,
                'total_chunks': len(chunks),
                'results': [{
                    'question': r['question'],
                    'answer': r['answer'],
                    'chunks': [{
                        'text': c['text'],
                        'score': c['similarity_score']
                    } for c in r['retrieved_chunks']]
                } for r in results]
            }, f, indent=2)
        logger.info(f"Full results saved to {results_file}")
        
        return {
            'document_url': doc_url,
            'total_chunks': len(chunks),
            'results': results
        }

def main():
    """Main function to run the improved semantic chunker"""
    # Test payload
    test_payload_1 = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    try:
        # Initialize improved chunker
        chunker = ImprovedSemanticChunker()
        
        # Process payload
        results = chunker.process_payload(test_payload_1)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Document processed: {results['document_url']}")
        print(f"Total chunks created: {results['total_chunks']}")
        print(f"Questions answered: {len(results['results'])}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
