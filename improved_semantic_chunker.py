import os
import json
import uuid
import requests
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from io import BytesIO
import re
import google.generativeai as genai
import tiktoken
import torch

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
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# Load environment variables
load_dotenv()

class ImprovedSemanticChunker:
    def __init__(self):
        """Initialize the improved semantic chunker with better models and configurations"""
        # Configure Google Gemini - using the full model instead of lite
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-en-v1.5")  # or whichever 1024-dim model you're using

        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.google_api_key)
        # Using the full Gemini model for better reasoning
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Initialize memory-efficient embedding model
        print("Loading BGE-Small-EN embedding model (memory optimized)...")
        # Using BGE-Small-EN for good performance with lower memory usage
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_model = self.embedding_model.to("cuda" if torch.cuda.is_available() else "cpu")

        
        
        # Initialize ChromaDB with persistent storage
        os.makedirs("vector_store", exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path="vector_store")
        self.collection_name = "improved_documents_bge_small"
        
        # Create a ChromaDB compatible embedding function with BGE-Small model
        # self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize tokenizer for token-based chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Chunking parameters (optimized for large documents)
        self.chunk_size_tokens = 500  # Larger chunk size to reduce total chunk count
        self.overlap_tokens = 100       # Overlap size in tokens
        
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        print(f"Downloading document from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract and clean text from PDF content"""
        print("Extracting text from PDF...")
        text = ""
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            num_pages = len(reader.pages)
            print(f"PDF has {num_pages} pages.")
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n" # Add space between pages
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'(\n\s*)+', '\n', text).strip() # Normalize newlines
            
            print(f"Successfully extracted and cleaned text. Total characters: {len(text)}")
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Return empty string or handle error as appropriate
            return ""
            
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
    
    def token_based_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks based on token count with specified overlap, with safeguards"""
        
        print(f"Creating token-based chunks with dynamic overlap...")

        # Tokenize the entire text (with memory optimization)
        try:
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
            print(f"Document tokenized: {total_tokens} tokens")
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Fallback to character-based chunking if tokenization fails
            return self.fallback_character_chunking(text)

        # Adjust overlap based on chunk size
        self.overlap_tokens = 100  # ~10% of chunk size is good for context continuity


        chunks = []
        chunk_id = 0
        start_idx = 0
        max_chunks = 2000  # Prevent crazy chunk counts
        previous_start_idx = -1

        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size_tokens, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]

            try:
                chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                if not chunk_text or len(chunk_text) < 50:
                    print(f"⚠️ Skipping chunk {chunk_id} due to insufficient content.")
                    start_idx = end_idx
                    continue
            except Exception as e:
                print(f"Decoding error for chunk {chunk_id}: {e}")
                start_idx = end_idx
                continue

            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': chunk_text,
                'size': len(chunk_tokens),
                'token_start': start_idx,
                'token_end': end_idx,
                'section': chunk_id // 10
            })
            chunk_id += 1

            if chunk_id % 100 == 0:
                print(f"Processed {chunk_id} chunks...")

            if chunk_id >= max_chunks:
                print(f"⚠️ Reached max chunk limit ({max_chunks}). Stopping chunking.")
                break

            previous_start_idx = start_idx
            start_idx = end_idx - self.overlap_tokens

            if start_idx <= previous_start_idx:
                print("⚠️ Infinite loop prevention triggered. Advancing start_idx.")
                start_idx = end_idx

            if start_idx >= total_tokens:
                break

        avg_tokens = total_tokens // len(chunks) if chunks else 0
        print(f"✅ Created {len(chunks)} token-based chunks (avg {avg_tokens} tokens per chunk)")
        return chunks

    
    def fallback_character_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Fallback character-based chunking if tokenization fails"""
        print("Using fallback character-based chunking...")
        
        chunk_size = 1500  # Character-based chunk size
        overlap = 200      # Character-based overlap
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'size': len(chunk_text),
                    'char_start': start,
                    'char_end': end,
                    'section': chunk_id // 10
                })
                if chunk_id >= max_chunks:
                    print(f"⚠️ Reached max chunk limit ({max_chunks}). Stopping chunking.")
                    break
                chunk_id += 1
            
            start = end - overlap
            if start >= end:
                break
        
        print(f"Created {len(chunks)} character-based chunks")
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
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Failed to delete collection {self.collection_name}: {e}")

        
        # Create new collection with explicit embedding function to handle dimension issues
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # Generate embeddings and add to collection
        texts = [chunk['text'] for chunk in chunks]
        
        # Step 1: Batched embedding on GPU using your initialized self.embedding_model
        batch_size = 128  # You can increase this if memory allows (128 or 256)
        self.collection.add(
            documents=texts,
            ids=[chunk['id'] for chunk in chunks],
            metadatas=[{
                'size': chunk['size'], 
                'section': chunk.get('section', 0)
            } for chunk in chunks]
        )

        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch = texts[i:i + batch_size]
        #     batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False, device='cuda', batch_size=batch_size, normalize_embeddings=True)
        #     embeddings.extend(batch_embeddings)

        # # Step 2: Add manually to ChromaDB collection with precomputed embeddings
        # self.collection.add(
        #     documents=texts,
        #     embeddings=embeddings,
        #     ids=[chunk['id'] for chunk in chunks],
        #     metadatas=[{
        #         'size': chunk['size'], 
        #         'section': chunk.get('section', 0)
        #     } for chunk in chunks]
        # )

        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def advanced_semantic_search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Perform semantic search with BGE embeddings"""
        print(f"\nSearching for: '{query}'")
        print("-" * 80)
        
        # Semantic search with BGE embeddings
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0], 
                results['distances'][0], 
                results['metadatas'][0]
            )):
                chunk_info = {
                    'rank': i + 1,
                    'text': doc,
                    'similarity_score': 1 - distance,
                    'metadata': metadata
                }
                retrieved_chunks.append(chunk_info)
                
                # Display full retrieved chunk
                print(f"RETRIEVED CHUNK {i+1} (Similarity Score: {chunk_info['similarity_score']:.3f}):")
                print(f"{doc}")
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

    test_payload_2 = {
        "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "When will my root canal claim of Rs 25,000 be settled?",
            "I have done an IVF for Rs 56,000. Is it covered?",
            "I did a cataract treatment of Rs 100,000. Will you settle full?",
            "Give me a list of documents to be uploaded for hospitalization due to heart surgery.",
            "I have raised a claim for hospitalization for Rs 25,000. What will I get?"
        ]
    }


    test_payload_3 = {
        "documents": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
        "questions": [
            "What is the ideal spark plug gap recommended?",
            "Does this come in tubeless tyre version?",
            "Is it compulsory to have a disc brake?",
            "Can I put Thums Up instead of oil?",
            "Give me JS code to generate a random number between 1 and 100"
        ]
    }


    test_payload_4 = {
        "documents": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
        "questions": [
            "Is Non-infective Arthritis covered?",
            "I renewed my policy yesterday, and I have been a customer for 2 years. Is Hydrocele claimable?",
            "Is abortion covered?"
        ]
    }


    test_payload_5 = {
        "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "questions": [
            "What is the official name of India according to Article 1?",
            "Which Article guarantees equality before the law and equal protection of the laws?",
            "What is abolished by Article 17 of the Constitution?",
            "What are the key ideals mentioned in the Preamble of the Indian Constitution?",
            "Under which Article can Parliament alter the boundaries of states?",
            "According to Article 24, children below what age are prohibited from working?",
            "What is the significance of Article 21 in the Indian Constitution?",
            "Article 15 prohibits discrimination on certain grounds. What are they?",
            "Which Article allows Parliament to regulate the right to form associations?",
            "What restrictions can the State impose on the right to freedom of speech?"
        ]
    }


    test_payload_5b = {
        "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "questions": [
            "If my car is stolen, what case will it be in law?",
            "If I am arrested without a warrant, is that legal?",
            "If someone denies me a job because of my caste, is that legal?",
            "If the government takes my land for a project, can I stop it?",
            "If my child is forced to work in a factory, is that legal?",
            "If I am stopped from speaking at a protest, is that a violation of my rights?",
            "If a religious place stops me from entering because of my caste, what can I do?",
            "If I change my religion, can the government stop me?",
            "If the police torture someone in custody, what right is violated?",
            "If I'm denied admission to a public university because of my caste, what law applies?"
        ]
    }
    
    # try:
    #     # Initialize improved chunker
    #     chunker = ImprovedSemanticChunker()
        
    #     # Process payload
    #     results = chunker.process_payload(test_payload_1)
        
    #     print("\n" + "="*80)
    #     print("PROCESSING COMPLETE")
    #     print("="*80)
    #     print(f"Document processed: {results['document_url']}")
    #     print(f"Total chunks created: {results['total_chunks']}")
    #     print(f"Questions answered: {len(results['results'])}")
        
    # except Exception as e:
    #     print(f"Error: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    # Define all test payloads in a list
    all_test_payloads = [
        test_payload_1,
        test_payload_2,
        test_payload_3,
        test_payload_4,
        test_payload_5,
        test_payload_5b
    ]


    total_questions = 0
    not_found_count = 0


    try:
        # Initialize improved chunker once
        chunker = ImprovedSemanticChunker()

        # Process each payload
        for idx, payload in enumerate(all_test_payloads, start=1):
            print("\n" + "="*100)
            print(f"PROCESSING PAYLOAD {idx}")
            print("="*100)
            results = chunker.process_payload(payload)
            
            print(f"Document processed: {results['document_url']}")
            print(f"Total chunks created: {results['total_chunks']}")
            print(f"Questions answered: {len(results['results'])}")
            
            for i, (q, a) in enumerate(zip(payload["questions"], results["results"]), start=1):
                answer_text = a['answer']
                print(f"\nQ{i}: {q}\nA{i}: {answer_text}")
                
                # Accuracy tracking
                total_questions += 1
                if "information not found" in answer_text.lower():
                    not_found_count += 1

        # Final Accuracy Report
        print("\n" + "="*50)
        print("📊 FINAL ACCURACY REPORT")
        print(f"Total Questions Processed: {total_questions}")
        print(f"'Information not found': {not_found_count}")
        accuracy = (total_questions - not_found_count) / total_questions if total_questions > 0 else 0
        print(f"✅ Accuracy: {accuracy:.2%}")
        print("="*50)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    print("="*80)
    print("🔍 RESOURCE CHECK: PyTorch GPU (CUDA) Availability")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU is available. Using CUDA device: {device_name}")
    else:
        print("❌ GPU not available. Falling back to CPU for embedding operations.")
    print("="*80)
    main()