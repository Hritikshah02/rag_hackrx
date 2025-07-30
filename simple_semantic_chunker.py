import os
import json
import requests
import tempfile
from typing import List, Dict, Any
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
from io import BytesIO
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleSemanticChunker:
    def __init__(self):
        """Initialize the semantic chunker with required models and configurations"""
        # Configure Google Gemini
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.google_api_key)
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = "temp_documents"
        
    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        print(f"Downloading document from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        print("Extracting text from PDF...")
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        
        return text
    
    def semantic_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create semantic chunks from text with overlap"""
        print("Creating semantic chunks...")
        
        # Clean and normalize text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split into sentences for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': current_chunk.strip(),
                    'size': current_size
                })
                
                # Create overlap by keeping last part of current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': current_chunk.strip(),
                'size': current_size
            })
        
        print(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Create vector store from chunks"""
        print("Creating vector embeddings...")
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate embeddings and add to collection
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=[chunk['id'] for chunk in chunks],
            metadatas=[{'size': chunk['size']} for chunk in chunks]
        )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search to find relevant chunks"""
        print(f"\nSearching for: '{query}'")
        print("-" * 60)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0], 
            results['metadatas'][0]
        )):
            chunk_info = {
                'rank': i + 1,
                'text': doc,
                'similarity_score': 1 - distance,  # Convert distance to similarity
                'metadata': metadata
            }
            retrieved_chunks.append(chunk_info)
            
            # Display full retrieved chunk
            print(f"RETRIEVED CHUNK {i+1} (Similarity Score: {chunk_info['similarity_score']:.3f}):")
            print(f"{doc}")
            print("-" * 60)
        
        return retrieved_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Gemini LLM"""
        print("\nGenerating answer with Gemini...")
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Context {i+1}: {chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Create prompt for short, precise answers
        prompt = f"""You are an expert insurance policy analyst. Based on the following context from an insurance policy document, provide a short, precise, one-sentence answer to the question.

Context from Document:
{context}

Question: {query}

Instructions:
- Provide a single, concise sentence that directly answers the question
- Include specific numbers, percentages, time periods, and conditions mentioned in the document
- Use exact terminology and values from the document
- Start with a direct answer (Yes/No if applicable) followed by key details
- If multiple conditions exist, mention the most important ones in one flowing sentence
- If the information is not in the context, state "Information not found in the document"
- Keep the answer under 50 words while being complete and accurate

Example format:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
- "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy, with eligibility requiring 24 months of continuous coverage and limited to two deliveries per policy period."

Answer:"""
        
        try:
            response = self.llm_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process the complete payload with document and questions"""
        print("=" * 80)
        print("SIMPLE SEMANTIC CHUNKER - PROCESSING PAYLOAD")
        print("=" * 80)
        
        # Download and process document
        doc_url = payload['documents']
        questions = payload['questions']
        
        # Download document
        doc_content = self.download_document(doc_url)
        
        # Extract text (assuming PDF for now)
        text = self.extract_text_from_pdf(doc_content)
        print(f"Extracted {len(text)} characters from document")
        
        # Create semantic chunks
        chunks = self.semantic_chunk_text(text)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Process each question
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"QUESTION {i}: {question}")
            print('='*80)
            
            # Semantic search
            retrieved_chunks = self.semantic_search(question)
            
            # Generate answer
            answer = self.generate_answer(question, retrieved_chunks)
            
            print(f"\n{'*'*20} FINAL ANSWER {'*'*20}")
            print(f"{answer}")
            print('*'*60)
            
            results.append({
                'question': question,
                'retrieved_chunks': retrieved_chunks,
                'answer': answer
            })
        
        return {
            'document_url': doc_url,
            'total_chunks': len(chunks),
            'results': results
        }

def main():
    """Main function to run the semantic chunker"""
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
        # Initialize chunker
        chunker = SimpleSemanticChunker()
        
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
