import os
import pickle
import re
from typing import List, Dict, Any, Optional
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import Config

class VectorStore:
    """Handles vector storage and semantic search using ChromaDB"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and ChromaDB client"""
        # Initialize ChromaDB client first
        try:
            # Try with simple client first
            self.client = chromadb.PersistentClient(
                path=self.config.VECTOR_STORE_PATH
            )
        except Exception as e:
            print(f"Warning: PersistentClient failed: {e}")
            # Fallback to in-memory client for testing
            print("Falling back to in-memory ChromaDB client")
            self.client = chromadb.Client()
        
        # Initialize collection as None - will be created when needed
        self.collection = None
        
        # Try to get existing collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.COLLECTION_NAME
            )
        except Exception:
            # Collection doesn't exist, will be created when needed
            pass
        
        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer embedding model with direct meta tensor bypass"""
        # Set environment variables to avoid PyTorch issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        print("🔄 Attempting to initialize sentence transformer model...")
        
        # Try multiple approaches to avoid meta tensor issues
        models_to_try = [
            self.config.EMBEDDING_MODEL,
            'all-MiniLM-L6-v2',
            'sentence-transformers/all-MiniLM-L6-v2'
        ]
        
        for i, model_name in enumerate(models_to_try):
            try:
                print(f"🔄 Attempt {i+1}: Trying model '{model_name}'...")
                
                # Import fresh each time to avoid cached issues
                import importlib
                import sys
                if 'sentence_transformers' in sys.modules:
                    importlib.reload(sys.modules['sentence_transformers'])
                
                from sentence_transformers import SentenceTransformer
                import torch
                
                # Force CPU device without using .to() method that causes meta tensor issues
                # Initialize without specifying device first, then handle device separately
                self.embedding_model = SentenceTransformer(model_name)
                
                # Manually move all parameters to CPU to avoid meta tensor issues
                if hasattr(self.embedding_model, '_modules'):
                    for module in self.embedding_model._modules.values():
                        if hasattr(module, 'cpu'):
                            module.cpu()
                
                # Test the model with a simple encoding
                test_result = self.embedding_model.encode(
                    ["This is a test sentence."], 
                    convert_to_numpy=True,
                    device='cpu'
                )
                
                if test_result is not None and len(test_result) > 0:
                    print(f"✅ Model '{model_name}' initialized and tested successfully!")
                    print(f"📊 Test embedding shape: {test_result.shape}")
                    return
                else:
                    raise Exception("Model test failed - no embeddings generated")
                    
            except Exception as e:
                print(f"❌ Attempt {i+1} failed: {str(e)}")
                if i < len(models_to_try) - 1:
                    print(f"🔄 Trying next model...")
                    continue
                else:
                    print(f"❌ Final fallback failed: {str(e)}")
                    print("💡 This appears to be a PyTorch/sentence-transformers compatibility issue.")
                    print("💡 Recommended solutions:")
                    print("   1. Restart your terminal/IDE completely")
                    print("   2. Try: pip install sentence-transformers==2.2.2")
                    print("   3. Clear cache: rm -rf ~/.cache/huggingface/")
                    raise RuntimeError(f"Could not initialize sentence transformer model: {str(e)}")
    
    def _ensure_embedding_model(self):
        """Ensure embedding model is initialized before use"""
        if self.embedding_model is None:
            self._initialize_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError("Sentence transformer model could not be initialized")
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create vector store from document chunks using normalized content for embeddings
        
        Args:
            chunks: List of document chunks with metadata
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        self._ensure_embedding_model()
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection(
                name=self.config.COLLECTION_NAME
            )
        
        # Prepare data for ChromaDB
        documents = []  # Original content for display
        embedding_texts = []  # Normalized content for embeddings
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['content'])
            # Use normalized content for embeddings if available, otherwise original
            embedding_text = chunk.get('normalized_content', chunk['content'].lower())
            embedding_texts.append(embedding_text)
            
            metadatas.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'token_count': chunk['token_count'],
                'char_count': chunk['char_count'],
                'keywords': ', '.join(chunk.get('keywords', []))  # Convert list to string
            })
            ids.append(f"chunk_{i}")
        
        # Generate embeddings using normalized text
        print(f"🔄 Generating embeddings for {len(embedding_texts)} chunks using normalized content...")
        embeddings = self._generate_embeddings(embedding_texts)
        
        # Add to collection (store original documents but use normalized embeddings)
        self.collection.add(
            documents=documents,  # Original content for display
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        
        print(f"✅ Vector store created with {len(chunks)} chunks using enhanced normalization")
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform enhanced semantic search with improved query processing and retrieval
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with metadata
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized. Please upload documents first.")
        
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        # Normalize and expand query for better matching
        normalized_query = self._normalize_query(query)
        expanded_queries = self._expand_query(normalized_query)
        
        # Generate embeddings for all query variations
        all_embeddings = self._generate_embeddings(expanded_queries)
        
        # Perform multiple search strategies
        all_results = []
        
        # Strategy 1: Direct semantic search with expanded queries
        for i, query_embedding in enumerate(all_embeddings):
            weight = 1.0 if i == 0 else 0.8  # Original query gets higher weight
            
            # Get more results initially to improve recall
            search_count = min(top_k * 3, 20)  # Increased search count
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=search_count
            )
            
            # Add weighted results
            for j in range(len(results['documents'][0])):
                distance = results['distances'][0][j]
                similarity_score = 1 - distance
                
                result = {
                    'content': results['documents'][0][j],
                    'metadata': results['metadatas'][0][j],
                    'score': similarity_score * weight,
                    'source': results['metadatas'][0][j]['source'],
                    'chunk_id': results['metadatas'][0][j]['chunk_id'],
                    'id': results['ids'][0][j],
                    'distance': distance
                }
                all_results.append(result)
        
        # Strategy 2: Keyword-based boost for exact matches
        query_lower = query.lower()
        for result in all_results:
            content_lower = result['content'].lower()
            
            # Boost score for exact keyword matches
            keyword_boost = 0
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    keyword_boost += 0.1
            
            # Boost for important terms
            important_terms = ['grace period', 'waiting period', 'coverage', 'benefit', 'premium']
            for term in important_terms:
                if term in query_lower and term in content_lower:
                    keyword_boost += 0.15
            
            result['score'] += keyword_boost
        
        # Deduplicate and merge results by ID (keep highest score)
        unique_results = {}
        for result in all_results:
            result_id = result['id']
            if result_id not in unique_results or result['score'] > unique_results[result_id]['score']:
                unique_results[result_id] = result
        
        # Convert back to list and sort by score
        search_results = list(unique_results.values())
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply dynamic similarity threshold based on result quality
        if search_results:
            # Use adaptive threshold based on top result
            top_score = search_results[0]['score']
            min_threshold = max(0.2, top_score * 0.4)  # At least 40% of top score
        else:
            min_threshold = 0.2
        
        filtered_results = []
        for result in search_results[:top_k * 2]:  # Consider more results
            if result['score'] >= min_threshold:
                # Clean up result and add relevance info
                result.pop('id', None)
                result.pop('distance', None)
                # Add content preview for debugging
                result['content_preview'] = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                filtered_results.append(result)
        
        # Ensure we return at least one result if any exist
        if not filtered_results and search_results:
            best_result = search_results[0]
            best_result.pop('id', None)
            best_result.pop('distance', None)
            best_result['content_preview'] = best_result['content'][:200] + "..." if len(best_result['content']) > 200 else best_result['content']
            filtered_results = [best_result]
        
        # Return top results
        return filtered_results[:top_k]
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better semantic matching"""
        # Convert to lowercase
        normalized = query.lower()
        
        # Apply same normalizations as document processing
        normalizations = {
            r'\bpre-existing\b': 'preexisting',
            r'\bpre existing\b': 'preexisting',
            r'\bco-payment\b': 'copayment',
            r'\bco payment\b': 'copayment',
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
            r'\bNCD\b': 'no claim discount',
            r'\bAYUSH\b': 'ayush',
            r'\bICU\b': 'icu intensive care',
            r'\bOPD\b': 'opd outpatient',
            r'\bIPD\b': 'ipd inpatient'
        }
        
        for pattern, replacement in normalizations.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations for better semantic matching"""
        expanded_queries = [query]  # Start with original query
        query_lower = query.lower()
        
        # Medical/insurance term expansions with more variations
        if 'waiting period' in query_lower or 'wait' in query_lower:
            expanded_queries.extend([
                query.replace('waiting period', 'wait time'),
                query + " coverage delay",
                query + " eligibility period",
                "how long wait " + query.replace('waiting period', ''),
                query.replace('waiting', 'wait')
            ])
        
        if 'grace period' in query_lower or 'grace' in query_lower:
            expanded_queries.extend([
                query.replace('grace period', 'payment grace time'),
                query + " premium payment delay",
                query + " late payment allowed",
                "payment deadline " + query
            ])
        
        # Coverage/benefit variations
        if any(word in query_lower for word in ['cover', 'coverage', 'benefit']):
            expanded_queries.extend([
                query.replace('covered', 'included'),
                query.replace('coverage', 'benefits'),
                query.replace('cover', 'include'),
                query + " policy benefits",
                query + " insurance coverage"
            ])
        
        # Surgery/procedure variations
        if any(word in query_lower for word in ['surgery', 'procedure', 'operation']):
            expanded_queries.extend([
                query + " medical treatment",
                query.replace('surgery', 'surgical procedure'),
                query.replace('surgery', 'operation'),
                query + " hospitalization",
                query + " medical procedure"
            ])
        
        # Maternity variations
        if 'maternity' in query_lower or 'pregnancy' in query_lower:
            expanded_queries.extend([
                query.replace('maternity', 'pregnancy'),
                query.replace('pregnancy', 'maternity'),
                query + " childbirth",
                query + " delivery expenses"
            ])
        
        # Dental variations
        if 'dental' in query_lower or 'teeth' in query_lower:
            expanded_queries.extend([
                query + " teeth treatment",
                query.replace('dental', 'tooth'),
                query.replace('teeth', 'dental'),
                "oral health " + query
            ])
        
        # Amount/percentage variations
        if any(word in query_lower for word in ['amount', 'limit', 'percent', '%']):
            expanded_queries.extend([
                query + " coverage amount",
                query + " benefit limit",
                query.replace('%', 'percent')
            ])
        
        # Remove duplicates, empty strings, and limit results
        expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
        expanded_queries = list(dict.fromkeys(expanded_queries))  # Preserve order while removing duplicates
        return expanded_queries[:5]  # Increased from 3 to 5
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using sentence transformers"""
        self._ensure_embedding_model()
        
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if self.collection is None:
            return {"status": "not_initialized"}
        
        count = self.collection.count()
        return {
            "status": "ready",
            "document_count": count,
            "collection_name": self.config.COLLECTION_NAME
        }
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new documents to existing vector store
        
        Args:
            chunks: List of new document chunks
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized")
        
        if not chunks:
            return
        
        # Get current count for ID generation
        current_count = self.collection.count()
        
        # Prepare data
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'token_count': chunk['token_count'],
                'char_count': chunk['char_count'],
                'keywords': ', '.join(chunk.get('keywords', []))  # Convert list to string
            })
            ids.append(f"chunk_{current_count + i}")
        
        # Generate embeddings
        embeddings = self._generate_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        
        print(f"Added {len(chunks)} new chunks to vector store")
    
    def delete_documents_by_source(self, source: str) -> None:
        """
        Delete all documents from a specific source
        
        Args:
            source: Source filename to delete
        """
        if self.collection is None:
            return
        
        # Query for documents from this source
        results = self.collection.get(
            where={"source": source},
            include=['metadatas']
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks from source: {source}")
    
    def reset_vector_store(self) -> None:
        """Reset the vector store by deleting all data"""
        try:
            self.client.delete_collection(name=self.config.COLLECTION_NAME)
            self.collection = None
            print("Vector store reset successfully")
        except ValueError:
            print("Vector store was already empty")
