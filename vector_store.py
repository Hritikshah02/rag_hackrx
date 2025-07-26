import os
import pickle
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
                    print("❌ All models failed. Trying final fallback approach...")
                    
                    # Final fallback: Use a simple embedding approach
                    try:
                        print("🔄 Using minimal initialization approach...")
                        
                        # Clear any torch caches
                        import torch
                        if hasattr(torch, 'cuda') and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Try with no device specification at all
                        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                        # Test without device specification
                        test_result = self.embedding_model.encode(["test"])
                        
                        print("✅ Minimal initialization successful!")
                        return
                        
                    except Exception as final_e:
                        print(f"❌ Final fallback failed: {str(final_e)}")
                        print("💡 This appears to be a PyTorch/sentence-transformers compatibility issue.")
                        print("💡 Recommended solutions:")
                        print("   1. Restart your terminal/IDE completely")
                        print("   2. Try: pip install sentence-transformers==2.2.2")
                        print("   3. Clear cache: rm -rf ~/.cache/huggingface/")
                        raise RuntimeError(f"Could not initialize sentence transformer model: {str(final_e)}")
    
    def _ensure_embedding_model(self):
        """Ensure embedding model is initialized before use"""
        if self.embedding_model is None:
            self._initialize_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError("Sentence transformer model could not be initialized")
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks with metadata
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.config.COLLECTION_NAME)
        except Exception:
            pass  # Collection doesn't exist or other error
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'token_count': chunk['token_count'],
                'char_count': chunk['char_count']
            })
            ids.append(f"chunk_{i}")
        
        # Generate embeddings
        embeddings = self._generate_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        
        print(f"Vector store created with {len(chunks)} chunks")
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the vector store with improved retrieval
        
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
        
        # Expand query for better semantic matching
        expanded_queries = self._expand_query(query)
        
        # Generate embeddings for all query variations
        all_embeddings = self._generate_embeddings(expanded_queries)
        
        # Perform search with multiple query embeddings and combine results
        all_results = []
        for i, query_embedding in enumerate(all_embeddings):
            weight = 1.0 if i == 0 else 0.7  # Original query gets higher weight
            
            # Perform search for this query variation
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 2  # Get more results to combine
            )
            
            # Add weighted results
            for j in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][j],
                    'metadata': results['metadatas'][0][j],
                    'score': (1 - results['distances'][0][j]) * weight,  # Weighted similarity
                    'source': results['metadatas'][0][j]['source'],
                    'chunk_id': results['metadatas'][0][j]['chunk_id'],
                    'id': results['ids'][0][j]
                }
                all_results.append(result)
        
        # Deduplicate and merge results by ID
        unique_results = {}
        for result in all_results:
            result_id = result['id']
            if result_id not in unique_results or result['score'] > unique_results[result_id]['score']:
                unique_results[result_id] = result
        
        # Convert back to list and sort by score
        search_results = list(unique_results.values())
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter by similarity threshold and limit results
        filtered_results = []
        for result in search_results[:top_k]:
            if result['score'] >= self.config.SIMILARITY_THRESHOLD:
                # Remove the 'id' field from final results
                result.pop('id', None)
                filtered_results.append(result)
        
        return filtered_results
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations for better semantic matching"""
        expanded_queries = [query]  # Start with original query
        
        # Add common insurance/medical synonyms and variations
        query_lower = query.lower()
        
        # Pre-existing conditions variations
        if 'pre-existing' in query_lower or 'preexisting' in query_lower:
            expanded_queries.extend([
                query + " waiting period",
                query.replace('pre-existing', 'existing medical conditions'),
                "medical conditions before policy",
                "waiting period for existing conditions"
            ])
        
        # Coverage/covered variations
        if 'covered' in query_lower or 'coverage' in query_lower:
            expanded_queries.extend([
                query.replace('covered', 'included'),
                query.replace('coverage', 'benefits'),
                query + " policy benefits"
            ])
        
        # Surgery/procedure variations
        if 'surgery' in query_lower or 'procedure' in query_lower:
            expanded_queries.extend([
                query + " medical treatment",
                query.replace('surgery', 'surgical procedure'),
                query + " hospitalization"
            ])
        
        # Dental variations
        if 'dental' in query_lower:
            expanded_queries.extend([
                query + " teeth treatment",
                query.replace('dental', 'tooth'),
                "oral health " + query
            ])
        
        # Remove duplicates and limit to avoid too many queries
        expanded_queries = list(set(expanded_queries))[:3]
        return expanded_queries
    
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
                'char_count': chunk['char_count']
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
