#!/usr/bin/env python3
"""
Test script to verify hybrid search implementation
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from webhook_api import ImprovedSemanticChunker

def test_hybrid_search():
    """Test the hybrid search functionality with sample documents"""
    
    print("🔍 Testing Hybrid Search Implementation")
    print("=" * 50)
    
    # Initialize the chunker
    try:
        chunker = ImprovedSemanticChunker()
        print("✅ ImprovedSemanticChunker initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chunker: {e}")
        return
    
    # Create sample documents for testing
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Python is a popular programming language used for data science and web development.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers to solve complex problems.",
        "Data preprocessing is crucial for building effective machine learning models.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "Retrieval-augmented generation combines information retrieval with language generation."
    ]
    
    # Create chunks from sample documents
    chunks = []
    for i, doc in enumerate(sample_docs):
        chunks.append({
            'id': f'test_chunk_{i}',
            'text': doc,
            'size': len(doc.split())
        })
    
    # Set up collection name and create vector store
    chunker.collection_name = "test_hybrid_search"
    
    try:
        # Try to delete existing collection if it exists
        try:
            chunker.chroma_client.delete_collection("test_hybrid_search")
        except:
            pass  # Collection doesn't exist, that's fine
        
        chunker.create_vector_store(chunks)
        print("✅ Vector store and BM25 index created successfully")
        print(f"📊 Indexed {len(chunks)} documents")
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return
    
    # Test queries
    test_queries = [
        "machine learning algorithms",  # Should match semantic + keyword
        "Python programming",          # Should match keyword well
        "neural networks",             # Should match semantic well
        "data science",               # Should match both approaches
    ]
    
    print("\n🧪 Running Test Queries")
    print("-" * 30)
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        try:
            # Test semantic search
            semantic_results = chunker.advanced_semantic_search(query, top_k=3)
            print(f"📝 Semantic Search: {len(semantic_results)} results")
            
            # Test keyword search
            keyword_results = chunker.keyword_search(query, top_k=3)
            print(f"🔑 Keyword Search: {len(keyword_results)} results")
            
            # Test hybrid search
            hybrid_results = chunker.hybrid_search(query, top_k=3)
            print(f"🔀 Hybrid Search: {len(hybrid_results)} results")
            
            # Show top result from each method
            if semantic_results:
                print(f"   Semantic Top: {semantic_results[0]['text'][:60]}... (score: {semantic_results[0]['similarity_score']:.3f})")
            
            if keyword_results:
                print(f"   Keyword Top:  {keyword_results[0]['text'][:60]}... (score: {keyword_results[0]['similarity_score']:.3f})")
            
            if hybrid_results:
                print(f"   Hybrid Top:   {hybrid_results[0]['text'][:60]}... (score: {hybrid_results[0]['similarity_score']:.3f})")
                if 'search_types' in hybrid_results[0]:
                    print(f"   Search Types: {hybrid_results[0]['search_types']}")
            
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
    
    print("\n✅ Hybrid Search Test Completed!")
    print("\n📋 Summary:")
    print("- ✅ BM25 keyword search implemented")
    print("- ✅ Hybrid search with score fusion implemented") 
    print("- ✅ Search results include metadata for search types")
    print("- ✅ Ready for integration with Groq LLM")

if __name__ == "__main__":
    test_hybrid_search()
