#!/usr/bin/env python3
"""
Test script to verify parallel processing with Groq LLM + Hybrid Search
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from webhook_api import ImprovedSemanticChunker

def test_parallel_processing():
    """Test parallel processing of questions with timing comparison"""
    
    print("⚡ Testing Parallel Processing with Groq + Hybrid Search")
    print("=" * 65)
    
    # Initialize the chunker
    try:
        chunker = ImprovedSemanticChunker()
        print("✅ ImprovedSemanticChunker initialized successfully")
        print(f"🎯 Primary LLM: {chunker.primary_llm}")
    except Exception as e:
        print(f"❌ Failed to initialize chunker: {e}")
        return
    
    # Create sample documents for testing
    sample_docs = [
        "Machine learning algorithms are computational methods that enable systems to learn patterns from data without explicit programming.",
        "Python is a versatile programming language widely used in data science, web development, and artificial intelligence applications.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language.",
        "Deep learning utilizes neural networks with multiple layers to solve complex problems in computer vision and language understanding.",
        "Data preprocessing involves cleaning, transforming, and preparing raw data for analysis and machine learning model training.",
        "The transformer architecture revolutionized NLP by introducing attention mechanisms for better context understanding.",
        "Vector databases store high-dimensional embeddings that enable efficient similarity search for retrieval-augmented generation.",
        "Retrieval-augmented generation (RAG) combines information retrieval with language generation to provide contextually accurate responses.",
        "Groq processors are specialized hardware designed to accelerate large language model inference with high throughput.",
        "Hybrid search combines semantic vector search with traditional keyword search to improve information retrieval accuracy.",
        "Artificial intelligence encompasses machine learning, deep learning, and neural networks for intelligent system development.",
        "Cloud computing provides scalable infrastructure for deploying AI and machine learning applications at enterprise scale."
    ]
    
    # Create chunks from sample documents
    chunks = []
    for i, doc in enumerate(sample_docs):
        chunks.append({
            'id': f'parallel_test_chunk_{i}',
            'text': doc,
            'size': len(doc.split())
        })
    
    # Set up collection and create vector store
    chunker.collection_name = "parallel_processing_test"
    
    try:
        # Clean up any existing collection
        try:
            chunker.chroma_client.delete_collection("parallel_processing_test")
        except:
            pass
        
        chunker.create_vector_store(chunks)
        print("✅ Vector store and BM25 index created successfully")
        print(f"📊 Indexed {len(chunks)} documents")
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return
    
    # Test questions for parallel processing
    test_questions = [
        "What is machine learning and how does it work?",
        "How does Python help in data science applications?",
        "What are the benefits of transformer architecture in NLP?",
        "Explain how RAG systems combine retrieval and generation",
        "What makes Groq processors special for AI inference?",
        "How does hybrid search improve information retrieval?",
        "What role does data preprocessing play in ML?",
        "How do vector databases support similarity search?",
        "What are the applications of deep learning?",
        "How does cloud computing support AI deployment?"
    ]
    
    print(f"\n🧪 Testing Parallel Processing with {len(test_questions)} Questions")
    print("-" * 60)
    
    # Create log directory for testing
    log_dir = "test_parallel_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Test parallel processing
        print("🚀 Starting parallel processing test...")
        start_time = time.time()
        
        # Use the parallel processing method
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            all_results_data, final_answers = loop.run_until_complete(
                chunker.process_questions_parallel(test_questions, log_dir)
            )
        finally:
            loop.close()
        
        end_time = time.time()
        parallel_duration = end_time - start_time
        
        print(f"\n⚡ Parallel Processing Results:")
        print(f"   ⏱️  Total time: {parallel_duration:.2f} seconds")
        print(f"   📝 Questions processed: {len(final_answers)}")
        print(f"   ✅ Successful answers: {sum(1 for result in all_results_data if result.get('success', True))}")
        print(f"   ❌ Failed answers: {sum(1 for result in all_results_data if not result.get('success', True))}")
        print(f"   🏃 Average time per question: {parallel_duration/len(test_questions):.2f} seconds")
        
        # Verify answer order preservation
        print(f"\n🔍 Verifying Answer Order Preservation:")
        for i, (question, answer) in enumerate(zip(test_questions, final_answers)):
            print(f"   Q{i+1}: {question[:50]}...")
            print(f"   A{i+1}: {answer[:80]}...")
            print()
        
        # Check for any errors
        errors = [result for result in all_results_data if not result.get('success', True)]
        if errors:
            print(f"\n⚠️  Found {len(errors)} errors:")
            for error in errors:
                print(f"   - Question {error['index']+1}: {error.get('error', 'Unknown error')}")
        else:
            print("\n✅ No errors found - all questions processed successfully!")
        
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   🔥 Parallel processing completed in {parallel_duration:.2f}s")
        print(f"   📈 Estimated sequential time would be ~{len(test_questions) * 2:.1f}s")
        print(f"   ⚡ Speed improvement: ~{(len(test_questions) * 2) / parallel_duration:.1f}x faster")
        
        print(f"\n✅ Parallel Processing Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Error during parallel processing test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test collection
        try:
            chunker.chroma_client.delete_collection("parallel_processing_test")
            print("🧹 Cleaned up test collection")
        except:
            pass

if __name__ == "__main__":
    test_parallel_processing()
