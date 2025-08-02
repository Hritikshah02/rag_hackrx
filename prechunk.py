import os
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document as LlamaDocument
from webhook_api import ImprovedSemanticChunker

# URLs for pre-chunking
DOCS = {
    "indian_constitution": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"
}

# Unique collection names for isolation
doc_to_collection = {
    "indian_constitution": "indian_constitution_collection"
}

def parse_and_chunk_with_llamaparse(file_url: str) -> list:
    """Use LlamaParse to extract and chunk document content semantically."""
    print(f"Using LlamaParse to process: {file_url}")
    parser = LlamaParse()
    
    # LlamaParse can ingest URLs directly
    docs = parser.load_data(file_url)
    
    # Each doc is a LlamaDocument, which contains nodes (chunks)
    all_chunks = []
    chunk_id = 0
    
    for doc in docs:
        # Use LlamaIndex's SimpleNodeParser to get semantic chunks
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=300,  # Match webhook_api chunk size
            chunk_overlap=50  # Match webhook_api overlap
        )
        nodes = node_parser.get_nodes_from_documents([doc])
        
        for node in nodes:
            chunk_text = node.get_content().strip()
            if chunk_text:
                all_chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'size': len(chunk_text),
                    'section': getattr(node, 'metadata', {}).get('section', 0)
                })
                chunk_id += 1
    
    print(f"LlamaParse created {len(all_chunks)} semantic chunks.")
    return all_chunks

if __name__ == "__main__":
    chunker = ImprovedSemanticChunker()
    for doc_key, url in DOCS.items():
        print(f"\nProcessing: {doc_key}")
        
        # Use LlamaParse for extraction and chunking
        chunks = parse_and_chunk_with_llamaparse(url)
        
        # Set unique collection name for this doc
        chunker.collection_name = doc_to_collection[doc_key]
        chunker.create_vector_store(chunks)
        print(f"Done: {doc_key} ({len(chunks)} chunks)")