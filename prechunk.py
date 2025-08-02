import os
from llama_parse import LlamaParse
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document as LlamaDocument
from webhook_api import ImprovedSemanticChunker

# URLs for pre-chunking
DOCS = {
    "indian_constitution": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "principia_newton": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "doc_1": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "doc_2": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "doc_3": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "doc_4": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
    "doc_5": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
    "uni_group_health": "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D"
}

# Unique collection names for isolation
doc_to_collection = {
    "indian_constitution": "indian_constitution_collection",
    "principia_newton": "principia_newton_collection",
    "doc_1": "doc_1",
    "doc_2": "doc_2",
    "doc_3": "doc_3",
    "doc_4": "doc_4",
    "doc_5": "doc_5",
    "uni_group_health": "uni_group_health_collection"
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