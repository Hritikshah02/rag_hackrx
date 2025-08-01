import os
from improved_semantic_chunker import ImprovedSemanticChunker

# URLs for pre-chunking
DOCS = {
    "indian_constitution": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "principia_newton": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
}

# Unique collection names for isolation
doc_to_collection = {
    "indian_constitution": "indian_constitution_collection",
    "principia_newton": "principia_newton_collection"
}

if __name__ == "__main__":
    chunker = ImprovedSemanticChunker()
    for doc_key, url in DOCS.items():
        print(f"\nProcessing: {doc_key}")
        # Download and extract
        pdf_bytes = chunker.download_document(url)
        text = chunker.extract_text_from_pdf(pdf_bytes)
        chunks = chunker.token_based_chunking(text)
        # Set unique collection name for this doc
        chunker.collection_name = doc_to_collection[doc_key]
        chunker.create_vector_store(chunks)
        print(f"Done: {doc_key} ({len(chunks)} chunks)")