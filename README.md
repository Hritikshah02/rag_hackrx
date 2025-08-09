# Agentic RAG API (Runtime-only)

A production-style Agentic Retrieval-Augmented Generation (RAG) API that processes documents at runtime only. No pre-chunking, no pre-caching. Each request fetches the document, chunks it, embeds it, retrieves context, and generates answers. Agentic tool-use augments the flow when needed.

## Highlights

- Runtime processing only: parse → chunk → embed → retrieve → answer per request
- Hybrid retrieval: semantic (Chroma + BGE embeddings) + BM25 keyword fusion
- Agentic actions: web fetch/search decisions and mission-style flows (restricted allowlist)
- LLM stack: Groq (primary) with Gemini fallback
- Language-aware: non-English PDF handling via OCR full-text path when available
- Secure API: FastAPI with Bearer auth

## Architecture

1. Receive payload: `documents` (URL) + `questions` (list)
2. Optional agentic decision: use web fetch/search if appropriate
3. Document parsing: LlamaParse (or agentic fetch) → normalized text → runtime chunking
4. Embedding: BAAI/bge-large-en-v1.5 (sentence-transformers)
5. Vector store: temporary Chroma collection (cleaned after answering)
6. Retrieval: hybrid search (semantic + BM25)
7. Answer: LLM generation (Groq primary, Gemini fallback)

## Quick Start

### Requirements
- Python 3.10+
- GPU optional (CPU supported)

### Install
```bash
pip install -r requirements.txt
```

### Environment
Create `.env` in project root:
```ini
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
# Optional: Tesseract OCR languages (for non-English PDFs)
TESSERACT_LANGS=eng+mal
```

### Run API
```bash
uvicorn webhook_api:app --host 0.0.0.0 --port 8000
```

### Auth
The API uses a static Bearer token (see `EXPECTED_TOKEN` in `webhook_api.py`).
Send header:
```
Authorization: Bearer <token>
```

### Endpoint
POST `/hackrx/run`
Request body:
```json
{
  "documents": "https://example.com/file.pdf",
  "questions": ["What is X?", "How many Y?"]
}
```
Response:
```json
{ "answers": ["...", "..."] }
```

## Key Components

- `webhook_api.py`
  - `ImprovedSemanticChunker`: end-to-end runtime pipeline
  - Chunking: token-based + LlamaParse nodes
  - Embeddings: BGE-large
  - Vector DB: Chroma (temporary collection per request)
  - Retrieval: `advanced_semantic_search`, `keyword_search`, `hybrid_search`
  - Agentic helpers: tool decision, mission resolution with strict host allowlist
  - OCR path for non-English PDFs: full-text fixed-context when OCR is available

## Configuration knobs (code-level)

- Chunking: `self.chunk_size_tokens` (default 300), `self.overlap_tokens` (50)
- Embedding batch size: `create_vector_store` uses `batch_size=64`
- Hybrid weights: `hybrid_search(semantic_weight, keyword_weight)`
- Allowed agentic hosts: `self.allowed_action_hosts`

## Troubleshooting

- Missing API keys: ensure `GROQ_API_KEY` and `GOOGLE_API_KEY` are set
- Slow embeddings: reduce batch size; prefer GPU when available
- Memory pressure: reduce chunk size; limit document size upstream
- OCR unavailable: non-English PDFs proceed without OCR full-text

## Security

- Bearer token auth (demo); replace with proper auth in production
- Agentic HTTP actions restricted to an allowlist

## License

MIT
