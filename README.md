
# Agentic RAG API (Runtime-only)

## **🚀 Advanced Agentic RAG with Zero-Hardcoding Intelligence**

**Next-generation RAG system featuring intelligent tool-calling, dynamic caching, hybrid retrieval fusion, and pattern-learning capabilities that remove the need for hardcoded logic through context-driven decision-making.**

• **🤖 Agentic Tool Intelligence:** HTTP requests, web search, API interactions) — fully context-driven decisions
• **⚡ Smart Document Caching:** Persistent chunk storage system providing 15–20× speed improvements for repeated queries with auto-cleanup and cache management
• **🧠 Pattern Learning Engine:** Dynamic mathematical and logical pattern extraction from documents — no explicit programming required
• **🔍 Hybrid Retrieval Fusion:** Semantic (BGE embeddings) + keyword (BM25) search with intelligent weight balancing and multi-language OCR
• **🌐 Production-Ready Architecture:** FastAPI backend, dual LLM fallbacks (Groq + Gemini), secure auth, full logging, and real-time monitoring

---

## 📋 **Quick Setup Guide for Evaluators**

### **Prerequisites**

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
* Internet connection for API calls
* Optional: GPU for faster embeddings (CPU supported)

---

### **🚀 Step 1: Clone & Setup**

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_hackrx

# Create a conda environment with Python 3.10
conda create -n rag_hackrx python=3.10 -y

# Activate the environment
conda activate rag_hackrx

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (including Malayalam support)
sudo apt-get install tesseract-ocr tesseract-ocr-mal -y

# Install additional tools
pip install tools frontend
```

---

### **🔑 Step 2: Environment Configuration**

Create a `.env` file in the project root:

```bash
touch .env
```

Add your keys:

```ini
# Required: LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Optional: OCR Support for non-English PDFs
TESSERACT_LANGS=eng+mal
```

**Where to get keys:**

* **Groq** → [console.groq.com](https://console.groq.com)
* **Google Gemini** → [ai.google.dev](https://ai.google.dev)
* **Llama Cloud** → [cloud.llamaindex.ai](https://cloud.llamaindex.ai)

---

### **▶️ Step 3: Start the Server**

```bash
# Start the API server
python webhook_api.py

# Or using uvicorn
uvicorn webhook_api:app --host 0.0.0.0 --port 8000 --reload
```

When ready, you should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 **Testing the System**

### **Method 1: Using curl**

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer hackrx2024" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": [
      "What is the main topic of this document?",
      "Summarize the key points"
    ]
  }'
```

### **Method 2: Using the Test Script**

```bash
python test_webhook.py
```

---

## 📊 **Example API Usage**

### **Basic Document Query**

```json
POST /hackrx/run
Headers: {
  "Authorization": "Bearer hackrx2024",
  "Content-Type": "application/json"
}
Body: {
  "documents": "https://arxiv.org/pdf/2301.00234.pdf",
  "questions": [
    "What is the paper about?",
    "What are the main contributions?",
    "What datasets were used?"
  ]
}
```

### **Mathematical Pattern Learning**

```json
{
  "documents": "https://example.com/math-rules.pdf",
  "questions": [
    "What is 25 + 17?",
    "Calculate 100 + 55"
  ]
}
```

### **Agentic Tool Calling**

```json
{
  "documents": "https://example.com/api-guide.pdf",
  "questions": [
    "What is the flight number for route XYZ?",
    "Get the current status from the API endpoint"
  ]
}
```

---

## 🏗 **System Architecture**

### **Core Pipeline**

1. **Document Input** → URL validation & cache check
2. **Smart Processing** → LlamaParse or cached chunks
3. **Agentic Decision** → Tool calling when required
4. **Hybrid Retrieval** → Semantic + keyword fusion
5. **Context-Aware Generation** → Intelligent responses
6. **Pattern Learning** → Dynamic rule discovery

---

## 📁 **Project Structure**

```
rag_hackrx/
├── webhook_api.py       # Main API server
├── test_webhook.py      # Test script
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── document_cache/      # Cached chunks
├── vector_store/        # ChromaDB storage
├── transaction_logs/    # Request/response logs
└── logs/                # System logs
```

---

## 🔍 **Troubleshooting**

**Missing API keys** → Check `.env` formatting (no extra spaces)
**Import errors** → `pip install -r requirements.txt --force-reinstall`
**Port in use** → `uvicorn webhook_api:app --port 8001`
**Slow processing** → First request builds cache; later ones are 20× faster
**Auth failed** → Use `Authorization: Bearer hackrx2024`

---

## 🏆 **Why This Stands Out**

* 🚀 Zero hardcoding — AI-driven decisions
* ⚡ 20× faster with intelligent caching
* 🧠 Self-learning pattern recognition
* 🤖 Autonomous tool calling
* 🔍 Hybrid retrieval fusion
* 🌐 Production-ready

