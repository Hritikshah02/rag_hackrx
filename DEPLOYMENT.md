# HackRX Webhook Deployment Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Make sure your `.env` file contains:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Test Locally
```bash
# Terminal 1: Start the webhook API
python webhook_api.py

# Terminal 2: Test the webhook
python test_webhook.py
```

## 📋 API Specification

### Endpoint
```
POST /hackrx/run
```

### Authentication
```
Authorization: Bearer 0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30
```

### Request Format
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

### Response Format
```json
{
    "answers": [
        "A grace period of thirty days is provided...",
        "There is a waiting period of thirty-six months..."
    ]
}
```

## 🌐 Deployment Options

### Option 1: Render (Recommended)
1. Push code to GitHub
2. Connect Render to your repository
3. Set environment variables in Render dashboard
4. Deploy automatically

**Render Configuration:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn webhook_api:app --host 0.0.0.0 --port $PORT`
- Environment Variables: `GOOGLE_API_KEY`

### Option 2: Railway
1. Connect Railway to GitHub
2. Set environment variables
3. Deploy with one click

### Option 3: Heroku
1. Create Heroku app
2. Set environment variables
3. Deploy using Git or GitHub integration

**Heroku Commands:**
```bash
heroku create your-app-name
heroku config:set GOOGLE_API_KEY=your_key
git push heroku main
```

## 🧪 Testing Your Deployed API

### Test Health Check
```bash
curl https://your-app.herokuapp.com/health
```

### Test Main Endpoint
```bash
curl -X POST https://your-app.herokuapp.com/hackrx/run \
  -H "Authorization: Bearer 0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period?"]
  }'
```

## 📊 Performance Optimization

### Memory Management
- Temporary PDF files are automatically cleaned up
- Vector store is recreated for each request (stateless)
- ChromaDB uses in-memory storage for better performance

### Response Time
- Target: < 30 seconds per request
- Optimized chunking and embedding process
- Efficient semantic search with lower thresholds

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

2. **Authentication Errors**
   - Verify Bearer token matches exactly
   - Check Authorization header format

3. **PDF Download Errors**
   - Ensure PDF URL is accessible
   - Check network connectivity

4. **LLM Errors**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API quota and limits

### Debug Mode
Set `log_level="debug"` in uvicorn.run() for detailed logs.

## 📝 Submission Checklist

- ✅ API responds to POST /hackrx/run
- ✅ Bearer token authentication works
- ✅ Handles PDF download from URL
- ✅ Returns JSON with answers array
- ✅ Response time < 30 seconds
- ✅ HTTPS enabled (via deployment platform)
- ✅ Public URL accessible
- ✅ Tested with sample payload

## 🎯 Expected Performance

Based on your enhanced RAG system:
- **Improved Recall**: Better text normalization and chunking
- **Lower Similarity Threshold**: 0.15 for better retrieval
- **Enhanced Query Processing**: 5 query variations per question
- **Keyword Boosting**: Exact term matching
- **Precise Answers**: Direct, factual responses

Your system should now successfully answer questions that were previously missed due to the comprehensive improvements made to the document processing and search pipeline.
