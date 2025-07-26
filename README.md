# LLM-Powered Document Query System

A comprehensive system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

## Features

- **Multi-format Document Support**: Upload PDFs, Word documents, text files, and emails
- **Semantic Search**: Advanced vector-based search using state-of-the-art embeddings
- **LLM Reasoning**: Powered by Google's Gemini Flash 2.5 for intelligent decision-making
- **Structured Responses**: Returns JSON responses with decisions, amounts, justifications, and clause references
- **Interactive UI**: Beautiful Streamlit interface for easy document upload and querying
- **Query History**: Track and review previous queries and responses

## System Architecture

```
User Query → Document Upload → Chunking → Embedding → Vector Store
                                                           ↓
Response ← LLM Reasoning ← Semantic Search ← Query Processing
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hackrx
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

4. **Get Google API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file as `GOOGLE_API_KEY=your_key_here`

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Documents**:
   - Use the file uploader to add your documents
   - Supported formats: PDF, DOCX, DOC, TXT, EML
   - Documents are automatically processed and embedded

3. **Query Your Documents**:
   - Enter natural language queries
   - Example: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
   - Get structured responses with decisions and justifications

## Sample Queries

- `"46M, knee surgery, Pune, 3-month policy"`
- `"What is the coverage for dental procedures?"`
- `"Are pre-existing conditions covered?"`
- `"What is the waiting period for surgery claims?"`

## Response Format

The system returns structured JSON responses:

```json
{
  "decision": "APPROVED",
  "amount": "₹50,000",
  "confidence": 85,
  "justification": "Knee surgery is covered under the policy...",
  "referenced_clauses": [
    {
      "clause_number": "Section 4.2",
      "source": "policy_document.pdf",
      "content": "Surgical procedures are covered...",
      "relevance_explanation": "Directly addresses surgical coverage"
    }
  ],
  "key_factors": ["Policy active", "Procedure covered", "No waiting period"],
  "additional_requirements": []
}
```

## Configuration

Key configuration options in `config.py`:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `gemini-1.5-flash`
- **Chunk Size**: 1000 tokens
- **Similarity Threshold**: 0.7

## Project Structure

```
hackrx/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── document_processor.py # Document parsing and chunking
├── vector_store.py       # Vector storage and semantic search
├── llm_reasoner.py       # LLM reasoning and response generation
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md            # This file
```

## Technical Details

### Document Processing
- **PDF**: Uses PyPDF2 for text extraction
- **Word**: Uses python-docx for document parsing
- **Email**: Parses EML files with email headers and body
- **Chunking**: Intelligent text chunking with overlap for context preservation

### Vector Storage
- **ChromaDB**: Persistent vector database
- **Embeddings**: Sentence-transformers for high-quality embeddings
- **Search**: Cosine similarity-based semantic search

### LLM Integration
- **Model**: Google Gemini Flash 2.5
- **Reasoning**: Structured prompting for consistent responses
- **Validation**: Response parsing and error handling

## Applications

This system can be applied in various domains:

- **Insurance**: Claim processing and policy queries
- **Legal**: Contract analysis and compliance checking
- **HR**: Policy interpretation and employee queries
- **Healthcare**: Medical guideline consultation
- **Finance**: Regulatory compliance and document analysis

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Google API key is correctly set in the `.env` file
2. **Memory Issues**: For large documents, consider reducing chunk size in config
3. **Slow Processing**: Vector embedding can be slow for large document sets

### Performance Tips

- Upload documents in batches for better performance
- Use specific queries for more accurate results
- Clear vector store periodically to maintain performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration options

---

**Note**: This system requires a Google API key for Gemini LLM access. Make sure to keep your API key secure and never commit it to version control.
