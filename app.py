import streamlit as st
import os
import tempfile
import json
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

# Import custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_reasoner import LLMReasoner
from config import Config

# Page configuration
st.set_page_config(
    page_title="LLM Document Query System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DocumentQueryApp:
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_reasoner = LLMReasoner()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = []
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">🤖 LLM Document Query System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Welcome to the LLM-Powered Document Query System!</strong><br>
            This system uses advanced AI to understand your queries and retrieve relevant information from uploaded documents.
            Upload your documents (PDFs, Word files, emails) and ask questions in natural language.
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration and status"""
        with st.sidebar:
            st.header("📊 System Status")
            
            # Document status
            st.subheader("📄 Documents")
            if st.session_state.documents_processed:
                st.success(f"✅ {len(st.session_state.documents_processed)} documents processed")
                for doc in st.session_state.documents_processed:
                    st.write(f"• {doc['name']} ({doc['chunks']} chunks)")
            else:
                st.info("No documents uploaded yet")
            
            # Vector store status
            st.subheader("🗄️ Vector Store")
            if st.session_state.vector_store_ready:
                st.success("✅ Vector store ready")
            else:
                st.info("Vector store not initialized")
            
            # Query history
            st.subheader("📝 Query History")
            if st.session_state.query_history:
                st.write(f"Total queries: {len(st.session_state.query_history)}")
                if st.button("Clear History"):
                    st.session_state.query_history = []
                    st.rerun()
            else:
                st.info("No queries yet")
            
            # Configuration
            st.subheader("⚙️ Configuration")
            st.write(f"Embedding Model: {self.config.EMBEDDING_MODEL}")
            st.write(f"LLM Model: {self.config.LLM_MODEL}")
            st.write(f"Chunk Size: {self.config.CHUNK_SIZE}")
    
    def render_document_upload(self):
        """Render document upload section"""
        st.markdown('<div class="section-header">📁 Document Upload</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'doc', 'txt', 'eml'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word documents, text files, and email files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and create vector store"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            all_chunks = []
            processed_docs = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process document
                    chunks = self.doc_processor.process_document(tmp_file_path, uploaded_file.name)
                    all_chunks.extend(chunks)
                    
                    processed_docs.append({
                        'name': uploaded_file.name,
                        'chunks': len(chunks),
                        'processed_at': datetime.now().isoformat()
                    })
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            # Create vector store
            status_text.text("Creating vector embeddings...")
            progress_bar.progress(0.8)
            
            self.vector_store.create_vector_store(all_chunks)
            
            # Update session state
            st.session_state.documents_processed.extend(processed_docs)
            st.session_state.vector_store_ready = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ All documents processed successfully!")
            
            st.markdown(f"""
            <div class="success-box">
                <strong>Success!</strong> Processed {len(uploaded_files)} documents with {len(all_chunks)} total chunks.
                You can now start querying your documents.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>Error:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    def render_query_interface(self):
        """Render the query interface"""
        st.markdown('<div class="section-header">🔍 Query Interface</div>', unsafe_allow_html=True)
        
        if not st.session_state.vector_store_ready:
            st.warning("Please upload and process documents first before querying.")
            return
        
        # Sample queries
        st.markdown("**Sample Queries:**")
        sample_queries = [
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "What is the coverage for dental procedures?",
            "Are pre-existing conditions covered?",
            "What is the waiting period for surgery claims?"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(sample_queries):
            with cols[i % 2]:
                if st.button(f"📝 {query[:30]}...", key=f"sample_{i}"):
                    st.session_state.current_query = query
        
        # Query input
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.get('current_query', ''),
            height=100,
            placeholder="Ask anything about your uploaded documents..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Query", type="primary", disabled=not query.strip()):
                self.process_query(query.strip())
        
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.current_query = ""
                st.rerun()
    
    def process_query(self, query: str):
        """Process user query and generate response"""
        with st.spinner("Processing your query..."):
            try:
                # Perform semantic search
                search_results = self.vector_store.semantic_search(query, top_k=5)
                
                # Generate response using LLM
                response = self.llm_reasoner.generate_response(query, search_results)
                
                # Store in history
                query_record = {
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'search_results_count': len(search_results)
                }
                st.session_state.query_history.append(query_record)
                
                # Display results
                self.display_query_results(query, response, search_results)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    def display_query_results(self, query: str, response: Dict[str, Any], search_results: List[Dict]):
        """Display query results"""
        st.markdown('<div class="section-header">📋 Query Results</div>', unsafe_allow_html=True)
        
        # Display structured response
        st.subheader("🎯 AI Response")
        
        # Create columns for structured display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Decision", response.get('decision', 'N/A'))
        
        with col2:
            amount = response.get('amount', 'N/A')
            st.metric("Amount", amount if amount != 'N/A' else 'Not applicable')
        
        with col3:
            confidence = response.get('confidence', 'N/A')
            st.metric("Confidence", f"{confidence}%" if confidence != 'N/A' else 'N/A')
        
        # Justification
        st.subheader("📝 Justification")
        st.write(response.get('justification', 'No justification provided'))
        
        # Referenced clauses
        if 'referenced_clauses' in response and response['referenced_clauses']:
            st.subheader("📄 Referenced Clauses")
            for i, clause in enumerate(response['referenced_clauses'], 1):
                with st.expander(f"Clause {i}: {clause.get('source', 'Unknown source')}"):
                    st.write(clause.get('content', 'No content available'))
                    if 'relevance_score' in clause:
                        st.caption(f"Relevance Score: {clause['relevance_score']:.3f}")
        
        # Raw JSON response
        with st.expander("🔧 Raw JSON Response"):
            st.json(response)
        
        # Search results
        with st.expander(f"🔍 Search Results ({len(search_results)} chunks)"):
            for i, result in enumerate(search_results, 1):
                st.write(f"**Result {i}** (Score: {result.get('score', 'N/A'):.3f})")
                st.write(f"Source: {result.get('source', 'Unknown')}")
                st.write(result.get('content', 'No content')[:500] + "...")
                st.divider()
    
    def render_query_history(self):
        """Render query history"""
        if not st.session_state.query_history:
            return
        
        st.markdown('<div class="section-header">📚 Query History</div>', unsafe_allow_html=True)
        
        for i, record in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {record['query'][:50]}..."):
                st.write(f"**Query:** {record['query']}")
                st.write(f"**Decision:** {record['response'].get('decision', 'N/A')}")
                st.write(f"**Amount:** {record['response'].get('amount', 'N/A')}")
                st.write(f"**Timestamp:** {record['timestamp']}")
                if st.button(f"View Full Response", key=f"history_{i}"):
                    st.json(record['response'])
    
    def run(self):
        """Main application runner"""
        self.initialize_session_state()
        self.render_header()
        self.render_sidebar()
        
        # Main content
        tab1, tab2 = st.tabs(["📁 Upload & Query", "📚 History"])
        
        with tab1:
            self.render_document_upload()
            st.divider()
            self.render_query_interface()
        
        with tab2:
            self.render_query_history()

if __name__ == "__main__":
    app = DocumentQueryApp()
    app.run()
