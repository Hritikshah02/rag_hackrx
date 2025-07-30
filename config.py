import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the LLM Document Query System"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.5-flash-lite"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector store
    VECTOR_STORE_PATH = "./vector_store"
    COLLECTION_NAME = "documents"
    
    # Search parameters
    DEFAULT_TOP_K = 3  # Focus on only the 3 BEST chunks for better accuracy
    SIMILARITY_THRESHOLD = 0.3  # Higher threshold to ensure quality chunks
    
    # LLM parameters
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.eml']
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        return True
