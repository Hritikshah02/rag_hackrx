import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing environment...")
google_api_key = os.getenv('GOOGLE_API_KEY')
print(f"Google API Key present: {'Yes' if google_api_key else 'No'}")

if google_api_key:
    print("Environment looks good. Testing the chunker...")
    try:
        from simple_semantic_chunker import SimpleSemanticChunker
        
        # Test payload - just one question for quick testing
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
            ]
        }
        
        # Initialize and run
        chunker = SimpleSemanticChunker()
        results = chunker.process_payload(test_payload)
        
        print("SUCCESS: Script completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("ERROR: GOOGLE_API_KEY not found in environment variables")
    print("Please check your .env file")
