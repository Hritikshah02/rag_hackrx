import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing improved chunker...")
google_api_key = os.getenv('GOOGLE_API_KEY')
print(f"Google API Key present: {'Yes' if google_api_key else 'No'}")

if google_api_key:
    print("Environment looks good. Testing the improved chunker...")
    try:
        from improved_semantic_chunker import ImprovedSemanticChunker
        
        # Test with just 2 questions first
        test_payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "questions": [
        "What is the official name of India according to Article 1?",
        "Which Article guarantees equality before the law and equal protection of the laws?",
        "What is abolished by Article 17 of the Constitution?",
        "What are the key ideals mentioned in the Preamble of the Indian Constitution?",
        "Under which Article can Parliament alter the boundaries of states?",
        "According to Article 24, children below what age are prohibited from working?",
        "What is the significance of Article 21 in the Indian Constitution?",
        "Article 15 prohibits discrimination on certain grounds. What are they?",
        "Which Article allows Parliament to regulate the right to form associations?",
        "What restrictions can the State impose on the right to freedom of speech?"
    ]
}
    #     test_payload = {
    #         "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    #         "questions": [
    #         "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    #         "What is the waiting period for pre-existing diseases (PED) to be covered?",
    #         "Does this policy cover maternity expenses, and what are the conditions?",
    #         "What is the waiting period for cataract surgery?",
    #         "Are the medical expenses for an organ donor covered under this policy?",
    #         "What is the No Claim Discount (NCD) offered in this policy?",
    #         "Is there a benefit for preventive health check-ups?",
    #         "How does the policy define a 'Hospital'?",
    #         "What is the extent of coverage for AYUSH treatments?",
    #         "Are there any sub-limits on room rent and ICU charges for Plan A?"
    #     ]
    # }
        
        # Initialize and run
        chunker = ImprovedSemanticChunker()
        results = chunker.process_payload(test_payload)
        
        # Display all final answers at the end
        print("\n" + "="*80)
        print("SUMMARY OF ALL ANSWERS")
        print("="*80)
        
        # Ensure output is flushed
        import sys
        sys.stdout.flush()
        
        for i, result in enumerate(results['results'], 1):
            question = result['question']
            answer = result['answer']
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            print("-"*80)
        
        # Get the log directory from the chunker
        log_dir = os.path.join(os.getcwd(), "logs")
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log") or f.endswith(".json")]
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        
        print("\nLOG FILES CREATED:")
        for i, log_file in enumerate(log_files[:4]):
            print(f"- {log_file}")
            
        print(f"\nLog files are saved in: {log_dir}")
        print("\nSUCCESS: Improved script completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("ERROR: GOOGLE_API_KEY not found in environment variables")
    print("Please check your .env file")
