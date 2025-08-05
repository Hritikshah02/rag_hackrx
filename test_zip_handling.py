#!/usr/bin/env python3
"""
Test script to verify ZIP file handling in the RAG system.
"""

import requests
import json

# Test configuration
API_URL = "http://localhost:8000/hackrx/run"
BEARER_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Test payloads with ZIP files
test_payloads = [
    {
        "name": "ZIP File Test 1",
        "payload": {
            "documents": "https://example.com/test_document.zip",
            "questions": [
                "What is the content of this document?",
                "Can you summarize the main points?"
            ]
        }
    },
    {
        "name": "ZIP File Test 2", 
        "payload": {
            "documents": "https://hackrx.blob.core.windows.net/assets/sample_files.ZIP",
            "questions": [
                "What are the key features mentioned?",
                "What is the pricing information?",
                "How do I contact support?"
            ]
        }
    },
    {
        "name": "Valid PDF Test (Control)",
        "payload": {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the policy coverage?"
            ]
        }
    }
]

def test_zip_handling():
    """Test ZIP file handling functionality."""
    print("🧪 Testing ZIP File Handling in RAG System")
    print("=" * 60)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    for i, test_case in enumerate(test_payloads, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"📄 Document: {test_case['payload']['documents']}")
        print(f"❓ Questions: {len(test_case['payload']['questions'])}")
        
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=test_case['payload'],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get('answers', [])
                
                print(f"✅ Status: Success ({response.status_code})")
                print(f"📝 Answers received: {len(answers)}")
                
                # Check if ZIP file error message is returned
                zip_error_message = "ZIP file is not allowed, please upload a valid file"
                zip_detected = any(zip_error_message in answer for answer in answers)
                
                if zip_detected:
                    print(f"🚫 ZIP File Detected: All answers contain error message")
                    print(f"📋 Sample Answer: {answers[0]}")
                else:
                    print(f"📋 Sample Answer: {answers[0][:100]}...")
                    
            else:
                print(f"❌ Status: Failed ({response.status_code})")
                print(f"📋 Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Failed: {str(e)}")
        
        print("-" * 40)
    
    print("\n✅ ZIP File Handling Test Complete!")

if __name__ == "__main__":
    print("🚀 Starting ZIP File Handling Test...")
    print("⚠️  Make sure the API server is running on localhost:8000")
    print()
    
    try:
        test_zip_handling()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
