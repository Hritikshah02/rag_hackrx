"""
Windows-specific test script for the HackRX webhook API
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test data
    headers = {
        "Authorization": "Bearer 0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30",
        "Content-Type": "application/json"
    }
    
    test_payload = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is the grace period?",
            "What are the waiting periods for different treatments?"
        ]
    }
    
    print("🚀 Testing HackRX Webhook API...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Main endpoint
    print("\n2. Testing main /hackrx/run endpoint...")
    try:
        response = requests.post(
            f"{base_url}/hackrx/run",
            headers=headers,
            json=test_payload,
            timeout=60
        )
        
        print(f"✅ API Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Got {len(result['answers'])} answers:")
            print("-" * 30)
            
            for i, (question, answer) in enumerate(zip(test_payload['questions'], result['answers']), 1):
                print(f"\nQ{i}: {question}")
                print(f"A{i}: {answer}")
                print("-" * 30)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_api()
