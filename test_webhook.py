"""
Test script for the HackRX webhook API
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Test payload (from HackRX requirements)
test_payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_webhook():
    """Test the main webhook endpoint"""
    print("\n🚀 Testing webhook endpoint...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📤 Sending request to {BASE_URL}/hackrx/run")
        print(f"📋 Questions: {len(test_payload['questions'])}")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=test_payload,
            timeout=60  # 60 second timeout
        )
        
        print(f"📥 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Got {len(result['answers'])} answers")
            
            # Print first few answers
            for i, answer in enumerate(result['answers'][:3], 1):
                print(f"\n📝 Answer {i}: {answer[:100]}...")
            
            if len(result['answers']) > 3:
                print(f"\n... and {len(result['answers']) - 3} more answers")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook test failed: {e}")
        return False

def test_auth():
    """Test authentication"""
    print("\n🔐 Testing authentication...")
    
    # Test without token
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_payload)
        print(f"No token - Status: {response.status_code} (should be 403)")
    except Exception as e:
        print(f"No token test error: {e}")
    
    # Test with wrong token
    try:
        headers = {"Authorization": "Bearer wrong_token"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_payload)
        print(f"Wrong token - Status: {response.status_code} (should be 401)")
    except Exception as e:
        print(f"Wrong token test error: {e}")

if __name__ == "__main__":
    print("🧪 HackRX Webhook API Test")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ Health check failed. Make sure the API is running.")
        exit(1)
    
    # Test authentication
    test_auth()
    
    # Test main functionality
    success = test_webhook()
    
    if success:
        print("\n✅ All tests passed! Webhook is ready for submission.")
    else:
        print("\n❌ Tests failed. Check the API logs for details.")
