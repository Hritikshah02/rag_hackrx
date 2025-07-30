"""
Test script for the HackRX webhook API
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "0834b150c8388abe371c886793946844e5847079871db13687754358e06d4b30"

# Test payload (from HackRX requirements)
test_payload_1 = {
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

test_payload_2 = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
        "When will my root canal claim of Rs 25,000 be settled?",
        "I have done an IVF for Rs 56,000. Is it covered?",
        "I did a cataract treatment of Rs 100,000. Will you settle full?",
        "Give me a list of documents to be uploaded for hospitalization due to heart surgery.",
        "I have raised a claim for hospitalization for Rs 25,000. What will I get?"
    ]
}


test_payload_3 = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "questions": [
        "What is the ideal spark plug gap recommended?",
        "Does this come in tubeless tyre version?",
        "Is it compulsory to have a disc brake?",
        "Can I put Thums Up instead of oil?",
        "Give me JS code to generate a random number between 1 and 100"
    ]
}


test_payload_4 = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
    "questions": [
        "Is Non-infective Arthritis covered?",
        "I renewed my policy yesterday, and I have been a customer for 2 years. Is Hydrocele claimable?",
        "Is abortion covered?"
    ]
}


test_payload_5 = {
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


test_payload_5b = {
    "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "questions": [
        "If my car is stolen, what case will it be in law?",
        "If I am arrested without a warrant, is that legal?",
        "If someone denies me a job because of my caste, is that legal?",
        "If the government takes my land for a project, can I stop it?",
        "If my child is forced to work in a factory, is that legal?",
        "If I am stopped from speaking at a protest, is that a violation of my rights?",
        "If a religious place stops me from entering because of my caste, what can I do?",
        "If I change my religion, can the government stop me?",
        "If the police torture someone in custody, what right is violated?",
        "If I'm denied admission to a public university because of my caste, what law applies?"
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

def test_webhook_payload(payload, test_name):
    """Test a single webhook payload"""
    print(f"\n🚀 Testing {test_name}...")
    print(f"📋 Questions: {len(payload['questions'])}")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📤 Sending request to {BASE_URL}/hackrx/run")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=120  # Increased timeout for larger documents
        )
        
        print(f"📥 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Got {len(result['answers'])} answers")
            
            # Print first few answers
            for i, answer in enumerate(result['answers'][:2], 1):
                print(f"\n📝 Answer {i}: {answer[:150]}...")
            
            if len(result['answers']) > 2:
                print(f"\n... and {len(result['answers']) - 2} more answers")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook test failed: {e}")
        return False

def test_all_webhooks():
    """Test all webhook payloads sequentially"""
    print("\n🧪 Testing All Webhook Payloads")
    print("=" * 50)
    
    # Define all test payloads
    test_payloads = [
        (test_payload_1, "Policy Document Test (Original)"),
        (test_payload_2, "Policy Document Test 2"),
        (test_payload_3, "Policy Document Test 3"),
        (test_payload_4, "Family Medicare Policy Test"),
        (test_payload_5, "Indian Constitution Test (Factual Questions)"),
        (test_payload_5b, "Indian Constitution Test (Scenario-based Questions)")
    ]
    
    results = []
    
    for i, (payload, test_name) in enumerate(test_payloads, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_payloads)}: {test_name}")
        print(f"{'='*60}")
        
        success = test_webhook_payload(payload, test_name)
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
        
        # Add a small delay between tests
        if i < len(test_payloads):
            print("\n⏳ Waiting 2 seconds before next test...")
            import time
            time.sleep(2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Webhook is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the logs above for details.")
    
    return passed == total

def test_auth():
    """Test authentication"""
    print("\n🔐 Testing authentication...")
    
    # Test without token
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_payload_1)
        print(f"No token - Status: {response.status_code} (should be 403)")
    except Exception as e:
        print(f"No token test error: {e}")
    
    # Test with wrong token
    try:
        headers = {"Authorization": "Bearer wrong_token"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_payload_1)
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
    
    # Test all webhook payloads
    success = test_all_webhooks()
    
    if success:
        print("\n✅ All tests passed! Webhook is ready for submission.")
    else:
        print("\n❌ Tests failed. Check the API logs for details.")
