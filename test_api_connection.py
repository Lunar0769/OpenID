"""
Quick test to verify API connectivity and response times
"""
import requests
import time

BASE_URL = "https://jumeirah-ai.testyourapp.online"

print("Testing API connectivity...\n")

# Test 1: Health check
print("1. Testing /health endpoint...")
try:
    start = time.time()
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    elapsed = time.time() - start
    print(f"   ✓ Status: {response.status_code}")
    print(f"   ✓ Response time: {elapsed:.2f}s")
    print(f"   ✓ Response: {response.text}\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 2: Check if extract_passport endpoint exists (without auth)
print("2. Testing /extract_passport endpoint (expect 401 or 422)...")
try:
    start = time.time()
    response = requests.post(f"{BASE_URL}/extract_passport", timeout=10)
    elapsed = time.time() - start
    print(f"   ✓ Status: {response.status_code}")
    print(f"   ✓ Response time: {elapsed:.2f}s")
    print(f"   ✓ Response: {response.text[:200]}\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 3: Try with a test image and API key
print("3. Testing /extract_passport with image (using 'your_key' as API key)...")
try:
    # Use one of the captured images
    import os
    captures_dir = "captures"
    if os.path.exists(captures_dir):
        images = [f for f in os.listdir(captures_dir) if f.endswith('.jpg')]
        if images:
            test_image = os.path.join(captures_dir, images[-1])
            print(f"   Using image: {test_image}")
            
            start = time.time()
            with open(test_image, 'rb') as f:
                files = {'front_file': f}
                headers = {'Authorization': 'Bearer your_key'}
                response = requests.post(
                    f"{BASE_URL}/extract_passport",
                    files=files,
                    headers=headers,
                    timeout=120
                )
            elapsed = time.time() - start
            
            print(f"   ✓ Status: {response.status_code}")
            print(f"   ✓ Response time: {elapsed:.2f}s")
            print(f"   ✓ Response: {response.text[:500]}\n")
        else:
            print("   ⚠ No images found in captures directory\n")
    else:
        print("   ⚠ Captures directory not found\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

print("\nTest complete!")
