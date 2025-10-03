#!/usr/bin/env python3
"""
Test script to verify the KD-Pruning backend functionality
"""

import requests
import json
import time

def test_backend():
    """Test the backend endpoints"""
    base_url = "http://127.0.0.1:5001"
    
    print("Testing KD-Pruning Backend...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/test", timeout=5)
        if response.status_code == 200:
            print("OK Server is running")
        else:
            print("FAIL Server not responding properly")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL Cannot connect to server: {e}")
        return False
    
    # Test 2: Test model loading
    models_to_test = ["distillBert", "T5-small", "MobileNetV2", "ResNet-18"]

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            response = requests.post(f"{base_url}/test_model",
                                    json={"model_name": model_name},
                                    timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
                if result.get("success"):
                    print(f"OK {model_name} loads successfully")
                else:
                    print(f"FAIL {model_name} failed to load: {result.get('error', 'Unknown error')}")
            else:
                print(f"FAIL {model_name} test failed with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"FAIL {model_name} test failed: {e}")
    
    # Test 3: Test metrics calculation
    print("\nTesting metrics calculation...")
    try:
        response = requests.get(f"{base_url}/test_metrics", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✓ Metrics calculation works")
            else:
                print(f"✗ Metrics calculation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Metrics test failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Metrics test failed: {e}")
    
    # Test 4: Test MATLAB metrics endpoint
    print("\nTesting MATLAB metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/matlab_metrics", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if not result.get("success") and "not trained" in result.get("error", ""):
                print("✓ MATLAB metrics endpoint works (no model trained yet)")
            else:
                print("✓ MATLAB metrics endpoint works")
        else:
            print(f"✗ MATLAB metrics test failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ MATLAB metrics test failed: {e}")
    
    print("\nBackend testing completed!")
    return True

if __name__ == "__main__":
    test_backend()
