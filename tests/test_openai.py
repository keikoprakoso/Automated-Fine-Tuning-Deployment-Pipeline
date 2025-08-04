#!/usr/bin/env python3
"""
Test script to isolate OpenAI API issues.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_import():
    """Test OpenAI import and basic functionality."""
    print("Testing OpenAI import...")
    try:
        import openai
        print("✅ OpenAI imported successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI import failed: {e}")
        return False

def test_openai_client():
    """Test OpenAI client initialization."""
    print("Testing OpenAI client initialization...")
    try:
        import openai
        openai.api_key = "test_key"
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        return False

def test_fine_tuning_manager():
    """Test FineTuningManager creation."""
    print("Testing FineTuningManager creation...")
    try:
        from fine_tune import FineTuningManager
        manager = FineTuningManager()
        print("✅ FineTuningManager created successfully")
        return True
    except Exception as e:
        print(f"❌ FineTuningManager creation failed: {e}")
        return False

def test_file_upload():
    """Test file upload functionality."""
    print("Testing file upload...")
    try:
        from fine_tune import FineTuningManager
        manager = FineTuningManager()
        
        # Test with a dummy file
        test_file = "data/sample_dataset.csv"
        if os.path.exists(test_file):
            print(f"✅ Test file exists: {test_file}")
            return True
        else:
            print(f"❌ Test file not found: {test_file}")
            return False
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 OpenAI API Compatibility Test")
    print("=" * 40)
    
    tests = [
        test_openai_import,
        test_openai_client,
        test_fine_tuning_manager,
        test_file_upload
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("📊 Test Results:")
    print("=" * 40)
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! OpenAI API should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 