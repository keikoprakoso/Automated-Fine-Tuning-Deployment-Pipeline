#!/usr/bin/env python3
"""
Test script to test the run_fine_tuning_pipeline method specifically.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pipeline_method():
    """Test the run_fine_tuning_pipeline method."""
    print("Testing run_fine_tuning_pipeline method...")
    try:
        from fine_tune import FineTuningManager
        manager = FineTuningManager()
        
        # Test with a real processed file
        test_file = "data/processed_sample_dataset.jsonl"
        
        if not os.path.exists(test_file):
            print(f"⚠️ Test file not found: {test_file}")
            print("Creating a test file...")
            
            # Create a simple test file
            os.makedirs("data", exist_ok=True)
            with open(test_file, "w") as f:
                f.write('{"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "test response"}]}\n')
        
        print(f"✅ Test file ready: {test_file}")
        
        # Test the method (this should fail with a real API call, but shouldn't fail with the proxies error)
        try:
            result = manager.run_fine_tuning_pipeline(
                training_file_path=test_file,
                model="gpt-3.5-turbo",
                hyperparameters={"n_epochs": 1},
                wait_for_completion=False
            )
            print("✅ Method call succeeded!")
            print(f"Result: {result}")
            return True
        except Exception as e:
            if "proxies" in str(e):
                print(f"❌ Still getting proxies error: {e}")
                return False
            elif "authentication" in str(e) or "invalid" in str(e):
                print(f"✅ Method works! Got expected API error: {e}")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run the test."""
    print("🔍 Testing run_fine_tuning_pipeline Method")
    print("=" * 50)
    
    result = test_pipeline_method()
    
    if result:
        print("\n🎉 Method test passed! The issue might be in the Streamlit context.")
    else:
        print("\n❌ Method test failed. The issue is in the method itself.")

if __name__ == "__main__":
    main() 