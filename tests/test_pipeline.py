"""
Test Script for LLM Fine-Tuning Pipeline

This script tests the complete pipeline workflow from data preprocessing
to fine-tuning and inference. Use this to validate your setup.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import DataPreprocessor
from fine_tune import FineTuningManager
from inference import InferenceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_preprocessing():
    """Test data preprocessing functionality."""
    logger.info("Testing data preprocessing...")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Test with sample dataset
        input_path = "data/sample_dataset.csv"
        output_path = "data/test_processed_data.jsonl"
        
        if not os.path.exists(input_path):
            logger.error(f"Sample dataset not found: {input_path}")
            return False
        
        # Process dataset
        summary = preprocessor.process_dataset(input_path, output_path)
        
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Summary: {summary}")
        
        # Verify output file exists
        if os.path.exists(output_path):
            logger.info(f"Output file created: {output_path}")
            return True
        else:
            logger.error("Output file not created")
            return False
            
    except Exception as e:
        logger.error(f"Preprocessing test failed: {str(e)}")
        return False


def test_fine_tuning_manager():
    """Test fine-tuning manager initialization and basic functionality."""
    logger.info("Testing fine-tuning manager...")
    
    try:
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            logger.warning("OpenAI API key not set. Skipping fine-tuning tests.")
            return True
        
        # Initialize manager
        manager = FineTuningManager()
        
        # Test database initialization
        if os.path.exists("jobs.db"):
            logger.info("Database initialized successfully")
        
        # Test listing jobs (should be empty initially)
        jobs = manager.list_jobs()
        logger.info(f"Found {len(jobs)} existing jobs")
        
        # Test getting active models (should be empty initially)
        models = manager.get_active_models()
        logger.info(f"Found {len(models)} active models")
        
        return True
        
    except Exception as e:
        logger.error(f"Fine-tuning manager test failed: {str(e)}")
        return False


def test_inference_manager():
    """Test inference manager initialization and basic functionality."""
    logger.info("Testing inference manager...")
    
    try:
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            logger.warning("OpenAI API key not set. Skipping inference tests.")
            return True
        
        # Initialize manager
        inference = InferenceManager()
        
        # Test log file initialization
        if os.path.exists("logs/inference_logs.csv"):
            logger.info("Log file initialized successfully")
        
        # Test usage summary (should be empty initially)
        summary = inference.get_usage_summary(days=30)
        logger.info(f"Usage summary: {summary}")
        
        # Test recent logs (should be empty initially)
        logs = inference.get_recent_logs(limit=5)
        logger.info(f"Recent logs: {len(logs)} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference manager test failed: {str(e)}")
        return False


def test_file_structure():
    """Test that all required files and directories exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        "preprocess.py",
        "fine_tune.py", 
        "inference.py",
        "app.py",
        "dashboard.py",
        "requirements.txt",
        "env.example",
        "README.md"
    ]
    
    required_dirs = [
        "data",
        "logs"
    ]
    
    # Check files
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    # Check directories
    missing_dirs = []
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False
    
    logger.info("All required files and directories found")
    return True


def test_environment():
    """Test environment configuration."""
    logger.info("Testing environment configuration...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'openai',
        'streamlit',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All required packages are installed")
    return True


def main():
    """Run all tests."""
    logger.info("Starting pipeline tests...")
    
    tests = [
        ("Environment", test_environment),
        ("File Structure", test_file_structure),
        ("Preprocessing", test_preprocessing),
        ("Fine-tuning Manager", test_fine_tuning_manager),
        ("Inference Manager", test_inference_manager)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your pipeline is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Set your OpenAI API key in .env file")
        logger.info("2. Run: python app.py (for API server)")
        logger.info("3. Run: streamlit run dashboard.py (for dashboard)")
    else:
        logger.error("⚠️  Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 