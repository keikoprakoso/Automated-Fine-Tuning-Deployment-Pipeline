#!/usr/bin/env python3
"""
Startup Script for LLM Fine-Tuning Pipeline

This script provides an easy way to start both the FastAPI server
and Streamlit dashboard with proper configuration.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating from template...")
        if os.path.exists('env.example'):
            subprocess.run(['cp', 'env.example', '.env'])
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your OpenAI API key")
        else:
            print("❌ env.example not found")
            return False
    
    # Check if required directories exist
    for directory in ['data', 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory} directory")
    
    return True

def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting FastAPI server...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ FastAPI server started successfully")
            print("📡 API available at: http://localhost:8000")
            print("📚 API docs at: http://localhost:8000/docs")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start API server: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        return None

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("📊 Starting Streamlit dashboard...")
    try:
        # Start the dashboard
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for dashboard to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Streamlit dashboard started successfully")
            print("📈 Dashboard available at: http://localhost:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start dashboard: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {str(e)}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Shutting down pipeline...")
    sys.exit(0)

def main():
    """Main startup function."""
    print("🤖 LLM Fine-Tuning Pipeline Startup")
    print("=" * 50)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Start services
    api_process = None
    dashboard_process = None
    
    try:
        # Start API server
        api_process = start_api_server()
        if not api_process:
            print("❌ Failed to start API server. Exiting.")
            sys.exit(1)
        
        # Start dashboard
        dashboard_process = start_dashboard()
        if not dashboard_process:
            print("⚠️  Failed to start dashboard. API server is still running.")
        
        print("\n🎉 Pipeline started successfully!")
        print("=" * 50)
        print("📡 API Server: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("📈 Dashboard: http://localhost:8501")
        print("=" * 50)
        print("Press Ctrl+C to stop all services")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process and api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
                
            if dashboard_process and dashboard_process.poll() is not None:
                print("❌ Dashboard stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
    
    finally:
        # Cleanup
        print("🧹 Cleaning up processes...")
        
        if api_process:
            try:
                api_process.terminate()
                api_process.wait(timeout=5)
                print("✅ API server stopped")
            except subprocess.TimeoutExpired:
                api_process.kill()
                print("⚠️  Force killed API server")
        
        if dashboard_process:
            try:
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
                print("✅ Dashboard stopped")
            except subprocess.TimeoutExpired:
                dashboard_process.kill()
                print("⚠️  Force killed dashboard")
        
        print("👋 Pipeline shutdown complete")

if __name__ == "__main__":
    main() 