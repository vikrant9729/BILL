#!/usr/bin/env python3
"""
Enhanced Medical Billing System - Deployment Script
==================================================

This script helps deploy and test the enhanced medical billing system.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    requirements_files = [
        'requirements_enhanced.txt',
        'requirements.txt'
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"Installing from {req_file}...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_file], 
                             check=True, capture_output=True, text=True)
                print(f"✅ Successfully installed dependencies from {req_file}")
                break
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing from {req_file}: {e}")
                continue
    else:
        print("❌ No requirements file found")
        return False
    
    return True

def check_optional_dependencies():
    """Check optional dependencies for PDF generation."""
    print("\n🔍 Checking optional dependencies...")
    
    # Check pdfkit
    try:
        import pdfkit
        print("✅ pdfkit available (primary PDF generator)")
        
        # Check wkhtmltopdf
        try:
            subprocess.run(['wkhtmltopdf', '--version'],
                         capture_output=True, check=True)
            print("✅ wkhtmltopdf available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  wkhtmltopdf not found - install for better PDF generation")
            print("   Ubuntu/Debian: sudo apt-get install wkhtmltopdf")
            print("   macOS: brew install wkhtmltopdf")
            print("   Windows: Download from https://wkhtmltopdf.org/downloads.html")
    except ImportError:
        print("⚠️  pdfkit not available")
    
    # Check xhtml2pdf
    try:
        from xhtml2pdf import pisa
        print("✅ xhtml2pdf available (fallback PDF generator)")
    except ImportError:
        print("⚠️  xhtml2pdf not available")
        print("   Install with: pip install xhtml2pdf")

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        'uploads',
        'bills',
        'bills/excel',
        'bills/pdf',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist."""
    print("\n⚙️  Setting up environment...")
    
    if not os.path.exists('.env'):
        env_content = """# Enhanced Medical Billing System Configuration
# AI API Keys (optional)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here_change_this_in_production
FLASK_ENV=development

# File Upload Settings
MAX_CONTENT_LENGTH=16777216

# PDF Generation (optional)
WKHTMLTOPDF_PATH=/usr/local/bin/wkhtmltopdf
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please update with your API keys")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        import pandas as pd
        import flask
        from io import BytesIO
        print("✅ Core imports successful")
        
        # Test enhanced app import
        sys.path.insert(0, '.')
        from app_enhanced import app, DataProcessor, ExcelExporter, PDFExporter
        print("✅ Enhanced app imports successful")
        
        # Test data processor
        print("Testing DataProcessor...")
        test_data = pd.DataFrame({
            'MobileNumber': ['HLM', 'B2B', 'HLM'],
            'CENTER NAME': ['Center A', 'Center B', 'Center A'],
            'TEST TYPE': ['Blood', 'X-Ray', 'Urine']
        })
        
        segmented = DataProcessor.segment_data_by_mobile_number(test_data)
        assert len(segmented['hlm_data']) == 2
        assert len(segmented['b2b_data']) == 1
        print("✅ DataProcessor tests passed")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def start_development_server():
    """Start the development server."""
    print("\n🚀 Starting development server...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        os.system(f"{sys.executable} app_enhanced.py")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main deployment function."""
    print("🏥 Enhanced Medical Billing System - Deployment Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Run tests
    if not run_tests():
        print("❌ Tests failed - please check the installation")
        sys.exit(1)
    
    print("\n✅ Deployment completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Update .env file with your API keys")
    print("2. Install wkhtmltopdf for better PDF generation (optional)")
    print("3. Run: python app_enhanced.py")
    print("4. Open: http://localhost:5000")
    
    # Ask if user wants to start server
    response = input("\n🚀 Start development server now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        start_development_server()

if __name__ == "__main__":
    main()