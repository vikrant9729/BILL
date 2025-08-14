#!/usr/bin/env python3
"""
Test script for Medical Billing Automation System
"""

import os
import sys
import tempfile
import pandas as pd
from datetime import datetime

def create_test_data():
    """Create test Excel data for testing the billing app."""
    
    # Create test main data
    main_data = {
        'PatientVisitCode': [1001, 1002, 1003, 1004],
        'RegisteredDate': ['2024-01-15', '2024-01-15', '2024-01-16', '2024-01-16'],
        'PatientName': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'Age': [35, 28, 45, 32],
        'AgeUnit': ['Years', 'Years', 'Years', 'Years'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'MobileNumber': ['B2B', 'B2B', 'HLM', 'HLM'],
        'TEST NAME': ['Blood Test', 'X-Ray', 'MRI Scan', 'ECG'],
        'CODE NO': ['BT001', 'XR001', 'MRI001', 'ECG001'],
        'CENTER NAME': ['Test Hospital', 'Test Hospital', 'Test Clinic', 'Test Clinic'],
        'Modality': ['Lab', 'Radiology', 'Radiology', 'Cardiology'],
        'MRP': [500, 800, 1500, 300],
        'CentreTestRate': [400, 600, 1200, 250],
        'TEST TYPE': ['Laboratory', 'Radiology', 'Radiology', 'Cardiology']
    }
    
    # Create test supporting data
    supporting_data = {
        'CENTER NAME': ['Test Clinic'],
        'TEST TYPE': ['Radiology'],
        'SHARE_PERCENTAGE': [60.0]
    }
    
    return main_data, supporting_data

def test_billing_app():
    """Test the medical billing app with sample data."""
    
    try:
        # Import the billing app
        from medical_billing_app import MedicalBillingProcessor
        
        print("🧪 Testing Medical Billing Automation System")
        print("=" * 50)
        
        # Create test data
        main_data, supporting_data = create_test_data()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as main_file:
            main_df = pd.DataFrame(main_data)
            main_df.to_excel(main_file.name, index=False)
            main_file_path = main_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as supporting_file:
            supporting_df = pd.DataFrame(supporting_data)
            supporting_df.to_excel(supporting_file.name, index=False)
            supporting_file_path = supporting_file.name
        
        print(f"✅ Created test files:")
        print(f"   Main data: {main_file_path}")
        print(f"   Supporting data: {supporting_file_path}")
        
        # Test the processor
        config = {
            'invoice_date': datetime.now(),
            'period_start': datetime(2024, 1, 1),
            'period_end': datetime(2024, 1, 31),
            'invoice_sequence_start': 1
        }
        
        processor = MedicalBillingProcessor(config)
        
        print("\n🔄 Testing billing data processing...")
        bills_data = processor.process_billing_data(main_file_path, supporting_file_path)
        
        print(f"✅ Successfully processed {len(bills_data)} center bills")
        
        for bill in bills_data:
            print(f"   📋 Center: {bill['center_name']}")
            print(f"   🏷️  Type: {bill['center_type']}")
            print(f"   💰 Total Net: ₹{bill['total_net']:.2f}")
            print(f"   📊 Tests: {len(bill['test_summary'])} types")
            print()
        
        # Test bill generation
        print("🔄 Testing bill generation...")
        result = processor.generate_all_bills(main_file_path, supporting_file_path)
        
        print(f"✅ Successfully generated bills:")
        print(f"   📊 Excel files: {len(result['excel_files'])}")
        print(f"   📄 PDF files: {len(result['pdf_files'])}")
        
        # Clean up temporary files
        os.unlink(main_file_path)
        os.unlink(supporting_file_path)
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_billing_app()
    sys.exit(0 if success else 1) 