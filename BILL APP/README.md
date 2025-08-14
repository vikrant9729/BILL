# 🏥 Enhanced Medical Billing System - Professional Edition

A comprehensive, AI-powered medical billing automation system with advanced data filtering, dynamic bill generation, and multi-format export capabilities. This enhanced version provides intelligent data segmentation, two-stage interactive workflows, and robust export systems.

## ✨ Key Features

### 🎯 Advanced Data Processing
- **Intelligent Data Segmentation**: Automatic routing based on MobileNumber column values ("HLM" or "B2B")
- **Dynamic Center Detection**: Real-time extraction of centers from uploaded data
- **TEST TYPE Recognition**: Automatic detection of unique test types per center
- **Memory-Efficient Processing**: In-memory data handling for large datasets

### 🔄 Two-Stage Interactive Workflow for HLM
1. **Stage 1: Center Selection**
   - Display all available HLM centers from uploaded data
   - Dynamic center list generation from CENTER NAME column
   - User-friendly center selection interface

2. **Stage 2: Sharing Configuration**
   - Dynamic TEST TYPE detection for selected center
   - Custom sharing percentage configuration per test type
   - Live calculation preview with real-time updates
   - Flexible default percentage with test-specific overrides

### 📊 Enhanced Export System
- **Excel Generation**: pandas.to_excel() with in-memory processing
- **Dual PDF System**: Primary pdfkit with xhtml2pdf fallback
- **Multi-Format Downloads**: HTML, Excel, PDF with bulk ZIP downloads
- **Professional Formatting**: Statutory compliance and company branding

### 🤖 AI-Powered Intelligence
- **Error Handling**: Contextual AI suggestions for troubleshooting
- **Data Validation**: Comprehensive checks with intelligent feedback
- **User Assistance**: Interactive AI chat for guidance

## 🚀 Quick Start Guide

### Prerequisites
```bash
Python 3.8+
pip install -r requirements_enhanced.txt
```

### Installation & Setup

1. **Clone or Download the Project**
   ```bash
   # Navigate to your project directory
   cd BILL APP
   ```

2. **Install Dependencies**
   ```bash
   # Install enhanced dependencies
   pip install -r requirements_enhanced.txt
   
   # Or install basic requirements
   pip install -r requirements.txt
   ```

3. **Optional: Install PDF Generation Tools**
   ```bash
   # For pdfkit (primary PDF generator)
   pip install pdfkit
   # Install wkhtmltopdf system dependency
   
   # For xhtml2pdf (fallback PDF generator)
   pip install xhtml2pdf
   ```

4. **Set Up Environment Variables (Optional)**
   Create a `.env` file:
   ```env
   # AI API Keys
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Flask Configuration
   FLASK_SECRET_KEY=your_secret_key_here
   FLASK_ENV=development
   ```

### Running the System

#### Option 1: Enhanced Version (Recommended)
```bash
python app_enhanced.py
```

#### Option 2: Standard Version
```bash
python app.py
```

#### Option 3: Medical Billing App
```bash
python medical_billing_app.py
```

#### Option 4: Automated Deployment
```bash
python deploy.py
```

**Access the system at:** `http://localhost:5000`

## 📋 Excel File Format Requirements

### Required Columns
| Column Name | Description | Example |
|-------------|-------------|---------|
| `CENTER NAME` | Diagnostic center name | "AMANDEEP_HOSPITAL" |
| `RegisteredDate` | Patient registration date | "2024-01-15" |
| `PatientVisitCode` | Unique visit identifier | "12345" |
| `PatientName` | Patient's full name | "John Doe" |
| `TEST NAME` | Medical test name | "Blood Test" |
| `MRP` | Maximum Retail Price | 1500.00 |
| `CentreTestRate` | Center's test rate | 1200.00 |
| `MobileNumber` | **Data Segmentation Key** | "HLM" or "B2B" |
| `TEST TYPE` | Test category for dynamic processing | "Laboratory" |

### Data Segmentation Logic
- **MobileNumber = "HLM"**: Routes to HLM processing workflow
- **MobileNumber = "B2B"**: Routes to B2B processing workflow
- **Other values**: Default to B2B processing

## 🎯 Enhanced Workflows

### HLM Processing Workflow
```
1. Upload Excel File
   ↓
2. System detects HLM records (MobileNumber = "HLM")
   ↓
3. Stage 1: Select HLM Center
   - Dynamic center list from data
   - User selects specific center
   ↓
4. Stage 2: Configure Sharing
   - System detects TEST TYPEs for selected center
   - User sets sharing percentages per test type
   - Live calculation preview
   ↓
5. Generate HLM Bills
   - Formula: Sharing = MRP × Percentage
   - Rate = MRP - Sharing
```

### B2B Processing Workflow
```
1. Upload Excel File
   ↓
2. System detects B2B records (MobileNumber = "B2B")
   ↓
3. Single-Click Bulk Processing
   - Formula: Sharing = MRP - CentreTestRate
   - Rate = CentreTestRate
   ↓
4. Generate B2B Bills
```

## 💰 Enhanced Billing Logic

### HLM Centers (Healthcare Laboratory Management)
```python
# Dynamic sharing calculation
for test_type in test_types:
    sharing_percentage = user_defined_percentages.get(test_type, default_percentage)
    sharing_amount = mrp * (sharing_percentage / 100)
    rate = mrp - sharing_amount
```

**Example:**
- MRP: 1500.00
- Blood Test Sharing: 60%
- X-Ray Sharing: 50%
- Blood Test: Sharing = 900.00, Rate = 600.00
- X-Ray: Sharing = 750.00, Rate = 750.00

### B2B Centers (Business-to-Business)
```python
# Standard B2B calculation
sharing_amount = mrp - centre_test_rate
rate = centre_test_rate
```

**Example:**
- MRP: 1500.00
- CentreTestRate: 1200.00
- Sharing: 300.00, Rate: 1200.00

## 📁 Project Structure

```
BILL APP/
├── app_enhanced.py                 # Enhanced Flask application (MAIN)
├── app.py                         # Standard Flask application
├── medical_billing_app.py         # Medical billing specific app
├── deploy.py                      # Automated deployment script
├── test_enhanced.py               # Comprehensive test suite
├── requirements_enhanced.txt      # Enhanced dependencies
├── requirements.txt               # Basic dependencies
├── README.md                      # This documentation
├── IMPLEMENTATION_SUMMARY.md      # Technical implementation details
├── BILL.xlsx                      # Sample Excel file
├── uploads/                       # Uploaded Excel files
├── bills/                         # Generated bill files
├── templates/
│   ├── hlm_bills_enhanced.html    # Stage 1: Center selection
│   ├── hlm_bills_stage2.html      # Stage 2: Sharing configuration
│   ├── bill_pdf_enhanced.html     # Enhanced PDF template
│   ├── bill_detail.html           # Bill detail view
│   ├── bills.html                 # Bills overview
│   └── [other templates]          # Additional templates
└── __pycache__/                   # Python cache files
```

## 🛠️ API Endpoints

### Enhanced Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/upload` | POST | Upload Excel file |
| `/get_centers_for_hlm` | GET | Get available HLM centers |
| `/get_test_types_for_center` | GET | Get test types for specific center |
| `/generate_hlm_bills` | POST | Two-stage HLM bill generation |
| `/generate_b2b_bills` | GET | Single-click B2B bill generation |
| `/bills` | GET | View all generated bills |
| `/bill/<id>` | GET | View specific bill details |
| `/download_bill/<id>?format=excel` | GET | Enhanced Excel download |
| `/download_bill/<id>?format=pdf` | GET | Enhanced PDF download |
| `/download_bill/<id>?format=html` | GET | HTML download |
| `/download_all_bills` | GET | Bulk download all bills |

### API Usage Examples
```javascript
// Get HLM centers
fetch('/get_centers_for_hlm')
  .then(response => response.json())
  .then(data => console.log(data.centers));

// Get test types for center
fetch('/get_test_types_for_center?center_name=Hospital_A')
  .then(response => response.json())
  .then(data => console.log(data.test_types));
```

## 📊 Export Capabilities

### Excel Export (Enhanced)
```python
# In-memory processing with pandas
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Summary')
    detailed_df.to_excel(writer, sheet_name='Detailed')
```

### PDF Export (Dual System)
```python
# Primary: pdfkit
if PDFKIT_AVAILABLE:
    pdf = pdfkit.from_string(html_content, options=pdf_options)

# Fallback: xhtml2pdf
else:
    pisa.CreatePDF(html_content, dest=pdf_buffer)
```

### Bulk Downloads
- **ZIP Archives**: All bills in single download
- **Format Options**: HTML, Excel, PDF bulk downloads
- **Memory Efficient**: Streaming for large datasets

## 🔧 Configuration

### PDF Generation Setup

#### Primary: pdfkit + wkhtmltopdf
```bash
# Ubuntu/Debian
sudo apt-get install wkhtmltopdf

# macOS
brew install wkhtmltopdf

# Windows
# Download from: https://wkhtmltopdf.org/downloads.html
```

#### Fallback: xhtml2pdf
```bash
pip install xhtml2pdf
# No additional system dependencies required
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
python test_enhanced.py
```

### Test Categories
1. **File Upload Tests**: Excel file validation and processing
2. **Data Segmentation Tests**: HLM vs B2B routing
3. **HLM Workflow Tests**: Two-stage processing
4. **B2B Processing Tests**: Single-click generation
5. **Export Tests**: Excel, PDF, HTML generation
6. **API Tests**: Endpoint functionality
7. **Error Handling Tests**: Edge cases and validation

### Manual Testing
```bash
# Test individual components
python test_ai.py              # AI integration tests
python test_billing_app.py     # Billing logic tests
```

## 🔍 Troubleshooting

### Common Issues

#### PDF Generation Issues
```bash
# If pdfkit fails
pip install xhtml2pdf  # Fallback will be used automatically

# If both fail
# Check system dependencies and permissions
```

#### Large File Processing
```python
# Memory optimization for large datasets
# System automatically uses chunked processing
# Monitor memory usage with large Excel files
```

#### Data Segmentation Issues
```python
# Ensure MobileNumber column contains exactly "HLM" or "B2B"
# Case-sensitive matching
# Other values default to B2B processing
```

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements_enhanced.txt

# For optional PDF libraries
pip install pdfkit xhtml2pdf
```

### AI Assistance
- Use the built-in AI chat for contextual help
- AI provides specific troubleshooting for your data
- Error messages include AI-generated solutions

## 🚀 Performance Optimizations

### Memory Management
- **In-memory Processing**: Efficient handling of large datasets
- **Streaming Downloads**: Memory-efficient bulk exports
- **Session Management**: Optimized data storage

### Processing Speed
- **Vectorized Operations**: pandas for fast data processing
- **Concurrent Processing**: Multi-threaded export generation
- **Caching**: Intelligent caching of processed data

## 🔒 Security Features

### Data Protection
- **Secure File Handling**: Validated uploads with size limits
- **Session Security**: Encrypted session data
- **Input Validation**: Comprehensive data sanitization

### API Security
- **Rate Limiting**: Protection against abuse
- **Error Handling**: Secure error messages
- **File Cleanup**: Automatic temporary file removal

## 📈 Scalability Features

### Horizontal Scaling
- **Stateless Design**: Session-based architecture
- **Database Ready**: Easy migration to persistent storage
- **Load Balancer Compatible**: Multiple instance support

### Vertical Scaling
- **Memory Efficient**: Optimized for large datasets
- **CPU Optimized**: Efficient processing algorithms
- **Storage Optimized**: Minimal disk usage

## 🤝 Migration Guide

### From Original to Enhanced
1. **Backup existing data**
2. **Install enhanced dependencies**
3. **Update templates (optional - backward compatible)**
4. **Configure PDF generation tools**
5. **Test with sample data**

### Backward Compatibility
- All existing templates work unchanged
- Original API endpoints preserved
- Existing Excel formats supported
- No breaking changes to core functionality

## 📞 Support & Documentation

### Getting Help
1. **AI Assistant**: Built-in contextual help
2. **Error Messages**: Detailed troubleshooting information
3. **Logs**: Comprehensive logging for debugging
4. **Documentation**: This comprehensive guide

### File Locations
- **Main Application**: `app_enhanced.py`
- **Documentation**: `README.md`, `IMPLEMENTATION_SUMMARY.md`
- **Tests**: `test_enhanced.py`
- **Templates**: `templates/` directory
- **Sample Data**: `BILL.xlsx`

## 🎯 Usage Instructions

### Step 1: Start the Application
```bash
python app_enhanced.py
```

### Step 2: Upload Excel File
1. Navigate to `http://localhost:5000`
2. Click "Upload Excel File"
3. Select your Excel file with required columns
4. Click "Upload"

### Step 3: Process Bills

#### For HLM Centers:
1. System detects HLM records automatically
2. **Stage 1**: Select the HLM center from dropdown
3. **Stage 2**: Configure sharing percentages for each test type
4. Click "Generate HLM Bills"

#### For B2B Centers:
1. System detects B2B records automatically
2. Click "Generate B2B Bills" (single-click processing)

### Step 4: View and Download Bills
1. Navigate to "View Bills" to see all generated bills
2. Click on any bill to view details
3. Download in multiple formats:
   - **Excel**: Detailed spreadsheet with calculations
   - **PDF**: Professional invoice format
   - **HTML**: Web-friendly format

### Step 5: Bulk Operations
1. Use "Download All Bills" for bulk export
2. Choose format (Excel, PDF, HTML)
3. System generates ZIP archive with all bills

## 🔄 Version History

- **v2.0 Enhanced**: Complete rewrite with intelligent data segmentation, two-stage workflows, and dual PDF generation
- **v1.5**: Added AI integration and enhanced templates
- **v1.0**: Basic medical billing functionality

---

**🏥 Enhanced Medical Billing System - Professional Edition**  
*AI-Powered Medical Test Billing Automation with Advanced Data Processing*

**Version**: 2.0 Enhanced  
**Last Updated**: January 2024  
**License**: MIT License

For technical implementation details, see [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)