# 🏥 Medical Test Billing Automation System

A professional Python application for automating medical test billing for diagnostic centers. This system supports both B2B and HLM center types with configurable client sharing percentages.

## ✨ Features

- **📊 Excel & PDF Bill Generation**: Professional invoice formatting with summary and detailed sheets
- **🤖 AI Integration**: Gemini/OpenAI API support for error handling and user assistance
- **💰 Smart Billing Logic**: Automatic calculation based on center type (B2B/HLM)
- **📅 Configurable Parameters**: Invoice dates, periods, sequence numbers
- **🔢 Amount to Words**: Indian numbering system conversion
- **📋 Statutory Compliance**: GST exemption and TDS notes
- **🛡️ Error Handling**: Comprehensive validation and error recovery

## 📋 Requirements

### System Requirements
- Python 3.7 or higher
- Windows/Linux/macOS

### Dependencies
Install all required packages:
```bash
pip install -r requirements_medical_billing.txt
```

## 🚀 Quick Start

### 1. Basic Usage
```bash
python medical_billing_app.py --main-data BILL.xlsx
```

### 2. With Supporting Data (for HLM centers)
```bash
python medical_billing_app.py --main-data BILL.xlsx --supporting-data sharing.xlsx
```

### 3. Custom Invoice Date
```bash
python medical_billing_app.py --main-data BILL.xlsx --invoice-date 2024-01-15
```

### 4. Custom Period
```bash
python medical_billing_app.py --main-data BILL.xlsx --period-start 2024-01-01 --period-end 2024-01-31
```

### 5. AI Chat Mode
```bash
python medical_billing_app.py --ask
```

## 📊 Input Data Format

### Main Data Excel (Required)
Must contain these columns:
- `PatientVisitCode`: Unique patient visit identifier
- `RegisteredDate`: Date of registration
- `PatientName`: Patient's full name
- `Age`: Patient's age
- `AgeUnit`: Age unit (Years, Months, etc.)
- `Gender`: Patient's gender
- `MobileNumber`: Contains center type (B2B/HLM)
- `TEST NAME`: Name of the medical test
- `CODE NO`: Test code
- `CENTER NAME`: Name of the diagnostic center
- `Modality`: Test modality
- `MRP`: Maximum retail price
- `CentreTestRate`: Actual test rate
- `TEST TYPE`: Category of test

### Supporting Data Excel (Optional)
For HLM centers with custom sharing percentages:
- `CENTER NAME`: Center name
- `TEST TYPE`: Test type
- `SHARE_PERCENTAGE`: Client sharing percentage (0-100)

## 💰 Billing Logic

### B2B Centers
- **Net Amount** = CentreTestRate × count
- No percentage sharing applied

### HLM Centers
- **Net Amount** = CentreTestRate × (Client Share %) × count
- Default sharing percentage: 50%
- Can be customized via supporting data

## 📁 Output Structure

Generated bills are saved in:
```
bills/
├── excel/
│   ├── Center_Name_1.xlsx
│   └── Center_Name_2.xlsx
└── pdf/
    ├── Center_Name_1.pdf
    └── Center_Name_2.pdf
```

### Excel Bill Structure
- **Sheet 1: Summary Bill**
  - Company header and invoice details
  - Test summary table
  - Amount in words
  - Narration and statutory notes
  - Bank details

- **Sheet 2: Detailed Bill**
  - Patient-wise details
  - SR.NO, Date, Patient Name, Age, Gender, Test Name, Value

### PDF Bill Structure
- Same content as Excel summary sheet
- Professional formatting
- Print-ready layout

## 🤖 AI Integration

### Setup AI API Keys
Create a `.env` file in the project directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### AI Features
- **Error Handling**: Automatic troubleshooting suggestions
- **Interactive Chat**: Ask questions about the system
- **Context-Aware**: Understands billing-specific issues

### Using AI Chat
```bash
python medical_billing_app.py --ask
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python test_billing_app.py
```

## 📝 Command Line Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `--main-data` | Path to main data Excel file | Yes* | - |
| `--supporting-data` | Path to supporting data Excel file | No | - |
| `--invoice-date` | Invoice date (YYYY-MM-DD) | No | Today |
| `--period-start` | Period start date (YYYY-MM-DD) | No | - |
| `--period-end` | Period end date (YYYY-MM-DD) | No | - |
| `--invoice-sequence-start` | Starting invoice sequence | No | 1 |
| `--ask` | Enter AI chat mode | No | False |

*Required unless using `--ask` for AI chat mode only.

## 🔧 Configuration

### Invoice Number Format
- Format: `KRPL/YY-YY/MM/NNN`
- Example: `KRPL/2024-2025/01/001`
- Automatically increments per month

### Default Settings
- **Default Client Share**: 50% (for HLM centers)
- **Invoice Sequence**: Starts from 1
- **Output Format**: Both Excel and PDF
- **File Naming**: Center name with spaces replaced by underscores

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Columns Error**
   - Ensure all required columns are present in main data
   - Check column names match exactly (case-sensitive)

2. **File Permission Error**
   - Ensure write permissions for output directories
   - Close any open Excel/PDF files

3. **AI API Errors**
   - Verify API keys in `.env` file
   - Check internet connection
   - Ensure API quotas are not exceeded

4. **Date Format Errors**
   - Use YYYY-MM-DD format for dates
   - Ensure dates are valid

### Getting Help
1. Check the log file: `billing.log`
2. Use AI chat mode: `python medical_billing_app.py --ask`
3. Review error messages for specific guidance

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Use AI chat mode: `python medical_billing_app.py --ask`
- Check the log file for detailed error information
- Review this README for common solutions

---

**Version**: 1.0.0  
**Last Updated**: January 2024 