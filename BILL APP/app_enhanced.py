from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
import pandas as pd
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import tempfile
import zipfile
from io import BytesIO
import logging
import requests
from dotenv import load_dotenv
import re
import numpy as np
from utils import AmountToWords, InvoiceNumberGenerator, AIIntegration, safe_float_conversion, safe_int_conversion, safe_date_conversion

# PDF generation imports with fallback
PDFKIT_AVAILABLE = False
XHTML2PDF_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    pass
    
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    pass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}


class DataProcessor:
    """Advanced data processing with intelligent segmentation."""
    
    @staticmethod
    def detect_center_type_from_mobile(mobile_value):
        """Detect center type from MobileNumber column value."""
        if pd.isna(mobile_value):
            return 'B2B'  # Default
        
        mobile_str = str(mobile_value).strip().upper()
        if mobile_str == 'HLM':
            return 'HLM'
        elif mobile_str == 'B2B':
            return 'B2B'
        else:
            return 'B2B'  # Default for other values
    
    @staticmethod
    def extract_centers_from_data(df):
        """Extract unique centers from data."""
        if 'CENTER NAME' not in df.columns:
            return []
        
        centers = df['CENTER NAME'].dropna().unique().tolist()
        return sorted(centers)
    
    @staticmethod
    def extract_test_types_for_center(df, center_name):
        """Extract unique test types for a specific center."""
        if 'TEST TYPE' not in df.columns:
            return []
        
        center_data = df[df['CENTER NAME'] == center_name]
        test_types = center_data['TEST TYPE'].dropna().unique().tolist()
        return sorted(test_types)
    
    @staticmethod
    def segment_data_by_mobile_number(df):
        """Segment data based on MobileNumber column values."""
        hlm_data = df[df['MobileNumber'].astype(str).str.strip().str.upper() == 'HLM'].copy()
        b2b_data = df[df['MobileNumber'].astype(str).str.strip().str.upper() == 'B2B'].copy()
        
        # Handle cases where MobileNumber doesn't contain HLM or B2B
        other_data = df[~df['MobileNumber'].astype(str).str.strip().str.upper().isin(['HLM', 'B2B'])].copy()
        
        return {
            'hlm_data': hlm_data,
            'b2b_data': b2b_data,
            'other_data': other_data
        }

class ExcelExporter:
    """Enhanced Excel export with pandas.to_excel() and in-memory processing."""
    
    @staticmethod
    def generate_excel_bill(bill_data):
        """Generate Excel bill with in-memory processing."""
        try:
            # Create in-memory buffer
            buffer = BytesIO()
            
            # Create Excel writer
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Description': ['Invoice Number', 'Center Name', 'Bill Date', 'Total Tests', 'Total MRP', 'Total Rate', 'Total Sharing', 'Amount in Words'],
                    'Value': [
                        bill_data.get('bill_number', 'N/A'),
                        bill_data.get('centre_name', 'N/A'),
                        bill_data.get('bill_date', 'N/A'),
                        len(bill_data.get('test_items', [])),
                        f"₹{bill_data.get('total_mrp', 0):.2f}",
                        f"₹{bill_data.get('total_rate', 0):.2f}",
                        f"₹{bill_data.get('total_sharing', 0):.2f}",
                        bill_data.get('amount_in_words', 'N/A')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed sheet
                if bill_data.get('test_items'):
                    detailed_data = []
                    for i, item in enumerate(bill_data['test_items'], 1):
                        detailed_data.append({
                            'Sr. No.': i,
                            'Date': item.get('registered_date', 'N/A'),
                            'Patient Name': item.get('patient_name', 'N/A'),
                            'Visit Code': item.get('visit_code', 'N/A'),
                            'Test Name': item.get('test_name', 'N/A'),
                            'MRP': f"₹{item.get('mrp', 0):.2f}",
                            'Rate': f"₹{item.get('rate', 0):.2f}",
                            'Sharing': f"₹{item.get('sharing_amount', 0):.2f}"
                        })
                    
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
            
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating Excel: {e}")
            raise

class PDFExporter:
    """Dual PDF generation system with pdfkit and xhtml2pdf fallback."""
    
    @staticmethod
    def generate_pdf_with_pdfkit(html_content):
        """Generate PDF using pdfkit (primary method)."""
        if not PDFKIT_AVAILABLE:
            raise Exception("pdfkit not available")
        
        try:
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            pdf_buffer = BytesIO()
            pdf = pdfkit.from_string(html_content, False, options=options)
            pdf_buffer.write(pdf)
            pdf_buffer.seek(0)
            return pdf_buffer
            
        except Exception as e:
            logger.error(f"pdfkit generation failed: {e}")
            raise
    
    @staticmethod
    def generate_pdf_with_xhtml2pdf(html_content):
        """Generate PDF using xhtml2pdf (fallback method)."""
        if not XHTML2PDF_AVAILABLE:
            raise Exception("xhtml2pdf not available")
        
        try:
            pdf_buffer = BytesIO()
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
            
            if pisa_status.err:
                raise Exception("xhtml2pdf generation failed")
            
            pdf_buffer.seek(0)
            return pdf_buffer
            
        except Exception as e:
            logger.error(f"xhtml2pdf generation failed: {e}")
            raise
    
    @staticmethod
    def generate_pdf_bill(bill_data, html_template):
        """Generate PDF with fallback system."""
        try:
            # Try pdfkit first
            if PDFKIT_AVAILABLE:
                try:
                    return PDFExporter.generate_pdf_with_pdfkit(html_template)
                except Exception as e:
                    logger.warning(f"pdfkit failed, trying fallback: {e}")
            
            # Fallback to xhtml2pdf
            if XHTML2PDF_AVAILABLE:
                return PDFExporter.generate_pdf_with_xhtml2pdf(html_template)
            else:
                raise Exception("No PDF generation library available")
                
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise

# Initialize global objects
amount_converter = AmountToWords()
invoice_generator = InvoiceNumberGenerator()
ai_integration = AIIntegration()
data_processor = DataProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_excel_data(df):
    """Enhanced data validation with comprehensive checks."""
    required_columns = ['RegisteredDate', 'PatientVisitCode', 'PatientName', 'TEST NAME', 'MRP', 'CentreTestRate', 'CENTER NAME']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "Excel file is empty"
    
    # Check for required data in key columns
    if df['CENTER NAME'].isna().all():
        return False, "No center names found in the data"
    
    if df['MRP'].isna().all():
        return False, "No MRP values found in the data"
    
    # Validate MobileNumber column for data segmentation
    if 'MobileNumber' not in df.columns:
        logger.warning("MobileNumber column not found, will default to B2B processing")
    
    return True, None

...existing code...
def df_for_session(df):
    """Convert datetime columns to string and replace NaT/empty with None."""
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].apply(lambda x: None if pd.isna(x) else x.strftime('%Y-%m-%d'))
    return df.to_dict('records')

def process_excel_file_enhanced(file_path, sharing_percentages=None, center_type=None):
    """Enhanced Excel file processing with intelligent data segmentation."""
    try:
        # Read Excel file with error handling
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            return None, f"Error reading Excel file: {str(e)}"
        
        # Validate data structure
        is_valid, error_msg = validate_excel_data(df)
        if not is_valid:
            return None, error_msg
        
        # Clean and process the data
        df = df.fillna('')
        
        # Intelligent data segmentation based on MobileNumber column
        segmented_data = data_processor.segment_data_by_mobile_number(df)
        
        # Store segmented data in session for two-stage processing
        session['segmented_data'] = {
            'hlm_centers': data_processor.extract_centers_from_data(segmented_data['hlm_data']),
            'b2b_centers': data_processor.extract_centers_from_data(segmented_data['b2b_data']),
           'hlm_data': df_for_session(segmented_data['hlm_data']),
            'b2b_data': df_for_session(segmented_data['b2b_data']),
            'other_data': df_for_session(segmented_data['other_data'])
        }
        
        # Process bills based on center type
        bills = []
        
        # Process HLM centers
        if not segmented_data['hlm_data'].empty:
            hlm_bills = process_hlm_data(segmented_data['hlm_data'], sharing_percentages)
            bills.extend(hlm_bills)
        
        # Process B2B centers
        if not segmented_data['b2b_data'].empty:
            b2b_bills = process_b2b_data(segmented_data['b2b_data'])
            bills.extend(b2b_bills)
        
        # Process other data as B2B by default
        if not segmented_data['other_data'].empty:
            other_bills = process_b2b_data(segmented_data['other_data'])
            bills.extend(other_bills)
        
        if not bills:
            return None, "No valid bills could be generated from the data"
        
        return bills, None
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return None, f"Error processing file: {str(e)}"

def process_hlm_data(hlm_df, sharing_percentages=None):
    """Process HLM data with configurable sharing percentages."""
    bills = []
    
    for centre_name, group in hlm_df.groupby('CENTER NAME'):
        if pd.isna(centre_name) or centre_name == '':
            continue
        
        # Extract test types for this center
        test_types = data_processor.extract_test_types_for_center(hlm_df, centre_name)
        
        # Process test items for this center
        test_items = []
        total_mrp = 0
        total_rate = 0
        total_sharing = 0
        
        for _, row in group.iterrows():
            try:
                mrp = safe_float_conversion(row['MRP'])
                test_type = str(row.get('TEST TYPE', 'Other')).strip()
                
                # Get sharing percentage for this test type
                if sharing_percentages and test_type in sharing_percentages:
                    sharing_percentage = sharing_percentages[test_type]
                else:
                    sharing_percentage = sharing_percentages.get('default', 55.0) if sharing_percentages else 55.0
                
                # Calculate sharing amount and rate for HLM
                sharing_amount = mrp * (sharing_percentage / 100)
                rate = mrp - sharing_amount
                
                test_item = {
                    'registered_date': safe_date_conversion(row['RegisteredDate']),
                    'visit_code': str(safe_int_conversion(row['PatientVisitCode'])),
                    'patient_name': str(row['PatientName']) if pd.notna(row['PatientName']) else 'N/A',
                    'test_name': str(row['TEST NAME']) if pd.notna(row['TEST NAME']) else 'N/A',
                    'test_type': test_type,
                    'mrp': mrp,
                    'rate': rate,
                    'sharing_amount': sharing_amount,
                    'sharing_percentage': sharing_percentage
                }
                test_items.append(test_item)
                total_mrp += mrp
                total_rate += rate
                total_sharing += sharing_amount
                
            except Exception as e:
                logger.error(f"Error processing HLM row: {e}")
                continue
        
        if test_items:
            # Generate professional invoice number
            invoice_number = invoice_generator.generate()
            bill = {
                'centre_name': str(centre_name),
                'test_items': test_items,
                'test_types': test_types,
                'total_mrp': total_mrp,
                'total_rate': total_rate,
                'total_sharing': total_sharing,
                'bill_date': datetime.now().strftime('%Y-%m-%d'),
                'bill_number': invoice_number,
                'center_type': 'HLM',
                'amount_in_words': amount_converter.convert(total_rate)
            }
            bills.append(bill)
    
    return bills

def process_b2b_data(b2b_df):
    """Process B2B data with standard billing logic."""
    bills = []
    
    for centre_name, group in b2b_df.groupby('CENTER NAME'):
        if pd.isna(centre_name) or centre_name == '':
            continue
        
        # Process test items for this center
        test_items = []
        total_mrp = 0
        total_rate = 0
        total_sharing = 0
        
        for _, row in group.iterrows():
            try:
                mrp = safe_float_conversion(row['MRP'])
                rate = safe_float_conversion(row['CentreTestRate'])
                
                # Calculate sharing amount for B2B: Sharing = MRP - Rate
                sharing_amount = mrp - rate
                
                test_item = {
                    'registered_date': safe_date_conversion(row['RegisteredDate']),
                    'visit_code': str(safe_int_conversion(row['PatientVisitCode'])),
                    'patient_name': str(row['PatientName']) if pd.notna(row['PatientName']) else 'N/A',
                    'test_name': str(row['TEST NAME']) if pd.notna(row['TEST NAME']) else 'N/A',
                    'test_type': str(row.get('TEST TYPE', 'Other')).strip(),
                    'mrp': mrp,
                    'rate': rate,
                    'sharing_amount': sharing_amount
                }
                test_items.append(test_item)
                total_mrp += mrp
                total_rate += rate
                total_sharing += sharing_amount
                
            except Exception as e:
                logger.error(f"Error processing B2B row: {e}")
                continue
        
        if test_items:
            # Generate professional invoice number
            invoice_number = invoice_generator.generate()
            bill = {
                'centre_name': str(centre_name),
                'test_items': test_items,
                'total_mrp': total_mrp,
                'total_rate': total_rate,
                'total_sharing': total_sharing,
                'bill_date': datetime.now().strftime('%Y-%m-%d'),
                'bill_number': invoice_number,
                'center_type': 'B2B',
                'amount_in_words': amount_converter.convert(total_rate)
            }
            bills.append(bill)
    
    return bills

@app.route('/')
def index():
    try:
        return render_template('index.html', app=app)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        flash('An error occurred while loading the page', 'error')
    return render_template('index.html', app=app)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not file or not allowed_file(file.filename):
            flash('Invalid file type. Please upload an Excel file (.xlsx or .xls)', 'error')
            return redirect(url_for('index'))
        
        # Secure filename and save file
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            flash('Error saving uploaded file', 'error')
            return redirect(url_for('index'))
        
        # Process the Excel file with enhanced processing
        bills, error = process_excel_file_enhanced(file_path)
        
        if error:
            # Get AI assistance for error handling
            ai_suggestion = ai_integration.handle_error(error, "File upload and processing")
            flash(f'Error processing file: {error}\n\nAI Suggestion:\n{ai_suggestion}', 'error')
            return redirect(url_for('index'))
        
        if not bills:
            flash('No bills could be generated from the uploaded file', 'error')
            return redirect(url_for('index'))
        
        # Store bills in session
        session['bills'] = bills
        
        flash(f'Successfully processed {len(bills)} bills from {filename}', 'success')
        return redirect(url_for('bills'))
    
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        ai_suggestion = ai_integration.handle_error(str(e), "File upload process")
        flash(f'An unexpected error occurred while processing the file\n\nAI Suggestion:\n{ai_suggestion}', 'error')
        return redirect(url_for('index'))

@app.route('/get_centers_for_hlm')
def get_centers_for_hlm():
    """API endpoint to get available HLM centers from uploaded data."""
    try:
        segmented_data = session.get('segmented_data', {})
        hlm_centers = segmented_data.get('hlm_centers', [])
        
        return jsonify({
            'success': True,
            'centers': hlm_centers
        })
    except Exception as e:
        logger.error(f"Error getting HLM centers: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_test_types_for_center')
def get_test_types_for_center():
    """API endpoint to get test types for a specific center."""
    try:
        center_name = request.args.get('center_name')
        if not center_name:
            return jsonify({'success': False, 'error': 'Center name required'})
        
        segmented_data = session.get('segmented_data', {})
        hlm_data = segmented_data.get('hlm_data', [])
        
        # Convert back to DataFrame for processing
        hlm_df = pd.DataFrame(hlm_data)
        test_types = data_processor.extract_test_types_for_center(hlm_df, center_name)
        
        return jsonify({
            'success': True,
            'test_types': test_types
        })
    except Exception as e:
        logger.error(f"Error getting test types: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/generate_hlm_bills', methods=['GET', 'POST'])
def generate_hlm_bills():
    """Enhanced HLM bill generation with two-stage interactive workflow."""
    try:
        if request.method == 'POST':
            # Check if this is stage 1 (center selection) or stage 2 (sharing configuration)
            stage = request.form.get('stage', '1')
            
            if stage == '1':
                # Stage 1: Center selection
                selected_center = request.form.get('selected_center')
                if not selected_center:
                    flash('Please select a center', 'error')
                    return redirect(url_for('generate_hlm_bills'))
                
                # Get test types for selected center
                segmented_data = session.get('segmented_data', {})
                hlm_data = segmented_data.get('hlm_data', [])
                hlm_df = pd.DataFrame(hlm_data)
                test_types = data_processor.extract_test_types_for_center(hlm_df, selected_center)
                
                # Store selected center and test types in session
                session['selected_hlm_center'] = selected_center
                session['center_test_types'] = test_types
                
                # Render stage 2 template with dynamic test types
                return render_template('hlm_bills_stage2.html',
                                     center_name=selected_center,
                                     test_types=test_types,
                                     app=app)
            
            elif stage == '2':
                # Stage 2: Sharing percentage configuration
                selected_center = session.get('selected_hlm_center')
                test_types = session.get('center_test_types', [])
                
                if not selected_center:
                    flash('Session expired. Please start again.', 'error')
                    return redirect(url_for('generate_hlm_bills'))
                
                # Collect sharing percentages for each test type
                sharing_percentages = {}
                for test_type in test_types:
                    percentage_key = f"sharing_{test_type.replace(' ', '_').lower()}"
                    percentage = request.form.get(percentage_key)
                    if percentage:
                        try:
                            sharing_percentages[test_type] = float(percentage)
                        except ValueError:
                            flash(f'Invalid percentage for {test_type}', 'error')
                            return redirect(url_for('generate_hlm_bills'))
                
                # Set default percentage if provided
                default_percentage = request.form.get('default_percentage')
                if default_percentage:
                    try:
                        sharing_percentages['default'] = float(default_percentage)
                    except ValueError:
                        flash('Invalid default percentage', 'error')
                        return redirect(url_for('generate_hlm_bills'))
                
                # Generate HLM bills with custom sharing percentages
                segmented_data = session.get('segmented_data', {})
                hlm_data = segmented_data.get('hlm_data', [])
                hlm_df = pd.DataFrame(hlm_data)
                
                # Filter for selected center
                center_data = hlm_df[hlm_df['CENTER NAME'] == selected_center]
                
                if center_data.empty:
                    flash('No data found for selected center', 'error')
                    return redirect(url_for('generate_hlm_bills'))
                
                # Process HLM bills with custom sharing
                hlm_bills = process_hlm_data(center_data, sharing_percentages)
                
                if not hlm_bills:
                    flash('No HLM bills could be generated', 'error')
                    return redirect(url_for('generate_hlm_bills'))
                
                # Store bills in session
                session['bills'] = hlm_bills
                
                flash(f'Generated {len(hlm_bills)} HLM bills for {selected_center}', 'success')
                return redirect(url_for('bills'))
        
        # GET request - show stage 1 (center selection)
        segmented_data = session.get('segmented_data', {})
        hlm_centers = segmented_data.get('hlm_centers', [])
        
        if not hlm_centers:
            flash('No HLM centers found in uploaded data. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        return render_template('hlm_bills.html', hlm_centers=hlm_centers, app=app)
        
    except Exception as e:
        logger.error(f"Error in generate_hlm_bills: {e}")
        flash('An error occurred while processing HLM bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_b2b_bills')
def generate_b2b_bills():
    """Generate bills for B2B centers with single-click bulk processing."""
    try:
        segmented_data = session.get('segmented_data', {})
        b2b_data = segmented_data.get('b2b_data', [])
        other_data = segmented_data.get('other_data', [])
        
        # Combine B2B and other data (other data defaults to B2B)
        all_b2b_data = b2b_data + other_data
        
        if not all_b2b_data:
            flash('No B2B data found in uploaded file', 'error')
            return redirect(url_for('bills'))
        
        # Convert to DataFrame and process
        b2b_df = pd.DataFrame(all_b2b_data)
        b2b_bills = process_b2b_data(b2b_df)
        
        if not b2b_bills:
            flash('No B2B bills could be generated', 'error')
            return redirect(url_for('bills'))
        
        # Store bills in session
        session['bills'] = b2b_bills
        
        flash(f'Generated {len(b2b_bills)} B2B bills', 'success')
        return redirect(url_for('bills'))
        
    except Exception as e:
        logger.error(f"Error in generate_b2b_bills: {e}")
        flash('An error occurred while processing B2B bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/bills')
def bills():
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        # Calculate total tests and amounts
        total_tests = sum(len(bill.get('test_items', [])) for bill in bills_data)
        total_mrp = sum(bill.get('total_mrp', 0) for bill in bills_data)
        total_rate = sum(bill.get('total_rate', 0) for bill in bills_data)
        total_sharing = sum(bill.get('total_sharing', 0) for bill in bills_data)
        
        return render_template('bills.html',
                             bills=bills_data,
                             total_tests=total_tests,
                             total_mrp=total_mrp,
                             total_rate=total_rate,
                             total_sharing=total_sharing,
                             app=app)
    except Exception as e:
        logger.error(f"Error in bills route: {e}")
        flash('An error occurred while loading bills', 'error')
        return redirect(url_for('index'))

@app.route('/bill/<int:bill_index>')
def view_bill(bill_index):
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        
        if bill_index < 0 or bill_index >= len(bills_data):
            flash('Bill not found', 'error')
            return redirect(url_for('bills'))
        
        bill = bills_data[bill_index]
        return render_template('bill_detail.html', bill=bill, bill_index=bill_index, app=app)
    except Exception as e:
        logger.error(f"Error in view_bill: {e}")
        flash('An error occurred while viewing the bill', 'error')
        return redirect(url_for('bills'))

@app.route('/download_bill/<int:bill_index>')
def download_bill(bill_index):
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        
        if bill_index < 0 or bill_index >= len(bills_data):
            flash('Bill not found', 'error')
            return redirect(url_for('bills'))
        
        bill = bills_data[bill_index]
        fmt = request.args.get('format', 'html').lower()
        
        if fmt == 'excel':
            try:
                excel_buffer = ExcelExporter.generate_excel_bill(bill)
                filename = f"{bill['bill_number']}.xlsx"
                
                return send_file(
                    excel_buffer,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name=filename
                )
            except Exception as e:
                logger.error(f"Error generating Excel: {e}")
                flash('Could not generate Excel file for this bill', 'error')
                return redirect(url_for('view_bill', bill_index=bill_index))
        
        elif fmt == 'pdf':
            try:
                # Render HTML template for PDF
                html_content = render_template('bill_pdf.html', bill=bill)
                pdf_buffer = PDFExporter.generate_pdf_bill(bill, html_content)
                filename = f"{bill['bill_number']}.pdf"
                
                return send_file(
                    pdf_buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=filename
                )
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                flash('Could not generate PDF file for this bill', 'error')
                return redirect(url_for('view_bill', bill_index=bill_index))
        
        else:
            # HTML format
            html_content = render_template('bill_pdf.html', bill=bill)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_path = f.name
            return send_file(temp_path, as_attachment=True, download_name=f"{bill['bill_number']}.html")
            
    except Exception as e:
        logger.error(f"Error in download_bill: {e}")
        flash('An error occurred while downloading the bill', 'error')
        return redirect(url_for('view_bill', bill_index=bill_index))

@app.route('/download_all_bills')
def download_all_bills():
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available', 'error')
            return redirect(url_for('index'))
        
        # Create a ZIP file containing all bills
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for i, bill in enumerate(bills_data):
                try:
                    html_content = render_template('bill_pdf.html', bill=bill)
                    zf.writestr(f"{bill['bill_number']}.html", html_content)
                except Exception as e:
                    logger.error(f"Error processing bill {i}: {e}")
                    continue
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"all_bills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
    except Exception as e:
        logger.error(f"Error in download_all_bills: {e}")
        flash('An error occurred while downloading all bills', 'error')
        return redirect(url_for('bills'))

@app.route('/download_all_excel')
def download_all_excel():
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for bill in bills_data:
                try:
                    excel_buffer = ExcelExporter.generate_excel_bill(bill)
                    filename = f"{bill['bill_number']}.xlsx"
                    zf.writestr(filename, excel_buffer.getvalue())
                except Exception as e:
                    logger.error(f"Error generating Excel for {bill.get('bill_number', 'unknown')}: {e}")
                    continue
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"all_bills_excel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
    except Exception as e:
        logger.error(f"Error in download_all_excel: {e}")
        flash('An error occurred while downloading all Excel bills', 'error')
        return redirect(url_for('bills'))

@app.route('/download_all_pdf')
def download_all_pdf():
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for bill in bills_data:
                try:
                    html_content = render_template('bill_pdf.html', bill=bill)
                    pdf_buffer = PDFExporter.generate_pdf_bill(bill, html_content)
                    filename = f"{bill['bill_number']}.pdf"
                    zf.writestr(filename, pdf_buffer.getvalue())
                except Exception as e:
                    logger.error(f"Error generating PDF for {bill.get('bill_number', 'unknown')}: {e}")
                    continue
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"all_bills_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
    except Exception as e:
        logger.error(f"Error in download_all_pdf: {e}")
        flash('An error occurred while downloading all PDF bills', 'error')
        return redirect(url_for('bills'))

@app.route('/api/bills')
def api_bills():
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            return jsonify({'error': 'No bills available'}), 404
        
        return jsonify(bills_data)
    except Exception as e:
        logger.error(f"Error in api_bills: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ai_assistance', methods=['GET', 'POST'])
def ai_assistance():
    """AI assistance page for user queries"""
    if request.method == 'POST':
        user_query = request.form.get('user_query', '')
        if user_query:
            try:
                ai_response = ai_integration.get_ai_response(user_query)
                return jsonify({'response': ai_response})
            except Exception as e:
                logger.error(f"AI assistance error: {e}")
                return jsonify({'response': f"AI assistance temporarily unavailable: {str(e)}"})
    
    return render_template('ai_assistance.html', app=app)

# Legacy routes for backward compatibility
@app.route('/generate_all_bills')
def generate_all_bills():
    """Generate bills for all centers"""
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        flash('All bills generated successfully!', 'success')
        return redirect(url_for('bills'))
    except Exception as e:
        logger.error(f"Error in generate_all_bills: {e}")
        flash('An error occurred while generating bills', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_manual_bill', methods=['GET', 'POST'])
def generate_manual_bill():
    """Show manual bill generation page or process single bill generation"""
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            center_name = request.form.get('center_name')
            if not center_name:
                flash('Please select a center', 'error')
                return redirect(url_for('generate_manual_bill'))
            
            # Find the selected bill
            selected_bill = None
            for bill in bills_data:
                if bill['centre_name'] == center_name:
                    selected_bill = bill
                    break
            
            if selected_bill:
                # Create a new bills list with only the selected bill
                session['bills'] = [selected_bill]
                flash(f'Generated bill for {center_name}', 'success')
                return redirect(url_for('bills'))
            else:
                flash('Selected center not found', 'error')
                return redirect(url_for('generate_manual_bill'))
        
        return render_template('manual_bill.html', bills=bills_data, app=app)
    except Exception as e:
        logger.error(f"Error in generate_manual_bill: {e}")
        flash('An error occurred while processing manual bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_multiple_bills', methods=['GET', 'POST'])
def generate_multiple_bills():
    """Show multiple bill generation page or process multiple bill generation"""
    try:
        bills_data = session.get('bills', [])
        if not bills_data:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            selected_centers = request.form.getlist('selected_centers')
            if not selected_centers:
                flash('Please select at least one center', 'error')
                return redirect(url_for('generate_multiple_bills'))
            
            # Filter bills for selected centers
            filtered_bills = [bill for bill in bills_data if bill['centre_name'] in selected_centers]
            
            if not filtered_bills:
                flash('No bills found for selected centers', 'error')
                return redirect(url_for('generate_multiple_bills'))
            
            session['bills'] = filtered_bills
            flash(f'Generated {len(filtered_bills)} bills for selected centers', 'success')
            return redirect(url_for('bills'))
        
        return render_template('multiple_bills.html', bills=bills_data, app=app)
    except Exception as e:
        logger.error(f"Error in generate_multiple_bills: {e}")
        flash('An error occurred while processing multiple bill generation', 'error')
        return redirect(url_for('bills'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)