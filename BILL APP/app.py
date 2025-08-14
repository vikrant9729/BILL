from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
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

class AmountToWords:
    """Convert numerical amounts to words for invoice generation."""
    
    def __init__(self):
        self.units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        self.teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        self.tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        self.scales = ["", "Thousand", "Lakh", "Crore"]
    
    def convert(self, amount: float) -> str:
        """Convert amount to words in Indian numbering system."""
        if amount == 0:
            return "Zero Rupees Only"
        
        # Split into rupees and paise
        rupees = int(amount)
        paise = int(round((amount - rupees) * 100))
        
        rupees_words = self._convert_rupees(rupees)
        paise_words = self._convert_paise(paise) if paise > 0 else ""
        
        if paise_words:
            return f"{rupees_words} and {paise_words} Only"
        else:
            return f"{rupees_words} Only"
    
    def _convert_rupees(self, rupees: int) -> str:
        """Convert rupees to words."""
        if rupees == 0:
            return "Zero Rupees"
        
        words = []
        scale_index = 0
        
        while rupees > 0:
            chunk = rupees % 1000
            if chunk != 0:
                chunk_words = self._convert_chunk(chunk)
                if scale_index > 0:
                    chunk_words += f" {self.scales[scale_index]}"
                words.insert(0, chunk_words)
            
            rupees //= 1000
            scale_index += 1
        
        return " ".join(words) + " Rupees"
    
    def _convert_chunk(self, chunk: int) -> str:
        """Convert a chunk of up to 3 digits to words."""
        if chunk == 0:
            return ""
        
        words = []
        
        # Handle hundreds
        if chunk >= 100:
            words.append(f"{self.units[chunk // 100]} Hundred")
            chunk %= 100
        
        # Handle tens and units
        if chunk >= 20:
            words.append(self.tens[chunk // 10])
            if chunk % 10 > 0:
                words.append(self.units[chunk % 10])
        elif chunk >= 10:
            words.append(self.teens[chunk - 10])
        elif chunk > 0:
            words.append(self.units[chunk])
        
        return " ".join(words)
    
    def _convert_paise(self, paise: int) -> str:
        """Convert paise to words."""
        if paise == 0:
            return ""
        
        if paise < 20:
            return f"{self.units[paise]} Paise"
        else:
            tens = paise // 10
            units = paise % 10
            if units > 0:
                return f"{self.tens[tens]} {self.units[units]} Paise"
            else:
                return f"{self.tens[tens]} Paise"

class InvoiceNumberGenerator:
    """Generate sequential invoice numbers in the format KRPL/YY-YY/MM/NNN."""
    
    def __init__(self, start_sequence: int = 1):
        self.sequence = start_sequence
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
    
    def generate(self, invoice_date: datetime = None) -> str:
        """Generate invoice number for the given date."""
        if invoice_date is None:
            invoice_date = datetime.now()
        
        year = invoice_date.year
        month = invoice_date.month
        
        # Check if year or month has changed, reset sequence if needed
        if year != self.current_year or month != self.current_month:
            self.sequence = 1
            self.current_year = year
            self.current_month = month
        
        # Format: KRPL/YY-YY/MM/NNN
        year_range = f"{year-1}-{year}" if month < 4 else f"{year}-{year+1}"
        month_str = f"{month:02d}"
        sequence_str = f"{self.sequence:03d}"
        
        invoice_number = f"KRPL/{year_range}/{month_str}/{sequence_str}"
        
        # Increment sequence for next invoice
        self.sequence += 1
        
        # Validate sequence doesn't exceed 999
        if self.sequence > 999:
            logger.warning("Invoice sequence exceeded 999, resetting to 1")
            self.sequence = 1
        
        return invoice_number

class AIIntegration:
    """AI integration for error handling and user assistance."""
    
    def __init__(self):
        # Hardcode the Gemini API key directly
        self.gemini_api_key = "AIzaSyCISRlocKiVnAlakm5GEllJu6VVnrBdP6s"
        self.openai_api_key = None  # Disable OpenAI
        self.gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    
    def get_ai_response(self, prompt: str, use_gemini: bool = True) -> str:
        """Get AI response for error handling or user assistance."""
        try:
            if use_gemini and self.gemini_api_key:
                return self._call_gemini(prompt)
            else:
                return "AI assistance not available. Please check your API keys."
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            return f"AI assistance temporarily unavailable: {str(e)}"
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API via REST."""
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(
            f"{self.gemini_url}?key={self.gemini_api_key}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"Gemini API error: {response.status_code}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                timeout=30
            )
            
            return response.choices[0].message.content
        except ImportError:
            raise Exception("OpenAI library not installed. Run: pip install openai")
    
    def handle_error(self, error_message: str, context: str = "") -> str:
        """Get AI suggestions for error handling."""
        prompt = f"""
        I'm working with a medical billing automation system and encountered an error:
        
        Error: {error_message}
        Context: {context}
        
        Please provide:
        1. A brief explanation of what might be causing this error
        2. Step-by-step troubleshooting suggestions
        3. Any preventive measures to avoid this error in the future
        
        Keep the response concise and practical.
        """
        
        return self.get_ai_response(prompt)

# Initialize global objects
amount_converter = AmountToWords()
invoice_generator = InvoiceNumberGenerator()
ai_integration = AIIntegration()

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_excel_data(df):
    """Validate Excel data structure and required columns"""
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
    
    return True, None

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float with error handling"""
    try:
        if pd.isna(value) or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} to float, using default {default}")
        return default

def safe_int_conversion(value, default=0):
    """Safely convert value to int with error handling"""
    try:
        if pd.isna(value) or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} to int, using default {default}")
        return default

def safe_date_conversion(value):
    """Safely convert date value with error handling"""
    try:
        if pd.isna(value) or value == '':
            return 'N/A'
        if hasattr(value, 'strftime'):
            return value.strftime('%Y-%m-%d')
        return str(value)
    except Exception as e:
        logger.warning(f"Could not convert date {value}: {e}")
        return 'N/A'

def process_excel_file(file_path, sharing_percentage=None, center_type=None):
    """Process Excel file and return billing data grouped by Center Name"""
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
        
        # Group by CENTER NAME to create center-wise bills
        bills = []
        hlm_centers = get_hlm_centers()
        
        for centre_name, group in df.groupby('CENTER NAME'):
            if pd.isna(centre_name) or centre_name == '':
                continue
            # Determine center type
            is_hlm = centre_name in hlm_centers
            current_center_type = 'HLM' if is_hlm else 'B2B'
            # Process test items for this center
            test_items = []
            total_mrp = 0
            total_rate = 0
            total_sharing = 0
            for _, row in group.iterrows():
                try:
                    mrp = safe_float_conversion(row['MRP'])
                    rate = safe_float_conversion(row['CentreTestRate'])
                    # Calculate sharing amount based on center type
                    if is_hlm:
                        # For HLM: Calculate sharing based on percentage, then rate = MRP - sharing
                        if sharing_percentage:
                            try:
                                sharing_percentage_float = float(sharing_percentage)
                                if sharing_percentage_float < 0 or sharing_percentage_float > 100:
                                    sharing_percentage_float = 55.0  # Default to 55%
                            except (ValueError, TypeError):
                                sharing_percentage_float = 55.0  # Default to 55%
                            sharing_amount = mrp * (sharing_percentage_float / 100)
                            rate = mrp - sharing_amount
                        else:
                            # Default sharing calculation for HLM
                            sharing_amount = mrp * 0.55  # 55% default
                            rate = mrp - sharing_amount
                    else:
                        # For B2B: Sharing = MRP - Rate (existing logic)
                        sharing_amount = mrp - rate
                    test_item = {
                        'registered_date': safe_date_conversion(row['RegisteredDate']),
                        'visit_code': str(safe_int_conversion(row['PatientVisitCode'])),
                        'patient_name': str(row['PatientName']) if pd.notna(row['PatientName']) else 'N/A',
                        'test_name': str(row['TEST NAME']) if pd.notna(row['TEST NAME']) else 'N/A',
                        'mrp': mrp,
                        'rate': rate,
                        'sharing_amount': sharing_amount
                    }
                    test_items.append(test_item)
                    total_mrp += mrp
                    total_rate += rate
                    total_sharing += sharing_amount
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
            if test_items:  # Only create bill if there are test items
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
                    'center_type': current_center_type,
                    'amount_in_words': amount_converter.convert(total_rate)
                }
                bills.append(bill)
        if not bills:
            return None, "No valid bills could be generated from the data"
        return bills, None
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return None, f"Error processing file: {str(e)}"

def get_hlm_centers():
    """Get list of HLM centers"""
    hlm_centers = [
        'Manglam_Diagnostics_Agroha',
        'Hansi_Lab_MANGLAM',
        'JIND_MANGLAM_LAB_HISAR',
        'Narwana_Manglam_Lab',
        'Sanyam_Lab,_Inside_Satija_Healthcare,_H..',
        'Shri_Bala_JI_Lab_Isherwal,_Miran',
        'Vikash_Laboratory,_Java_Hospital,_Tosha..'
    ]
    return hlm_centers

def get_b2b_centers():
    """Get list of B2B centers (non-HLM)"""
    all_centers = [
        'Aarogya_Hospital_Hisar',
        'AMANDEEP_HOSPITAL',
        'AMARAVATI_HOSPITAL',
        'APEX_DIAGNOSTICS',
        'Barwala_Jansevarth_Lab,_Barwala',
        'CITY_CENTER_HISAR',
        'DR_ANKIT_GOYAL',
        'DR._RAJESH_MEHTA',
        'ECHS',
        'Elora_Dass_Gupta',
        'Fatehabad_Manglam_Diagnostices',
        'Geetanjali_Hospital',
        'GOBIND_NARSING_HOME',
        'GUPTA_NEWTON_HOSPITAL',
        'Guru_Jambheshwar_Multispeciality_Hosp....',
        'HISAR_DIAGNOSTICS_JHAJHPUL',
        'HISAR_GESTRO_HOSPITAL',
        'Hisar_Hospital_Nursery_Inside_Hsr_Hosp..',
        'HOLY_HELP_HOSPITAL',
        'INSURANCE,_HISAR',
        'JANKI_HOSPITAL',
        'LIFE_LINE_HOSPITAL',
        'MEYANSH_HOSPITAL',
        'Navjeevan_Hospital',
        'Onquest_Laboratories_Ltd..',
        'Pathkind_Diagnostics',
        'Ram_Niwas\'s_Centre',
        'Ravindra_Hospital',
        'RMCT_TOHANA',
        'SACHIN_MITTAL',
        'SADBHAVNA_HOSPITAL',
        'Sai_Hospital',
        'Sapra_Hospital,_Hisar',
        'SARVODYA_HOSPITAL',
        'SHANI_MANAV_SEVA_TRUST',
        'SHANTI_GI_HOSPITAL',
        'SHARDHA_HOSPITAL',
        'Shree_Krishna_Pranami_Multi_speciality_H..'
    ]
    hlm_centers = get_hlm_centers()
    b2b_centers = [center for center in all_centers if center not in hlm_centers]
    return b2b_centers

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
        
        # Get sharing percentage if provided
        sharing_percentage = request.form.get('sharing_percentage')
        center_type = request.form.get('center_type', 'B2B')
        
        # Validate sharing percentage if provided
        if sharing_percentage:
            try:
                sharing_float = float(sharing_percentage)
                if sharing_float < 0 or sharing_float > 100:
                    flash('Sharing percentage must be between 0 and 100', 'error')
                    return redirect(url_for('index'))
            except ValueError:
                flash('Invalid sharing percentage format', 'error')
                return redirect(url_for('index'))
        
        # Process the Excel file
        bills, error = process_excel_file(file_path, sharing_percentage, center_type)
        
        if error:
            # Get AI assistance for error handling
            ai_suggestion = ai_integration.handle_error(error, "File upload and processing")
            flash(f'Error processing file: {error}\n\nAI Suggestion:\n{ai_suggestion}', 'error')
            return redirect(url_for('index'))
        
        if not bills:
            flash('No bills could be generated from the uploaded file', 'error')
            return redirect(url_for('index'))
        
        # Store bills in session or temporary storage
        app.bills = bills
        
        flash(f'Successfully processed {len(bills)} bills from {filename}', 'success')
        return redirect(url_for('bills'))
    
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        ai_suggestion = ai_integration.handle_error(str(e), "File upload process")
        flash(f'An unexpected error occurred while processing the file\n\nAI Suggestion:\n{ai_suggestion}', 'error')
        return redirect(url_for('index'))

@app.route('/generate_all_bills')
def generate_all_bills():
    """Generate bills for all centers"""
    try:
        if not hasattr(app, 'bills') or not app.bills:
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
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            center_name = request.form.get('center_name')
            if not center_name:
                flash('Please select a center', 'error')
                return redirect(url_for('generate_manual_bill'))
            
            # Find the selected bill
            selected_bill = None
            for bill in app.bills:
                if bill['centre_name'] == center_name:
                    selected_bill = bill
                    break
            
            if selected_bill:
                # Create a new bills list with only the selected bill
                app.bills = [selected_bill]
                flash(f'Generated bill for {center_name}', 'success')
                return redirect(url_for('bills'))
            else:
                flash('Selected center not found', 'error')
                return redirect(url_for('generate_manual_bill'))
        
        return render_template('manual_bill.html', bills=app.bills, app=app)
    except Exception as e:
        logger.error(f"Error in generate_manual_bill: {e}")
        flash('An error occurred while processing manual bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_multiple_bills', methods=['GET', 'POST'])
def generate_multiple_bills():
    """Show multiple bill generation page or process multiple bill generation"""
    try:
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            selected_centers = request.form.getlist('selected_centers')
            if not selected_centers:
                flash('Please select at least one center', 'error')
                return redirect(url_for('generate_multiple_bills'))
            
            # Filter bills for selected centers
            filtered_bills = [bill for bill in app.bills if bill['centre_name'] in selected_centers]
            
            if not filtered_bills:
                flash('No bills found for selected centers', 'error')
                return redirect(url_for('generate_multiple_bills'))
            
            app.bills = filtered_bills
            flash(f'Generated {len(filtered_bills)} bills for selected centers', 'success')
            return redirect(url_for('bills'))
        
        return render_template('multiple_bills.html', bills=app.bills, app=app)
    except Exception as e:
        logger.error(f"Error in generate_multiple_bills: {e}")
        flash('An error occurred while processing multiple bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_hlm_bills', methods=['GET', 'POST'])
def generate_hlm_bills():
    """Generate bills for HLM centers with sharing percentage"""
    try:
        if request.method == 'POST':
            sharing_percentage = request.form.get('sharing_percentage')
            sharing_type = request.form.get('sharing_type', 'overall')
            
            if not sharing_percentage:
                flash('Please provide sharing percentage for HLM bills', 'error')
                return redirect(url_for('generate_hlm_bills'))
            
            try:
                sharing_percentage = float(sharing_percentage)
                if sharing_percentage < 0 or sharing_percentage > 100:
                    flash('Sharing percentage must be between 0 and 100', 'error')
                    return redirect(url_for('generate_hlm_bills'))
            except ValueError:
                flash('Invalid sharing percentage', 'error')
                return redirect(url_for('generate_hlm_bills'))
            
            # Get test-type specific sharing percentages if provided
            test_type_sharing = {}
            if sharing_type == 'test_type':
                try:
                    test_type_sharing = {
                        'blood_test_sharing': float(request.form.get('blood_test_sharing', sharing_percentage)),
                        'urine_test_sharing': float(request.form.get('urine_test_sharing', sharing_percentage)),
                        'xray_test_sharing': float(request.form.get('xray_test_sharing', sharing_percentage)),
                        'other_test_sharing': float(request.form.get('other_test_sharing', sharing_percentage))
                    }
                    
                    # Validate test-type sharing percentages
                    for key, value in test_type_sharing.items():
                        if value < 0 or value > 100:
                            flash(f'{key.replace("_", " ").title()} sharing percentage must be between 0 and 100', 'error')
                            return redirect(url_for('generate_hlm_bills'))
                except ValueError:
                    flash('Invalid test-type sharing percentage format', 'error')
                    return redirect(url_for('generate_hlm_bills'))
            
            # Get HLM centers
            hlm_centers = get_hlm_centers()
            
            if not hasattr(app, 'bills') or not app.bills:
                flash('No bills available. Please upload an Excel file first.', 'error')
                return redirect(url_for('index'))
            
            # Filter bills for HLM centers and apply sharing percentage
            hlm_bills = []
            for bill in app.bills:
                if bill['centre_name'] in hlm_centers:
                    try:
                        # Apply sharing percentage to each test item
                        for item in bill['test_items']:
                            mrp = item['mrp']
                            
                            if sharing_type == 'overall':
                                # Apply overall sharing percentage
                                sharing_amount = mrp * (sharing_percentage / 100)
                                item['rate'] = mrp - sharing_amount
                                item['sharing_amount'] = sharing_amount
                            else:
                                # Apply test-type specific sharing
                                test_name = item['test_name'].lower()
                                if 'blood' in test_name:
                                    test_sharing = test_type_sharing.get('blood_test_sharing', sharing_percentage)
                                elif 'urine' in test_name:
                                    test_sharing = test_type_sharing.get('urine_test_sharing', sharing_percentage)
                                elif 'x-ray' in test_name or 'xray' in test_name:
                                    test_sharing = test_type_sharing.get('xray_test_sharing', sharing_percentage)
                                else:
                                    test_sharing = test_type_sharing.get('other_test_sharing', sharing_percentage)
                                
                                sharing_amount = mrp * (test_sharing / 100)
                                item['rate'] = mrp - sharing_amount
                                item['sharing_amount'] = sharing_amount
                        
                        # Recalculate totals
                        bill['total_rate'] = sum(item['rate'] for item in bill['test_items'])
                        bill['total_sharing'] = sum(item['sharing_amount'] for item in bill['test_items'])
                        bill['center_type'] = 'HLM'
                        bill['amount_in_words'] = amount_converter.convert(bill['total_rate'])
                        hlm_bills.append(bill)
                    except Exception as e:
                        logger.error(f"Error processing HLM bill {bill['centre_name']}: {e}")
                        continue
            
            if not hlm_bills:
                flash('No HLM bills found in the uploaded data', 'error')
                return redirect(url_for('generate_hlm_bills'))
            
            app.bills = hlm_bills
            flash(f'Generated {len(hlm_bills)} HLM bills with {sharing_percentage}% sharing', 'success')
            return redirect(url_for('bills'))
        
        return render_template('hlm_bills.html', app=app)
    except Exception as e:
        logger.error(f"Error in generate_hlm_bills: {e}")
        flash('An error occurred while processing HLM bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/generate_b2b_bills')
def generate_b2b_bills():
    """Generate bills for B2B centers"""
    try:
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        b2b_centers = get_b2b_centers()
        # Filter bills for B2B centers only
        b2b_bills = [bill for bill in app.bills if bill['centre_name'] in b2b_centers]
        
        if not b2b_bills:
            flash('No B2B bills found in the uploaded data', 'error')
            return redirect(url_for('bills'))
        
        app.bills = b2b_bills
        flash(f'Generated {len(b2b_bills)} B2B bills', 'success')
        return redirect(url_for('bills'))
    except Exception as e:
        logger.error(f"Error in generate_b2b_bills: {e}")
        flash('An error occurred while processing B2B bill generation', 'error')
        return redirect(url_for('bills'))

@app.route('/bills')
def bills():
    try:
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available. Please upload an Excel file first.', 'error')
            return redirect(url_for('index'))
        
        # Calculate total tests and amounts
        total_tests = sum(len(bill['test_items']) for bill in app.bills)
        total_mrp = sum(bill['total_mrp'] for bill in app.bills)
        total_rate = sum(bill['total_rate'] for bill in app.bills)
        total_sharing = sum(bill['total_sharing'] for bill in app.bills)
        
        return render_template('bills.html', 
                             bills=app.bills, 
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
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        
        if bill_index < 0 or bill_index >= len(app.bills):
            flash('Bill not found', 'error')
            return redirect(url_for('bills'))
        
        bill = app.bills[bill_index]
        return render_template('bill_detail.html', bill=bill, bill_index=bill_index, app=app)
    except Exception as e:
        logger.error(f"Error in view_bill: {e}")
        flash('An error occurred while viewing the bill', 'error')
        return redirect(url_for('bills'))

@app.route('/download_bill/<int:bill_index>')
def download_bill(bill_index):
    try:
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        if bill_index < 0 or bill_index >= len(app.bills):
            flash('Bill not found', 'error')
            return redirect(url_for('bills'))
        bill = app.bills[bill_index]
        fmt = request.args.get('format', 'html').lower()
        if fmt == 'excel':
            filename = f"{bill['bill_number']}.xlsx"
            filepath = os.path.join('bills', 'excel', filename)
            if not os.path.exists(filepath):
                try:
                    from medical_billing_app import MedicalBillingProcessor
                    processor = MedicalBillingProcessor()
                    processor.generate_excel_bill(bill)
                except Exception as e:
                    logger.error(f"Error generating Excel for bill: {e}")
                    flash('Could not generate Excel file for this bill', 'error')
                    return redirect(url_for('view_bill', bill_index=bill_index))
            return send_file(filepath, as_attachment=True, download_name=filename)
        elif fmt == 'pdf':
            filename = f"{bill['bill_number']}.pdf"
            filepath = os.path.join('bills', 'pdf', filename)
            if not os.path.exists(filepath):
                try:
                    from medical_billing_app import MedicalBillingProcessor
                    processor = MedicalBillingProcessor()
                    processor.generate_pdf_bill(bill)
                except Exception as e:
                    logger.error(f"Error generating PDF for bill: {e}")
                    flash('Could not generate PDF file for this bill', 'error')
                    return redirect(url_for('view_bill', bill_index=bill_index))
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
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
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available', 'error')
            return redirect(url_for('index'))
        
        # Create a ZIP file containing all bills
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for i, bill in enumerate(app.bills):
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
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for bill in app.bills:
                filename = f"{bill['bill_number']}.xlsx"
                filepath = os.path.join('bills', 'excel', filename)
                if os.path.exists(filepath):
                    zf.write(filepath, arcname=filename)
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
        if not hasattr(app, 'bills') or not app.bills:
            flash('No bills available', 'error')
            return redirect(url_for('bills'))
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for bill in app.bills:
                filename = f"{bill['bill_number']}.pdf"
                filepath = os.path.join('bills', 'pdf', filename)
                if os.path.exists(filepath):
                    zf.write(filepath, arcname=filename)
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
        if not hasattr(app, 'bills') or not app.bills:
            return jsonify({'error': 'No bills available'}), 404
        
        return jsonify(app.bills)
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