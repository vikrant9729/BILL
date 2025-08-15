"""
Shared utilities for billing apps: amount conversion, invoice generation, AI integration, and type conversion.
"""
import logging
from datetime import datetime
import requests
import pandas as pd

class AmountToWords:
    """Convert numerical amounts to words for invoice generation."""
    def __init__(self):
        self.units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        self.teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        self.tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        self.scales = ["", "Thousand", "Lakh", "Crore"]
    def convert(self, amount: float) -> str:
        if amount == 0:
            return "Zero Rupees Only"
        rupees = int(amount)
        paise = int(round((amount - rupees) * 100))
        rupees_words = self._convert_rupees(rupees)
        paise_words = self._convert_paise(paise) if paise > 0 else ""
        if paise_words:
            return f"{rupees_words} and {paise_words} Only"
        else:
            return f"{rupees_words} Only"
    def _convert_rupees(self, rupees: int) -> str:
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
        if chunk == 0:
            return ""
        words = []
        if chunk >= 100:
            words.append(f"{self.units[chunk // 100]} Hundred")
            chunk %= 100
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
        if invoice_date is None:
            invoice_date = datetime.now()
        year = invoice_date.year
        month = invoice_date.month
        if year != self.current_year or month != self.current_month:
            self.sequence = 1
            self.current_year = year
            self.current_month = month
        year_range = f"{year-1}-{year}" if month < 4 else f"{year}-{year+1}"
        month_str = f"{month:02d}"
        sequence_str = f"{self.sequence:03d}"
        invoice_number = f"KRPL/{year_range}/{month_str}/{sequence_str}"
        self.sequence += 1
        if self.sequence > 999:
            logging.warning("Invoice sequence exceeded 999, resetting to 1")
            self.sequence = 1
        return invoice_number

class AIIntegration:
    """AI integration for error handling and user assistance."""
    def __init__(self):
        self.gemini_api_key = "AIzaSyCISRlocKiVnAlakm5GEllJu6VVnrBdP6s"
        self.openai_api_key = None
        self.gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    def get_ai_response(self, prompt: str, use_gemini: bool = True) -> str:
        try:
            if use_gemini and self.gemini_api_key:
                return self._call_gemini(prompt)
            else:
                return "AI assistance not available. Please check your API keys."
        except Exception as e:
            logging.error(f"AI API call failed: {e}")
            return f"AI assistance temporarily unavailable: {str(e)}"
    def _call_gemini(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(f"{self.gemini_url}?key={self.gemini_api_key}", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"Gemini API error: {response.status_code}")
    def handle_error(self, error_message: str, context: str = "") -> str:
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

def safe_float_conversion(value, default=0.0):
    try:
        if pd.isna(value) or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert {value} to float, using default {default}")
        return default

def safe_int_conversion(value, default=0):
    try:
        if pd.isna(value) or value == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        logging.warning(f"Could not convert {value} to int, using default {default}")
        return default

def safe_date_conversion(value):
    try:
        if pd.isna(value) or value == '':
            return 'N/A'
        if hasattr(value, 'strftime'):
            return value.strftime('%Y-%m-%d')
        return str(value)
    except Exception as e:
        logging.warning(f"Could not convert date {value}: {e}")
        return 'N/A'
