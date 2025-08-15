"""Medical billing processor with type safety."""

from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import logging
import google.generativeai as genai

from medical_billing_types import (
    BillData, CenterConfig, PDFOptions,
    BillingDataFrame, AIResponse
)

logger = logging.getLogger(__name__)

class BillingProcessor:
    """Process medical test billing data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.ai = AIHelper()
    
    def process_billing_data(
        self,
        main_data_path: str,
        supporting_data_path: Optional[str] = None
    ) -> List[BillData]:
        """Process billing data from Excel files."""
        try:
            main_df = BillingDataFrame.read_excel(main_data_path)
            bills: List[BillData] = []
            
            if supporting_data_path:
                supporting_df = BillingDataFrame.read_excel(supporting_data_path)
            else:
                supporting_df = None
            
            for center_name, center_data in main_df.groupby('CENTER NAME'):
                bill_data = self._process_center_billing(
                    center_name,
                    center_data,
                    supporting_df
                )
                bills.append(bill_data)
            
            logger.info(f"Successfully processed {len(bills)} center bills")
            return bills
            
        except Exception as e:
            error_msg = f"Failed to process billing data: {str(e)}"
            logger.error(error_msg)
            self.ai.handle_error(error_msg)
            raise

class AIHelper:
    """AI-powered error handling and assistance."""
    
    def __init__(self) -> None:
        genai.configure(api_key="AIzaSyCISRlocKiVnAlakm5GEllJu6VVnrBdP6s")
        self.model = genai.GenerativeModel('gemini-1.0-pro')
    
    def get_response(self, prompt: str) -> str:
        """Get AI response."""
        try:
            response = self.model.generate_content(prompt)
            return str(response.text)
        except Exception as e:
            logger.error(f"AI API call failed: {e}")
            return f"AI assistance unavailable: {str(e)}"
    
    def handle_error(self, error_msg: str, context: str = "") -> str:
        """Get AI help for error handling."""
        prompt = f"""
        Medical billing system error:
        Error: {error_msg}
        Context: {context or 'No context provided'}
        
        Please provide:
        1. Likely cause
        2. Solution steps
        3. Prevention tips
        
        Keep it concise and practical.
        """
        return self.get_response(prompt)
