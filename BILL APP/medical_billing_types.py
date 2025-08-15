"""Type definitions for medical billing application."""

from typing import TypedDict, List, Dict, Union, Optional, Any
import pandas as pd
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

class BillData(TypedDict):
    patient_name: str
    patient_id: str
    tests: List[str]
    amounts: List[float]
    total: float
    date: str
    doctor: str
    center_type: str
    share_percentage: float

class CenterConfig(TypedDict):
    name: str
    type: str
    share_percentage: float
    address: str
    contact: str

class PDFOptions(TypedDict):
    page_size: tuple[float, float]
    margin: float
    font_name: str
    font_size: int

class BillingDataFrame:
    data: pd.DataFrame
    
    @staticmethod
    def read_excel(path: Union[str, Path]) -> 'BillingDataFrame':
        df = pd.read_excel(path)
        result = BillingDataFrame()
        result.data = df
        return result
    
    def groupby(self, column: str) -> pd.core.groupby.DataFrameGroupBy:
        return self.data.groupby(column)
    
    def fillna(self, value: Any) -> 'BillingDataFrame':
        self.data = self.data.fillna(value)
        return self
    
    def to_numeric(self, column: str) -> 'BillingDataFrame':
        self.data[column] = pd.to_numeric(self.data[column], errors='coerce').fillna(0)
        return self

class AIResponse(TypedDict):
    text: str
    model: str
    status: str

NumberType = Union[int, float]
OptionalStr = Optional[str]
JsonDict = Dict[str, Union[str, int, float, List[Union[str, int, float]], Dict[str, Union[str, int, float]]]]
GenerativeModel = genai.GenerativeModel
AIResponseType = GenerateContentResponse
DataFrame = pd.DataFrame
Series = pd.Series
