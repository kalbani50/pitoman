"""
نظام معالجة البيانات المتقدم
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def process_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """معالجة البيانات التاريخية"""
        data = self.clean_data(data)
        data = self.handle_missing_values(data)
        data = self.normalize_data(data)
        return data
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """تنظيف البيانات"""
        # إزالة القيم المتطرفة
        data = self.remove_outliers(data)
        # إزالة القيم المكررة
        data = data.drop_duplicates()
        # تنسيق التواريخ
        data = self.format_dates(data)
        return data
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المفقودة"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = self.imputer.fit_transform(data[numerical_cols])
        return data
        
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """تطبيع البيانات"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        return data
        
    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """إزالة القيم المتطرفة"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        return data
        
    def format_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """تنسيق التواريخ"""
        date_columns = data.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            data[col] = pd.to_datetime(data[col])
        return data

class FeatureEngineering:
    def __init__(self):
        self.features = {}
        
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """إنشاء المؤشرات الفنية"""
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['EMA_20'] = data['close'].ewm(span=20).mean()
        data['RSI'] = self.calculate_rsi(data['close'])
        data['MACD'] = self.calculate_macd(data['close'])
        return data
        
    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """حساب مؤشر RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """حساب مؤشر MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

class DataValidator:
    def __init__(self):
        self.validation_rules = {}
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """التحقق من صحة البيانات"""
        return all([
            self.check_data_types(data),
            self.check_value_ranges(data),
            self.check_data_consistency(data)
        ])
        
    def check_data_types(self, data: pd.DataFrame) -> bool:
        """التحقق من أنواع البيانات"""
        expected_types = {
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.int64,
            'date': 'datetime64[ns]'
        }
        return all(data[col].dtype == dtype for col, dtype in expected_types.items())
