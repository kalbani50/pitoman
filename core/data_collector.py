"""
نظام جمع وتحليل بيانات السوق
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from sklearn.preprocessing import StandardScaler

class MarketDataCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = DataCache()
        self.min_volume = 1000000  # الحد الأدنى للحجم اليومي
        self.update_interval = 60  # ثانية
        
    async def collect_market_data(self, exchanges: List[str], 
                                trading_type: str) -> Dict:
        """جمع بيانات السوق"""
        all_data = {}
        
        for exchange in exchanges:
            # جمع البيانات الأساسية
            basic_data = await self._fetch_basic_data(exchange, trading_type)
            
            # تصفية العملات حسب الحجم
            filtered_data = self._filter_by_volume(basic_data)
            
            # جمع بيانات إضافية للعملات المختارة
            detailed_data = await self._fetch_detailed_data(
                exchange, filtered_data, trading_type
            )
            
            all_data[exchange] = detailed_data
            
        return all_data
        
    async def _fetch_basic_data(self, exchange: str, trading_type: str) -> Dict:
        """جلب البيانات الأساسية"""
        # التحقق من الكاش
        cache_key = f"{exchange}_{trading_type}_basic"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        # جلب البيانات من المنصة
        if trading_type == 'spot':
            data = await self._fetch_spot_data(exchange)
        else:
            data = await self._fetch_futures_data(exchange)
            
        # تحديث الكاش
        self.cache.set(cache_key, data, self.update_interval)
        
        return data
        
    def _filter_by_volume(self, data: Dict) -> Dict:
        """تصفية العملات حسب الحجم"""
        filtered = {}
        
        for symbol, info in data.items():
            if info['volume_24h'] >= self.min_volume:
                filtered[symbol] = info
                
        return filtered
        
    async def _fetch_detailed_data(self, exchange: str, basic_data: Dict,
                                 trading_type: str) -> Dict:
        """جلب البيانات التفصيلية"""
        detailed_data = {}
        
        for symbol in basic_data:
            # جلب بيانات الأوردر بوك
            orderbook = await self._fetch_orderbook(exchange, symbol)
            
            # جلب بيانات التداول الأخيرة
            trades = await self._fetch_recent_trades(exchange, symbol)
            
            # جلب البيانات التاريخية
            historical = await self._fetch_historical_data(
                exchange, symbol, trading_type
            )
            
            detailed_data[symbol] = {
                **basic_data[symbol],
                'orderbook': orderbook,
                'recent_trades': trades,
                'historical': historical,
                'analysis': self._analyze_symbol_data(
                    orderbook, trades, historical
                )
            }
            
        return detailed_data

class DataCache:
    def __init__(self):
        self.cache = {}
        
    def get(self, key: str) -> Optional[Dict]:
        """استرجاع البيانات من الكاش"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=60):
                return data
        return None
        
    def set(self, key: str, data: Dict, ttl: int):
        """تخزين البيانات في الكاش"""
        self.cache[key] = (data, datetime.now())
        
    def clear(self):
        """مسح الكاش"""
        self.cache.clear()

class MarketAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def analyze_market_data(self, data: Dict) -> Dict:
        """تحليل بيانات السوق"""
        analysis = {}
        
        for symbol, symbol_data in data.items():
            analysis[symbol] = {
                'volatility': self._calculate_volatility(symbol_data),
                'trend': self._analyze_trend(symbol_data),
                'liquidity': self._analyze_liquidity(symbol_data),
                'momentum': self._calculate_momentum(symbol_data),
                'risk_metrics': self._calculate_risk_metrics(symbol_data)
            }
            
        return analysis
        
    def _calculate_volatility(self, data: Dict) -> float:
        """حساب التقلب"""
        prices = data['historical']['close']
        returns = np.log(prices / prices.shift(1))
        return returns.std() * np.sqrt(252)
        
    def _analyze_trend(self, data: Dict) -> Dict:
        """تحليل الاتجاه"""
        df = pd.DataFrame(data['historical'])
        
        # حساب المتوسطات المتحركة
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # تحديد الاتجاه
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            trend = 'صعودي'
        elif current_price < sma_20 < sma_50:
            trend = 'هبوطي'
        else:
            trend = 'متذبذب'
            
        return {
            'direction': trend,
            'strength': self._calculate_trend_strength(df)
        }
        
    def _analyze_liquidity(self, data: Dict) -> Dict:
        """تحليل السيولة"""
        orderbook = data['orderbook']
        trades = data['recent_trades']
        
        return {
            'bid_ask_spread': self._calculate_spread(orderbook),
            'depth': self._calculate_market_depth(orderbook),
            'trade_frequency': self._calculate_trade_frequency(trades)
        }
