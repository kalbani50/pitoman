"""
نظام المقارنة الذكي بين منصات التداول
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging

class ExchangeComparator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.history = {}
        
    async def compare_exchanges(self, symbol: str, order_size: float) -> Dict:
        """مقارنة شاملة بين المنصات"""
        comparison = {
            'binance': await self._analyze_exchange('binance', symbol, order_size),
            'okx': await self._analyze_exchange('okx', symbol, order_size)
        }
        
        return {
            'best_overall': self._find_best_overall(comparison),
            'best_by_metric': self._find_best_by_metric(comparison),
            'detailed_comparison': comparison
        }
        
    async def _analyze_exchange(self, exchange: str, symbol: str, order_size: float) -> Dict:
        """تحليل منصة محددة"""
        return {
            'execution': await self._analyze_execution(exchange, symbol),
            'costs': self._analyze_costs(exchange, order_size),
            'reliability': self._analyze_reliability(exchange),
            'features': self._analyze_features(exchange)
        }
        
    def _analyze_costs(self, exchange: str, order_size: float) -> Dict:
        """تحليل التكاليف"""
        fee_structures = {
            'binance': {
                'spot': {'maker': 0.001, 'taker': 0.001},
                'futures': {'maker': 0.0002, 'taker': 0.0004},
                'withdrawal': {'BTC': 0.0005, 'ETH': 0.005, 'USDT': 1}
            },
            'okx': {
                'spot': {'maker': 0.0008, 'taker': 0.001},
                'futures': {'maker': 0.0002, 'taker': 0.0005},
                'withdrawal': {'BTC': 0.0004, 'ETH': 0.004, 'USDT': 1}
            }
        }
        
        exchange_fees = fee_structures[exchange]
        
        return {
            'trading_fees': {
                'spot': self._calculate_trading_fees(order_size, exchange_fees['spot']),
                'futures': self._calculate_trading_fees(order_size, exchange_fees['futures'])
            },
            'withdrawal_fees': exchange_fees['withdrawal'],
            'total_cost': self._calculate_total_cost(order_size, exchange_fees)
        }
        
    def _analyze_reliability(self, exchange: str) -> Dict:
        """تحليل موثوقية المنصة"""
        metrics = {
            'binance': {
                'uptime': 0.9995,
                'api_stability': 0.998,
                'order_execution': 0.997
            },
            'okx': {
                'uptime': 0.9993,
                'api_stability': 0.997,
                'order_execution': 0.996
            }
        }
        
        return {
            'metrics': metrics[exchange],
            'score': np.mean(list(metrics[exchange].values()))
        }
        
    def _analyze_features(self, exchange: str) -> Dict:
        """تحليل الميزات"""
        features = {
            'binance': {
                'spot_trading': True,
                'futures_trading': True,
                'margin_trading': True,
                'lending': True,
                'staking': True,
                'api_features': ['REST', 'WebSocket', 'FIX'],
                'order_types': ['market', 'limit', 'stop', 'oco']
            },
            'okx': {
                'spot_trading': True,
                'futures_trading': True,
                'margin_trading': True,
                'lending': True,
                'staking': True,
                'api_features': ['REST', 'WebSocket'],
                'order_types': ['market', 'limit', 'stop', 'oco']
            }
        }
        
        return {
            'available_features': features[exchange],
            'feature_score': self._calculate_feature_score(features[exchange])
        }
        
    def _calculate_trading_fees(self, order_size: float, fee_structure: Dict) -> Dict:
        """حساب رسوم التداول"""
        maker_fee = order_size * fee_structure['maker']
        taker_fee = order_size * fee_structure['taker']
        
        return {
            'maker_fee': maker_fee,
            'taker_fee': taker_fee,
            'average_fee': (maker_fee + taker_fee) / 2
        }
        
    def _calculate_total_cost(self, order_size: float, fee_structure: Dict) -> float:
        """حساب التكلفة الإجمالية"""
        trading_fees = self._calculate_trading_fees(order_size, fee_structure['spot'])
        withdrawal_fee = fee_structure['withdrawal']['USDT']  # افتراض USDT
        
        return trading_fees['average_fee'] + withdrawal_fee
        
    def _find_best_overall(self, comparison: Dict) -> Dict:
        """تحديد أفضل منصة إجمالاً"""
        scores = {}
        
        for exchange, metrics in comparison.items():
            scores[exchange] = (
                metrics['execution']['score'] * 0.3 +
                (1 - metrics['costs']['total_cost']) * 0.3 +
                metrics['reliability']['score'] * 0.2 +
                metrics['features']['feature_score'] * 0.2
            )
            
        best_exchange = max(scores.items(), key=lambda x: x[1])
        return {
            'exchange': best_exchange[0],
            'score': best_exchange[1],
            'reasons': self._get_selection_reasons(comparison, best_exchange[0])
        }
        
    def _find_best_by_metric(self, comparison: Dict) -> Dict:
        """تحديد أفضل منصة لكل معيار"""
        return {
            'execution': min(comparison.items(), key=lambda x: x[1]['execution']['latency'])[0],
            'costs': min(comparison.items(), key=lambda x: x[1]['costs']['total_cost'])[0],
            'reliability': max(comparison.items(), key=lambda x: x[1]['reliability']['score'])[0],
            'features': max(comparison.items(), key=lambda x: x[1]['features']['feature_score'])[0]
        }
        
    def _get_selection_reasons(self, comparison: Dict, selected_exchange: str) -> List[str]:
        """تحديد أسباب اختيار المنصة"""
        reasons = []
        exchange_data = comparison[selected_exchange]
        
        if exchange_data['execution']['score'] > 0.9:
            reasons.append("سرعة تنفيذ ممتازة")
            
        if exchange_data['costs']['total_cost'] < 0.001:
            reasons.append("رسوم منخفضة")
            
        if exchange_data['reliability']['score'] > 0.99:
            reasons.append("موثوقية عالية")
            
        if exchange_data['features']['feature_score'] > 0.9:
            reasons.append("ميزات متقدمة")
            
        return reasons
