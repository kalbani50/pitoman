import unittest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_manager import StrategyManager

class TestTradingStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'strategy_params': {
                'trend_following': {
                    'ma_fast': 10,
                    'ma_slow': 30,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                },
                'mean_reversion': {
                    'lookback_period': 20,
                    'std_dev_threshold': 2.0,
                    'max_holding_period': 48  # hours
                }
            },
            'risk_params': {
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.05
            }
        }
        self.strategy_manager = StrategyManager(self.config)

    def test_strategy_selection(self):
        """Test strategy selection based on market conditions"""
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'close': np.random.normal(42000, 1000, 100),
            'volume': np.random.normal(1000, 100, 100),
            'high': np.random.normal(42500, 1000, 100),
            'low': np.random.normal(41500, 1000, 100)
        })
        
        market_conditions = self.strategy_manager.analyze_market_conditions(market_data)
        strategy = self.strategy_manager.select_strategy(market_conditions)
        
        # Verify strategy selection
        self.assertIsNotNone(strategy)
        self.assertTrue(hasattr(strategy, 'generate_signals'))
        self.assertTrue(hasattr(strategy, 'calculate_indicators'))

    def test_signal_generation(self):
        """Test signal generation with market data"""
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'close': np.random.normal(42000, 1000, 100),
            'volume': np.random.normal(1000, 100, 100),
            'high': np.random.normal(42500, 1000, 100),
            'low': np.random.normal(41500, 1000, 100)
        })
        
        strategy = self.strategy_manager.get_strategy('trend_following')
        signals = strategy.generate_signals(market_data)
        
        # Verify signals
        self.assertIsInstance(signals, dict)
        self.assertTrue('action' in signals)
        self.assertTrue('confidence' in signals)
        self.assertTrue('entry_price' in signals)
        self.assertTrue('stop_loss' in signals)
        self.assertTrue('take_profit' in signals)

    def test_indicator_calculation(self):
        """Test technical indicator calculations"""
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'close': np.random.normal(42000, 1000, 100),
            'volume': np.random.normal(1000, 100, 100),
            'high': np.random.normal(42500, 1000, 100),
            'low': np.random.normal(41500, 1000, 100)
        })
        
        strategy = self.strategy_manager.get_strategy('mean_reversion')
        indicators = strategy.calculate_indicators(market_data)
        
        # Verify indicators
        self.assertIsInstance(indicators, dict)
        self.assertTrue('ma' in indicators)
        self.assertTrue('std_dev' in indicators)
        self.assertTrue(len(indicators['ma']) == len(market_data))

if __name__ == '__main__':
    unittest.main()
