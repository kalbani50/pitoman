import unittest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk_limits': {
                'max_position_size': 0.1,  # 10% of portfolio
                'max_total_exposure': 0.5,  # 50% of portfolio
                'stop_loss_threshold': 0.02  # 2% stop loss
            },
            'portfolio_params': {
                'initial_balance': 100000,
                'base_currency': 'USDT'
            }
        }
        self.risk_manager = RiskManager(self.config)

    def test_position_sizing(self):
        """Test position sizing calculations"""
        portfolio_value = 100000
        risk_params = {
            'risk_per_trade': 0.02,  # 2%
            'entry_price': 42000,
            'stop_loss': 41000,
            'leverage': 1
        }
        
        position_size = self.risk_manager.calculate_position_size(
            portfolio_value,
            risk_params
        )
        
        # Verify position sizing
        self.assertIsInstance(position_size, float)
        self.assertTrue(position_size > 0)
        
        # Verify risk amount
        risk_amount = (risk_params['entry_price'] - risk_params['stop_loss']) * position_size
        self.assertLessEqual(risk_amount, portfolio_value * risk_params['risk_per_trade'])

    def test_risk_exposure(self):
        """Test risk exposure calculations"""
        positions = [
            {
                'symbol': 'BTC/USDT',
                'size': 0.5,
                'entry_price': 42000,
                'current_price': 42500,
                'unrealized_pnl': 250
            },
            {
                'symbol': 'ETH/USDT',
                'size': 5,
                'entry_price': 3000,
                'current_price': 3100,
                'unrealized_pnl': 500
            }
        ]
        
        exposure = self.risk_manager.calculate_exposure(positions)
        
        # Verify exposure calculations
        self.assertIsInstance(exposure, dict)
        self.assertTrue('total_exposure' in exposure)
        self.assertTrue('exposure_ratio' in exposure)
        self.assertTrue(0 <= exposure['exposure_ratio'] <= 1)

    def test_risk_metrics(self):
        """Test risk metrics calculations"""
        trade_history = [
            {'pnl': 100, 'duration': 3600},
            {'pnl': -50, 'duration': 7200},
            {'pnl': 200, 'duration': 1800},
            {'pnl': -30, 'duration': 5400}
        ]
        
        metrics = self.risk_manager.calculate_risk_metrics(trade_history)
        
        # Verify risk metrics
        self.assertIsInstance(metrics, dict)
        self.assertTrue('sharpe_ratio' in metrics)
        self.assertTrue('max_drawdown' in metrics)
        self.assertTrue('win_rate' in metrics)
        self.assertTrue(0 <= metrics['win_rate'] <= 1)

if __name__ == '__main__':
    unittest.main()
