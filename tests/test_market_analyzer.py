import unittest
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.models.market_analyzer import MarketAnalyzer

class TestMarketAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = {
            'model_params': {
                'lstm_units': 128,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            },
            'training_params': {
                'batch_size': 32,
                'epochs': 100
            }
        }
        self.analyzer = MarketAnalyzer(self.config)

    def test_model_initialization(self):
        """Test if all models are initialized correctly"""
        self.assertIsNotNone(self.analyzer.price_predictor)
        self.assertIsNotNone(self.analyzer.sentiment_analyzer)
        self.assertIsNotNone(self.analyzer.pattern_recognizer)
        self.assertIsNotNone(self.analyzer.risk_evaluator)

    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        # Test market data preprocessing
        market_data = {
            'BTC/USDT': pd.DataFrame({
                'close': [40000, 41000, 42000, 41500, 42500],
                'volume': [100, 150, 200, 180, 220],
                'timestamp': [1, 2, 3, 4, 5]
            })
        }
        
        processed_data = self.analyzer.preprocess_data(market_data)
        
        # Verify preprocessing results
        self.assertIsNotNone(processed_data)
        self.assertTrue(isinstance(processed_data, np.ndarray))
        self.assertEqual(processed_data.shape[1], 5)  # Number of features

    def test_prediction_shape(self):
        """Test if predictions have correct shape"""
        input_data = np.random.rand(1, 100, 5)  # Batch size 1, 100 time steps, 5 features
        prediction = self.analyzer.predict_price(input_data)
        
        # Verify prediction shape
        self.assertEqual(prediction.shape, (1, 1))  # Single prediction value

if __name__ == '__main__':
    unittest.main()
