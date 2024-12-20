"""
Advanced Market Analysis System using AGI
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MarketAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_models()
        
    def setup_models(self):
        """Initialize all AI models"""
        self.price_predictor = self._create_price_predictor()
        self.sentiment_analyzer = self._create_sentiment_analyzer()
        self.pattern_recognizer = self._create_pattern_recognizer()
        self.risk_evaluator = self._create_risk_evaluator()
        
    def _create_price_predictor(self) -> Sequential:
        """Create LSTM model for price prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(100, 5)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _create_sentiment_analyzer(self):
        """Create sentiment analysis model"""
        model_name = 'finbert-tone'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return AutoModel.from_pretrained(model_name)
        
    def _create_pattern_recognizer(self) -> nn.Module:
        """Create pattern recognition model"""
        class PatternNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(5, 32, 3)
                self.conv2 = nn.Conv1d(32, 64, 3)
                self.pool = nn.MaxPool1d(2)
                self.fc1 = nn.Linear(1920, 128)
                self.fc2 = nn.Linear(128, 10)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 1920)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
                
        return PatternNet()
        
    def _create_risk_evaluator(self) -> nn.Module:
        """Create risk evaluation model"""
        class RiskNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(5, 64, batch_first=True)
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 3)
                
            def forward(self, x):
                x, _ = self.lstm(x)
                x = x[:, -1, :]
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
                
        return RiskNet()
        
    async def analyze_market(self, market_data: Dict) -> Dict:
        """Comprehensive market analysis"""
        analysis = {}
        
        # Price prediction
        price_pred = await self._predict_prices(market_data)
        analysis['price_predictions'] = price_pred
        
        # Sentiment analysis
        sentiment = await self._analyze_sentiment(market_data)
        analysis['market_sentiment'] = sentiment
        
        # Pattern recognition
        patterns = await self._detect_patterns(market_data)
        analysis['detected_patterns'] = patterns
        
        # Risk evaluation
        risk = await self._evaluate_risk(market_data)
        analysis['risk_assessment'] = risk
        
        # Combined analysis
        analysis['trading_signals'] = self._generate_trading_signals(
            price_pred, sentiment, patterns, risk
        )
        
        return analysis
        
    async def _predict_prices(self, market_data: Dict) -> Dict:
        """Predict future price movements"""
        predictions = {}
        for symbol, data in market_data.items():
            # Prepare data
            X = self._prepare_price_data(data)
            
            # Make prediction
            pred = self.price_predictor.predict(X)
            predictions[symbol] = {
                'next_price': float(pred[0]),
                'confidence': self._calculate_prediction_confidence(pred, data)
            }
            
        return predictions
        
    async def _analyze_sentiment(self, market_data: Dict) -> Dict:
        """Analyze market sentiment"""
        sentiments = {}
        for symbol, data in market_data.items():
            # Get news and social media data
            texts = data.get('news', []) + data.get('social_media', [])
            
            if texts:
                # Tokenize and analyze
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
                outputs = self.sentiment_analyzer(**inputs)
                
                # Calculate sentiment scores
                sentiments[symbol] = {
                    'score': float(torch.mean(outputs.last_hidden_state[:, 0, :])),
                    'confidence': float(torch.std(outputs.last_hidden_state[:, 0, :]))
                }
                
        return sentiments
        
    async def _detect_patterns(self, market_data: Dict) -> Dict:
        """Detect technical patterns"""
        patterns = {}
        for symbol, data in market_data.items():
            # Prepare data
            X = self._prepare_pattern_data(data)
            
            # Detect patterns
            with torch.no_grad():
                pattern_scores = self.pattern_recognizer(X)
                patterns[symbol] = {
                    'patterns': self._decode_patterns(pattern_scores),
                    'strength': float(torch.max(pattern_scores))
                }
                
        return patterns
        
    async def _evaluate_risk(self, market_data: Dict) -> Dict:
        """Evaluate market risk"""
        risks = {}
        for symbol, data in market_data.items():
            # Prepare data
            X = self._prepare_risk_data(data)
            
            # Evaluate risk
            with torch.no_grad():
                risk_scores = self.risk_evaluator(X)
                risks[symbol] = {
                    'level': self._decode_risk_level(risk_scores),
                    'factors': self._analyze_risk_factors(data)
                }
                
        return risks
        
    def _generate_trading_signals(self, prices: Dict, sentiment: Dict,
                                patterns: Dict, risk: Dict) -> Dict:
        """Generate trading signals from all analyses"""
        signals = {}
        for symbol in prices.keys():
            # Combine all factors
            signal = self._combine_analysis_factors(
                prices.get(symbol, {}),
                sentiment.get(symbol, {}),
                patterns.get(symbol, {}),
                risk.get(symbol, {})
            )
            
            signals[symbol] = {
                'action': signal['action'],
                'confidence': signal['confidence'],
                'factors': signal['factors']
            }
            
        return signals
        
    def _combine_analysis_factors(self, price: Dict, sentiment: Dict,
                                pattern: Dict, risk: Dict) -> Dict:
        """Combine different analysis factors"""
        # Calculate combined signal
        factors = {
            'price_prediction': price.get('next_price', 0),
            'price_confidence': price.get('confidence', 0),
            'sentiment_score': sentiment.get('score', 0),
            'pattern_strength': pattern.get('strength', 0),
            'risk_level': risk.get('level', 'high')
        }
        
        # Weight and combine factors
        weights = self.config['ai']['signal_weights']
        combined_score = (
            weights['price'] * factors['price_confidence'] * 
            (factors['price_prediction'] - 1) +
            weights['sentiment'] * factors['sentiment_score'] +
            weights['pattern'] * factors['pattern_strength']
        ) * self._risk_multiplier(factors['risk_level'])
        
        # Determine action
        if combined_score > self.config['ai']['buy_threshold']:
            action = 'buy'
        elif combined_score < self.config['ai']['sell_threshold']:
            action = 'sell'
        else:
            action = 'hold'
            
        return {
            'action': action,
            'confidence': abs(combined_score),
            'factors': factors
        }
        
    def _risk_multiplier(self, risk_level: str) -> float:
        """Get risk multiplier based on risk level"""
        multipliers = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4
        }
        return multipliers.get(risk_level, 0.4)
