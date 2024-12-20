"""
Advanced Market Analysis System with Enhanced Learning and Multi-Model Integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from lightgbm import LGBMClassifier

class ModelEnsemble:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'gb': GradientBoostingClassifier(n_estimators=100),
            'xgb': xgb.XGBClassifier(n_estimators=100),
            'lgbm': LGBMClassifier(n_estimators=100)
        }
        self.weights = {model: 0.25 for model in self.models}
        self.is_trained = {model: False for model in self.models}
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train all models in the ensemble"""
        for name, model in self.models.items():
            model.fit(X, y)
            self.is_trained[name] = True
            
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make weighted ensemble prediction"""
        predictions = []
        confidences = []
        
        for name, model in self.models.items():
            if self.is_trained[name]:
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba[:, 1])
                confidences.append(self.weights[name])
                
        if not predictions:
            return np.array([0.5]), 0.0
            
        weighted_pred = np.average(predictions, weights=confidences, axis=0)
        confidence = np.sum(confidences)
        
        return weighted_pred, confidence
        
    def update_weights(self, performances: Dict[str, float]):
        """Update model weights based on performance"""
        total_perf = sum(performances.values())
        if total_perf > 0:
            self.weights = {
                model: perf / total_perf
                for model, perf in performances.items()
            }

class AdvancedLearningSystem:
    """Enhanced learning system with multiple models and strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.experiences = deque(maxlen=config.get('max_experiences', 100000))
        self.models = {
            'short_term': ModelEnsemble(config),
            'medium_term': ModelEnsemble(config),
            'long_term': ModelEnsemble(config)
        }
        self.feature_importance = {}
        self.performance_metrics = {}
        self.market_regime = 'normal'
        self.regime_detector = self._create_regime_detector()
        
    def _create_regime_detector(self):
        """Create market regime detection model"""
        return GradientBoostingClassifier(n_estimators=50)
        
    def detect_market_regime(self, state: Dict) -> str:
        """Detect current market regime"""
        features = self._extract_regime_features(state)
        prediction = self.regime_detector.predict([features])[0]
        regimes = ['normal', 'volatile', 'trending', 'ranging']
        return regimes[prediction]
        
    def _extract_regime_features(self, state: Dict) -> List[float]:
        """Extract features for regime detection"""
        market = state.get('market', {})
        return [
            market.get('volatility', 0),
            market.get('trend_strength', 0),
            market.get('volume_profile', 0),
            market.get('price_momentum', 0),
            market.get('market_depth', 0)
        ]
        
    def add_experience(self, state: Dict, action: Dict,
                      reward: float, next_state: Dict,
                      metadata: Dict = None):
        """Add trading experience with enhanced metadata"""
        # Detect market regime
        regime = self.detect_market_regime(state)
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': datetime.utcnow(),
            'market_conditions': self._extract_market_conditions(state),
            'success_factors': self._analyze_success_factors(state, reward),
            'market_regime': regime,
            'metadata': metadata or {}
        }
        
        self.experiences.append(experience)
        self._update_metrics(experience)
        
    def train(self, force: bool = False) -> bool:
        """Train all models with different time horizons"""
        if len(self.experiences) < self.config.get('min_experiences', 1000) and not force:
            return False
            
        try:
            for timeframe, ensemble in self.models.items():
                X, y = self._prepare_training_data(timeframe)
                if len(X) < 100:
                    continue
                    
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train ensemble
                ensemble.train(X_train, y_train)
                
                # Evaluate models
                performances = {}
                for name, model in ensemble.models.items():
                    y_pred = model.predict(X_val)
                    performances[name] = f1_score(y_val, y_pred)
                    
                # Update ensemble weights
                ensemble.update_weights(performances)
                
                # Store metrics
                self.performance_metrics[timeframe] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred),
                    'recall': recall_score(y_val, y_pred),
                    'f1': f1_score(y_val, y_pred),
                    'classification_report': classification_report(y_val, y_pred),
                    'confusion_matrix': confusion_matrix(y_val, y_pred)
                }
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
            
    def predict(self, state: Dict, timeframe: str = 'short_term') -> Dict:
        """Make predictions with confidence scores"""
        try:
            features = self._extract_features(state, timeframe)
            ensemble = self.models[timeframe]
            
            # Get ensemble prediction
            prob, confidence = ensemble.predict(np.array([features]))
            
            # Get important factors
            factors = self._get_important_factors(features, timeframe)
            
            # Get market regime
            regime = self.detect_market_regime(state)
            
            return {
                'probability': float(prob[0]),
                'confidence': float(confidence),
                'important_factors': factors,
                'market_regime': regime
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {
                'probability': 0.5,
                'confidence': 0.0,
                'important_factors': [],
                'market_regime': 'unknown'
            }

class MarketAnalyzer:
    """Advanced market analysis using multiple AI models and adaptive learning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.models = {}
        self.scalers = {}
        self.sentiment_analyzer = None
        self.pattern_recognizer = None
        
        # Enhanced caching system
        self.analysis_cache = {}
        self.last_update = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes default
        
        # Advanced learning system
        self.learning_system = AdvancedLearningSystem(config)
        
    async def analyze_markets(self, market_data: Dict) -> Dict:
        """Perform comprehensive market analysis with enhanced learning"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(market_data)
            if self._is_cache_valid(cache_key):
                return self.analysis_cache[cache_key]
                
            # Get base analysis
            analysis = await self._get_base_analysis(market_data)
            
            # Analyze different timeframes
            for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
                timeframe_analysis = await self._analyze_timeframe(
                    market_data, timeframe
                )
                analysis[f'timeframe_{timeframe}'] = timeframe_analysis
                
            # Get market regime
            analysis['market_regime'] = self.learning_system.detect_market_regime(
                {'market': market_data}
            )
            
            # Adjust signals based on predictions
            self._adjust_signals_with_predictions(analysis)
            
            # Cache results
            self._cache_analysis(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return {}
            
    def _generate_cache_key(self, market_data: Dict) -> str:
        """Generate cache key from market data"""
        return f"{market_data.get('symbol')}:{int(time.time() / self.cache_ttl)}"
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached analysis is still valid"""
        if cache_key not in self.analysis_cache:
            return False
            
        last_update = self.last_update.get(cache_key, 0)
        return time.time() - last_update < self.cache_ttl
        
    def _cache_analysis(self, cache_key: str, analysis: Dict):
        """Cache analysis results"""
        self.analysis_cache[cache_key] = analysis
        self.last_update[cache_key] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, update_time in self.last_update.items()
            if current_time - update_time > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.analysis_cache[key]
            del self.last_update[key]

    async def load_models(self) -> bool:
        """Load all required AI models"""
        try:
            # Load price prediction model
            self.models['price'] = self._load_price_model()
            
            # Load sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            
            # Load pattern recognition model
            self.models['pattern'] = self._load_pattern_model()
            
            # Initialize scalers
            self.scalers['price'] = StandardScaler()
            self.scalers['volume'] = StandardScaler()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
            
    async def _get_base_analysis(self) -> Dict:
        """Get base market analysis"""
        analysis = {
            'timestamp': datetime.utcnow(),
            'markets': {},
            'global': {}
        }
        
        # Analyze each timeframe
        for timeframe in self.config['timeframes']:
            analysis['markets'][timeframe] = await self._analyze_timeframe(timeframe)
            
        # Global market analysis
        analysis['global'] = await self._analyze_global_market()
        
        return analysis
        
    async def _analyze_timeframe(self, timeframe: str) -> Dict:
        """Analyze market data for a specific timeframe"""
        analysis = {}
        
        try:
            # Get market data
            data = await self._get_market_data(timeframe)
            
            # Price prediction
            analysis['price_prediction'] = await self._predict_price(data)
            
            # Technical analysis
            analysis['technical'] = await self._perform_technical_analysis(data)
            
            # Pattern recognition
            analysis['patterns'] = await self._recognize_patterns(data)
            
            # Volume analysis
            analysis['volume'] = await self._analyze_volume(data)
            
            # Sentiment analysis
            analysis['sentiment'] = await self._analyze_sentiment(timeframe)
            
            # Calculate combined signals
            analysis['signals'] = self._calculate_signals(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            return {}
            
    async def _analyze_global_market(self) -> Dict:
        """Analyze global market conditions"""
        try:
            analysis = {
                'market_state': await self._determine_market_state(),
                'correlation_matrix': await self._calculate_correlations(),
                'volatility_index': await self._calculate_volatility_index(),
                'market_sentiment': await self._analyze_global_sentiment(),
                'trend_strength': await self._calculate_trend_strength()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in global market analysis: {str(e)}")
            return {}
            
    async def _predict_price(self, data: pd.DataFrame) -> Dict:
        """Predict future price movements"""
        try:
            # Prepare features
            features = self._prepare_features(data)
            
            # Scale features
            scaled_features = self.scalers['price'].transform(features)
            
            # Make predictions
            predictions = self.models['price'].predict(scaled_features)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(predictions)
            
            return {
                'direction': np.sign(predictions[-1]),
                'magnitude': abs(predictions[-1]),
                'confidence': confidence,
                'timeframes': self._get_prediction_timeframes(confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error in price prediction: {str(e)}")
            return {}
            
    async def _analyze_sentiment(self, timeframe: str) -> Dict:
        """Analyze market sentiment"""
        try:
            # Get recent news and social media data
            texts = await self._get_sentiment_data(timeframe)
            
            # Analyze sentiment
            sentiments = self.sentiment_analyzer(texts)
            
            # Aggregate results
            sentiment_score = np.mean([s['score'] for s in sentiments])
            sentiment_label = 'bullish' if sentiment_score > 0.6 else 'bearish' if sentiment_score < 0.4 else 'neutral'
            
            return {
                'score': sentiment_score,
                'label': sentiment_label,
                'confidence': np.std([s['score'] for s in sentiments]),
                'sources': len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {}
            
    async def _recognize_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Recognize chart patterns"""
        try:
            patterns = []
            
            # Prepare data for pattern recognition
            prepared_data = self._prepare_pattern_data(data)
            
            # Run pattern recognition model
            detected_patterns = self.models['pattern'].predict(prepared_data)
            
            # Process and filter patterns
            for pattern in detected_patterns:
                if pattern['confidence'] >= self.config['min_confidence']:
                    patterns.append({
                        'type': pattern['type'],
                        'start_idx': pattern['start'],
                        'end_idx': pattern['end'],
                        'confidence': pattern['confidence'],
                        'significance': pattern['significance']
                    })
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern recognition: {str(e)}")
            return []
            
    def _calculate_signals(self, analysis: Dict) -> Dict:
        """Calculate trading signals from analysis"""
        try:
            signals = {
                'strength': 0,
                'direction': 0,
                'confidence': 0,
                'timeframe': None
            }
            
            # Combine price predictions
            if 'price_prediction' in analysis:
                signals['direction'] += analysis['price_prediction'].get('direction', 0) * 0.3
                signals['confidence'] += analysis['price_prediction'].get('confidence', 0) * 0.3
                
            # Add sentiment impact
            if 'sentiment' in analysis:
                sentiment_score = analysis['sentiment'].get('score', 0.5)
                signals['direction'] += (sentiment_score - 0.5) * 2 * 0.2
                signals['confidence'] += analysis['sentiment'].get('confidence', 0) * 0.2
                
            # Consider patterns
            if 'patterns' in analysis:
                for pattern in analysis['patterns']:
                    signals['strength'] += pattern.get('significance', 0) * pattern.get('confidence', 0)
                    
            # Normalize signals
            signals['strength'] = min(max(signals['strength'], -1), 1)
            signals['direction'] = min(max(signals['direction'], -1), 1)
            signals['confidence'] = min(max(signals['confidence'], 0), 1)
            
            # Determine best timeframe
            signals['timeframe'] = self._determine_best_timeframe(analysis)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating signals: {str(e)}")
            return {}
            
    def _determine_best_timeframe(self, analysis: Dict) -> Optional[str]:
        """Determine the most suitable timeframe for trading"""
        try:
            timeframe_scores = {}
            
            for timeframe in self.config['timeframes']:
                score = 0
                
                # Consider prediction confidence
                if 'price_prediction' in analysis:
                    score += analysis['price_prediction'].get('confidence', 0) * 0.4
                    
                # Consider pattern significance
                if 'patterns' in analysis:
                    pattern_score = sum(p.get('significance', 0) * p.get('confidence', 0)
                                     for p in analysis['patterns'])
                    score += pattern_score * 0.3
                    
                # Consider sentiment stability
                if 'sentiment' in analysis:
                    score += analysis['sentiment'].get('confidence', 0) * 0.3
                    
                timeframe_scores[timeframe] = score
                
            # Select timeframe with highest score
            if timeframe_scores:
                return max(timeframe_scores.items(), key=lambda x: x[1])[0]
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining timeframe: {str(e)}")
            return None
            
    def _load_price_model(self) -> tf.keras.Model:
        """Load price prediction model"""
        try:
            model_path = f"{self.config['model_path']}/price_prediction"
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            self.logger.error(f"Error loading price model: {str(e)}")
            return self._create_default_price_model()
            
    def _load_pattern_model(self) -> tf.keras.Model:
        """Load pattern recognition model"""
        try:
            model_path = f"{self.config['model_path']}/pattern_recognition"
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            self.logger.error(f"Error loading pattern model: {str(e)}")
            return self._create_default_pattern_model()
            
    def _create_default_price_model(self) -> tf.keras.Model:
        """Create a default price prediction model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(100, 5)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _create_default_pattern_model(self) -> tf.keras.Model:
        """Create a default pattern recognition model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, input_shape=(100, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.config['patterns']), activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
