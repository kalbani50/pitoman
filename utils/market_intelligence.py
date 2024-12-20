import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
import logging
from datetime import datetime, timezone
import aiohttp
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from textblob import TextBlob
import yfinance as yf

class MarketIntelligence:
    def __init__(self, config: Dict):
        self.config = config
        self.market_data = {}
        self.analysis_results = {}
        self.correlations = {}
        self.market_patterns = {}
        
    async def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyzes market microstructure"""
        try:
            # Analyze order book
            order_book = await self._analyze_order_book(data)
            
            # Analyze market depth
            depth = await self._analyze_market_depth(data)
            
            # Analyze liquidity
            liquidity = await self._analyze_liquidity(data)
            
            # Analyze volatility clusters
            volatility = await self._analyze_volatility_clusters(data)
            
            return {
                'order_book': order_book,
                'depth': depth,
                'liquidity': liquidity,
                'volatility': volatility
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market structure: {e}")
            raise
            
    async def detect_market_regimes(self, data: pd.DataFrame) -> Dict:
        """Detects market regimes using clustering"""
        try:
            # Prepare features
            features = self._prepare_regime_features(data)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Detect regimes using DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=5)
            regimes = clustering.fit_predict(scaled_features)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regimes(data, regimes)
            
            return {
                'regimes': regimes.tolist(),
                'analysis': regime_analysis
            }
            
        except Exception as e:
            logging.error(f"Error detecting market regimes: {e}")
            raise
            
    async def analyze_market_networks(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyzes market networks and relationships"""
        try:
            # Create market network
            network = self._create_market_network(data)
            
            # Analyze network metrics
            metrics = self._analyze_network_metrics(network)
            
            # Detect communities
            communities = self._detect_network_communities(network)
            
            # Analyze influence
            influence = self._analyze_market_influence(network)
            
            return {
                'metrics': metrics,
                'communities': communities,
                'influence': influence
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market networks: {e}")
            raise
            
    async def analyze_market_sentiment(self, data: Dict) -> Dict:
        """Analyzes market sentiment from multiple sources"""
        try:
            # Analyze news sentiment
            news_sentiment = await self._analyze_news_sentiment(data['news'])
            
            # Analyze social media sentiment
            social_sentiment = await self._analyze_social_sentiment(
                data['social_media']
            )
            
            # Analyze market indicators sentiment
            market_sentiment = self._analyze_market_sentiment_indicators(
                data['market_data']
            )
            
            # Combine sentiments
            combined_sentiment = self._combine_sentiment_analysis(
                news_sentiment,
                social_sentiment,
                market_sentiment
            )
            
            return {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_sentiment': market_sentiment,
                'combined_sentiment': combined_sentiment
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market sentiment: {e}")
            raise
            
    async def predict_market_movements(self, data: pd.DataFrame) -> Dict:
        """Predicts market movements using multiple approaches"""
        try:
            # Technical analysis predictions
            technical = await self._technical_analysis_prediction(data)
            
            # Statistical predictions
            statistical = await self._statistical_prediction(data)
            
            # Machine learning predictions
            ml_predictions = await self._ml_prediction(data)
            
            # Combine predictions
            combined = self._combine_predictions(
                technical,
                statistical,
                ml_predictions
            )
            
            return {
                'technical': technical,
                'statistical': statistical,
                'ml_predictions': ml_predictions,
                'combined': combined
            }
            
        except Exception as e:
            logging.error(f"Error predicting market movements: {e}")
            raise
            
    def _create_market_network(self, data: Dict[str, pd.DataFrame]) -> nx.Graph:
        """Creates market network from price data"""
        network = nx.Graph()
        
        # Add nodes
        for asset in data.keys():
            network.add_node(asset)
            
        # Add edges based on correlations
        for asset1 in data.keys():
            for asset2 in data.keys():
                if asset1 != asset2:
                    correlation = data[asset1]['close'].corr(data[asset2]['close'])
                    if abs(correlation) > self.config['correlation_threshold']:
                        network.add_edge(asset1, asset2, weight=correlation)
                        
        return network
        
    async def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyzes sentiment from news articles"""
        sentiments = []
        
        for article in news_data:
            # Analyze text sentiment
            blob = TextBlob(article['text'])
            sentiment = blob.sentiment.polarity
            
            # Weight by article relevance
            weighted_sentiment = sentiment * article.get('relevance', 1.0)
            sentiments.append(weighted_sentiment)
            
        return {
            'average_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'sentiment_distribution': np.histogram(sentiments)[0].tolist()
        }
        
    async def generate_market_report(self) -> Dict:
        """Generates comprehensive market analysis report"""
        try:
            report = {
                'market_structure': self.analysis_results.get('structure'),
                'market_regimes': self.analysis_results.get('regimes'),
                'market_networks': self.analysis_results.get('networks'),
                'market_sentiment': self.analysis_results.get('sentiment'),
                'market_predictions': self.analysis_results.get('predictions'),
                'timestamp': datetime.now(timezone.utc)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating market report: {e}")
            raise
