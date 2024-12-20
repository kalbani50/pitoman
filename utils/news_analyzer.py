import aiohttp
import asyncio
from typing import Dict, List
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
import json

class NewsAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.news_cache = {}
        self.impact_scores = {}
        
    async def analyze_market_news(self, symbol: str) -> Dict:
        try:
            # Fetch news from multiple sources
            news_data = await self._fetch_news(symbol)
            
            # Analyze news sentiment
            sentiment_analysis = self._analyze_news_sentiment(news_data)
            
            # Analyze news impact
            impact_analysis = self._analyze_news_impact(news_data)
            
            # Analyze economic events
            economic_events = await self._analyze_economic_events(symbol)
            
            # Calculate combined news score
            news_score = self._calculate_news_score(
                sentiment_analysis,
                impact_analysis,
                economic_events
            )
            
            return {
                'sentiment': sentiment_analysis,
                'impact': impact_analysis,
                'economic_events': economic_events,
                'news_score': news_score,
                'recommendations': self._generate_news_recommendations(news_score)
            }
            
        except Exception as e:
            logging.error(f"Error in news analysis: {e}", exc_info=True)
            return {}
            
    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """Fetches news from multiple sources"""
        try:
            news_data = []
            
            # Cryptocurrency news sources
            crypto_sources = [
                'cryptopanic',
                'coindesk',
                'cointelegraph',
                'bitcoinmagazine'
            ]
            
            # Traditional finance sources
            finance_sources = [
                'reuters',
                'bloomberg',
                'wsj',
                'ft'
            ]
            
            async with aiohttp.ClientSession() as session:
                # Fetch from crypto sources
                for source in crypto_sources:
                    news = await self._fetch_source(session, source, symbol)
                    news_data.extend(news)
                    
                # Fetch from finance sources
                for source in finance_sources:
                    news = await self._fetch_source(session, source, symbol)
                    news_data.extend(news)
                    
            return news_data
            
        except Exception as e:
            logging.error(f"Error fetching news: {e}", exc_info=True)
            return []
            
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyzes sentiment of news articles"""
        try:
            sentiments = []
            
            for article in news_data:
                # Analyze title sentiment
                title_sentiment = self.sentiment_analyzer(article['title'])[0]
                
                # Analyze content sentiment
                content_blob = TextBlob(article['content'])
                content_sentiment = content_blob.sentiment.polarity
                
                # Calculate weighted sentiment
                weighted_sentiment = (
                    float(title_sentiment['score']) * 0.4 +
                    content_sentiment * 0.6
                )
                
                sentiments.append({
                    'title': article['title'],
                    'sentiment': weighted_sentiment,
                    'confidence': float(title_sentiment['score']),
                    'timestamp': article['published_at']
                })
                
            # Calculate aggregate sentiment
            if sentiments:
                avg_sentiment = np.mean([s['sentiment'] for s in sentiments])
                sentiment_std = np.std([s['sentiment'] for s in sentiments])
                
                return {
                    'aggregate_sentiment': float(avg_sentiment),
                    'sentiment_volatility': float(sentiment_std),
                    'recent_sentiment_change': self._calculate_sentiment_change(sentiments),
                    'detailed_sentiments': sentiments
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error analyzing news sentiment: {e}", exc_info=True)
            return {}
            
    def _analyze_news_impact(self, news_data: List[Dict]) -> Dict:
        """Analyzes potential market impact of news"""
        try:
            impacts = []
            
            for article in news_data:
                # Calculate reach score
                reach_score = self._calculate_reach_score(article)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(article)
                
                # Calculate timeliness score
                timeliness_score = self._calculate_timeliness_score(article)
                
                # Calculate overall impact
                impact_score = (
                    reach_score * 0.3 +
                    relevance_score * 0.4 +
                    timeliness_score * 0.3
                )
                
                impacts.append({
                    'title': article['title'],
                    'impact_score': impact_score,
                    'reach': reach_score,
                    'relevance': relevance_score,
                    'timeliness': timeliness_score
                })
                
            # Calculate aggregate impact
            if impacts:
                avg_impact = np.mean([i['impact_score'] for i in impacts])
                max_impact = max([i['impact_score'] for i in impacts])
                
                return {
                    'aggregate_impact': float(avg_impact),
                    'max_impact': float(max_impact),
                    'high_impact_news': [i for i in impacts if i['impact_score'] > 0.7],
                    'detailed_impacts': impacts
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error analyzing news impact: {e}", exc_info=True)
            return {}
            
    async def _analyze_economic_events(self, symbol: str) -> Dict:
        """Analyzes economic events and their potential impact"""
        try:
            # Fetch economic calendar
            events = await self._fetch_economic_calendar(symbol)
            
            # Analyze event importance
            analyzed_events = []
            for event in events:
                importance = self._calculate_event_importance(event)
                potential_impact = self._calculate_event_impact(event)
                
                analyzed_events.append({
                    'event': event['title'],
                    'importance': importance,
                    'potential_impact': potential_impact,
                    'timestamp': event['timestamp']
                })
                
            # Calculate aggregate event impact
            if analyzed_events:
                avg_importance = np.mean([e['importance'] for e in analyzed_events])
                max_importance = max([e['importance'] for e in analyzed_events])
                
                return {
                    'aggregate_importance': float(avg_importance),
                    'max_importance': float(max_importance),
                    'upcoming_events': [e for e in analyzed_events if e['importance'] > 0.7],
                    'detailed_events': analyzed_events
                }
                
            return {}
            
        except Exception as e:
            logging.error(f"Error analyzing economic events: {e}", exc_info=True)
            return {}
            
    def _calculate_news_score(self,
                            sentiment: Dict,
                            impact: Dict,
                            events: Dict) -> float:
        """Calculates combined news score"""
        try:
            if not all([sentiment, impact, events]):
                return 0.0
                
            # Weight components
            sentiment_weight = 0.4
            impact_weight = 0.3
            events_weight = 0.3
            
            # Calculate weighted score
            score = (
                sentiment['aggregate_sentiment'] * sentiment_weight +
                impact['aggregate_impact'] * impact_weight +
                events['aggregate_importance'] * events_weight
            )
            
            return float(score)
            
        except Exception as e:
            logging.error(f"Error calculating news score: {e}", exc_info=True)
            return 0.0
            
    def _generate_news_recommendations(self, news_score: float) -> Dict:
        """Generates trading recommendations based on news analysis"""
        try:
            if news_score > 0.7:
                action = 'strong_buy'
                confidence = news_score
            elif news_score > 0.3:
                action = 'buy'
                confidence = news_score
            elif news_score < -0.7:
                action = 'strong_sell'
                confidence = abs(news_score)
            elif news_score < -0.3:
                action = 'sell'
                confidence = abs(news_score)
            else:
                action = 'neutral'
                confidence = 1 - abs(news_score)
                
            return {
                'action': action,
                'confidence': float(confidence),
                'score': float(news_score),
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}", exc_info=True)
            return {}
