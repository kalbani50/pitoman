import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging
from datetime import datetime, timezone
import ta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from collections import deque

class SmartTradingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.market_memory = deque(maxlen=1000)  # Store market patterns
        self.trade_memory = deque(maxlen=1000)   # Store trade results
        self.pattern_memory = {}  # Store successful patterns
        
    async def analyze_market_psychology(self, df: pd.DataFrame) -> Dict:
        try:
            psychology = {}
            
            # 1. Market Sentiment Analysis
            psychology['sentiment'] = self._analyze_sentiment(df)
            
            # 2. Fear & Greed Analysis
            psychology['fear_greed'] = self._analyze_fear_greed(df)
            
            # 3. Smart Money Analysis
            psychology['smart_money'] = self._analyze_smart_money(df)
            
            # 4. Market Manipulation Detection
            psychology['manipulation'] = self._detect_manipulation(df)
            
            # 5. Crowd Psychology
            psychology['crowd_behavior'] = self._analyze_crowd_behavior(df)
            
            return psychology
            
        except Exception as e:
            logging.error(f"Error in market psychology analysis: {e}", exc_info=True)
            return {}
            
    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """Analyzes market sentiment using advanced indicators"""
        try:
            # Calculate price momentum
            momentum = df['close'].pct_change(5)
            
            # Calculate volume force
            volume_force = df['volume'] * momentum
            
            # Calculate buying/selling pressure
            buying_pressure = volume_force[volume_force > 0].sum()
            selling_pressure = abs(volume_force[volume_force < 0].sum())
            
            # Calculate sentiment score
            sentiment_score = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure)
            
            # Determine sentiment levels
            if sentiment_score > 0.7:
                sentiment = 'extremely_bullish'
            elif sentiment_score > 0.3:
                sentiment = 'bullish'
            elif sentiment_score > -0.3:
                sentiment = 'neutral'
            elif sentiment_score > -0.7:
                sentiment = 'bearish'
            else:
                sentiment = 'extremely_bearish'
                
            return {
                'score': sentiment_score,
                'level': sentiment,
                'buying_pressure': float(buying_pressure),
                'selling_pressure': float(selling_pressure)
            }
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}", exc_info=True)
            return {}
            
    def _analyze_fear_greed(self, df: pd.DataFrame) -> Dict:
        """Analyzes market fear and greed levels"""
        try:
            # 1. Volatility Analysis
            volatility = df['close'].pct_change().std() * np.sqrt(252)
            
            # 2. Market Momentum
            momentum = ta.momentum.rsi(df['close'], window=10)
            
            # 3. Volume Analysis
            volume_ma = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'] / volume_ma
            
            # 4. RSI Analysis
            rsi = ta.momentum.rsi(df['close'], window=14)
            
            # Calculate Fear & Greed Index
            fear_greed_score = (
                (rsi.iloc[-1] / 100) * 0.25 +
                (momentum.iloc[-1] / momentum.abs().max()) * 0.25 +
                (volume_ratio.iloc[-1] / volume_ratio.max()) * 0.25 +
                (1 - volatility / volatility.max()) * 0.25
            )
            
            # Determine market state
            if fear_greed_score > 0.8:
                state = 'extreme_greed'
            elif fear_greed_score > 0.6:
                state = 'greed'
            elif fear_greed_score > 0.4:
                state = 'neutral'
            elif fear_greed_score > 0.2:
                state = 'fear'
            else:
                state = 'extreme_fear'
                
            return {
                'score': float(fear_greed_score),
                'state': state,
                'volatility': float(volatility),
                'momentum': float(momentum.iloc[-1]),
                'volume_ratio': float(volume_ratio.iloc[-1])
            }
            
        except Exception as e:
            logging.error(f"Error in fear/greed analysis: {e}", exc_info=True)
            return {}
            
    def _analyze_smart_money(self, df: pd.DataFrame) -> Dict:
        """Analyzes smart money movements"""
        try:
            # 1. Large Volume Analysis
            volume_std = df['volume'].std()
            large_volumes = df[df['volume'] > df['volume'].mean() + 2 * volume_std]
            
            # 2. Price Impact Analysis
            price_impact = (df['high'] - df['low']) / df['volume']
            
            # 3. Order Flow Analysis
            delta = df['close'] - df['open']
            cumulative_delta = delta.cumsum()
            
            # 4. Hidden Orders Analysis
            hidden_buying = (
                (df['close'] > df['open']) &
                (df['volume'] < df['volume'].mean()) &
                (df['high'] - df['close'] < df['close'] - df['open'])
            )
            
            hidden_selling = (
                (df['close'] < df['open']) &
                (df['volume'] < df['volume'].mean()) &
                (df['close'] - df['low'] < df['open'] - df['close'])
            )
            
            return {
                'large_volume_ratio': len(large_volumes) / len(df),
                'price_impact': float(price_impact.mean()),
                'cumulative_delta': float(cumulative_delta.iloc[-1]),
                'hidden_buying': float(hidden_buying.sum() / len(df)),
                'hidden_selling': float(hidden_selling.sum() / len(df))
            }
            
        except Exception as e:
            logging.error(f"Error in smart money analysis: {e}", exc_info=True)
            return {}
            
    def _detect_manipulation(self, df: pd.DataFrame) -> Dict:
        """Detects potential market manipulation"""
        try:
            # 1. Price Manipulation
            price_changes = df['close'].pct_change()
            unusual_moves = price_changes[abs(price_changes) > 3 * price_changes.std()]
            
            # 2. Volume Manipulation
            volume_changes = df['volume'].pct_change()
            unusual_volume = volume_changes[abs(volume_changes) > 3 * volume_changes.std()]
            
            # 3. Spoofing Detection
            potential_spoofing = (
                (df['high'] - df['low']) > 3 * df['high'].rolling(20).std()
            ) & (df['volume'] < df['volume'].rolling(20).mean())
            
            # 4. Wash Trading Detection
            potential_wash = (
                (abs(df['close'] - df['open']) < 0.1 * df['close'].std()) &
                (df['volume'] > 2 * df['volume'].rolling(20).mean())
            )
            
            return {
                'unusual_price_moves': len(unusual_moves) / len(df),
                'unusual_volume': len(unusual_volume) / len(df),
                'potential_spoofing': float(potential_spoofing.sum() / len(df)),
                'potential_wash_trading': float(potential_wash.sum() / len(df))
            }
            
        except Exception as e:
            logging.error(f"Error in manipulation detection: {e}", exc_info=True)
            return {}
            
    def _analyze_crowd_behavior(self, df: pd.DataFrame) -> Dict:
        """Analyzes crowd behavior patterns"""
        try:
            # 1. Trend Following Analysis
            trend = ta.trend.adx(df['high'], df['low'], df['close'])
            
            # 2. Momentum Analysis
            momentum = ta.momentum.mom(df['close'], window=10)
            
            # 3. Herd Behavior Detection
            herd_behavior = (
                (df['volume'] > 2 * df['volume'].rolling(20).mean()) &
                (abs(df['close'].pct_change()) > 2 * df['close'].pct_change().std())
            )
            
            # 4. FOMO Detection
            fomo = (
                (df['close'].pct_change() > 0) &
                (df['volume'] > df['volume'].rolling(20).mean()) &
                (df['close'] > df['close'].rolling(20).mean())
            )
            
            return {
                'trend_strength': float(trend.mean()),
                'momentum_strength': float(momentum.iloc[-1] / df['close'].iloc[-1]),
                'herd_behavior': float(herd_behavior.sum() / len(df)),
                'fomo_indicator': float(fomo.sum() / len(df))
            }
            
        except Exception as e:
            logging.error(f"Error in crowd behavior analysis: {e}", exc_info=True)
            return {}
            
    async def generate_smart_trading_decision(self,
                                           df: pd.DataFrame,
                                           psychology: Dict) -> Dict:
        try:
            # 1. Pattern Recognition
            current_pattern = self._identify_pattern(df)
            
            # 2. Risk Assessment
            risk_level = self._assess_risk(df, psychology)
            
            # 3. Opportunity Assessment
            opportunity = self._assess_opportunity(df, psychology)
            
            # 4. Position Sizing
            position_size = self._calculate_position_size(
                risk_level,
                opportunity['score']
            )
            
            # 5. Entry/Exit Points
            entry_exit = self._calculate_entry_exit_points(df, psychology)
            
            # 6. Generate Decision
            decision = {
                'action': self._determine_action(
                    psychology,
                    risk_level,
                    opportunity
                ),
                'position_size': position_size,
                'entry_price': entry_exit['entry'],
                'stop_loss': entry_exit['stop_loss'],
                'take_profit': entry_exit['take_profit'],
                'confidence': self._calculate_confidence(
                    psychology,
                    risk_level,
                    opportunity
                )
            }
            
            # 7. Store Decision Pattern
            self._store_pattern(current_pattern, decision)
            
            return decision
            
        except Exception as e:
            logging.error(f"Error generating trading decision: {e}", exc_info=True)
            return {}
            
    def _identify_pattern(self, df: pd.DataFrame) -> Dict:
        """Identifies current market pattern"""
        try:
            # Calculate pattern features
            pattern = {
                'price_pattern': self._calculate_price_pattern(df),
                'volume_pattern': self._calculate_volume_pattern(df),
                'momentum_pattern': self._calculate_momentum_pattern(df)
            }
            
            # Match with successful patterns
            pattern['similarity'] = self._match_patterns(pattern)
            
            return pattern
            
        except Exception as e:
            logging.error(f"Error identifying pattern: {e}", exc_info=True)
            return {}
            
    def _assess_risk(self, df: pd.DataFrame, psychology: Dict) -> Dict:
        """Assesses current market risk"""
        try:
            # Calculate various risk metrics
            volatility_risk = self._calculate_volatility_risk(df)
            liquidity_risk = self._calculate_liquidity_risk(df)
            sentiment_risk = self._calculate_sentiment_risk(psychology)
            
            # Combine risk metrics
            total_risk = (
                volatility_risk * 0.4 +
                liquidity_risk * 0.3 +
                sentiment_risk * 0.3
            )
            
            return {
                'total_risk': total_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'sentiment_risk': sentiment_risk
            }
            
        except Exception as e:
            logging.error(f"Error assessing risk: {e}", exc_info=True)
            return {}
            
    def _assess_opportunity(self, df: pd.DataFrame, psychology: Dict) -> Dict:
        """Assesses trading opportunity"""
        try:
            # Calculate opportunity metrics
            trend_score = self._calculate_trend_score(df)
            momentum_score = self._calculate_momentum_score(df)
            sentiment_score = self._calculate_sentiment_score(psychology)
            
            # Combine opportunity metrics
            total_score = (
                trend_score * 0.4 +
                momentum_score * 0.3 +
                sentiment_score * 0.3
            )
            
            return {
                'score': total_score,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            logging.error(f"Error assessing opportunity: {e}", exc_info=True)
            return {}
            
    def _calculate_position_size(self,
                               risk: Dict,
                               opportunity_score: float) -> Decimal:
        """Calculates optimal position size"""
        try:
            # Base position size on risk and opportunity
            base_size = Decimal('0.02')  # 2% of capital
            
            # Adjust for risk
            risk_adjustment = Decimal(str(1 - risk['total_risk']))
            
            # Adjust for opportunity
            opportunity_adjustment = Decimal(str(opportunity_score))
            
            # Calculate final position size
            position_size = (
                base_size *
                risk_adjustment *
                opportunity_adjustment
            )
            
            # Apply limits
            max_size = Decimal('0.1')  # 10% of capital
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}", exc_info=True)
            return Decimal('0')
            
    def _calculate_entry_exit_points(self,
                                   df: pd.DataFrame,
                                   psychology: Dict) -> Dict:
        """Calculates optimal entry and exit points"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate entry point
            entry = self._calculate_entry_point(df, psychology)
            
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(df, entry)
            
            # Calculate take profit
            take_profit = self._calculate_take_profit(
                df,
                entry,
                stop_loss
            )
            
            return {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            logging.error(f"Error calculating entry/exit points: {e}", exc_info=True)
            return {}
            
    def _determine_action(self,
                         psychology: Dict,
                         risk: Dict,
                         opportunity: Dict) -> str:
        """Determines trading action"""
        try:
            # No action if risk is too high
            if risk['total_risk'] > 0.7:
                return 'no_action'
                
            # No action if opportunity is too low
            if opportunity['score'] < 0.3:
                return 'no_action'
                
            # Determine action based on psychology
            if psychology['sentiment']['level'] in ['extremely_bullish', 'bullish']:
                return 'buy'
            elif psychology['sentiment']['level'] in ['extremely_bearish', 'bearish']:
                return 'sell'
                
            return 'no_action'
            
        except Exception as e:
            logging.error(f"Error determining action: {e}", exc_info=True)
            return 'no_action'
            
    def _calculate_confidence(self,
                            psychology: Dict,
                            risk: Dict,
                            opportunity: Dict) -> float:
        """Calculates confidence level in trading decision"""
        try:
            # Combine various factors
            sentiment_confidence = 1 - abs(psychology['sentiment']['score'])
            risk_confidence = 1 - risk['total_risk']
            opportunity_confidence = opportunity['score']
            
            # Calculate weighted confidence
            confidence = (
                sentiment_confidence * 0.3 +
                risk_confidence * 0.3 +
                opportunity_confidence * 0.4
            )
            
            return float(confidence)
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {e}", exc_info=True)
            return 0.0
            
    def _store_pattern(self, pattern: Dict, decision: Dict):
        """Stores successful patterns for future reference"""
        try:
            pattern_key = f"{pattern['price_pattern']}_{pattern['volume_pattern']}"
            
            if pattern_key not in self.pattern_memory:
                self.pattern_memory[pattern_key] = {
                    'decisions': [],
                    'success_rate': 0.0
                }
                
            self.pattern_memory[pattern_key]['decisions'].append(decision)
            
        except Exception as e:
            logging.error(f"Error storing pattern: {e}", exc_info=True)
            
    async def update_trading_performance(self, trade_result: Dict):
        """Updates trading performance metrics"""
        try:
            # Store trade result
            self.trade_memory.append(trade_result)
            
            # Update pattern success rates
            self._update_pattern_success_rates()
            
            # Analyze performance
            performance = self._analyze_performance()
            
            # Adjust trading parameters
            self._adjust_parameters(performance)
            
        except Exception as e:
            logging.error(f"Error updating performance: {e}", exc_info=True)
            
    def _update_pattern_success_rates(self):
        """Updates success rates for stored patterns"""
        try:
            for pattern_key, pattern_data in self.pattern_memory.items():
                successful_trades = sum(
                    1 for decision in pattern_data['decisions']
                    if decision.get('result', {}).get('profit', 0) > 0
                )
                
                total_trades = len(pattern_data['decisions'])
                
                if total_trades > 0:
                    pattern_data['success_rate'] = successful_trades / total_trades
                    
        except Exception as e:
            logging.error(f"Error updating pattern success rates: {e}", exc_info=True)
            
    def _analyze_performance(self) -> Dict:
        """Analyzes trading performance"""
        try:
            if not self.trade_memory:
                return {}
                
            trades = pd.DataFrame(self.trade_memory)
            
            return {
                'win_rate': len(trades[trades['profit'] > 0]) / len(trades),
                'average_profit': trades['profit'].mean(),
                'profit_factor': abs(
                    trades[trades['profit'] > 0]['profit'].sum() /
                    trades[trades['profit'] < 0]['profit'].sum()
                ),
                'max_drawdown': self._calculate_max_drawdown(trades),
                'sharpe_ratio': self._calculate_sharpe_ratio(trades)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing performance: {e}", exc_info=True)
            return {}
