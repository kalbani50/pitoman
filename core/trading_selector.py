"""
Smart Trading Type and Pair Selection System
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class TradingSetup:
    trading_type: str  # 'spot' or 'futures'
    selected_pairs: List[str]
    position_sizes: Dict[str, float]
    leverage: Dict[str, float]
    risk_levels: Dict[str, float]

class SmartTradingSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.market_state = {}
        self.trading_history = []
        
    async def select_trading_setup(self, capital: float) -> TradingSetup:
        """Select optimal trading setup"""
        try:
            # Analyze market conditions
            market_analysis = await self._analyze_market_conditions()
            
            # Select trading type
            trading_type = self._select_trading_type(market_analysis)
            
            # Select trading pairs
            selected_pairs = await self._select_trading_pairs(
                trading_type,
                market_analysis
            )
            
            # Calculate position sizes
            position_sizes = self._calculate_position_sizes(
                capital,
                selected_pairs,
                market_analysis
            )
            
            # Determine leverage
            leverage = self._determine_leverage(
                trading_type,
                selected_pairs,
                market_analysis
            )
            
            # Assess risk levels
            risk_levels = self._assess_risk_levels(
                selected_pairs,
                position_sizes,
                leverage,
                market_analysis
            )
            
            return TradingSetup(
                trading_type=trading_type,
                selected_pairs=selected_pairs,
                position_sizes=position_sizes,
                leverage=leverage,
                risk_levels=risk_levels
            )
            
        except Exception as e:
            self.logger.error(f"Error in trading setup selection: {e}")
            return self._get_default_setup(capital)
            
    async def _analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions"""
        analysis = {
            'volatility': self._analyze_market_volatility(),
            'trend_strength': self._analyze_trend_strength(),
            'liquidity': self._analyze_market_liquidity(),
            'correlation': self._analyze_correlation(),
            'sentiment': await self._analyze_market_sentiment()
        }
        
        # Update market state
        self.market_state = analysis
        
        return analysis
        
    def _select_trading_type(self, analysis: Dict) -> str:
        """Select between spot and futures trading"""
        # Calculate scores for each type
        spot_score = self._calculate_spot_score(analysis)
        futures_score = self._calculate_futures_score(analysis)
        
        # Compare scores
        if futures_score > spot_score * 1.2:  # 20% threshold
            return 'futures'
        return 'spot'
        
    async def _select_trading_pairs(self, trading_type: str,
                                  analysis: Dict) -> List[str]:
        """Select optimal trading pairs"""
        all_pairs = await self._get_available_pairs(trading_type)
        scored_pairs = []
        
        for pair in all_pairs:
            score = self._calculate_pair_score(pair, trading_type, analysis)
            scored_pairs.append((pair, score))
            
        # Sort by score and select top pairs
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        max_pairs = self.config['trading']['max_pairs']
        
        return [pair for pair, _ in scored_pairs[:max_pairs]]
        
    def _calculate_position_sizes(self, capital: float,
                                pairs: List[str],
                                analysis: Dict) -> Dict[str, float]:
        """Calculate optimal position sizes"""
        position_sizes = {}
        remaining_capital = capital
        
        # Sort pairs by opportunity score
        scored_pairs = [
            (pair, self._calculate_opportunity_score(pair, analysis))
            for pair in pairs
        ]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for pair, score in scored_pairs:
            # Calculate base position size
            base_size = remaining_capital * score
            
            # Adjust for risk
            risk_adjusted_size = self._adjust_size_for_risk(
                base_size,
                pair,
                analysis
            )
            
            # Ensure minimum and maximum limits
            final_size = self._apply_size_limits(risk_adjusted_size, pair)
            
            position_sizes[pair] = final_size
            remaining_capital -= final_size
            
            if remaining_capital <= 0:
                break
                
        return position_sizes
        
    def _determine_leverage(self, trading_type: str,
                          pairs: List[str],
                          analysis: Dict) -> Dict[str, float]:
        """Determine optimal leverage for each pair"""
        if trading_type == 'spot':
            return {pair: 1.0 for pair in pairs}
            
        leverage = {}
        for pair in pairs:
            # Calculate base leverage
            base_leverage = self._calculate_base_leverage(pair, analysis)
            
            # Adjust for volatility
            volatility_adjusted = self._adjust_leverage_for_volatility(
                base_leverage,
                pair,
                analysis
            )
            
            # Apply limits
            final_leverage = min(
                volatility_adjusted,
                self.config['trading']['max_leverage']
            )
            
            leverage[pair] = final_leverage
            
        return leverage
        
    def _assess_risk_levels(self, pairs: List[str],
                          position_sizes: Dict[str, float],
                          leverage: Dict[str, float],
                          analysis: Dict) -> Dict[str, float]:
        """Assess risk level for each position"""
        risk_levels = {}
        
        for pair in pairs:
            # Calculate individual risk factors
            volatility_risk = analysis['volatility'].get(pair, 1.0)
            liquidity_risk = 1 - analysis['liquidity'].get(pair, 0)
            leverage_risk = (leverage[pair] / self.config['trading']['max_leverage'])
            size_risk = position_sizes[pair] / sum(position_sizes.values())
            
            # Combine risk factors
            total_risk = (
                volatility_risk * 0.3 +
                liquidity_risk * 0.3 +
                leverage_risk * 0.2 +
                size_risk * 0.2
            )
            
            risk_levels[pair] = min(total_risk, 1.0)
            
        return risk_levels
        
    def _calculate_spot_score(self, analysis: Dict) -> float:
        """Calculate score for spot trading"""
        weights = self.config['trading']['spot_weights']
        
        score = (
            weights['volatility'] * (1 - analysis['volatility'].get('average', 0)) +
            weights['trend'] * analysis['trend_strength'].get('average', 0) +
            weights['liquidity'] * analysis['liquidity'].get('average', 0)
        )
        
        return score
        
    def _calculate_futures_score(self, analysis: Dict) -> float:
        """Calculate score for futures trading"""
        weights = self.config['trading']['futures_weights']
        
        score = (
            weights['volatility'] * analysis['volatility'].get('average', 0) +
            weights['trend'] * analysis['trend_strength'].get('average', 0) +
            weights['liquidity'] * analysis['liquidity'].get('average', 0)
        )
        
        return score
        
    def _calculate_pair_score(self, pair: str,
                            trading_type: str,
                            analysis: Dict) -> float:
        """Calculate score for a trading pair"""
        weights = self.config['trading']['pair_weights']
        
        score = (
            weights['volatility'] * analysis['volatility'].get(pair, 0) +
            weights['trend'] * analysis['trend_strength'].get(pair, 0) +
            weights['liquidity'] * analysis['liquidity'].get(pair, 0) +
            weights['sentiment'] * analysis['sentiment'].get(pair, 0)
        )
        
        if trading_type == 'futures':
            score *= 1.2  # Boost score for futures if conditions are right
            
        return score
        
    def _calculate_opportunity_score(self, pair: str,
                                   analysis: Dict) -> float:
        """Calculate opportunity score for position sizing"""
        weights = self.config['trading']['opportunity_weights']
        
        return (
            weights['trend'] * analysis['trend_strength'].get(pair, 0) +
            weights['sentiment'] * analysis['sentiment'].get(pair, 0) +
            weights['volatility'] * analysis['volatility'].get(pair, 0)
        )
        
    def _get_default_setup(self, capital: float) -> TradingSetup:
        """Get default conservative setup"""
        return TradingSetup(
            trading_type='spot',
            selected_pairs=['BTC/USDT'],
            position_sizes={'BTC/USDT': capital * 0.5},
            leverage={'BTC/USDT': 1.0},
            risk_levels={'BTC/USDT': 0.5}
        )
