import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from abc import ABC, abstractmethod
from utils.advanced_indicators import AdvancedIndicators

class AdvancedStrategy(ABC):
    def __init__(self, params: Dict = None):
        self.params = params or {}
        
    @abstractmethod
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        pass
        
    def get_signal_strength(self, signals: Dict[str, float]) -> float:
        return signals.get('overall_signal', 0)
        
class SmartMoneyStrategy(AdvancedStrategy):
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        try:
            signals = {}
            
            # Get indicator signals
            indicator_signals = AdvancedIndicators.get_indicator_signals(df)
            
            # Smart Money Conditions
            smart_money_active = (
                indicator_signals['smart_money'] > 0 and
                indicator_signals['volume_strength'] > 0 and
                indicator_signals['delta_strength'] > 0
            )
            
            # Order Flow Conditions
            order_flow_positive = (
                indicator_signals['buying_pressure'] > 0 and
                df['OF_Momentum'].iloc[-1] > 0 and
                df['Volume_Imbalance'].iloc[-1] > 0
            )
            
            # Market Structure Conditions
            structure_bullish = (
                df['Structure_Break'].iloc[-1] > 0 and
                df['Higher_High'].iloc[-1] == 1
            )
            
            # Entry Conditions
            signals['enter'] = (
                smart_money_active and
                order_flow_positive and
                structure_bullish and
                indicator_signals['overall_signal'] > 0.5
            )
            
            # Exit Conditions
            signals['exit'] = (
                indicator_signals['smart_money'] < 0 or
                (df['Structure_Break'].iloc[-1] < 0 and df['Lower_Low'].iloc[-1] == 1)
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error in SmartMoneyStrategy: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
class VolatilityBreakoutStrategy(AdvancedStrategy):
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        try:
            signals = {}
            
            # Get indicator signals
            indicator_signals = AdvancedIndicators.get_indicator_signals(df)
            
            # Volatility Conditions
            volatility_setup = (
                df['Vol_Breakout'].iloc[-1] > 0.8 and
                df['Normalized_Volatility'].iloc[-1] > df['Normalized_Volatility'].rolling(20).mean().iloc[-1]
            )
            
            # Momentum Conditions
            momentum_aligned = (
                indicator_signals['momentum_quality'] > 0 and
                indicator_signals['triple_momentum'] > 0 and
                df['Custom_Osc'].iloc[-1] > 50
            )
            
            # Volume Conditions
            volume_confirmed = (
                df['VZO'].iloc[-1] > 0 and
                df['MFI'].iloc[-1] > 50 and
                df['Volume_Imbalance'].iloc[-1] > 0
            )
            
            # Entry Conditions
            signals['enter'] = (
                volatility_setup and
                momentum_aligned and
                volume_confirmed and
                indicator_signals['overall_signal'] > 0.6
            )
            
            # Exit Conditions
            signals['exit'] = (
                df['Vol_Breakout'].iloc[-1] < 0.2 or
                (df['Custom_Osc'].iloc[-1] < 30 and df['Triple_Mom'].iloc[-1] < 0)
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error in VolatilityBreakoutStrategy: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
class MarketStructureStrategy(AdvancedStrategy):
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        try:
            signals = {}
            
            # Get indicator signals
            indicator_signals = AdvancedIndicators.get_indicator_signals(df)
            
            # Market Structure Conditions
            structure_aligned = (
                df['Structure_Break'].iloc[-1] > 0 and
                df['Higher_High'].iloc[-1] == 1 and
                df['Efficiency_Ratio'].iloc[-1] > 0.7
            )
            
            # Trend Conditions
            trend_confirmed = (
                df['AMA'].iloc[-1] > df['AMA'].iloc[-2] and
                df['TSI'].iloc[-1] > 0 and
                df['VPT'].iloc[-1] > df['VPT'].iloc[-2]
            )
            
            # Volume Structure
            volume_structure = (
                df['VWAP'].iloc[-1] < df['close'].iloc[-1] and
                df['MFI'].iloc[-1] > 50 and
                df['Force_Index'].iloc[-1] > 0
            )
            
            # Entry Conditions
            signals['enter'] = (
                structure_aligned and
                trend_confirmed and
                volume_structure and
                indicator_signals['overall_signal'] > 0.7
            )
            
            # Exit Conditions
            signals['exit'] = (
                df['Structure_Break'].iloc[-1] < 0 or
                (df['Efficiency_Ratio'].iloc[-1] < 0.3 and df['TSI'].iloc[-1] < 0)
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error in MarketStructureStrategy: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
class AdaptiveMomentumStrategy(AdvancedStrategy):
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        try:
            signals = {}
            
            # Get indicator signals
            indicator_signals = AdvancedIndicators.get_indicator_signals(df)
            
            # Momentum Conditions
            momentum_setup = (
                df['Vol_Adj_Mom'].iloc[-1] > 0 and
                df['Triple_Mom'].iloc[-1] > 0 and
                df['Mom_Quality'].iloc[-1] > 0
            )
            
            # Volatility Adjustment
            volatility_suitable = (
                df['Vol_Regime'].iloc[-1] != 'High' and
                df['ATR_Percent'].iloc[-1] < df['ATR_Percent'].rolling(20).mean().iloc[-1]
            )
            
            # Market Efficiency
            market_efficient = (
                df['Efficiency_Ratio'].iloc[-1] > 0.6 and
                df['Custom_Osc'].iloc[-1] > 40 and
                df['TSI'].iloc[-1] > 0
            )
            
            # Entry Conditions
            signals['enter'] = (
                momentum_setup and
                volatility_suitable and
                market_efficient and
                indicator_signals['overall_signal'] > 0.65
            )
            
            # Exit Conditions
            signals['exit'] = (
                df['Vol_Adj_Mom'].iloc[-1] < 0 or
                (df['Triple_Mom'].iloc[-1] < 0 and df['Custom_Osc'].iloc[-1] < 30)
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error in AdaptiveMomentumStrategy: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
class HybridFlowStrategy(AdvancedStrategy):
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        try:
            signals = {}
            
            # Get indicator signals
            indicator_signals = AdvancedIndicators.get_indicator_signals(df)
            
            # Order Flow Conditions
            flow_aligned = (
                df['OF_Momentum'].iloc[-1] > 0 and
                df['Delta'].iloc[-1] > 0 and
                df['Volume_Imbalance'].iloc[-1] > 0
            )
            
            # Smart Money Conditions
            smart_money_confirmed = (
                df['SMI'].iloc[-1] > 0 and
                df['VPT'].iloc[-1] > df['VPT'].iloc[-2] and
                df['Force_Index'].iloc[-1] > 0
            )
            
            # Technical Structure
            tech_structure = (
                df['RSI_Divergence'].iloc[-1] > 0 and
                df['MACD_Divergence'].iloc[-1] > 0 and
                df['AMA'].iloc[-1] > df['AMA'].iloc[-2]
            )
            
            # Entry Conditions
            signals['enter'] = (
                flow_aligned and
                smart_money_confirmed and
                tech_structure and
                indicator_signals['overall_signal'] > 0.75
            )
            
            # Exit Conditions
            signals['exit'] = (
                df['OF_Momentum'].iloc[-1] < 0 or
                (df['SMI'].iloc[-1] < 0 and df['RSI_Divergence'].iloc[-1] < 0)
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error in HybridFlowStrategy: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
