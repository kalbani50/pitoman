from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional
from decimal import Decimal
import logging

class StrategyInterface(ABC):
    @abstractmethod
    async def execute(self, bot, pair: str, df: pd.DataFrame, models_dict: Dict, balances: Dict):
        pass

class MACDRSIStrategy(StrategyInterface):
    async def execute(self, bot, pair: str, df: pd.DataFrame, models_dict: Dict, balances: Dict):
        try:
            if df is None or df.empty:
                return None

            current_price = Decimal(str(df['close'].iloc[-1]))
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            
            # Check SuperTrend and Hull MA for confirmation
            supertrend_signal = df['close'].iloc[-1] > df['SuperTrend'].iloc[-1]
            hull_trend = df['close'].iloc[-1] > df['HullMA'].iloc[-1]
            
            # Money flow conditions
            money_flow_healthy = 20 < df['MFI'].iloc[-1] < 80
            positive_flow = df['CMF'].iloc[-1] > 0

            if (rsi < 30 and macd > macd_signal and supertrend_signal and hull_trend and 
                money_flow_healthy and positive_flow):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'buy')
            elif (rsi > 70 and macd < macd_signal and not supertrend_signal and not hull_trend):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'sell')
            
            return None
            
        except Exception as e:
            logging.error(f"Error in MACD_RSI strategy: {e}", exc_info=True)
            return None

class VolatilityStrategy(StrategyInterface):
    async def execute(self, bot, pair: str, df: pd.DataFrame, models_dict: Dict, balances: Dict):
        try:
            if df is None or df.empty:
                return None

            current_price = Decimal(str(df['close'].iloc[-1]))
            atr = df['ATR'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            keltner_upper = df['KeltnerChannels_Upper'].iloc[-1]
            keltner_lower = df['KeltnerChannels_Lower'].iloc[-1]
            
            # Volatility squeeze condition
            squeeze = (bb_upper - bb_lower) < (keltner_upper - keltner_lower)
            
            # Momentum conditions
            momentum = df['Ultimate_Oscillator'].iloc[-1]
            trend_strength = df['ADX'].iloc[-1]
            
            if (squeeze and momentum > 70 and trend_strength > 25 and 
                df['close'].iloc[-1] > df['HullMA'].iloc[-1]):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'buy')
            elif (squeeze and momentum < 30 and trend_strength > 25 and 
                  df['close'].iloc[-1] < df['HullMA'].iloc[-1]):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'sell')
            
            return None
            
        except Exception as e:
            logging.error(f"Error in Volatility strategy: {e}", exc_info=True)
            return None

class MovingAverageCrossoverStrategy(StrategyInterface):
    async def execute(self, bot, pair: str, df: pd.DataFrame, models_dict: Dict, balances: Dict):
        try:
            if df is None or df.empty:
                return None

            current_price = Decimal(str(df['close'].iloc[-1]))
            
            # Moving average crossovers
            ema_short = df['EMA_20'].iloc[-1]
            ema_long = df['EMA_50'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            
            # Additional confirmations
            supertrend_bullish = df['close'].iloc[-1] > df['SuperTrend'].iloc[-1]
            hull_trend = df['close'].iloc[-1] > df['HullMA'].iloc[-1]
            psar_bullish = df['close'].iloc[-1] > df['PSar'].iloc[-1]
            
            # Volume and momentum confirmations
            volume_trend = df['OBV'].iloc[-1] > df['OBV'].iloc[-5]  # Volume increasing
            momentum_healthy = df['MFI'].iloc[-1] > 20 and df['MFI'].iloc[-1] < 80
            
            if (ema_short > ema_long and sma_50 > sma_200 and supertrend_bullish and 
                hull_trend and psar_bullish and volume_trend and momentum_healthy):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'buy')
            elif (ema_short < ema_long and sma_50 < sma_200 and not supertrend_bullish and 
                  not hull_trend and not psar_bullish):
                return await bot.execute_trade(pair, models_dict, current_price, balances, df, 'sell')
            
            return None
            
        except Exception as e:
            logging.error(f"Error in Moving Average Crossover strategy: {e}", exc_info=True)
            return None

class StrategyFactory:
    @staticmethod
    def get_strategies(strategy_names: list) -> list:
        strategy_map = {
            "MACD_RSI": MACDRSIStrategy(),
            "VOLATILITY": VolatilityStrategy(),
            "MOVING_AVERAGE": MovingAverageCrossoverStrategy()
        }
        
        return [strategy_map[name] for name in strategy_names if name in strategy_map]
