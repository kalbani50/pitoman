import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from scipy.signal import find_peaks
import logging

class AdvancedIndicators:
    @staticmethod
    def calculate_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Make copy to avoid modifying original
            df = df.copy()
            
            # Add Volume Profile
            df = AdvancedIndicators.add_volume_profile(df)
            
            # Add Order Flow Indicators
            df = AdvancedIndicators.add_order_flow_indicators(df)
            
            # Add Market Structure Indicators
            df = AdvancedIndicators.add_market_structure_indicators(df)
            
            # Add Volatility Indicators
            df = AdvancedIndicators.add_volatility_indicators(df)
            
            # Add Momentum Indicators
            df = AdvancedIndicators.add_momentum_indicators(df)
            
            # Add Custom Indicators
            df = AdvancedIndicators.add_custom_indicators(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating advanced indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Volume Weighted Average Price (VWAP)
            df['VWAP'] = AdvancedIndicators._volume_weighted_average_price(df)
            
            # Volume Force Index
            df['Force_Index'] = df['close'].diff() * df['volume']
            
            # Money Flow Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0.0, index=df.index)
            negative_flow = pd.Series(0.0, index=df.index)
            
            # Calculate positive and negative money flow
            mask = df['close'] > df['close'].shift(1)
            positive_flow[mask] = money_flow[mask]
            negative_flow[~mask] = money_flow[~mask]
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            df['MFI'] = 100 - (100 / (1 + positive_mf / negative_flow))
            
            # Volume Zone Oscillator (VZO)
            df['VZO'] = ((df['volume'] - df['volume'].rolling(window=20).mean()) / 
                        df['volume'].rolling(window=20).std())
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding volume profile indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_order_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Delta (Buying vs Selling Pressure)
            df['Delta'] = df['close'] - df['open']
            
            # Cumulative Delta
            df['Cumulative_Delta'] = df['Delta'].cumsum()
            
            # Buy/Sell Volume Imbalance
            df['Volume_Imbalance'] = df.apply(
                lambda x: x['volume'] if x['close'] > x['open'] else -x['volume'],
                axis=1
            )
            
            # Order Flow Momentum
            df['OF_Momentum'] = df['Volume_Imbalance'].rolling(window=20).mean()
            
            # Price Impact
            df['Price_Impact'] = df['Delta'] / df['volume']
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding order flow indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_market_structure_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Market Structure Levels
            highs = df['high'].values
            lows = df['low'].values
            
            # Find swing highs and lows
            high_peaks, _ = find_peaks(highs, distance=10)
            low_peaks, _ = find_peaks(-lows, distance=10)
            
            df['Swing_High'] = 0
            df['Swing_Low'] = 0
            df.loc[high_peaks, 'Swing_High'] = 1
            df.loc[low_peaks, 'Swing_Low'] = 1
            
            # Market Structure Trend
            df['Higher_High'] = df['Swing_High'].rolling(window=2).apply(
                lambda x: 1 if (x == 1).all() and 
                         df['high'][x.index].diff().iloc[-1] > 0 else 0
            )
            
            df['Lower_Low'] = df['Swing_Low'].rolling(window=2).apply(
                lambda x: 1 if (x == 1).all() and 
                         df['low'][x.index].diff().iloc[-1] < 0 else 0
            )
            
            # Market Structure Break
            df['Structure_Break'] = 0
            df.loc[(df['Higher_High'] == 1) & (df['Lower_Low'].shift(1) == 1), 'Structure_Break'] = 1
            df.loc[(df['Lower_Low'] == 1) & (df['Higher_High'].shift(1) == 1), 'Structure_Break'] = -1
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding market structure indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Enhanced ATR
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            df['ATR_Percent'] = df['ATR'] / df['close'] * 100
            
            # Volatility Ratio
            df['Volatility_Ratio'] = df['ATR'] / df['ATR'].rolling(window=100).mean()
            
            # Normalized Volatility
            returns = df['close'].pct_change()
            df['Normalized_Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Volatility Breakout
            df['Vol_High'] = df['ATR'].rolling(window=20).max()
            df['Vol_Low'] = df['ATR'].rolling(window=20).min()
            df['Vol_Breakout'] = (df['ATR'] - df['Vol_Low']) / (df['Vol_High'] - df['Vol_Low'])
            
            # Volatility Regime
            df['Vol_Regime'] = pd.qcut(
                df['Normalized_Volatility'].fillna(0),
                q=3,
                labels=['Low', 'Medium', 'High']
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding volatility indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Enhanced RSI
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            df['RSI_MA'] = df['RSI'].rolling(window=9).mean()
            df['RSI_Divergence'] = df['RSI'] - df['RSI_MA']
            
            # Enhanced MACD
            df['MACD'] = ta.trend.macd_diff(df['close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['close'])
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Divergence'] = df['MACD'] - df['MACD_Signal']
            
            # Momentum Quality
            df['Mom_Quality'] = ta.momentum.momentum(df['close'], window=10)
            df['Mom_Quality'] = df['Mom_Quality'] / df['close'] * 100
            
            # Triple Momentum
            df['Mom_Short'] = ta.momentum.momentum(df['close'], window=5)
            df['Mom_Medium'] = ta.momentum.momentum(df['close'], window=10)
            df['Mom_Long'] = ta.momentum.momentum(df['close'], window=20)
            
            df['Triple_Mom'] = (
                df['Mom_Short'] * 0.5 +
                df['Mom_Medium'] * 0.3 +
                df['Mom_Long'] * 0.2
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding momentum indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Smart Money Index
            df['SMI'] = df['close'] - df['open'].shift(1)
            df['SMI'] = df['SMI'].rolling(window=20).mean()
            
            # Volume Price Trend
            df['VPT'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
            df['VPT'] = df['VPT'].cumsum()
            
            # Trend Strength Index
            df['TSI'] = ta.trend.tsi(df['close'], window_slow=25, window_fast=13)
            
            # Efficiency Ratio
            price_change = abs(df['close'] - df['close'].shift(20))
            path_length = df['high'].rolling(20).max() - df['low'].rolling(20).min()
            df['Efficiency_Ratio'] = price_change / path_length
            
            # Volatility-Adjusted Momentum
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            df['Vol_Adj_Mom'] = returns / volatility
            
            # Adaptive Moving Average
            df['AMA'] = ta.trend.kama(df['close'], window=30)
            
            # Custom Oscillator
            df['Custom_Osc'] = (
                df['RSI'] * 0.3 +
                df['MFI'] * 0.3 +
                ((df['close'] - df['low']) / (df['high'] - df['low'])) * 100 * 0.4
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding custom indicators: {e}", exc_info=True)
            return df
            
    @staticmethod
    def get_indicator_signals(df: pd.DataFrame) -> Dict[str, float]:
        try:
            signals = {}
            
            # Volume Signals
            signals['volume_strength'] = 1 if df['VZO'].iloc[-1] > 0 else -1
            signals['money_flow'] = 1 if df['MFI'].iloc[-1] > 50 else -1
            
            # Order Flow Signals
            signals['buying_pressure'] = 1 if df['OF_Momentum'].iloc[-1] > 0 else -1
            signals['delta_strength'] = 1 if df['Cumulative_Delta'].iloc[-1] > 0 else -1
            
            # Market Structure Signals
            signals['trend_strength'] = 1 if df['Structure_Break'].iloc[-1] > 0 else -1
            signals['structure_quality'] = 1 if (
                df['Higher_High'].iloc[-1] == 1 and
                df['Lower_Low'].iloc[-1] == 0
            ) else -1
            
            # Volatility Signals
            signals['volatility_regime'] = 1 if df['Vol_Regime'].iloc[-1] == 'High' else 0
            signals['vol_breakout'] = 1 if df['Vol_Breakout'].iloc[-1] > 0.8 else -1
            
            # Momentum Signals
            signals['momentum_quality'] = 1 if df['Mom_Quality'].iloc[-1] > 0 else -1
            signals['triple_momentum'] = 1 if df['Triple_Mom'].iloc[-1] > 0 else -1
            
            # Custom Signals
            signals['smart_money'] = 1 if df['SMI'].iloc[-1] > 0 else -1
            signals['efficiency'] = 1 if df['Efficiency_Ratio'].iloc[-1] > 0.7 else -1
            
            # Calculate Overall Signal
            signal_weights = {
                'volume_strength': 0.15,
                'money_flow': 0.1,
                'buying_pressure': 0.15,
                'delta_strength': 0.1,
                'trend_strength': 0.15,
                'structure_quality': 0.1,
                'momentum_quality': 0.15,
                'triple_momentum': 0.1
            }
            
            signals['overall_signal'] = sum(
                signals[k] * v for k, v in signal_weights.items()
                if k in signals
            )
            
            return signals
            
        except Exception as e:
            logging.error(f"Error getting indicator signals: {e}", exc_info=True)
            return {}

    @staticmethod
    def _volume_weighted_average_price(df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
