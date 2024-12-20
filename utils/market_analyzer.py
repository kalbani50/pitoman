import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ta

class MarketAnalyzer:
    def __init__(self):
        self.market_states = {}
        self.trend_changes = {}
        self.volatility_levels = {}
        self.support_resistance = {}
        
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        try:
            # Identify market state
            market_state = self._identify_market_state(df)
            
            # Detect support and resistance levels
            support_resistance = self._find_support_resistance(df)
            
            # Analyze volume profile
            volume_profile = self._analyze_volume_profile(df)
            
            # Detect market patterns
            patterns = self._detect_patterns(df)
            
            # Calculate market strength
            strength_indicators = self._calculate_market_strength(df)
            
            return {
                'market_state': market_state,
                'support_resistance': support_resistance,
                'volume_profile': volume_profile,
                'patterns': patterns,
                'strength': strength_indicators
            }
        except Exception as e:
            logging.error(f"Error in market structure analysis: {e}", exc_info=True)
            return {}
            
    def _identify_market_state(self, df: pd.DataFrame) -> str:
        try:
            # Calculate trend using multiple timeframes
            short_trend = df['close'].rolling(window=20).mean()
            long_trend = df['close'].rolling(window=50).mean()
            
            # Calculate volatility
            volatility = df['close'].pct_change().std()
            
            # Determine market state
            current_close = df['close'].iloc[-1]
            current_short = short_trend.iloc[-1]
            current_long = long_trend.iloc[-1]
            
            if current_short > current_long:
                if volatility > 0.02:
                    return 'volatile_uptrend'
                return 'uptrend'
            elif current_short < current_long:
                if volatility > 0.02:
                    return 'volatile_downtrend'
                return 'downtrend'
            else:
                if volatility > 0.02:
                    return 'volatile_sideways'
                return 'sideways'
                
        except Exception as e:
            logging.error(f"Error identifying market state: {e}", exc_info=True)
            return 'unknown'
            
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        try:
            # Use fractal method to identify potential levels
            highs = df['high'].values
            lows = df['low'].values
            
            support_levels = []
            resistance_levels = []
            
            # Find resistance levels
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
                    
            # Find support levels
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
                    
            # Cluster levels to remove noise
            if support_levels:
                support_levels = self._cluster_levels(support_levels)
            if resistance_levels:
                resistance_levels = self._cluster_levels(resistance_levels)
                
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logging.error(f"Error finding support/resistance: {e}", exc_info=True)
            return {'support': [], 'resistance': []}
            
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        try:
            if not levels:
                return []
                
            levels = np.array(levels).reshape(-1, 1)
            scaler = StandardScaler()
            levels_scaled = scaler.fit_transform(levels)
            
            # Determine optimal number of clusters
            n_clusters = min(len(levels), 5)  # Maximum 5 levels
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(levels_scaled)
            
            # Get cluster centers and convert back to original scale
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            return sorted(centers.flatten().tolist())
            
        except Exception as e:
            logging.error(f"Error clustering levels: {e}", exc_info=True)
            return []
            
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        try:
            # Calculate volume-weighted average price (VWAP)
            df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Analyze volume trends
            volume_sma = df['volume'].rolling(window=20).mean()
            current_vol = df['volume'].iloc[-1]
            avg_vol = volume_sma.iloc[-1]
            
            # Volume profile analysis
            return {
                'volume_trend': 'increasing' if current_vol > avg_vol else 'decreasing',
                'volume_ratio': float(current_vol / avg_vol),
                'vwap': float(df['VWAP'].iloc[-1])
            }
            
        except Exception as e:
            logging.error(f"Error analyzing volume profile: {e}", exc_info=True)
            return {}
            
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        try:
            patterns = []
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Detect double top
            if self._is_double_top(highs[-100:]):
                patterns.append('double_top')
                
            # Detect double bottom
            if self._is_double_bottom(lows[-100:]):
                patterns.append('double_bottom')
                
            # Detect head and shoulders
            if self._is_head_and_shoulders(highs[-100:]):
                patterns.append('head_and_shoulders')
                
            # Detect triangle patterns
            if self._is_triangle(highs[-50:], lows[-50:]):
                patterns.append('triangle')
                
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting patterns: {e}", exc_info=True)
            return []
            
    def _is_double_top(self, prices: np.ndarray, tolerance: float = 0.02) -> bool:
        try:
            peaks = []
            for i in range(1, len(prices)-1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    peaks.append((i, prices[i]))
                    
            if len(peaks) < 2:
                return False
                
            # Check if the two highest peaks are within tolerance
            peaks.sort(key=lambda x: x[1], reverse=True)
            top1, top2 = peaks[0], peaks[1]
            
            price_diff = abs(top1[1] - top2[1]) / top1[1]
            index_diff = abs(top1[0] - top2[0])
            
            return price_diff <= tolerance and index_diff >= 10
            
        except Exception as e:
            logging.error(f"Error detecting double top: {e}", exc_info=True)
            return False
            
    def _is_double_bottom(self, prices: np.ndarray, tolerance: float = 0.02) -> bool:
        try:
            bottoms = []
            for i in range(1, len(prices)-1):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    bottoms.append((i, prices[i]))
                    
            if len(bottoms) < 2:
                return False
                
            # Check if the two lowest bottoms are within tolerance
            bottoms.sort(key=lambda x: x[1])
            bottom1, bottom2 = bottoms[0], bottoms[1]
            
            price_diff = abs(bottom1[1] - bottom2[1]) / bottom1[1]
            index_diff = abs(bottom1[0] - bottom2[0])
            
            return price_diff <= tolerance and index_diff >= 10
            
        except Exception as e:
            logging.error(f"Error detecting double bottom: {e}", exc_info=True)
            return False
            
    def _is_head_and_shoulders(self, prices: np.ndarray, tolerance: float = 0.02) -> bool:
        try:
            peaks = []
            for i in range(1, len(prices)-1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    peaks.append((i, prices[i]))
                    
            if len(peaks) < 3:
                return False
                
            # Find three consecutive peaks
            for i in range(len(peaks)-2):
                left = peaks[i]
                head = peaks[i+1]
                right = peaks[i+2]
                
                # Check if head is higher than shoulders
                if (head[1] > left[1] and head[1] > right[1] and
                    abs(left[1] - right[1]) / left[1] <= tolerance):
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error detecting head and shoulders: {e}", exc_info=True)
            return False
            
    def _is_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        try:
            # Calculate trend lines
            high_trend = np.polyfit(range(len(highs)), highs, 1)
            low_trend = np.polyfit(range(len(lows)), lows, 1)
            
            # Check if trends are converging
            high_slope = high_trend[0]
            low_slope = low_trend[0]
            
            return (high_slope < 0 and low_slope > 0) or (high_slope > 0 and low_slope < 0)
            
        except Exception as e:
            logging.error(f"Error detecting triangle: {e}", exc_info=True)
            return False
            
    def _calculate_market_strength(self, df: pd.DataFrame) -> Dict:
        try:
            # Calculate various strength indicators
            rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            
            # Calculate price momentum
            momentum = df['close'].pct_change(periods=20).iloc[-1] * 100
            
            # Calculate volume strength
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_strength = current_volume / volume_ma
            
            return {
                'rsi_strength': float(rsi),
                'trend_strength': float(adx),
                'momentum': float(momentum),
                'volume_strength': float(volume_strength)
            }
            
        except Exception as e:
            logging.error(f"Error calculating market strength: {e}", exc_info=True)
            return {}
