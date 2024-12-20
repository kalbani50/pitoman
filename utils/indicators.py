import pandas as pd
import numpy as np
import ta
from typing import Optional
import logging

class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Basic Indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
            
            # Moving Averages
            df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['EMA_100'] = ta.trend.EMAIndicator(df['close'], window=100).ema_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['SMA_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            
            # Volatility Indicators
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_upper'] = bollinger.bollinger_hband()
            
            # Volume Indicators
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Trend Indicators
            ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
            df['ICHIMOKU_A'] = ichimoku.ichimoku_a()
            df['ICHIMOKU_B'] = ichimoku.ichimoku_b()
            
            # Momentum Indicators
            df['MOM'] = ta.momentum.ROCIndicator(df['close']).roc()
            df['ROC'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            df['WILLR'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['STOCH_k'] = stoch.stoch()
            df['STOCH_d'] = stoch.stoch_signal()
            
            # Additional Indicators
            df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['ADX'] = adx.adx()
            df['DMP'] = adx.adx_pos()
            df['DMN'] = adx.adx_neg()
            
            # Advanced Indicators
            TechnicalIndicators.add_supertrend(df)
            TechnicalIndicators.add_hull_ma(df)
            TechnicalIndicators.add_keltner_channels(df)
            TechnicalIndicators.add_money_flow_indicators(df)
            TechnicalIndicators.add_psar(df)
            
            # Clean up NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}", exc_info=True)
            raise

    @staticmethod
    def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
            
            hl2 = (df['high'] + df['low']) / 2
            final_upperband = hl2 + multiplier * atr
            final_lowerband = hl2 - multiplier * atr
            
            supertrend = pd.Series(0.0, index=df.index)
            
            for i in range(1, len(df.index)):
                if df['close'][i] > final_upperband[i-1]:
                    supertrend[i] = final_lowerband[i]
                elif df['close'][i] < final_lowerband[i-1]:
                    supertrend[i] = final_upperband[i]
                else:
                    supertrend[i] = supertrend[i-1]
                    
            df['SuperTrend'] = supertrend
            
        except Exception as e:
            logging.error(f"Error calculating SuperTrend: {e}", exc_info=True)
            raise

    @staticmethod
    def add_hull_ma(df: pd.DataFrame, period: int = 20):
        try:
            wma1 = df['close'].rolling(window=period//2).mean()
            wma2 = df['close'].rolling(window=period).mean()
            diff = 2 * wma1 - wma2
            df['HullMA'] = diff.rolling(window=int(np.sqrt(period))).mean()
            
        except Exception as e:
            logging.error(f"Error calculating Hull MA: {e}", exc_info=True)
            raise

    @staticmethod
    def add_keltner_channels(df: pd.DataFrame, period: int = 20, multiplier: int = 2):
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            ema = typical_price.ewm(span=period, adjust=False).mean()
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
            
            df['KeltnerChannels_Upper'] = ema + (multiplier * atr)
            df['KeltnerChannels_Lower'] = ema - (multiplier * atr)
            
        except Exception as e:
            logging.error(f"Error calculating Keltner Channels: {e}", exc_info=True)
            raise

    @staticmethod
    def add_money_flow_indicators(df: pd.DataFrame):
        try:
            # Chaikin Money Flow
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfv = mfm * df['volume']
            df['CMF'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Money Flow Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0.0, index=df.index)
            negative_flow = pd.Series(0.0, index=df.index)
            
            for i in range(1, len(df)):
                if typical_price[i] > typical_price[i-1]:
                    positive_flow[i] = money_flow[i]
                elif typical_price[i] < typical_price[i-1]:
                    negative_flow[i] = money_flow[i]
                    
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            df['MFI'] = mfi
            
        except Exception as e:
            logging.error(f"Error calculating Money Flow indicators: {e}", exc_info=True)
            raise

    @staticmethod
    def add_psar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2):
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            psar = close.copy()
            psarbull = [True] * len(close)
            bull = True
            
            af = acceleration
            hp = high[0]
            lp = low[0]
            
            for i in range(2, len(close)):
                if bull:
                    psar[i] = psar[i-1] + af * (hp - psar[i-1])
                else:
                    psar[i] = psar[i-1] + af * (lp - psar[i-1])
                
                reverse = False
                
                if bull:
                    if low[i] < psar[i]:
                        bull = False
                        reverse = True
                        psar[i] = hp
                        lp = low[i]
                        af = acceleration
                else:
                    if high[i] > psar[i]:
                        bull = True
                        reverse = True
                        psar[i] = lp
                        hp = high[i]
                        af = acceleration
            
                if not reverse:
                    if bull:
                        if high[i] > hp:
                            hp = high[i]
                            af = min(af + acceleration, maximum)
                        if low[i-1] < psar[i]:
                            psar[i] = low[i-1]
                        if low[i-2] < psar[i]:
                            psar[i] = low[i-2]
                    else:
                        if low[i] < lp:
                            lp = low[i]
                            af = min(af + acceleration, maximum)
                        if high[i-1] > psar[i]:
                            psar[i] = high[i-1]
                        if high[i-2] > psar[i]:
                            psar[i] = high[i-2]
                
            df['PSar'] = psar
            
        except Exception as e:
            logging.error(f"Error calculating PSAR: {e}", exc_info=True)
            raise
