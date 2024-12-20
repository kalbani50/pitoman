import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
import json
from datetime import datetime, timezone

class StrategyOptimizer:
    def __init__(self):
        self.optimization_history = {}
        self.best_parameters = {}
        self.performance_metrics = {}
        
    async def optimize_strategy_parameters(self,
                                        strategy_name: str,
                                        historical_data: pd.DataFrame,
                                        initial_params: Dict,
                                        optimization_target: str = 'sharpe_ratio') -> Dict:
        try:
            # Prepare data
            train_data, test_data = self._split_data(historical_data)
            
            # Define parameter bounds
            bounds = self._get_parameter_bounds(strategy_name)
            
            # Optimize parameters
            result = minimize(
                fun=lambda x: -self._evaluate_parameters(
                    strategy_name,
                    self._dict_from_array(x, initial_params),
                    train_data,
                    optimization_target
                ),
                x0=self._array_from_dict(initial_params),
                bounds=bounds,
                method='Powell'
            )
            
            # Convert optimized parameters back to dictionary
            optimized_params = self._dict_from_array(result.x, initial_params)
            
            # Validate parameters on test data
            test_performance = self._evaluate_parameters(
                strategy_name,
                optimized_params,
                test_data,
                optimization_target
            )
            
            # Store optimization results
            self._store_optimization_results(
                strategy_name,
                optimized_params,
                test_performance
            )
            
            return optimized_params
            
        except Exception as e:
            logging.error(f"Error optimizing strategy parameters: {e}", exc_info=True)
            return initial_params
            
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Use the last 20% of data for testing
            split_idx = int(len(data) * 0.8)
            return data[:split_idx], data[split_idx:]
            
        except Exception as e:
            logging.error(f"Error splitting data: {e}", exc_info=True)
            return data, data
            
    def _get_parameter_bounds(self, strategy_name: str) -> List[Tuple[float, float]]:
        try:
            # Define parameter bounds for each strategy
            bounds_map = {
                'MACD_RSI': [
                    (5, 30),    # RSI period
                    (5, 20),    # MACD fast period
                    (10, 40),   # MACD slow period
                    (5, 15)     # MACD signal period
                ],
                'VOLATILITY': [
                    (10, 30),   # ATR period
                    (1.5, 3.0), # ATR multiplier
                    (10, 30),   # Bollinger period
                    (1.5, 3.0)  # Bollinger std
                ],
                'MOVING_AVERAGE': [
                    (5, 30),    # Short MA period
                    (20, 100),  # Long MA period
                    (0.5, 2.0)  # Threshold multiplier
                ]
            }
            
            return bounds_map.get(strategy_name, [])
            
        except Exception as e:
            logging.error(f"Error getting parameter bounds: {e}", exc_info=True)
            return []
            
    def _evaluate_parameters(self,
                           strategy_name: str,
                           params: Dict,
                           data: pd.DataFrame,
                           target: str) -> float:
        try:
            # Simulate trading with given parameters
            trades = self._simulate_trading(strategy_name, params, data)
            
            if not trades:
                return float('-inf')
                
            # Calculate performance metrics
            returns = pd.Series([t['return'] for t in trades])
            sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = sum(1 for t in trades if t['return'] > 0) / len(trades)
            
            # Return metric based on optimization target
            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': -max_drawdown,  # Negative because we're minimizing
                'win_rate': win_rate
            }
            
            return metrics.get(target, sharpe_ratio)
            
        except Exception as e:
            logging.error(f"Error evaluating parameters: {e}", exc_info=True)
            return float('-inf')
            
    def _simulate_trading(self,
                         strategy_name: str,
                         params: Dict,
                         data: pd.DataFrame) -> List[Dict]:
        try:
            trades = []
            position = None
            
            for i in range(len(data)-1):
                signals = self._generate_signals(
                    strategy_name,
                    params,
                    data.iloc[i:i+2]
                )
                
                if signals['enter'] and position is None:
                    position = {
                        'entry_price': data['close'].iloc[i+1],
                        'entry_time': data.index[i+1]
                    }
                elif signals['exit'] and position is not None:
                    exit_price = data['close'].iloc[i+1]
                    returns = (exit_price - position['entry_price']) / position['entry_price']
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': data.index[i+1],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': returns
                    })
                    
                    position = None
                    
            return trades
            
        except Exception as e:
            logging.error(f"Error simulating trading: {e}", exc_info=True)
            return []
            
    def _generate_signals(self,
                         strategy_name: str,
                         params: Dict,
                         data: pd.DataFrame) -> Dict[str, bool]:
        try:
            if strategy_name == 'MACD_RSI':
                return self._macd_rsi_signals(params, data)
            elif strategy_name == 'VOLATILITY':
                return self._volatility_signals(params, data)
            elif strategy_name == 'MOVING_AVERAGE':
                return self._moving_average_signals(params, data)
            else:
                return {'enter': False, 'exit': False}
                
        except Exception as e:
            logging.error(f"Error generating signals: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
    def _macd_rsi_signals(self, params: Dict, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            # Calculate indicators
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_signal'].iloc[-1]
            
            # Generate signals
            enter = (rsi < params['rsi_oversold'] and 
                    macd > macd_signal and
                    macd < 0)  # Potential reversal
                    
            exit = (rsi > params['rsi_overbought'] or
                   macd < macd_signal)
                   
            return {'enter': enter, 'exit': exit}
            
        except Exception as e:
            logging.error(f"Error generating MACD/RSI signals: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
    def _volatility_signals(self, params: Dict, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            # Calculate indicators
            atr = data['ATR'].iloc[-1]
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            close = data['close'].iloc[-1]
            
            # Generate signals
            enter = (close < bb_lower and
                    atr > params['min_atr'])
                    
            exit = (close > bb_upper or
                   atr < params['min_atr'])
                   
            return {'enter': enter, 'exit': exit}
            
        except Exception as e:
            logging.error(f"Error generating volatility signals: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
    def _moving_average_signals(self, params: Dict, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            # Calculate indicators
            short_ma = data['EMA_20'].iloc[-1]
            long_ma = data['EMA_50'].iloc[-1]
            
            # Generate signals
            enter = short_ma > long_ma * (1 + params['threshold'])
            exit = short_ma < long_ma * (1 - params['threshold'])
            
            return {'enter': enter, 'exit': exit}
            
        except Exception as e:
            logging.error(f"Error generating moving average signals: {e}", exc_info=True)
            return {'enter': False, 'exit': False}
            
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / running_max - 1
            return abs(drawdowns.min())
            
        except Exception as e:
            logging.error(f"Error calculating max drawdown: {e}", exc_info=True)
            return 0.0
            
    def _store_optimization_results(self,
                                  strategy_name: str,
                                  params: Dict,
                                  performance: float) -> None:
        try:
            timestamp = datetime.now(timezone.utc)
            
            self.optimization_history[strategy_name] = {
                'timestamp': timestamp,
                'parameters': params,
                'performance': performance
            }
            
            # Save to file
            with open(f'optimization_history_{strategy_name}.json', 'w') as f:
                json.dump(
                    self.optimization_history[strategy_name],
                    f,
                    default=str,
                    indent=4
                )
                
        except Exception as e:
            logging.error(f"Error storing optimization results: {e}", exc_info=True)
            
    def _array_from_dict(self, params: Dict) -> np.ndarray:
        return np.array(list(params.values()))
        
    def _dict_from_array(self, arr: np.ndarray, template: Dict) -> Dict:
        return dict(zip(template.keys(), arr))
