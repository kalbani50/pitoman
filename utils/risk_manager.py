from decimal import Decimal
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

class RiskManager:
    def __init__(self, initial_capital: Decimal):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = Decimal('0.02')  # 2% max risk per trade
        self.max_total_risk = Decimal('0.06')      # 6% max total risk
        self.current_positions = {}
        self.trade_history = []
        self.drawdown_threshold = Decimal('0.10')  # 10% max drawdown
        
    def calculate_position_size(self, 
                              pair: str,
                              entry_price: Decimal,
                              stop_loss: Decimal,
                              market_volatility: float,
                              market_trend: str) -> Optional[Decimal]:
        try:
            # Calculate base position size using risk amount
            risk_amount = self.current_capital * self.max_risk_per_trade
            
            # Calculate position size based on stop loss distance
            stop_loss_distance = abs(entry_price - stop_loss)
            if stop_loss_distance == 0:
                logging.warning(f"Stop loss distance is zero for {pair}")
                return None
                
            base_position_size = risk_amount / stop_loss_distance
            
            # Adjust position size based on market conditions
            adjusted_position_size = self._adjust_position_size(
                base_position_size,
                market_volatility,
                market_trend
            )
            
            # Check if position size exceeds maximum risk
            if not self._validate_position_size(adjusted_position_size, entry_price):
                logging.warning(f"Position size exceeds maximum risk for {pair}")
                return None
                
            return adjusted_position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}", exc_info=True)
            return None
            
    def _adjust_position_size(self,
                            base_size: Decimal,
                            volatility: float,
                            trend: str) -> Decimal:
        try:
            # Volatility adjustment
            volatility_factor = Decimal(str(1 - min(volatility * 2, 0.5)))
            
            # Trend adjustment
            trend_factor = Decimal('1.2') if trend == 'strong_uptrend' else \
                         Decimal('1.0') if trend == 'uptrend' else \
                         Decimal('0.8') if trend == 'downtrend' else \
                         Decimal('0.6')  # strong_downtrend
                         
            # Calculate adjusted size
            adjusted_size = base_size * volatility_factor * trend_factor
            
            # Round to appropriate precision
            return self._round_position_size(adjusted_size)
            
        except Exception as e:
            logging.error(f"Error adjusting position size: {e}", exc_info=True)
            return base_size
            
    def _validate_position_size(self, position_size: Decimal, price: Decimal) -> bool:
        try:
            # Calculate total position value
            position_value = position_size * price
            
            # Calculate current total exposure
            current_exposure = sum(
                pos['size'] * pos['price'] 
                for pos in self.current_positions.values()
            )
            
            # Check if new position would exceed max total risk
            total_exposure = current_exposure + position_value
            max_exposure = self.current_capital * (Decimal('1') + self.max_total_risk)
            
            return total_exposure <= max_exposure
            
        except Exception as e:
            logging.error(f"Error validating position size: {e}", exc_info=True)
            return False
            
    def _round_position_size(self, size: Decimal) -> Decimal:
        try:
            # Round to appropriate number of decimal places based on size
            if size >= 1:
                return Decimal(str(round(float(size), 2)))
            elif size >= 0.1:
                return Decimal(str(round(float(size), 3)))
            else:
                return Decimal(str(round(float(size), 4)))
                
        except Exception as e:
            logging.error(f"Error rounding position size: {e}", exc_info=True)
            return size
            
    def update_position(self,
                       pair: str,
                       size: Decimal,
                       price: Decimal,
                       side: str) -> None:
        try:
            if side == 'buy':
                self.current_positions[pair] = {
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now(timezone.utc)
                }
            elif side == 'sell' and pair in self.current_positions:
                # Calculate profit/loss
                entry_price = self.current_positions[pair]['price']
                entry_size = self.current_positions[pair]['size']
                pnl = (price - entry_price) * entry_size
                
                # Update trade history
                self.trade_history.append({
                    'pair': pair,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'size': entry_size,
                    'pnl': pnl,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                # Update capital
                self.current_capital += pnl
                
                # Remove position
                del self.current_positions[pair]
                
        except Exception as e:
            logging.error(f"Error updating position: {e}", exc_info=True)
            
    def check_drawdown(self) -> bool:
        try:
            if self.initial_capital == 0:
                return False
                
            drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
            return drawdown > self.drawdown_threshold
            
        except Exception as e:
            logging.error(f"Error checking drawdown: {e}", exc_info=True)
            return False
            
    def get_risk_metrics(self) -> Dict:
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': Decimal('0'),
                    'average_profit': Decimal('0'),
                    'max_drawdown': Decimal('0'),
                    'risk_reward_ratio': Decimal('0'),
                    'sharpe_ratio': Decimal('0')
                }
                
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            profitable_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
            win_rate = Decimal(str(profitable_trades)) / Decimal(str(total_trades))
            
            # Calculate profit metrics
            profits = [trade['pnl'] for trade in self.trade_history]
            average_profit = sum(profits) / len(profits)
            
            # Calculate max drawdown
            cumulative_profits = np.cumsum([float(p) for p in profits])
            max_drawdown = min(0, np.min(cumulative_profits))
            
            # Calculate risk-reward ratio
            if profitable_trades > 0:
                avg_win = sum(p for p in profits if p > 0) / profitable_trades
                avg_loss = abs(sum(p for p in profits if p < 0) / (total_trades - profitable_trades))
                risk_reward = avg_win / avg_loss if avg_loss != 0 else 0
            else:
                risk_reward = 0
                
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            returns = pd.Series(profits)
            sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'average_profit': average_profit,
                'max_drawdown': Decimal(str(max_drawdown)),
                'risk_reward_ratio': Decimal(str(risk_reward)),
                'sharpe_ratio': Decimal(str(sharpe_ratio))
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}", exc_info=True)
            return {}
