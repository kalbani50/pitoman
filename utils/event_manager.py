import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging
import json
from decimal import Decimal

class EventManager:
    def __init__(self):
        self.events = []
        self.performance_metrics = {}
        self.strategy_performance = {}
        self.risk_metrics = {}
        
    async def log_trade_event(self, trade_data: Dict) -> None:
        try:
            event = {
                "timestamp": datetime.now(timezone.utc),
                "type": "trade",
                "data": trade_data
            }
            self.events.append(event)
            await self.update_metrics(trade_data)
        except Exception as e:
            logging.error(f"Error logging trade event: {e}", exc_info=True)
            
    async def log_error_event(self, error_data: Dict) -> None:
        try:
            event = {
                "timestamp": datetime.now(timezone.utc),
                "type": "error",
                "data": error_data
            }
            self.events.append(event)
        except Exception as e:
            logging.error(f"Error logging error event: {e}", exc_info=True)
            
    async def update_metrics(self, trade_data: Dict) -> None:
        try:
            pair = trade_data['pair']
            strategy = trade_data.get('strategy', 'unknown')
            profit = Decimal(str(trade_data.get('profit', '0')))
            
            # Update pair performance
            if pair not in self.performance_metrics:
                self.performance_metrics[pair] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': Decimal('0'),
                    'max_drawdown': Decimal('0'),
                    'best_trade': Decimal('0'),
                    'worst_trade': Decimal('0')
                }
                
            metrics = self.performance_metrics[pair]
            metrics['total_trades'] += 1
            if profit > 0:
                metrics['winning_trades'] += 1
            metrics['total_profit'] += profit
            
            metrics['best_trade'] = max(metrics['best_trade'], profit)
            metrics['worst_trade'] = min(metrics['worst_trade'], profit)
            
            # Update strategy performance
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': Decimal('0'),
                    'average_profit': Decimal('0')
                }
                
            strat_metrics = self.strategy_performance[strategy]
            strat_metrics['total_trades'] += 1
            if profit > 0:
                strat_metrics['winning_trades'] += 1
            strat_metrics['total_profit'] += profit
            strat_metrics['average_profit'] = (
                strat_metrics['total_profit'] / strat_metrics['total_trades']
            )
            
            # Update risk metrics
            self.update_risk_metrics(trade_data)
            
        except Exception as e:
            logging.error(f"Error updating metrics: {e}", exc_info=True)
            
    def update_risk_metrics(self, trade_data: Dict) -> None:
        try:
            pair = trade_data['pair']
            if pair not in self.risk_metrics:
                self.risk_metrics[pair] = {
                    'volatility': Decimal('0'),
                    'sharpe_ratio': Decimal('0'),
                    'max_drawdown': Decimal('0'),
                    'risk_reward_ratio': Decimal('0'),
                    'win_rate': Decimal('0')
                }
                
            metrics = self.risk_metrics[pair]
            pair_metrics = self.performance_metrics[pair]
            
            # Calculate win rate
            if pair_metrics['total_trades'] > 0:
                metrics['win_rate'] = (
                    Decimal(str(pair_metrics['winning_trades'])) /
                    Decimal(str(pair_metrics['total_trades']))
                ) * 100
                
            # Calculate risk-reward ratio
            if pair_metrics['worst_trade'] != 0:
                metrics['risk_reward_ratio'] = abs(
                    pair_metrics['best_trade'] / pair_metrics['worst_trade']
                )
                
        except Exception as e:
            logging.error(f"Error updating risk metrics: {e}", exc_info=True)
            
    def get_performance_summary(self) -> Dict:
        try:
            summary = {
                'overall': {
                    'total_profit': sum(
                        m['total_profit'] for m in self.performance_metrics.values()
                    ),
                    'total_trades': sum(
                        m['total_trades'] for m in self.performance_metrics.values()
                    ),
                    'winning_trades': sum(
                        m['winning_trades'] for m in self.performance_metrics.values()
                    )
                },
                'by_pair': self.performance_metrics,
                'by_strategy': self.strategy_performance,
                'risk_metrics': self.risk_metrics
            }
            
            # Calculate overall win rate
            if summary['overall']['total_trades'] > 0:
                summary['overall']['win_rate'] = (
                    Decimal(str(summary['overall']['winning_trades'])) /
                    Decimal(str(summary['overall']['total_trades']))
                ) * 100
                
            return summary
        except Exception as e:
            logging.error(f"Error getting performance summary: {e}", exc_info=True)
            return {}
            
    async def export_metrics(self, filepath: str) -> None:
        try:
            summary = self.get_performance_summary()
            # Convert Decimal objects to strings for JSON serialization
            summary_str = json.dumps(summary, default=str, indent=4)
            
            with open(filepath, 'w') as f:
                f.write(summary_str)
                
            logging.info(f"Metrics exported successfully to {filepath}")
        except Exception as e:
            logging.error(f"Error exporting metrics: {e}", exc_info=True)
            
    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        return self.events[-limit:] if self.events else []
