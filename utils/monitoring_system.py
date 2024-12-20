import asyncio
from typing import Dict, List
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from collections import deque

class MonitoringSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.active_trades = {}
        self.performance_metrics = deque(maxlen=1000)
        self.risk_metrics = deque(maxlen=1000)
        self.system_health = {}
        
    async def monitor_trading(self, trading_data: Dict) -> Dict:
        """Monitors trading activities in real-time"""
        try:
            # Update metrics
            self._update_metrics(trading_data)
            
            # Monitor active trades
            trade_status = await self._monitor_active_trades(
                trading_data
            )
            
            # Monitor system health
            system_health = self._monitor_system_health()
            
            # Check for interventions
            interventions = await self._check_interventions(
                trade_status,
                system_health
            )
            
            return {
                'trade_status': trade_status,
                'system_health': system_health,
                'interventions': interventions,
                'metrics': self._get_current_metrics()
            }
            
        except Exception as e:
            logging.error(f"Error monitoring trading: {e}", exc_info=True)
            return {}
            
    def _update_metrics(self, trading_data: Dict) -> None:
        """Updates performance and risk metrics"""
        try:
            # Update performance metrics
            self.performance_metrics.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'returns': trading_data.get('returns', 0),
                'pnl': trading_data.get('pnl', 0),
                'win_rate': trading_data.get('win_rate', 0)
            })
            
            # Update risk metrics
            self.risk_metrics.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'volatility': trading_data.get('volatility', 0),
                'drawdown': trading_data.get('drawdown', 0),
                'var': trading_data.get('var', 0)
            })
            
        except Exception as e:
            logging.error(f"Error updating metrics: {e}", exc_info=True)
            
    async def _monitor_active_trades(self, trading_data: Dict) -> Dict:
        """Monitors active trades"""
        try:
            active_trades = trading_data.get('active_trades', {})
            trade_status = {}
            
            for trade_id, trade in active_trades.items():
                # Calculate trade metrics
                metrics = self._calculate_trade_metrics(trade)
                
                # Check trade health
                health = self._check_trade_health(trade, metrics)
                
                # Generate warnings
                warnings = self._generate_trade_warnings(trade, metrics)
                
                trade_status[trade_id] = {
                    'metrics': metrics,
                    'health': health,
                    'warnings': warnings
                }
                
            return trade_status
            
        except Exception as e:
            logging.error(f"Error monitoring active trades: {e}", exc_info=True)
            return {}
            
    def _monitor_system_health(self) -> Dict:
        """Monitors overall system health"""
        try:
            return {
                'performance_health': self._check_performance_health(),
                'risk_health': self._check_risk_health(),
                'technical_health': self._check_technical_health(),
                'resource_health': self._check_resource_health()
            }
            
        except Exception as e:
            logging.error(f"Error monitoring system health: {e}", exc_info=True)
            return {}
            
    async def _check_interventions(self,
                                 trade_status: Dict,
                                 system_health: Dict) -> List[Dict]:
        """Checks if interventions are needed"""
        try:
            interventions = []
            
            # Check trade interventions
            if trade_interventions := self._check_trade_interventions(
                trade_status
            ):
                interventions.extend(trade_interventions)
                
            # Check system interventions
            if system_interventions := self._check_system_interventions(
                system_health
            ):
                interventions.extend(system_interventions)
                
            # Check risk interventions
            if risk_interventions := self._check_risk_interventions(
                trade_status,
                system_health
            ):
                interventions.extend(risk_interventions)
                
            return interventions
            
        except Exception as e:
            logging.error(f"Error checking interventions: {e}", exc_info=True)
            return []
            
    def _calculate_trade_metrics(self, trade: Dict) -> Dict:
        """Calculates metrics for a single trade"""
        try:
            entry_price = trade.get('entry_price', 0)
            current_price = trade.get('current_price', 0)
            position_size = trade.get('position_size', 0)
            
            return {
                'unrealized_pnl': (current_price - entry_price) * position_size,
                'duration': (datetime.now(timezone.utc) - datetime.fromisoformat(trade['entry_time'])).seconds,
                'risk_reward_ratio': self._calculate_risk_reward_ratio(trade),
                'max_adverse_excursion': self._calculate_max_adverse_excursion(trade)
            }
            
        except Exception as e:
            logging.error(f"Error calculating trade metrics: {e}", exc_info=True)
            return {}
            
    def _check_trade_health(self, trade: Dict, metrics: Dict) -> str:
        """Checks health of a single trade"""
        try:
            if metrics['unrealized_pnl'] <= -self.config['max_loss_threshold']:
                return 'critical'
            elif metrics['risk_reward_ratio'] < self.config['min_risk_reward_ratio']:
                return 'warning'
            elif metrics['duration'] > self.config['max_trade_duration']:
                return 'warning'
            else:
                return 'healthy'
                
        except Exception as e:
            logging.error(f"Error checking trade health: {e}", exc_info=True)
            return 'unknown'
            
    def _generate_trade_warnings(self, trade: Dict, metrics: Dict) -> List[str]:
        """Generates warnings for a single trade"""
        try:
            warnings = []
            
            if metrics['unrealized_pnl'] < 0:
                warnings.append('Negative unrealized P&L')
                
            if metrics['risk_reward_ratio'] < self.config['min_risk_reward_ratio']:
                warnings.append('Poor risk-reward ratio')
                
            if metrics['duration'] > self.config['max_trade_duration']:
                warnings.append('Trade duration exceeded')
                
            return warnings
            
        except Exception as e:
            logging.error(f"Error generating trade warnings: {e}", exc_info=True)
            return []
