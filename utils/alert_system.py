import tkinter as tk
from tkinter import ttk
from typing import Dict, List
import logging
from datetime import datetime, timezone
import asyncio
from plyer import notification
import json
import os

class AlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.active_alerts = {}
        self.alert_history = []
        
    async def show_popup_alert(self, alert_data: Dict) -> None:
        """Shows popup notification"""
        try:
            # Create system notification
            notification.notify(
                title=alert_data['title'],
                message=alert_data['message'],
                app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
                timeout=10,  # seconds
                toast=True
            )
            
            # Store alert in history
            self.alert_history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': alert_data['type'],
                'title': alert_data['title'],
                'message': alert_data['message']
            })
            
        except Exception as e:
            logging.error(f"Error showing alert: {e}", exc_info=True)
            
    async def process_trading_alert(self, trading_data: Dict) -> None:
        """Processes and shows trading alerts"""
        try:
            # Check for various alert conditions
            alerts = []
            
            # Price alerts
            if self._check_price_alert(trading_data):
                alerts.append({
                    'type': 'price',
                    'title': 'Price Alert',
                    'message': f"Price threshold reached for {trading_data['symbol']}"
                })
                
            # Volume alerts
            if self._check_volume_alert(trading_data):
                alerts.append({
                    'type': 'volume',
                    'title': 'Volume Alert',
                    'message': f"Unusual volume detected for {trading_data['symbol']}"
                })
                
            # Pattern alerts
            if patterns := self._check_pattern_alert(trading_data):
                alerts.append({
                    'type': 'pattern',
                    'title': 'Pattern Alert',
                    'message': f"Patterns detected: {', '.join(patterns)}"
                })
                
            # Risk alerts
            if risks := self._check_risk_alert(trading_data):
                alerts.append({
                    'type': 'risk',
                    'title': 'Risk Alert',
                    'message': f"Risk factors detected: {', '.join(risks)}"
                })
                
            # Show alerts
            for alert in alerts:
                await self.show_popup_alert(alert)
                
        except Exception as e:
            logging.error(f"Error processing trading alert: {e}", exc_info=True)
            
    def _check_price_alert(self, trading_data: Dict) -> bool:
        """Checks for price alert conditions"""
        try:
            current_price = trading_data.get('price', 0)
            
            # Check upper threshold
            if current_price >= self.config['price_upper_threshold']:
                return True
                
            # Check lower threshold
            if current_price <= self.config['price_lower_threshold']:
                return True
                
            # Check price change
            if abs(trading_data.get('price_change', 0)) >= self.config['price_change_threshold']:
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking price alert: {e}", exc_info=True)
            return False
            
    def _check_volume_alert(self, trading_data: Dict) -> bool:
        """Checks for volume alert conditions"""
        try:
            current_volume = trading_data.get('volume', 0)
            avg_volume = trading_data.get('average_volume', 0)
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                return volume_ratio >= self.config['volume_threshold']
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking volume alert: {e}", exc_info=True)
            return False
            
    def _check_pattern_alert(self, trading_data: Dict) -> List[str]:
        """Checks for pattern alert conditions"""
        try:
            patterns = []
            
            # Check various patterns
            if self._check_trend_reversal(trading_data):
                patterns.append('Trend Reversal')
                
            if self._check_breakout(trading_data):
                patterns.append('Breakout')
                
            if self._check_consolidation(trading_data):
                patterns.append('Consolidation')
                
            return patterns
            
        except Exception as e:
            logging.error(f"Error checking pattern alert: {e}", exc_info=True)
            return []
            
    def _check_risk_alert(self, trading_data: Dict) -> List[str]:
        """Checks for risk alert conditions"""
        try:
            risks = []
            
            # Check various risk factors
            if self._check_volatility_risk(trading_data):
                risks.append('High Volatility')
                
            if self._check_exposure_risk(trading_data):
                risks.append('High Exposure')
                
            if self._check_correlation_risk(trading_data):
                risks.append('High Correlation')
                
            return risks
            
        except Exception as e:
            logging.error(f"Error checking risk alert: {e}", exc_info=True)
            return []
            
    async def cleanup_alert_history(self) -> None:
        """Cleans up old alerts"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Keep only last month's alerts
            self.alert_history = [
                alert for alert in self.alert_history
                if (current_time - datetime.fromisoformat(alert['timestamp'])).days <= 30
            ]
            
        except Exception as e:
            logging.error(f"Error cleaning up alerts: {e}", exc_info=True)
