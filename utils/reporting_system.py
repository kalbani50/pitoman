import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timezone
import os
import json
import shutil
from pathlib import Path
import asyncio

class ReportingSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    async def generate_report(self, trading_data: Dict) -> str:
        """Generates and saves trading report"""
        try:
            # Generate report data
            report_data = self._prepare_report_data(trading_data)
            
            # Create report filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"trading_report_{timestamp}.json"
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=4)
                
            return str(report_file)
            
        except Exception as e:
            logging.error(f"Error generating report: {e}", exc_info=True)
            return ""
            
    def _prepare_report_data(self, trading_data: Dict) -> Dict:
        """Prepares report data"""
        try:
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'performance_metrics': self._calculate_performance_metrics(
                    trading_data
                ),
                'risk_metrics': self._calculate_risk_metrics(trading_data),
                'trading_summary': self._prepare_trading_summary(trading_data),
                'portfolio_analysis': self._analyze_portfolio(trading_data),
                'recommendations': self._generate_recommendations(trading_data)
            }
            
        except Exception as e:
            logging.error(f"Error preparing report data: {e}", exc_info=True)
            return {}
            
    def _calculate_performance_metrics(self, trading_data: Dict) -> Dict:
        """Calculates performance metrics"""
        try:
            returns = trading_data.get('returns', [])
            
            return {
                'total_return': np.sum(returns),
                'average_return': np.mean(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': self._calculate_win_rate(returns),
                'profit_factor': self._calculate_profit_factor(returns)
            }
            
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return {}
            
    def _calculate_risk_metrics(self, trading_data: Dict) -> Dict:
        """Calculates risk metrics"""
        try:
            returns = trading_data.get('returns', [])
            
            return {
                'volatility': np.std(returns),
                'var': self._calculate_var(returns),
                'cvar': self._calculate_cvar(returns),
                'beta': self._calculate_beta(returns, trading_data.get('market_returns', [])),
                'correlation': self._calculate_correlation(returns, trading_data.get('market_returns', []))
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}", exc_info=True)
            return {}
            
    def _prepare_trading_summary(self, trading_data: Dict) -> Dict:
        """Prepares trading summary"""
        try:
            trades = trading_data.get('trades', [])
            
            return {
                'total_trades': len(trades),
                'winning_trades': sum(1 for t in trades if t['profit'] > 0),
                'losing_trades': sum(1 for t in trades if t['profit'] < 0),
                'average_trade_duration': np.mean([t['duration'] for t in trades]),
                'best_trade': max(trades, key=lambda x: x['profit']),
                'worst_trade': min(trades, key=lambda x: x['profit'])
            }
            
        except Exception as e:
            logging.error(f"Error preparing trading summary: {e}", exc_info=True)
            return {}
            
    async def cleanup_old_reports(self) -> None:
        """Cleans up reports older than one month"""
        try:
            current_time = datetime.now(timezone.utc)
            
            for report_file in self.reports_dir.glob("*.json"):
                # Extract timestamp from filename
                timestamp_str = report_file.stem.split('_')[-1]
                report_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                report_time = report_time.replace(tzinfo=timezone.utc)
                
                # Remove if older than a month
                if (current_time - report_time).days > 30:
                    report_file.unlink()
                    
        except Exception as e:
            logging.error(f"Error cleaning up reports: {e}", exc_info=True)
            
    async def export_report(self, report_file: str, format: str = 'pdf') -> str:
        """Exports report to different formats"""
        try:
            report_path = Path(report_file)
            
            if not report_path.exists():
                raise FileNotFoundError(f"Report file not found: {report_file}")
                
            # Load report data
            with open(report_path, 'r') as f:
                report_data = json.load(f)
                
            # Create export filename
            export_file = report_path.with_suffix(f".{format}")
            
            if format == 'pdf':
                await self._export_to_pdf(report_data, export_file)
            elif format == 'excel':
                await self._export_to_excel(report_data, export_file)
            elif format == 'html':
                await self._export_to_html(report_data, export_file)
                
            return str(export_file)
            
        except Exception as e:
            logging.error(f"Error exporting report: {e}", exc_info=True)
            return ""
            
    def _analyze_portfolio(self, trading_data: Dict) -> Dict:
        """Analyzes portfolio composition and performance"""
        try:
            portfolio = trading_data.get('portfolio', {})
            
            return {
                'asset_allocation': self._calculate_asset_allocation(portfolio),
                'sector_exposure': self._calculate_sector_exposure(portfolio),
                'concentration_risk': self._calculate_concentration_risk(portfolio),
                'portfolio_metrics': self._calculate_portfolio_metrics(portfolio)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing portfolio: {e}", exc_info=True)
            return {}
            
    def _generate_recommendations(self, trading_data: Dict) -> List[Dict]:
        """Generates trading recommendations"""
        try:
            recommendations = []
            
            # Analyze performance
            if issues := self._analyze_performance_issues(trading_data):
                recommendations.extend(issues)
                
            # Analyze risk
            if risk_recommendations := self._analyze_risk_issues(trading_data):
                recommendations.extend(risk_recommendations)
                
            # Analyze portfolio
            if portfolio_recommendations := self._analyze_portfolio_issues(trading_data):
                recommendations.extend(portfolio_recommendations)
                
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}", exc_info=True)
            return []
