"""
AGI Trading Bot - Main Entry Point
Autonomous trading system with minimal human intervention
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from core.config import Config
from core.trading_engine import TradingEngine
from core.risk_manager import RiskManager
from core.market_analyzer import MarketAnalyzer
from core.position_manager import PositionManager
from core.exchange_manager import ExchangeManager
from utils.logger import setup_logging
from utils.notification import NotificationManager
from utils.system_monitor import SystemMonitor
from utils.database import Database
from utils.backup import BackupManager

class AGITradingBot:
    """Main bot class that orchestrates all components"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.maintenance_mode = False
        
        # Initialize core components
        self.db = Database(self.config.database)
        self.exchange = ExchangeManager(self.config.exchange)
        self.market_analyzer = MarketAnalyzer(self.config.analysis)
        self.risk_manager = RiskManager(self.config.risk)
        self.position_manager = PositionManager(self.config.position)
        self.trading_engine = TradingEngine(
            self.exchange,
            self.market_analyzer,
            self.risk_manager,
            self.position_manager
        )
        
        # Initialize utility components
        self.notification = NotificationManager(self.config.notification)
        self.system_monitor = SystemMonitor(self.config.monitoring)
        self.backup_manager = BackupManager(self.config.backup)
        
    async def initialize(self) -> bool:
        """Initialize all systems and perform safety checks"""
        try:
            self.logger.info("Initializing AGI Trading Bot...")
            
            # Validate configuration
            if not self.config.validate():
                self.logger.error("Invalid configuration")
                return False
                
            # Initialize database
            if not await self.db.initialize():
                self.logger.error("Database initialization failed")
                return False
                
            # Connect to exchange
            if not await self.exchange.connect():
                self.logger.error("Exchange connection failed")
                return False
                
            # Load AI models
            if not await self.market_analyzer.load_models():
                self.logger.error("Failed to load AI models")
                return False
                
            # Initialize risk management
            if not await self.risk_manager.initialize():
                self.logger.error("Risk management initialization failed")
                return False
                
            # Restore previous state if needed
            await self._restore_state()
            
            self.logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
            
    async def start(self) -> NoReturn:
        """Start the trading bot"""
        if not await self.initialize():
            self.logger.error("Failed to initialize. Exiting...")
            sys.exit(1)
            
        self.running = True
        self.logger.info("Starting AGI Trading Bot...")
        
        try:
            # Start system monitoring
            await self.system_monitor.start()
            
            # Start backup system
            await self.backup_manager.start()
            
            # Main trading loop
            while self.running:
                if not self.maintenance_mode:
                    await self._trading_cycle()
                else:
                    await self._maintenance_cycle()
                    
                # Check system health
                await self._health_check()
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(self.config.cycle_delay)
                
        except Exception as e:
            self.logger.error(f"Critical error: {str(e)}")
            await self.shutdown()
            
    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            # Update market data
            market_data = await self.market_analyzer.analyze_markets()
            
            # Update risk parameters
            risk_params = await self.risk_manager.evaluate_risk(market_data)
            
            # Get trading decisions
            decisions = await self.trading_engine.make_decisions(
                market_data,
                risk_params
            )
            
            # Execute trades
            if decisions:
                await self.trading_engine.execute_trades(decisions)
                
            # Update positions
            await self.position_manager.update_positions()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            await self.notification.send_alert(
                "Trading Cycle Error",
                str(e),
                "high"
            )
            
    async def _maintenance_cycle(self) -> None:
        """Execute maintenance tasks"""
        try:
            # Perform system maintenance
            await self.system_monitor.run_maintenance()
            
            # Backup data
            await self.backup_manager.backup()
            
            # Optimize AI models
            await self.market_analyzer.optimize_models()
            
            # Clean up old data
            await self.db.cleanup()
            
        except Exception as e:
            self.logger.error(f"Error in maintenance cycle: {str(e)}")
            await self.notification.send_alert(
                "Maintenance Error",
                str(e),
                "medium"
            )
            
    async def _health_check(self) -> None:
        """Check system health and respond to issues"""
        health_status = await self.system_monitor.check_health()
        
        if not health_status['healthy']:
            self.logger.warning("Health check failed")
            await self.notification.send_alert(
                "Health Check Failed",
                str(health_status['issues']),
                "high"
            )
            
            if health_status['critical']:
                await self.shutdown()
                
    async def _restore_state(self) -> None:
        """Restore previous state after restart"""
        try:
            state = await self.db.get_last_state()
            if state:
                await self.position_manager.restore_positions(state['positions'])
                await self.risk_manager.restore_state(state['risk'])
                self.logger.info("Previous state restored successfully")
                
        except Exception as e:
            self.logger.error(f"Error restoring state: {str(e)}")
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the bot"""
        self.logger.info("Initiating shutdown sequence...")
        self.running = False
        
        try:
            # Save current state
            await self.db.save_state({
                'positions': await self.position_manager.get_positions(),
                'risk': await self.risk_manager.get_state(),
                'timestamp': datetime.utcnow()
            })
            
            # Close positions if configured
            if self.config.close_positions_on_shutdown:
                await self.position_manager.close_all_positions()
                
            # Cleanup
            await self.exchange.disconnect()
            await self.db.close()
            await self.system_monitor.stop()
            await self.backup_manager.stop()
            
            self.logger.info("Shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            
def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    
    # Create and start bot
    bot = AGITradingBot()
    
    try:
        # Run the bot
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
        asyncio.run(bot.shutdown())
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
