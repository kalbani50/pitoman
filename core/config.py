"""
Configuration management for the AGI Trading Bot
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    
@dataclass
class ExchangeConfig:
    name: str
    api_key: str
    api_secret: str
    testnet: bool
    
@dataclass
class TradingConfig:
    max_pairs: int
    min_position_size: float
    max_position_size: float
    max_leverage: float
    close_positions_on_shutdown: bool
    
@dataclass
class RiskConfig:
    max_drawdown: float
    max_daily_loss: float
    max_position_risk: float
    risk_free_rate: float
    
@dataclass
class AIConfig:
    model_path: str
    update_interval: int
    min_confidence: float
    
@dataclass
class NotificationConfig:
    enabled: bool
    email: Optional[str]
    telegram_token: Optional[str]
    discord_webhook: Optional[str]
    
class Config:
    """Configuration management class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path("config")
        self.env_file = Path(".env")
        
        # Load configurations
        self._load_config()
        
    def _load_config(self) -> None:
        """Load all configuration files"""
        try:
            # Load main config
            main_config = self._load_yaml("config.yaml")
            
            # Load specific configs
            self.database = self._load_database_config()
            self.exchange = self._load_exchange_config()
            self.trading = self._load_trading_config(main_config)
            self.risk = self._load_risk_config(main_config)
            self.analysis = self._load_ai_config(main_config)
            self.notification = self._load_notification_config(main_config)
            
            # Additional settings
            self.cycle_delay = main_config.get('cycle_delay', 1)
            self.debug_mode = main_config.get('debug_mode', False)
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filename}")
            
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'trading_bot'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
    def _load_exchange_config(self) -> ExchangeConfig:
        """Load exchange configuration"""
        return ExchangeConfig(
            name=os.getenv('EXCHANGE_NAME', 'binance'),
            api_key=os.getenv('EXCHANGE_API_KEY', ''),
            api_secret=os.getenv('EXCHANGE_API_SECRET', ''),
            testnet=os.getenv('EXCHANGE_TESTNET', 'true').lower() == 'true'
        )
        
    def _load_trading_config(self, config: Dict) -> TradingConfig:
        """Load trading configuration"""
        trading = config.get('trading', {})
        return TradingConfig(
            max_pairs=trading.get('max_pairs', 5),
            min_position_size=trading.get('min_position_size', 0.01),
            max_position_size=trading.get('max_position_size', 0.1),
            max_leverage=trading.get('max_leverage', 3.0),
            close_positions_on_shutdown=trading.get('close_positions_on_shutdown', True)
        )
        
    def _load_risk_config(self, config: Dict) -> RiskConfig:
        """Load risk configuration"""
        risk = config.get('risk', {})
        return RiskConfig(
            max_drawdown=risk.get('max_drawdown', 0.2),
            max_daily_loss=risk.get('max_daily_loss', 0.05),
            max_position_risk=risk.get('max_position_risk', 0.02),
            risk_free_rate=risk.get('risk_free_rate', 0.02)
        )
        
    def _load_ai_config(self, config: Dict) -> AIConfig:
        """Load AI configuration"""
        ai = config.get('ai', {})
        return AIConfig(
            model_path=ai.get('model_path', 'models'),
            update_interval=ai.get('update_interval', 3600),
            min_confidence=ai.get('min_confidence', 0.7)
        )
        
    def _load_notification_config(self, config: Dict) -> NotificationConfig:
        """Load notification configuration"""
        notify = config.get('notification', {})
        return NotificationConfig(
            enabled=notify.get('enabled', True),
            email=os.getenv('NOTIFICATION_EMAIL'),
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            discord_webhook=os.getenv('DISCORD_WEBHOOK')
        )
        
    def validate(self) -> bool:
        """Validate the configuration"""
        try:
            # Check required directories
            if not self.config_dir.exists():
                self.logger.error("Config directory not found")
                return False
                
            # Validate database config
            if not self.database.host or not self.database.password:
                self.logger.error("Invalid database configuration")
                return False
                
            # Validate exchange config
            if not self.exchange.api_key or not self.exchange.api_secret:
                self.logger.error("Invalid exchange configuration")
                return False
                
            # Validate trading parameters
            if (self.trading.max_position_size <= 0 or
                self.trading.min_position_size <= 0 or
                self.trading.min_position_size >= self.trading.max_position_size):
                self.logger.error("Invalid trading configuration")
                return False
                
            # Validate risk parameters
            if (self.risk.max_drawdown <= 0 or
                self.risk.max_drawdown >= 1 or
                self.risk.max_daily_loss <= 0 or
                self.risk.max_daily_loss >= 1):
                self.logger.error("Invalid risk configuration")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
            
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a configuration value"""
        try:
            config_file = self.config_dir / "config.yaml"
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if section not in config:
                config[section] = {}
                
            config[section][key] = value
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
                
            # Reload configuration
            self._load_config()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return False
