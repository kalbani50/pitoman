import os
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Authentication
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Trading Parameters
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
TIMEFRAME = '1h'
INITIAL_CAPITAL = Decimal('10000')
ORDER_TYPE = 'market'  # 'limit' or 'market'
MAX_CONCURRENT_TASKS = 5
WAIT_INTERVAL = 60  # seconds
RETRY_DELAY = 30  # seconds

# Strategy Parameters
STRATEGY_PARAMS = {
    'SmartMoneyStrategy': {
        'volume_threshold': 1.5,
        'delta_threshold': 0.5,
        'structure_threshold': 0.7
    },
    'VolatilityBreakoutStrategy': {
        'volatility_threshold': 0.8,
        'momentum_threshold': 0.6,
        'volume_threshold': 0.7
    },
    'MarketStructureStrategy': {
        'efficiency_threshold': 0.7,
        'trend_strength': 0.6,
        'volume_confirmation': 0.65
    },
    'AdaptiveMomentumStrategy': {
        'momentum_threshold': 0.65,
        'volatility_adjustment': 0.5,
        'efficiency_minimum': 0.6
    },
    'HybridFlowStrategy': {
        'flow_threshold': 0.75,
        'smart_money_confirmation': 0.7,
        'technical_alignment': 0.8
    }
}

# Risk Management Parameters
RISK_PARAMS = {
    'max_position_size': Decimal('0.1'),  # 10% of capital
    'max_risk_per_trade': Decimal('0.02'),  # 2% risk per trade
    'max_daily_risk': Decimal('0.06'),  # 6% daily risk
    'max_correlated_positions': 3,
    'position_sizing_models': ['fixed', 'volatility_adjusted', 'kelly_criterion'],
    'stop_loss_models': ['fixed', 'atr_based', 'support_resistance'],
    'take_profit_models': ['fixed', 'risk_reward', 'trailing']
}

# Advanced Indicator Parameters
INDICATOR_PARAMS = {
    'volume_profile': {
        'vwap_period': 14,
        'mfi_period': 14,
        'vzo_period': 20
    },
    'order_flow': {
        'delta_period': 20,
        'imbalance_threshold': 0.7,
        'momentum_period': 20
    },
    'market_structure': {
        'swing_period': 10,
        'structure_threshold': 0.8,
        'break_confirmation': 2
    },
    'volatility': {
        'atr_period': 14,
        'normalized_period': 20,
        'regime_threshold': 3
    },
    'momentum': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    },
    'custom': {
        'efficiency_period': 20,
        'adaptive_period': 30,
        'oscillator_weights': [0.3, 0.3, 0.4]
    }
}

# AI and Machine Learning Parameters
AI_PARAMS = {
    'model_update_interval': 24 * 3600,  # 24 hours
    'training_window': 1000,
    'validation_split': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 10,
    'model_weights': {
        'market_state': 0.3,
        'strategy_selector': 0.4,
        'risk_manager': 0.3
    }
}

# Performance Monitoring Parameters
PERFORMANCE_PARAMS = {
    'metrics': [
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'recovery_factor'
    ],
    'monitoring_interval': 3600,  # 1 hour
    'alert_thresholds': {
        'drawdown_alert': Decimal('0.1'),  # 10% drawdown
        'profit_target': Decimal('0.5'),  # 50% profit
        'risk_alert': Decimal('0.15')  # 15% at risk
    }
}

# Database Collections
DB_COLLECTIONS = {
    'trades': 'trades',
    'performance': 'performance',
    'models': 'models',
    'events': 'events',
    'market_data': 'market_data',
    'analysis': 'analysis'
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'trading_bot.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}
