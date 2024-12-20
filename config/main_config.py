from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_config() -> Dict:
    """Returns complete system configuration"""
    return {
        # System Components
        'components': {
            'ai_manager': {
                'enabled': True,
                'update_interval': 1,  # seconds
                'model_update_interval': 3600,  # 1 hour
                'confidence_threshold': 0.85
            },
            'autonomous_ai': {
                'enabled': True,
                'decision_interval': 1,
                'learning_rate': 0.001,
                'memory_size': 10000
            },
            'swarm_intelligence': {
                'enabled': True,
                'swarm_size': 8,
                'update_interval': 1,
                'consensus_threshold': 0.75
            },
            'self_evolution': {
                'enabled': True,
                'evolution_interval': 3600,
                'generation_size': 100,
                'mutation_rate': 0.01
            },
            'advanced_simulation': {
                'enabled': True,
                'simulation_interval': 300,
                'scenarios_count': 1000,
                'confidence_threshold': 0.9
            },
            'crisis_predictor': {
                'enabled': True,
                'prediction_interval': 300,
                'alert_threshold': 0.7,
                'lookback_period': 30  # days
            },
            'news_events_integrator': {
                'enabled': True,
                'update_interval': 60,
                'impact_threshold': 0.6,
                'sources_count': 50
            },
            'advanced_verification': {
                'enabled': True,
                'verification_interval': 1,
                'confidence_threshold': 0.95,
                'proof_chain_size': 1000
            },
            'realtime_sync': {
                'enabled': True,
                'sync_interval': 0.1,
                'max_latency': 100,  # ms
                'time_servers': [
                    'pool.ntp.org',
                    'time.google.com',
                    'time.windows.com'
                ]
            },
            'system_orchestrator': {
                'enabled': True,
                'orchestration_interval': 0.5,
                'resource_check_interval': 5,
                'component_timeout': 10
            }
        },

        # Trading Parameters
        'trading': {
            'exchanges': {
                'binance': {
                    'api_key': os.getenv('BINANCE_API_KEY'),
                    'secret': os.getenv('BINANCE_SECRET_KEY'),
                    'enabled': True
                }
                # Add more exchanges as needed
            },
            'max_positions': 50,
            'max_leverage': 5,
            'min_volume': 100,  # USD
            'max_volume': 10000,  # USD
            'risk_per_trade': 0.02,  # 2% per trade
            'stop_loss_multiplier': 2,
            'take_profit_multiplier': 3
        },

        # Risk Management
        'risk_management': {
            'max_daily_loss': 0.05,  # 5%
            'max_position_size': 0.1,  # 10% of portfolio
            'correlation_threshold': 0.7,
            'volatility_threshold': 0.3,
            'drawdown_threshold': 0.15,  # 15%
            'risk_free_rate': 0.02  # 2%
        },

        # Performance Monitoring
        'monitoring': {
            'performance_window': 30,  # days
            'metrics_update_interval': 300,  # seconds
            'alert_threshold': 0.1,  # 10% deviation
            'log_level': 'INFO',
            'backup_interval': 3600  # 1 hour
        },

        # Database Configuration
        'database': {
            'uri': os.getenv('MONGO_URI'),
            'name': os.getenv('MONGO_DB_NAME'),
            'collections': {
                'trades': 'trades',
                'performance': 'performance',
                'models': 'models',
                'system_state': 'system_state'
            }
        },

        # API Configuration
        'api': {
            'openai_key': os.getenv('OPENAI_API_KEY'),
            'news_api_key': os.getenv('NEWSAPI_KEY'),
            'max_retries': 3,
            'timeout': 30,
            'rate_limit': 100  # requests per minute
        },

        # Security Configuration
        'security': {
            'encryption_algorithm': 'AES-256',
            'key_rotation_interval': 86400,  # 24 hours
            'max_failed_attempts': 3,
            'session_timeout': 3600,  # 1 hour
            'ip_whitelist': ['127.0.0.1']
        },

        # System Resources
        'resources': {
            'max_memory': '16G',
            'max_cpu_usage': 0.9,  # 90%
            'max_gpu_usage': 0.95,  # 95%
            'thread_pool_size': 16,
            'async_workers': 8
        },

        # Recovery Configuration
        'recovery': {
            'backup_interval': 3600,  # 1 hour
            'max_restore_time': 300,  # 5 minutes
            'checkpoint_interval': 900,  # 15 minutes
            'max_recovery_attempts': 3
        }
    }
