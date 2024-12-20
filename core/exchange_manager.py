"""
High-Performance Exchange Integration System with Enhanced Error Handling and Rate Limiting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import ccxt
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import sys
import time
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class PerformanceOptimizer:
    """Optimize exchange operations for better performance and reliability"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_times = {}
        self.request_times = deque(maxlen=1000)
        self.operation_stats = defaultdict(list)
        self.rate_limiter = RateLimiter(
            max_requests=config.get('max_requests_per_second', 10),
            time_window=1.0
        )
        
    async def execute_with_retry(self, operation: str, func: callable,
                               *args, **kwargs) -> Optional[Any]:
        """Execute operation with retry logic and rate limiting"""
        max_retries = self.config.get('max_retries', 3)
        base_delay = self.config.get('retry_base_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                # Execute operation
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                self.record_operation_time(operation, duration)
                
                return result
                
            except ccxt.RateLimitExceeded:
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Rate limit exceeded, waiting {delay}s")
                await asyncio.sleep(delay)
                
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Network error: {str(e)}, retrying in {delay}s")
                await asyncio.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Unhandled error in {operation}: {str(e)}")
                raise

    def should_cache(self, operation: str, data: Dict) -> bool:
        """Determine if an operation should be cached"""
        cache_config = self.config.get('cache_settings', {})
        
        if operation not in cache_config:
            return False
            
        ttl = cache_config[operation].get('ttl', 60)  # Default 60 seconds
        importance = cache_config[operation].get('importance', 0.5)
        
        # Check data size
        data_size = sys.getsizeof(str(data))
        if data_size > cache_config[operation].get('max_size', 1024 * 1024):
            return False
            
        return True
        
    def cache_data(self, operation: str, key: str, data: any):
        """Cache operation data"""
        if self.should_cache(operation, data):
            self.cache[f"{operation}:{key}"] = data
            self.cache_times[f"{operation}:{key}"] = datetime.utcnow()
            
    def get_cached_data(self, operation: str, key: str) -> Optional[any]:
        """Get cached data if valid"""
        cache_key = f"{operation}:{key}"
        if cache_key not in self.cache:
            return None
            
        cache_time = self.cache_times[cache_key]
        ttl = self.config.get('cache_settings', {}).get(operation, {}).get('ttl', 60)
        
        if (datetime.utcnow() - cache_time).total_seconds() > ttl:
            del self.cache[cache_key]
            del self.cache_times[cache_key]
            return None
            
        return self.cache[cache_key]
        
    def record_operation_time(self, operation: str, duration: float):
        """Record operation execution time"""
        self.operation_stats[operation].append({
            'duration': duration,
            'timestamp': datetime.utcnow()
        })
        
    def get_operation_stats(self, operation: str) -> Dict:
        """Get statistics for an operation"""
        if operation not in self.operation_stats:
            return {}
            
        durations = [stat['duration'] for stat in self.operation_stats[operation]]
        return {
            'avg_duration': np.mean(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'std_duration': np.std(durations),
            'total_operations': len(durations)
        }
        
    def optimize_batch_size(self, operation: str) -> int:
        """Optimize batch size based on performance stats"""
        stats = self.get_operation_stats(operation)
        if not stats:
            return self.config.get('default_batch_size', 100)
            
        # Adjust batch size based on performance
        avg_duration = stats['avg_duration']
        if avg_duration > self.config.get('target_response_time', 1.0):
            return max(10, self.config.get('default_batch_size', 100) // 2)
        elif avg_duration < self.config.get('target_response_time', 1.0) / 2:
            return min(1000, self.config.get('default_batch_size', 100) * 2)
            
        return self.config.get('default_batch_size', 100)

class RateLimiter:
    """Implements token bucket algorithm for rate limiting"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire a token for making a request"""
        async with self.lock:
            await self._add_tokens()
            
            if self.tokens <= 0:
                wait_time = self.time_window / self.max_requests
                await asyncio.sleep(wait_time)
                await self._add_tokens()
                
            self.tokens -= 1
            
    async def _add_tokens(self):
        """Add new tokens based on elapsed time"""
        now = time.time()
        time_passed = now - self.last_update
        new_tokens = time_passed * (self.max_requests / self.time_window)
        self.tokens = min(self.max_requests, self.tokens + new_tokens)
        self.last_update = now

class ExchangeManager:
    """Manages all exchange interactions with enhanced reliability"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.markets = {}
        self.balances = {}
        self.order_cache = {}
        self.position_cache = {}
        
        # Initialize components
        self.optimizer = PerformanceOptimizer(config)
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('max_threads', 4)
        )
        
    async def connect(self) -> bool:
        """Connect to the exchange with enhanced error handling"""
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, self.config.name)
            self.exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            
            # Load markets with retry
            await self.optimizer.execute_with_retry(
                'load_markets',
                self.exchange.load_markets
            )
            self.markets = self.exchange.markets
            
            # Initialize account
            await self._initialize_account()
            
            self.logger.info(f"Successfully connected to {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {str(e)}")
            return False
            
    async def _initialize_account(self):
        """Initialize account state"""
        # Update balances
        await self.update_balances()
        
        # Load open positions
        await self._load_positions()
        
        # Load open orders
        await self._load_open_orders()
        
    async def _load_positions(self):
        """Load all open positions"""
        try:
            positions = await self.optimizer.execute_with_retry(
                'fetch_positions',
                self.exchange.fetch_positions
            )
            
            self.position_cache = {
                pos['symbol']: pos for pos in positions if float(pos['contracts']) != 0
            }
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {str(e)}")
            
    async def _load_open_orders(self):
        """Load all open orders"""
        try:
            orders = await self.optimizer.execute_with_retry(
                'fetch_open_orders',
                self.exchange.fetch_open_orders
            )
            
            self.order_cache = {
                order['id']: order for order in orders
            }
            
        except Exception as e:
            self.logger.error(f"Error loading open orders: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from the exchange"""
        try:
            # Close any open connections
            if hasattr(self.exchange, 'close'):
                await self.exchange.close()
                
            self.exchange = None
            self.markets = {}
            self.balances = {}
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from exchange: {str(e)}")
            
    async def execute_order(self, order: Dict) -> Optional[Dict]:
        """Execute a trade order with performance optimization"""
        operation = 'execute_order'
        
        try:
            # Record start time
            start_time = time.time()
            
            # Execute order
            result = await self.optimizer.execute_with_retry(
                operation,
                self._execute_order_internal,
                order
            )
            
            # Record operation time
            duration = time.time() - start_time
            self.optimizer.record_operation_time(operation, duration)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None
            
    async def _execute_order_internal(self, order: Dict) -> Dict:
        """Internal order execution with retries and error handling"""
        max_retries = self.config.get('max_order_retries', 3)
        retry_delay = self.config.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                result = await self.exchange.create_order(
                    symbol=order['symbol'],
                    type=order['type'],
                    side=order['side'],
                    amount=order['amount'],
                    price=order.get('price'),
                    params=self._prepare_order_params(order)
                )
                
                # Cache successful order
                self.order_cache[result['id']] = result
                
                # Update balances
                await self.update_balances()
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                    
                self.logger.warning(f"Order attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        try:
            await self.optimizer.execute_with_retry(
                'cancel_order',
                self.exchange.cancel_order,
                order_id,
                symbol
            )
            
            # Remove from cache
            if order_id in self.order_cache:
                del self.order_cache[order_id]
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            return False
            
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol"""
        try:
            position = await self.optimizer.execute_with_retry(
                'fetch_position',
                self.exchange.fetch_position,
                symbol
            )
            return position
            
        except Exception as e:
            self.logger.error(f"Error fetching position: {str(e)}")
            return None
            
    async def close_position(self, symbol: str) -> bool:
        """Close an open position"""
        try:
            position = await self.get_position(symbol)
            if not position or float(position['contracts']) == 0:
                return True
                
            # Create closing order
            side = 'sell' if position['side'] == 'long' else 'buy'
            await self.execute_order({
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': abs(float(position['contracts'])),
                'reduce_only': True
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False
            
    async def update_balances(self) -> bool:
        """Update account balances"""
        try:
            balances = await self.optimizer.execute_with_retry(
                'fetch_balance',
                self.exchange.fetch_balance
            )
            self.balances = balances
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating balances: {str(e)}")
            return False
            
    async def get_market_data(self, symbol: str, timeframe: str,
                            limit: int = 100) -> Optional[List[Dict]]:
        """Get market OHLCV data with performance optimization"""
        operation = 'get_market_data'
        cache_key = f"{symbol}:{timeframe}:{limit}"
        
        try:
            # Check cache
            cached_data = self.optimizer.get_cached_data(operation, cache_key)
            if cached_data is not None:
                return cached_data
                
            # Record start time
            start_time = time.time()
            
            # Get data
            data = await self.optimizer.execute_with_retry(
                operation,
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Record operation time
            duration = time.time() - start_time
            self.optimizer.record_operation_time(operation, duration)
            
            # Cache data
            self.optimizer.cache_data(operation, cache_key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None
            
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book data with performance optimization"""
        operation = 'get_order_book'
        cache_key = f"{symbol}:{limit}"
        
        try:
            # Check cache
            cached_data = self.optimizer.get_cached_data(operation, cache_key)
            if cached_data is not None:
                return cached_data
                
            # Record start time
            start_time = time.time()
            
            # Get data
            data = await self.optimizer.execute_with_retry(
                operation,
                self.exchange.fetch_order_book,
                symbol,
                limit=limit
            )
            
            # Record operation time
            duration = time.time() - start_time
            self.optimizer.record_operation_time(operation, duration)
            
            # Cache data
            self.optimizer.cache_data(operation, cache_key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            return None
            
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker data"""
        try:
            ticker = await self.optimizer.execute_with_retry(
                'fetch_ticker',
                self.exchange.fetch_ticker,
                symbol
            )
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            return None
            
    def _validate_order(self, order: Dict) -> bool:
        """Validate order parameters"""
        try:
            required_fields = ['symbol', 'type', 'side', 'amount']
            if not all(field in order for field in required_fields):
                return False
                
            # Validate symbol
            if order['symbol'] not in self.markets:
                return False
                
            # Validate order type
            if order['type'] not in ['market', 'limit']:
                return False
                
            # Validate side
            if order['side'] not in ['buy', 'sell']:
                return False
                
            # Validate amount
            if not isinstance(order['amount'], (int, float, Decimal)):
                return False
                
            # Validate price for limit orders
            if order['type'] == 'limit' and 'price' not in order:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            return False
            
    def _prepare_order_params(self, order: Dict) -> Dict:
        """Prepare additional order parameters"""
        params = {}
        
        # Add leverage if specified
        if 'leverage' in order:
            params['leverage'] = order['leverage']
            
        # Add reduce_only flag if specified
        if 'reduce_only' in order:
            params['reduce_only'] = order['reduce_only']
            
        # Add stop loss and take profit if specified
        if 'stop_loss' in order:
            params['stopLoss'] = {'price': order['stop_loss']}
            
        if 'take_profit' in order:
            params['takeProfit'] = {'price': order['take_profit']}
            
        return params
