import ccxt
import pandas as pd
from typing import Dict, List, Union
import logging
from datetime import datetime, timezone
import asyncio
import aiohttp
import json
import hashlib
import hmac
import base64
from concurrent.futures import ThreadPoolExecutor

class PlatformIntegration:
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.active_connections = {}
        self.data_streams = {}
        self.supported_exchanges = ['binance', 'okx']
        
    async def initialize_platforms(self) -> None:
        """Initializes connections to Binance and OKX"""
        try:
            # Initialize Binance
            if 'binance' in self.config['exchanges']:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config['exchanges']['binance']['api_key'],
                    'secret': self.config['exchanges']['binance']['secret'],
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                        'recvWindow': 60000
                    }
                })
                
            # Initialize OKX
            if 'okx' in self.config['exchanges']:
                self.exchanges['okx'] = ccxt.okx({
                    'apiKey': self.config['exchanges']['okx']['api_key'],
                    'secret': self.config['exchanges']['okx']['secret'],
                    'password': self.config['exchanges']['okx']['password'],  # OKX requires password
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True
                    }
                })
                
            # Initialize websocket connections
            await self._initialize_websockets()
            
        except Exception as e:
            logging.error(f"Error initializing platforms: {e}", exc_info=True)
            
    async def _initialize_websockets(self) -> None:
        """Initializes websocket connections for both exchanges"""
        try:
            # Binance WebSocket
            if 'binance' in self.exchanges:
                self.active_connections['binance'] = await self._connect_binance_websocket()
                
            # OKX WebSocket
            if 'okx' in self.exchanges:
                self.active_connections['okx'] = await self._connect_okx_websocket()
                
        except Exception as e:
            logging.error(f"Error initializing websockets: {e}", exc_info=True)
            
    async def _connect_binance_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Connects to Binance WebSocket"""
        try:
            # Connect to Binance WebSocket stream
            session = aiohttp.ClientSession()
            ws = await session.ws_connect('wss://fstream.binance.com/ws')
            
            # Subscribe to relevant streams
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [
                    "bookTicker",  # Best bid/ask
                    "aggTrade",    # Aggregated trades
                    "markPrice",   # Mark price
                    "kline_1m"     # 1-minute klines
                ],
                "id": 1
            }
            await ws.send_json(subscribe_message)
            return ws
            
        except Exception as e:
            logging.error(f"Error connecting to Binance WebSocket: {e}", exc_info=True)
            return None
            
    async def _connect_okx_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Connects to OKX WebSocket"""
        try:
            # Connect to OKX WebSocket stream
            session = aiohttp.ClientSession()
            ws = await session.ws_connect('wss://ws.okx.com:8443/ws/v5/public')
            
            # Subscribe to relevant streams
            subscribe_message = {
                "op": "subscribe",
                "args": [
                    {
                        "channel": "tickers",
                        "instType": "SWAP"
                    },
                    {
                        "channel": "trades",
                        "instType": "SWAP"
                    },
                    {
                        "channel": "mark-price",
                        "instType": "SWAP"
                    }
                ]
            }
            await ws.send_json(subscribe_message)
            return ws
            
        except Exception as e:
            logging.error(f"Error connecting to OKX WebSocket: {e}", exc_info=True)
            return None
            
    async def execute_multi_platform_trade(self, trade_data: Dict) -> Dict:
        """Executes trades on Binance and OKX"""
        try:
            results = {}
            
            # Execute trades in parallel
            async with ThreadPoolExecutor() as executor:
                tasks = []
                for platform, order in trade_data['orders'].items():
                    if platform in self.supported_exchanges:
                        task = asyncio.create_task(
                            self._execute_platform_trade(platform, order)
                        )
                        tasks.append(task)
                        
                # Wait for all trades to complete
                trade_results = await asyncio.gather(*tasks)
                
                # Organize results
                for platform, result in zip(trade_data['orders'].keys(), trade_results):
                    results[platform] = result
                    
            return results
            
        except Exception as e:
            logging.error(f"Error executing multi-platform trade: {e}", exc_info=True)
            return {}
            
    async def _execute_platform_trade(self, platform: str, order: Dict) -> Dict:
        """Executes trade on specific platform"""
        try:
            exchange = self.exchanges[platform]
            
            # Prepare order parameters
            params = self._prepare_order_params(platform, order)
            
            # Execute order
            if platform == 'binance':
                result = await self._execute_binance_order(exchange, params)
            else:  # okx
                result = await self._execute_okx_order(exchange, params)
                
            return result
            
        except Exception as e:
            logging.error(f"Error executing {platform} trade: {e}", exc_info=True)
            return {}
            
    def _prepare_order_params(self, platform: str, order: Dict) -> Dict:
        """Prepares order parameters for specific platform"""
        try:
            if platform == 'binance':
                return {
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': order['amount'],
                    'price': order.get('price'),
                    'params': {
                        'timeInForce': 'GTC',
                        'reduceOnly': order.get('reduce_only', False),
                        'leverage': order.get('leverage', 1)
                    }
                }
            else:  # okx
                return {
                    'symbol': order['symbol'].replace('/', '-'),  # OKX uses different format
                    'type': order['type'],
                    'side': order['side'],
                    'amount': order['amount'],
                    'price': order.get('price'),
                    'params': {
                        'tdMode': 'cross',  # or 'isolated'
                        'posSide': order.get('pos_side', 'long')
                    }
                }
                
        except Exception as e:
            logging.error(f"Error preparing order params: {e}", exc_info=True)
            return {}
            
    async def sync_platform_data(self) -> Dict:
        """Synchronizes data from Binance and OKX"""
        try:
            synced_data = {
                'balances': await self._sync_balances(),
                'positions': await self._sync_positions(),
                'orders': await self._sync_orders(),
                'trades': await self._sync_trades(),
                'funding_rates': await self._sync_funding_rates(),
                'market_data': await self._sync_market_data()
            }
            
            return synced_data
            
        except Exception as e:
            logging.error(f"Error syncing platform data: {e}", exc_info=True)
            return {}
            
    async def _sync_funding_rates(self) -> Dict:
        """Synchronizes funding rates"""
        try:
            funding_rates = {}
            
            for exchange_id in self.supported_exchanges:
                if exchange_id in self.exchanges:
                    exchange = self.exchanges[exchange_id]
                    rates = await asyncio.to_thread(
                        exchange.fetch_funding_rates
                    )
                    funding_rates[exchange_id] = self._normalize_funding_rates(rates)
                    
            return funding_rates
            
        except Exception as e:
            logging.error(f"Error syncing funding rates: {e}", exc_info=True)
            return {}
            
    async def _sync_market_data(self) -> Dict:
        """Synchronizes market data"""
        try:
            market_data = {}
            
            for exchange_id in self.supported_exchanges:
                if exchange_id in self.exchanges:
                    exchange = self.exchanges[exchange_id]
                    tickers = await asyncio.to_thread(
                        exchange.fetch_tickers
                    )
                    market_data[exchange_id] = self._normalize_market_data(tickers)
                    
            return market_data
            
        except Exception as e:
            logging.error(f"Error syncing market data: {e}", exc_info=True)
            return {}
            
    def _normalize_funding_rates(self, rates: Dict) -> Dict:
        """Normalizes funding rate data format"""
        try:
            normalized = {}
            
            for symbol, rate in rates.items():
                normalized[symbol] = {
                    'rate': float(rate['fundingRate']),
                    'timestamp': rate['timestamp'],
                    'next_rate': float(rate.get('nextFundingRate', 0)),
                    'next_timestamp': rate.get('nextFundingTime', 0)
                }
                
            return normalized
            
        except Exception as e:
            logging.error(f"Error normalizing funding rates: {e}", exc_info=True)
            return {}
            
    async def manage_platform_connections(self) -> None:
        """Manages platform connections for Binance and OKX"""
        try:
            while True:
                # Check Binance connection
                if 'binance' in self.active_connections:
                    if not self.active_connections['binance'].closed:
                        self.active_connections['binance'] = await self._connect_binance_websocket()
                        
                # Check OKX connection
                if 'okx' in self.active_connections:
                    if not self.active_connections['okx'].closed:
                        self.active_connections['okx'] = await self._connect_okx_websocket()
                        
                # Monitor data streams
                await self._monitor_data_streams()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logging.error(f"Error managing connections: {e}", exc_info=True)
            
    async def _monitor_data_streams(self) -> None:
        """Monitors data streams for both platforms"""
        try:
            for exchange_id in self.supported_exchanges:
                if exchange_id in self.data_streams:
                    # Check stream health
                    if not await self._check_stream_health(exchange_id):
                        # Restart stream
                        await self._restart_data_stream(exchange_id)
                        
        except Exception as e:
            logging.error(f"Error monitoring data streams: {e}", exc_info=True)
            
    async def generate_unified_report(self) -> Dict:
        """Generates unified report for Binance and OKX"""
        try:
            # Sync latest data
            synced_data = await self.sync_platform_data()
            
            # Calculate unified metrics
            unified_metrics = self._calculate_unified_metrics(synced_data)
            
            # Calculate arbitrage opportunities
            arbitrage_ops = self._find_arbitrage_opportunities(synced_data)
            
            # Generate report
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'platforms': self.supported_exchanges,
                'unified_metrics': unified_metrics,
                'platform_metrics': self._calculate_platform_metrics(synced_data),
                'arbitrage_opportunities': arbitrage_ops,
                'recommendations': self._generate_platform_recommendations(unified_metrics)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating unified report: {e}", exc_info=True)
            return {}
