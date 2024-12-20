from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import logging
from typing import Dict, Optional
from decimal import Decimal

class DatabaseManager:
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.trades_collection = self.db["trades_binance"]
        
    async def save_trade(self, trade_data: Dict) -> None:
        try:
            trade_data['timestamp'] = datetime.now(timezone.utc)
            await self.trades_collection.insert_one(trade_data)
            logging.info(f"Trade saved successfully: {trade_data['pair']}")
        except Exception as e:
            logging.error(f"Error saving trade to database: {e}", exc_info=True)
            
    async def get_trade_history(self, pair: str, limit: int = 100) -> list:
        try:
            cursor = self.trades_collection.find(
                {"pair": pair}
            ).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logging.error(f"Error retrieving trade history: {e}", exc_info=True)
            return []
            
    async def calculate_performance_metrics(self, pair: str) -> Dict:
        try:
            trades = await self.get_trade_history(pair)
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "average_profit": Decimal("0"),
                    "total_profit": Decimal("0")
                }
                
            winning_trades = sum(1 for trade in trades if float(trade.get('profit', 0)) > 0)
            total_trades = len(trades)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            profits = [Decimal(str(trade.get('profit', 0))) for trade in trades]
            average_profit = sum(profits) / len(profits) if profits else Decimal("0")
            total_profit = sum(profits)
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "average_profit": average_profit,
                "total_profit": total_profit
            }
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return {
                "total_trades": 0,
                "win_rate": 0,
                "average_profit": Decimal("0"),
                "total_profit": Decimal("0")
            }
            
    async def update_trade_status(self, trade_id: str, status: str, 
                                additional_info: Optional[Dict] = None) -> None:
        try:
            update_data = {"status": status}
            if additional_info:
                update_data.update(additional_info)
            
            await self.trades_collection.update_one(
                {"_id": trade_id},
                {"$set": update_data}
            )
            logging.info(f"Trade status updated: {trade_id} -> {status}")
        except Exception as e:
            logging.error(f"Error updating trade status: {e}", exc_info=True)
            
    async def cleanup_old_trades(self, days: int = 30) -> None:
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.trades_collection.delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            logging.info(f"Cleaned up {result.deleted_count} old trades")
        except Exception as e:
            logging.error(f"Error cleaning up old trades: {e}", exc_info=True)
