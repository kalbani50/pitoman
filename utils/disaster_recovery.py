import json
import logging
from typing import Dict, List
from datetime import datetime, timezone
import asyncio
import aiofiles
import os
import shutil
from pathlib import Path
import pickle
import hashlib

class DisasterRecovery:
    def __init__(self, config: Dict):
        self.config = config
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.state_snapshots = {}
        self.recovery_points = []
        
    async def create_system_snapshot(self, system_state: Dict) -> str:
        """Creates a complete snapshot of the system state"""
        try:
            # Generate snapshot ID
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"snapshot_{timestamp}"
            
            # Create snapshot directory
            snapshot_dir = self.backup_dir / snapshot_id
            snapshot_dir.mkdir(exist_ok=True)
            
            # Save different components
            await self._save_trading_state(snapshot_dir, system_state.get('trading', {}))
            await self._save_portfolio_state(snapshot_dir, system_state.get('portfolio', {}))
            await self._save_model_state(snapshot_dir, system_state.get('models', {}))
            await self._save_config_state(snapshot_dir, system_state.get('config', {}))
            
            # Create checksum
            checksum = self._create_snapshot_checksum(snapshot_dir)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'checksum': checksum,
                'components': list(system_state.keys()),
                'version': self.config.get('version', '1.0.0')
            }
            
            await self._save_metadata(snapshot_dir, metadata)
            
            # Add to recovery points
            self.recovery_points.append({
                'id': snapshot_id,
                'timestamp': timestamp,
                'metadata': metadata
            })
            
            return snapshot_id
            
        except Exception as e:
            logging.error(f"Error creating system snapshot: {e}", exc_info=True)
            return ""
            
    async def restore_from_snapshot(self, snapshot_id: str) -> Dict:
        """Restores system from a snapshot"""
        try:
            snapshot_dir = self.backup_dir / snapshot_id
            
            if not snapshot_dir.exists():
                raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
                
            # Verify checksum
            if not self._verify_snapshot_checksum(snapshot_dir):
                raise ValueError("Snapshot checksum verification failed")
                
            # Load components
            restored_state = {
                'trading': await self._load_trading_state(snapshot_dir),
                'portfolio': await self._load_portfolio_state(snapshot_dir),
                'models': await self._load_model_state(snapshot_dir),
                'config': await self._load_config_state(snapshot_dir)
            }
            
            return restored_state
            
        except Exception as e:
            logging.error(f"Error restoring from snapshot: {e}", exc_info=True)
            return {}
            
    async def _save_trading_state(self, snapshot_dir: Path, trading_state: Dict) -> None:
        """Saves trading state"""
        try:
            trading_file = snapshot_dir / "trading_state.pkl"
            async with aiofiles.open(trading_file, 'wb') as f:
                await f.write(pickle.dumps(trading_state))
                
        except Exception as e:
            logging.error(f"Error saving trading state: {e}", exc_info=True)
            
    async def _save_portfolio_state(self, snapshot_dir: Path, portfolio_state: Dict) -> None:
        """Saves portfolio state"""
        try:
            portfolio_file = snapshot_dir / "portfolio_state.pkl"
            async with aiofiles.open(portfolio_file, 'wb') as f:
                await f.write(pickle.dumps(portfolio_state))
                
        except Exception as e:
            logging.error(f"Error saving portfolio state: {e}", exc_info=True)
            
    async def _save_model_state(self, snapshot_dir: Path, model_state: Dict) -> None:
        """Saves ML model states"""
        try:
            model_dir = snapshot_dir / "models"
            model_dir.mkdir(exist_ok=True)
            
            for model_name, model_data in model_state.items():
                model_file = model_dir / f"{model_name}.pkl"
                async with aiofiles.open(model_file, 'wb') as f:
                    await f.write(pickle.dumps(model_data))
                    
        except Exception as e:
            logging.error(f"Error saving model state: {e}", exc_info=True)
            
    def _create_snapshot_checksum(self, snapshot_dir: Path) -> str:
        """Creates checksum for snapshot verification"""
        try:
            hasher = hashlib.sha256()
            
            for root, _, files in os.walk(snapshot_dir):
                for file in sorted(files):
                    file_path = Path(root) / file
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                        
            return hasher.hexdigest()
            
        except Exception as e:
            logging.error(f"Error creating checksum: {e}", exc_info=True)
            return ""
            
    def _verify_snapshot_checksum(self, snapshot_dir: Path) -> bool:
        """Verifies snapshot integrity"""
        try:
            metadata_file = snapshot_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            stored_checksum = metadata['checksum']
            current_checksum = self._create_snapshot_checksum(snapshot_dir)
            
            return stored_checksum == current_checksum
            
        except Exception as e:
            logging.error(f"Error verifying checksum: {e}", exc_info=True)
            return False
            
    async def create_emergency_backup(self) -> str:
        """Creates emergency backup of critical data"""
        try:
            # Generate backup ID
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_id = f"emergency_{timestamp}"
            
            # Create backup directory
            backup_dir = self.backup_dir / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # Save critical data
            await self._save_critical_data(backup_dir)
            
            # Create compressed archive
            shutil.make_archive(str(backup_dir), 'zip', str(backup_dir))
            
            return backup_id
            
        except Exception as e:
            logging.error(f"Error creating emergency backup: {e}", exc_info=True)
            return ""
            
    async def validate_system_integrity(self) -> Dict:
        """Validates system integrity"""
        try:
            validation_results = {
                'database_integrity': await self._validate_database(),
                'file_integrity': await self._validate_files(),
                'model_integrity': await self._validate_models(),
                'config_integrity': await self._validate_config()
            }
            
            # Calculate overall health score
            validation_results['health_score'] = self._calculate_health_score(
                validation_results
            )
            
            return validation_results
            
        except Exception as e:
            logging.error(f"Error validating system integrity: {e}", exc_info=True)
            return {}
            
    async def cleanup_old_backups(self, days_to_keep: int = 30) -> None:
        """Cleans up old backups"""
        try:
            current_time = datetime.now(timezone.utc)
            
            for backup_dir in self.backup_dir.glob("*"):
                if backup_dir.is_dir():
                    # Extract timestamp from directory name
                    timestamp_str = backup_dir.name.split('_')[1]
                    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    backup_time = backup_time.replace(tzinfo=timezone.utc)
                    
                    # Remove if older than specified days
                    if (current_time - backup_time).days > days_to_keep:
                        shutil.rmtree(backup_dir)
                        
        except Exception as e:
            logging.error(f"Error cleaning up backups: {e}", exc_info=True)
            
    async def auto_recovery(self, error_type: str) -> Dict:
        """Attempts automatic recovery based on error type"""
        try:
            recovery_actions = {
                'database_error': self._recover_database,
                'connection_error': self._recover_connections,
                'model_error': self._recover_models,
                'system_error': self._recover_system
            }
            
            if error_type in recovery_actions:
                recovery_result = await recovery_actions[error_type]()
                
                # Log recovery attempt
                logging.info(f"Auto recovery attempted for {error_type}")
                
                return {
                    'error_type': error_type,
                    'recovery_result': recovery_result,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            return {'error': f"Unknown error type: {error_type}"}
            
        except Exception as e:
            logging.error(f"Error in auto recovery: {e}", exc_info=True)
            return {'error': str(e)}
