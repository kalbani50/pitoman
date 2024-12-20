import hashlib
import hmac
import base64
from typing import Dict, List, Union
import logging
from datetime import datetime, timezone
import asyncio
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import jwt

class AdvancedSecurity:
    def __init__(self, config: Dict):
        self.config = config
        self.key = self._generate_key()
        self.fernet = Fernet(self.key)
        self.security_logs = []
        
    def _generate_key(self) -> bytes:
        """Generates encryption key"""
        try:
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(
                self.config['secret_key'].encode()
            ))
            return key
            
        except Exception as e:
            logging.error(f"Error generating key: {e}", exc_info=True)
            return b''
            
    async def encrypt_sensitive_data(self, data: Union[Dict, str]) -> str:
        """Encrypts sensitive data"""
        try:
            # Convert data to string if dictionary
            if isinstance(data, dict):
                data = json.dumps(data)
                
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data.encode())
            
            # Log encryption event
            self._log_security_event('encryption', 'success')
            
            return encrypted_data.decode()
            
        except Exception as e:
            logging.error(f"Error encrypting data: {e}", exc_info=True)
            self._log_security_event('encryption', 'failed', str(e))
            return ''
            
    async def decrypt_sensitive_data(self, encrypted_data: str) -> Union[Dict, str]:
        """Decrypts sensitive data"""
        try:
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            
            # Try to parse as JSON
            try:
                data = json.loads(decrypted_data.decode())
            except json.JSONDecodeError:
                data = decrypted_data.decode()
                
            # Log decryption event
            self._log_security_event('decryption', 'success')
            
            return data
            
        except Exception as e:
            logging.error(f"Error decrypting data: {e}", exc_info=True)
            self._log_security_event('decryption', 'failed', str(e))
            return {}
            
    def generate_signature(self, data: Dict) -> str:
        """Generates digital signature for data"""
        try:
            # Convert data to string
            data_str = json.dumps(data, sort_keys=True)
            
            # Create signature
            signature = hmac.new(
                self.config['secret_key'].encode(),
                data_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logging.error(f"Error generating signature: {e}", exc_info=True)
            return ''
            
    def verify_signature(self, data: Dict, signature: str) -> bool:
        """Verifies digital signature"""
        try:
            # Generate new signature for comparison
            new_signature = self.generate_signature(data)
            
            # Compare signatures
            return hmac.compare_digest(new_signature, signature)
            
        except Exception as e:
            logging.error(f"Error verifying signature: {e}", exc_info=True)
            return False
            
    def generate_access_token(self, user_data: Dict) -> str:
        """Generates JWT access token"""
        try:
            # Add expiration time
            user_data['exp'] = datetime.now(timezone.utc).timestamp() + \
                             self.config['token_expiry']
                             
            # Generate token
            token = jwt.encode(
                user_data,
                self.config['secret_key'],
                algorithm='HS256'
            )
            
            return token
            
        except Exception as e:
            logging.error(f"Error generating token: {e}", exc_info=True)
            return ''
            
    def verify_access_token(self, token: str) -> Dict:
        """Verifies JWT access token"""
        try:
            # Decode and verify token
            decoded_data = jwt.decode(
                token,
                self.config['secret_key'],
                algorithms=['HS256']
            )
            
            return decoded_data
            
        except jwt.ExpiredSignatureError:
            logging.error("Token has expired")
            return {}
        except jwt.InvalidTokenError as e:
            logging.error(f"Invalid token: {e}")
            return {}
            
    def _log_security_event(self, event_type: str, status: str, details: str = '') -> None:
        """Logs security events"""
        try:
            event = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': event_type,
                'status': status,
                'details': details
            }
            
            self.security_logs.append(event)
            logging.info(f"Security event logged: {event}")
            
        except Exception as e:
            logging.error(f"Error logging security event: {e}", exc_info=True)
            
    async def analyze_security_logs(self) -> Dict:
        """Analyzes security logs for potential issues"""
        try:
            analysis = {
                'total_events': len(self.security_logs),
                'failed_events': self._count_failed_events(),
                'event_distribution': self._analyze_event_distribution(),
                'potential_threats': self._identify_potential_threats()
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing security logs: {e}", exc_info=True)
            return {}
            
    def _count_failed_events(self) -> int:
        """Counts failed security events"""
        try:
            return sum(1 for event in self.security_logs
                      if event['status'] == 'failed')
                      
        except Exception as e:
            logging.error(f"Error counting failed events: {e}", exc_info=True)
            return 0
            
    def _analyze_event_distribution(self) -> Dict:
        """Analyzes distribution of security events"""
        try:
            distribution = {}
            
            for event in self.security_logs:
                event_type = event['type']
                distribution[event_type] = distribution.get(event_type, 0) + 1
                
            return distribution
            
        except Exception as e:
            logging.error(f"Error analyzing event distribution: {e}", exc_info=True)
            return {}
            
    def _identify_potential_threats(self) -> List[Dict]:
        """Identifies potential security threats"""
        try:
            threats = []
            
            # Analyze failed events
            failed_events = [event for event in self.security_logs
                           if event['status'] == 'failed']
                           
            # Group by type
            failed_by_type = {}
            for event in failed_events:
                event_type = event['type']
                failed_by_type[event_type] = failed_by_type.get(event_type, 0) + 1
                
            # Identify suspicious patterns
            for event_type, count in failed_by_type.items():
                if count >= self.config['threat_threshold']:
                    threats.append({
                        'type': event_type,
                        'count': count,
                        'severity': 'high',
                        'description': f'Multiple failed {event_type} attempts'
                    })
                    
            return threats
            
        except Exception as e:
            logging.error(f"Error identifying threats: {e}", exc_info=True)
            return []
