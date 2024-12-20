import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import jwt
from datetime import datetime, timedelta
import logging
from typing import Dict, Union
import asyncio
import secrets

class SecuritySystem:
    def __init__(self, config: Dict):
        self.config = config
        self.key = self._generate_key()
        self.fernet = Fernet(self.key)
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    def _generate_key(self) -> bytes:
        """Generates a secure encryption key"""
        try:
            salt = secrets.token_bytes(16)
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
            return None
            
    async def secure_data(self, data: Dict) -> Dict:
        """Secures sensitive data"""
        try:
            # Encrypt sensitive data
            encrypted_data = self._encrypt_sensitive_data(data)
            
            # Generate signatures
            signatures = self._generate_signatures(encrypted_data)
            
            # Apply access controls
            access_controls = self._apply_access_controls(encrypted_data)
            
            return {
                'encrypted_data': encrypted_data,
                'signatures': signatures,
                'access_controls': access_controls,
                'metadata': self._generate_security_metadata()
            }
            
        except Exception as e:
            logging.error(f"Error securing data: {e}", exc_info=True)
            return {}
            
    def _encrypt_sensitive_data(self, data: Dict) -> Dict:
        """Encrypts sensitive data"""
        try:
            encrypted_data = {}
            
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    # Encrypt with Fernet (symmetric)
                    if isinstance(value, (str, bytes)):
                        encrypted_data[key] = self.fernet.encrypt(
                            value.encode() if isinstance(value, str) else value
                        )
                    # Encrypt with RSA (asymmetric)
                    else:
                        encrypted_data[key] = self.public_key.encrypt(
                            str(value).encode(),
                            padding.OAEP(
                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                algorithm=hashes.SHA256(),
                                label=None
                            )
                        )
                else:
                    encrypted_data[key] = value
                    
            return encrypted_data
            
        except Exception as e:
            logging.error(f"Error encrypting data: {e}", exc_info=True)
            return {}
            
    def _generate_signatures(self, data: Dict) -> Dict:
        """Generates digital signatures"""
        try:
            signatures = {}
            
            for key, value in data.items():
                if isinstance(value, (str, bytes)):
                    # Generate HMAC
                    signatures[key] = hmac.new(
                        self.key,
                        value.encode() if isinstance(value, str) else value,
                        hashlib.sha256
                    ).hexdigest()
                    
            return signatures
            
        except Exception as e:
            logging.error(f"Error generating signatures: {e}", exc_info=True)
            return {}
            
    def _apply_access_controls(self, data: Dict) -> Dict:
        """Applies access controls"""
        try:
            access_controls = {}
            
            for key in data.keys():
                # Generate JWT token
                access_controls[key] = self._generate_access_token(key)
                
            return access_controls
            
        except Exception as e:
            logging.error(f"Error applying access controls: {e}", exc_info=True)
            return {}
            
    def _generate_access_token(self, resource: str) -> str:
        """Generates JWT access token"""
        try:
            payload = {
                'resource': resource,
                'exp': datetime.utcnow() + timedelta(hours=1),
                'iat': datetime.utcnow(),
                'permissions': self._get_resource_permissions(resource)
            }
            
            return jwt.encode(
                payload,
                self.config['jwt_secret'],
                algorithm='HS256'
            )
            
        except Exception as e:
            logging.error(f"Error generating access token: {e}", exc_info=True)
            return ''
            
    def _is_sensitive_field(self, field: str) -> bool:
        """Checks if field contains sensitive data"""
        sensitive_fields = [
            'api_key',
            'secret_key',
            'private_key',
            'password',
            'token',
            'credentials',
            'wallet',
            'balance',
            'position'
        ]
        return any(sensitive in field.lower() for sensitive in sensitive_fields)
        
    def _get_resource_permissions(self, resource: str) -> List[str]:
        """Gets permissions for resource"""
        try:
            base_permissions = ['read']
            
            if resource in self.config['writable_resources']:
                base_permissions.append('write')
                
            if resource in self.config['deletable_resources']:
                base_permissions.append('delete')
                
            return base_permissions
            
        except Exception as e:
            logging.error(f"Error getting permissions: {e}", exc_info=True)
            return ['read']
            
    async def verify_security(self, secured_data: Dict) -> bool:
        """Verifies security measures"""
        try:
            # Verify signatures
            signatures_valid = self._verify_signatures(
                secured_data['encrypted_data'],
                secured_data['signatures']
            )
            
            # Verify access controls
            access_valid = self._verify_access_controls(
                secured_data['access_controls']
            )
            
            # Verify data integrity
            integrity_valid = self._verify_data_integrity(
                secured_data['encrypted_data']
            )
            
            return all([signatures_valid, access_valid, integrity_valid])
            
        except Exception as e:
            logging.error(f"Error verifying security: {e}", exc_info=True)
            return False
            
    def _verify_signatures(self,
                          data: Dict,
                          signatures: Dict) -> bool:
        """Verifies digital signatures"""
        try:
            for key, value in data.items():
                if key in signatures:
                    expected_signature = hmac.new(
                        self.key,
                        value.encode() if isinstance(value, str) else value,
                        hashlib.sha256
                    ).hexdigest()
                    
                    if signatures[key] != expected_signature:
                        return False
                        
            return True
            
        except Exception as e:
            logging.error(f"Error verifying signatures: {e}", exc_info=True)
            return False
            
    def _verify_access_controls(self, access_controls: Dict) -> bool:
        """Verifies access control tokens"""
        try:
            for token in access_controls.values():
                try:
                    jwt.decode(
                        token,
                        self.config['jwt_secret'],
                        algorithms=['HS256']
                    )
                except jwt.InvalidTokenError:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error verifying access controls: {e}", exc_info=True)
            return False
            
    def _verify_data_integrity(self, data: Dict) -> bool:
        """Verifies data integrity"""
        try:
            for key, value in data.items():
                if self._is_sensitive_field(key):
                    try:
                        # Try decrypting to verify integrity
                        if isinstance(value, (str, bytes)):
                            self.fernet.decrypt(
                                value.encode() if isinstance(value, str) else value
                            )
                    except Exception:
                        return False
                        
            return True
            
        except Exception as e:
            logging.error(f"Error verifying data integrity: {e}", exc_info=True)
            return False
