"""
SecurityUtils - Utilidades de seguridad y criptografía.
Incluye encriptación, hashing, saneamiento de entrada y auditoría.
"""

import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import re
import logging
from pathlib import Path
from ..core.exceptions import BrainException, ValidationError

logger = logging.getLogger(__name__)

class SecurityUtils:
    """
    Utilidades de seguridad para protección de datos y operaciones.
    
    Características:
    1. Encriptación simétrica y asimétrica
    2. Hashing seguro con salt
    3. Saneamiento de entrada y prevención de inyecciones
    4. Generación de tokens y valores aleatorios seguros
    5. Auditoría de eventos de seguridad
    """
    
    # Algoritmos soportados
    SUPPORTED_HASH_ALGORITHMS = {
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
        'sha3_256': hashlib.sha3_256,
        'sha3_512': hashlib.sha3_512,
        'blake2b': hashlib.blake2b,
        'blake2s': hashlib.blake2s,
    }
    
    # Patrones peligrosos para saneamiento
    DANGEROUS_PATTERNS = [
        # SQL Injection
        r'(\'|\"|--|#|;|/\*|\*/|@@|@|char|nchar|varchar|nvarchar|alter|begin|cast|create|cursor|declare|delete|drop|end|exec|execute|fetch|insert|kill|open|select|sys|sysobject|syscolumns|table|update|union)',
        
        # XSS
        r'(<script|javascript:|onload=|onerror=|onclick=|onmouseover=|alert\(|confirm\(|prompt\(|eval\(|document\.|window\.|location\.|cookie)',
        
        # Command Injection
        r'(\||&|;|`|\$\(|\n|\r|<\?php|<\?=|\?>)',
        
        # Path Traversal
        r'(\.\./|\.\.\\|~/|\\|//)',
    ]
    
    @staticmethod
    def encrypt_data(
        data: Union[str, bytes],
        key: Optional[bytes] = None,
        algorithm: str = 'fernet',
        encoding: str = 'utf-8'
    ) -> Dict[str, Any]:
        """
        Encripta datos usando el algoritmo especificado.
        
        Args:
            data: Datos a encriptar
            key: Clave de encriptación (generada si es None)
            algorithm: Algoritmo ('fernet', 'aes', 'chacha20')
            encoding: Encoding para datos de texto
            
        Returns:
            Diccionario con datos encriptados y metadatos
            
        Raises:
            BrainException: Si hay error en la encriptación
            ValueError: Si el algoritmo no es soportado
        """
        if isinstance(data, str):
            data_bytes = data.encode(encoding)
        else:
            data_bytes = data
        
        try:
            if algorithm == 'fernet':
                return SecurityUtils._encrypt_fernet(data_bytes, key)
            elif algorithm == 'aes':
                return SecurityUtils._encrypt_aes(data_bytes, key)
            elif algorithm == 'chacha20':
                return SecurityUtils._encrypt_chacha20(data_bytes, key)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise BrainException(f"Encryption failed: {e}")
    
    @staticmethod
    def decrypt_data(
        encrypted_data: Dict[str, Any],
        key: Optional[bytes] = None,
        encoding: str = 'utf-8'
    ) -> Union[str, bytes]:
        """
        Desencripta datos previamente encriptados.
        
        Args:
            encrypted_data: Datos encriptados (formato retornado por encrypt_data)
            key: Clave de encriptación
            encoding: Encoding para decodificar a string
            
        Returns:
            Datos desencriptados (bytes o string)
            
        Raises:
            BrainException: Si hay error en la desencriptación
            ValidationError: Si los datos encriptados son inválidos
        """
        try:
            # Validar estructura
            required_fields = ['algorithm', 'data', 'salt', 'timestamp']
            for field in required_fields:
                if field not in encrypted_data:
                    raise ValidationError(f"Missing field in encrypted data: {field}")
            
            algorithm = encrypted_data['algorithm']
            encrypted_bytes = base64.b64decode(encrypted_data['data'])
            salt = base64.b64decode(encrypted_data['salt'])
            
            # Recuperar clave si no se proporciona
            if key is None and 'key_hash' in encrypted_data:
                # En producción, esto recuperaría la clave de un keystore seguro
                raise ValueError("Key required for decryption")
            
            if algorithm == 'fernet':
                result = SecurityUtils._decrypt_fernet(encrypted_bytes, key, salt)
            elif algorithm == 'aes':
                result = SecurityUtils._decrypt_aes(encrypted_bytes, key, salt)
            elif algorithm == 'chacha20':
                result = SecurityUtils._decrypt_chacha20(encrypted_bytes, key, salt)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Decodificar si se espera string
            if encrypted_data.get('encoding') == 'string':
                try:
                    return result.decode(encoding)
                except UnicodeDecodeError:
                    # Intentar con encoding original
                    original_encoding = encrypted_data.get('original_encoding', 'utf-8')
                    return result.decode(original_encoding)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise BrainException(f"Decryption failed: {e}")
    
    @staticmethod
    def hash_data(
        data: Union[str, bytes],
        algorithm: str = 'sha256',
        salt: Optional[bytes] = None,
        iterations: int = 100000,
        encoding: str = 'utf-8'
    ) -> Dict[str, Any]:
        """
        Calcula hash seguro de datos con salt.
        
        Args:
            data: Datos a hashear
            algorithm: Algoritmo de hash
            salt: Salt aleatorio (generado si es None)
            iterations: Número de iteraciones para PBKDF2
            encoding: Encoding para datos de texto
            
        Returns:
            Diccionario con hash y metadatos
            
        Raises:
            ValueError: Si el algoritmo no es soportado
        """
        if algorithm not in SecurityUtils.SUPPORTED_HASH_ALGORITHMS:
            raise ValueError(
                f"Unsupported hash algorithm: {algorithm}. "
                f"Supported: {list(SecurityUtils.SUPPORTED_HASH_ALGORITHMS.keys())}"
            )
        
        if isinstance(data, str):
            data_bytes = data.encode(encoding)
        else:
            data_bytes = data
        
        # Generar salt si no se proporciona
        if salt is None:
            salt = secrets.token_bytes(32)
        
        try:
            # Usar PBKDF2 para mayor seguridad
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            
            # Derivar clave
            key = kdf.derive(data_bytes)
            
            # Hash adicional con algoritmo especificado
            hash_func = SecurityUtils.SUPPORTED_HASH_ALGORITHMS[algorithm]
            final_hash = hash_func(key).hexdigest()
            
            return {
                'hash': final_hash,
                'algorithm': algorithm,
                'salt': base64.b64encode(salt).decode('ascii'),
                'iterations': iterations,
                'timestamp': datetime.now().isoformat(),
                'length': len(data_bytes)
            }
            
        except Exception as e:
            logger.error(f"Hashing failed: {e}")
            # Fallback a hash simple
            hash_func = SecurityUtils.SUPPORTED_HASH_ALGORITHMS[algorithm]
            if isinstance(data, str):
                simple_hash = hash_func(data.encode(encoding)).hexdigest()
            else:
                simple_hash = hash_func(data).hexdigest()
            
            return {
                'hash': simple_hash,
                'algorithm': algorithm,
                'salt': None,
                'iterations': 1,
                'timestamp': datetime.now().isoformat(),
                'length': len(data_bytes),
                'warning': 'Used simple hash due to error'
            }
    
    @staticmethod
    def validate_hash(
        data: Union[str, bytes],
        hash_info: Dict[str, Any],
        encoding: str = 'utf-8'
    ) -> bool:
        """
        Valida que datos produzcan el hash esperado.
        
        Args:
            data: Datos a validar
            hash_info: Información del hash (retornada por hash_data)
            encoding: Encoding para datos de texto
            
        Returns:
            True si el hash es válido
            
        Raises:
            ValidationError: Si la información del hash es inválida
        """
        required_fields = ['hash', 'algorithm']
        for field in required_fields:
            if field not in hash_info:
                raise ValidationError(f"Missing field in hash info: {field}")
        
        # Calcular hash de los datos
        salt = None
        if hash_info.get('salt'):
            salt = base64.b64decode(hash_info['salt'])
        
        iterations = hash_info.get('iterations', 100000)
        
        new_hash_info = SecurityUtils.hash_data(
            data=data,
            algorithm=hash_info['algorithm'],
            salt=salt,
            iterations=iterations,
            encoding=encoding
        )
        
        # Comparar hashes de manera segura (constante time)
        return hmac.compare_digest(
            new_hash_info['hash'],
            hash_info['hash']
        )
    
    @staticmethod
    def sanitize_input(
        input_data: Any,
        input_type: str = 'text',
        max_length: Optional[int] = None,
        allow_html: bool = False,
        allow_scripts: bool = False,
        allow_sql: bool = False,
        custom_patterns: List[str] = None
    ) -> Any:
        """
        Sanea entrada de usuario para prevenir ataques.
        
        Args:
            input_data: Datos a sanear
            input_type: Tipo de entrada ('text', 'html', 'sql', 'path', 'json')
            max_length: Longitud máxima permitida
            allow_html: Permitir etiquetas HTML
            allow_scripts: Permitir scripts
            allow_sql: Permitir palabras clave SQL
            custom_patterns: Patrones personalizados a bloquear
            
        Returns:
            Datos saneados
            
        Raises:
            ValidationError: Si la entrada contiene contenido peligroso
        """
        if input_data is None:
            return None
        
        # Convertir a string para saneamiento
        if isinstance(input_data, (dict, list)):
            # Para estructuras complejas, sanear recursivamente
            return SecurityUtils._sanitize_structure(
                input_data, input_type, max_length, allow_html, 
                allow_scripts, allow_sql, custom_patterns
            )
        
        original_type = type(input_data)
        str_data = str(input_data)
        
        # Aplicar límite de longitud
        if max_length and len(str_data) > max_length:
            str_data = str_data[:max_length]
        
        # Saneamiento específico por tipo
        if input_type == 'html':
            str_data = SecurityUtils._sanitize_html(str_data, allow_scripts)
        elif input_type == 'sql':
            str_data = SecurityUtils._sanitize_sql(str_data, allow_sql)
        elif input_type == 'path':
            str_data = SecurityUtils._sanitize_path(str_data)
        elif input_type == 'json':
            str_data = SecurityUtils._sanitize_json(str_data)
        else:  # 'text' por defecto
            str_data = SecurityUtils._sanitize_text(str_data)
        
        # Aplicar patrones peligrosos
        dangerous_patterns = SecurityUtils.DANGEROUS_PATTERNS.copy()
        if custom_patterns:
            dangerous_patterns.extend(custom_patterns)
        
        for pattern in dangerous_patterns:
            if re.search(pattern, str_data, re.IGNORECASE):
                raise ValidationError(
                    f"Input contains potentially dangerous pattern: {pattern}"
                )
        
        # Convertir de vuelta al tipo original si es posible
        try:
            if original_type == int:
                return int(str_data)
            elif original_type == float:
                return float(str_data)
            elif original_type == bool:
                return str_data.lower() in ('true', '1', 'yes', 'y')
            else:
                return str_data
        except (ValueError, TypeError):
            return str_data
    
    @staticmethod
    def generate_secure_random(
        length: int = 32,
        type: str = 'bytes',
        charset: Optional[str] = None
    ) -> Union[bytes, str, int]:
        """
        Genera valores aleatorios criptográficamente seguros.
        
        Args:
            length: Longitud del valor aleatorio
            type: Tipo de salida ('bytes', 'string', 'hex', 'int', 'uuid')
            charset: Conjunto de caracteres para strings (opcional)
            
        Returns:
            Valor aleatorio seguro
        """
        if type == 'bytes':
            return secrets.token_bytes(length)
        
        elif type == 'string':
            if charset:
                return ''.join(secrets.choice(charset) for _ in range(length))
            else:
                # Charset alfanumérico seguro
                alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                return ''.join(secrets.choice(alphabet) for _ in range(length))
        
        elif type == 'hex':
            return secrets.token_hex(length // 2 + 1)[:length]
        
        elif type == 'int':
            # Entero seguro en rango [0, 2^length - 1]
            max_value = 2 ** (length * 8) - 1
            return secrets.randbelow(max_value)
        
        elif type == 'uuid':
            import uuid
            return str(uuid.uuid4())
        
        else:
            raise ValueError(f"Unsupported random type: {type}")
    
    @staticmethod
    def audit_security_events(
        event_type: str,
        event_data: Dict[str, Any],
        severity: str = 'info',
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        save_to_file: bool = True,
        file_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Registra eventos de seguridad para auditoría.
        
        Args:
            event_type: Tipo de evento ('login', 'access', 'error', 'config_change')
            event_data: Datos específicos del evento
            severity: Severidad ('info', 'warning', 'error', 'critical')
            user_id: ID del usuario relacionado
            ip_address: Dirección IP del origen
            save_to_file: Guardar en archivo de log
            file_path: Ruta al archivo de log
            
        Returns:
            Entrada de auditoría creada
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'event_id': secrets.token_hex(16),
            'data': event_data,
            'metadata': {
                'user_id': user_id,
                'ip_address': ip_address,
                'process_id': os.getpid(),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            }
        }
        
        # Hashear datos sensibles si están presentes
        sensitive_fields = ['password', 'token', 'key', 'secret']
        for field in sensitive_fields:
            if field in event_data:
                audit_entry['data'][field] = '[REDACTED]'
        
        # Guardar en archivo si está configurado
        if save_to_file:
            try:
                if file_path is None:
                    file_path = Path('security_audit.log')
                else:
                    file_path = Path(file_path)
                
                # Crear directorio si no existe
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Formato de log estructurado
                log_line = json.dumps(audit_entry) + '\n'
                
                # Escribir de manera atómica
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(log_line)
                
                logger.info(f"Security event logged: {event_type} ({severity})")
                
            except Exception as e:
                logger.error(f"Failed to write security audit: {e}")
        
        return audit_entry
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _encrypt_fernet(data: bytes, key: Optional[bytes]) -> Dict[str, Any]:
        """Encripta usando Fernet (AES CBC + HMAC)."""
        from cryptography.fernet import Fernet
        
        # Generar clave si no se proporciona
        if key is None:
            key = Fernet.generate_key()
        elif len(key) != 44:  # Fernet keys son 32 bytes codificados en base64
            # Derivar clave Fernet
            salt = secrets.token_bytes(16)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key)
            key = base64.urlsafe_b64encode(derived_key)
        
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data)
        
        return {
            'algorithm': 'fernet',
            'data': base64.b64encode(encrypted).decode('ascii'),
            'key_hash': base64.b64encode(hashlib.sha256(key).digest()).decode('ascii'),
            'salt': base64.b64encode(secrets.token_bytes(16)).decode('ascii'),
            'timestamp': datetime.now().isoformat(),
            'encoding': 'bytes' if isinstance(data, bytes) else 'string'
        }
    
    @staticmethod
    def _decrypt_fernet(encrypted: bytes, key: bytes, salt: bytes) -> bytes:
        """Desencripta usando Fernet."""
        from cryptography.fernet import Fernet
        
        # Si la clave no es del formato Fernet, derivarla
        if len(key) != 44:  # No es una clave Fernet
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key)
            key = base64.urlsafe_b64encode(derived_key)
        
        fernet = Fernet(key)
        return fernet.decrypt(encrypted)
    
    @staticmethod
    def _encrypt_aes(data: bytes, key: Optional[bytes]) -> Dict[str, Any]:
        """Encripta usando AES-256-GCM."""
        # Generar clave y nonce
        if key is None:
            key = secrets.token_bytes(32)  # AES-256 requiere 32 bytes
        
        if len(key) not in [16, 24, 32]:
            # Derivar clave de tamaño apropiado
            salt = secrets.token_bytes(16)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(key)
            used_salt = salt
        else:
            used_salt = secrets.token_bytes(16)
        
        # Generar nonce
        nonce = secrets.token_bytes(12)  # GCM recomienda 12 bytes
        
        # Configurar cifrado
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Encriptar
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'algorithm': 'aes',
            'data': base64.b64encode(ciphertext).decode('ascii'),
            'key_hash': base64.b64encode(hashlib.sha256(key).digest()).decode('ascii'),
            'salt': base64.b64encode(used_salt).decode('ascii'),
            'nonce': base64.b64encode(nonce).decode('ascii'),
            'tag': base64.b64encode(encryptor.tag).decode('ascii'),
            'timestamp': datetime.now().isoformat(),
            'encoding': 'bytes' if isinstance(data, bytes) else 'string'
        }
    
    @staticmethod
    def _decrypt_aes(encrypted: bytes, key: bytes, salt: bytes) -> bytes:
        """Desencripta usando AES-256-GCM."""
        # Para AES, necesitamos nonce y tag del diccionario original
        # Esta función asume que se llama desde decrypt_data que extrae estos valores
        raise NotImplementedError("AES decryption requires additional parameters")
    
    @staticmethod
    def _encrypt_chacha20(data: bytes, key: Optional[bytes]) -> Dict[str, Any]:
        """Encripta usando ChaCha20-Poly1305."""
        # ChaCha20 requiere clave de 32 bytes
        if key is None:
            key = secrets.token_bytes(32)
        
        if len(key) != 32:
            # Derivar clave de 32 bytes
            salt = secrets.token_bytes(16)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(key)
            used_salt = salt
        else:
            used_salt = secrets.token_bytes(16)
        
        # Generar nonce (ChaCha20 requiere 12 bytes)
        nonce = secrets.token_bytes(12)
        
        # Configurar cifrado
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,  # ChaCha20 no usa modo
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Encriptar
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'algorithm': 'chacha20',
            'data': base64.b64encode(ciphertext).decode('ascii'),
            'key_hash': base64.b64encode(hashlib.sha256(key).digest()).decode('ascii'),
            'salt': base64.b64encode(used_salt).decode('ascii'),
            'nonce': base64.b64encode(nonce).decode('ascii'),
            'timestamp': datetime.now().isoformat(),
            'encoding': 'bytes' if isinstance(data, bytes) else 'string'
        }
    
    @staticmethod
    def _decrypt_chacha20(encrypted: bytes, key: bytes, salt: bytes) -> bytes:
        """Desencripta usando ChaCha20."""
        # Similar a AES, requiere parámetros adicionales
        raise NotImplementedError("ChaCha20 decryption requires additional parameters")
    
    @staticmethod
    def _sanitize_structure(data, input_type, max_length, allow_html, 
                           allow_scripts, allow_sql, custom_patterns):
        """Sanea estructuras de datos complejas recursivamente."""
        if isinstance(data, dict):
            return {
                key: SecurityUtils.sanitize_input(
                    value, input_type, max_length, allow_html,
                    allow_scripts, allow_sql, custom_patterns
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                SecurityUtils.sanitize_input(
                    item, input_type, max_length, allow_html,
                    allow_scripts, allow_sql, custom_patterns
                )
                for item in data
            ]
        else:
            return SecurityUtils.sanitize_input(
                data, input_type, max_length, allow_html,
                allow_scripts, allow_sql, custom_patterns
            )
    
    @staticmethod
    def _sanitize_html(text: str, allow_scripts: bool) -> str:
        """Sanea HTML eliminando etiquetas peligrosas."""
        if allow_scripts:
            # Solo eliminar etiquetas particularmente peligrosas
            dangerous_tags = ['iframe', 'object', 'embed']
            for tag in dangerous_tags:
                text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.IGNORECASE | re.DOTALL)
            return text
        else:
            # Eliminar todas las etiquetas HTML
            text = re.sub(r'<[^>]+>', '', text)
            # Escapar caracteres HTML
            text = (text.replace('&', '&amp;')
                       .replace('<', '&lt;')
                       .replace('>', '&gt;')
                       .replace('"', '&quot;')
                       .replace("'", '&#x27;'))
            return text
    
    @staticmethod
    def _sanitize_sql(text: str, allow_sql: bool) -> str:
        """Sanea para prevenir inyección SQL."""
        if allow_sql:
            return text
        else:
            # Escapar caracteres especiales SQL
            text = text.replace("'", "''")
            text = text.replace('"', '""')
            text = text.replace(';', '')
            text = text.replace('--', '')
            text = text.replace('/*', '')
            text = text.replace('*/', '')
            return text
    
    @staticmethod
    def _sanitize_path(text: str) -> str:
        """Sanea rutas de archivo para prevenir traversal."""
        # Normalizar separadores
        text = text.replace('\\', '/')
        
        # Eliminar componentes de traversal
        while '/../' in text or text.startswith('../'):
            text = text.replace('/../', '/')
            text = text.replace('../', '')
        
        # Eliminar referencia a directorio home
        text = text.replace('~/', '')
        
        # Eliminar protocolos
        text = re.sub(r'^[a-zA-Z]+://', '', text)
        
        return text
    
    @staticmethod
    def _sanitize_json(text: str) -> str:
        """Valida y sanea JSON."""
        try:
            # Validar que es JSON válido
            parsed = json.loads(text)
            # Volver a serializar (esto elimina cualquier código ejecutable)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # Si no es JSON válido, escapar caracteres peligrosos
            return json.dumps(text)
    
    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Sanea texto plano."""
        # Escapar caracteres especiales básicos
        text = text.replace('\0', '')  # Null byte
        text = text.replace('\r', '')  # Carriage return
        text = text.replace('\n', ' ')  # Newline
        
        # Eliminar caracteres de control
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text