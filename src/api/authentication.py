"""
Authentication Module - Project Brain API
Manejo de autenticación, autorización, tokens y sesiones.
"""

import asyncio
import uuid
import time
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

from ..core.exceptions import (
    BrainException, 
    AuthenticationError, 
    AuthorizationError,
    ValidationError
)
from ..utils.security_utils import SecurityUtils
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# MODELOS DE DATOS
# ============================================================================

class AuthMethod(str, Enum):
    """Métodos de autenticación soportados."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    BASIC = "basic"

class TokenType(str, Enum):
    """Tipos de tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SESSION = "session"

class UserRole(str, Enum):
    """Roles de usuario."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SYSTEM = "system"
    GUEST = "guest"

class SessionStatus(str, Enum):
    """Estados de sesión."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

@dataclass
class UserCredentials:
    """Credenciales de usuario."""
    username: str
    password_hash: str
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    roles: List[UserRole] = field(default_factory=lambda: [UserRole.VIEWER])
    email: Optional[str] = None
    full_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0

@dataclass
class AuthToken:
    """Token de autenticación."""
    token: str
    token_type: TokenType
    user_id: str
    client_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verifica si el token ha expirado."""
        return datetime.now() > self.expires_at

@dataclass
class UserSession:
    """Sesión de usuario."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    auth_method: AuthMethod
    client_ip: str
    user_agent: Optional[str] = None
    issued_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=8))
    status: SessionStatus = SessionStatus.ACTIVE
    tokens: List[AuthToken] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Verifica si la sesión está activa."""
        return (self.status == SessionStatus.ACTIVE and 
                datetime.now() < self.expires_at)

class AuthRequest(BaseModel):
    """Solicitud de autenticación."""
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    auth_method: AuthMethod = AuthMethod.JWT
    client_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class AuthResponse(BaseModel):
    """Respuesta de autenticación."""
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 3600
    user_id: Optional[str] = None
    roles: List[UserRole] = Field(default_factory=list)
    session_id: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class TokenValidationResponse(BaseModel):
    """Respuesta de validación de token."""
    valid: bool
    user_id: Optional[str] = None
    roles: List[UserRole] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# CLASE PRINCIPAL AUTHENTICATION
# ============================================================================

class Authentication:
    """
    Sistema de autenticación y gestión de sesiones para Project Brain.
    
    Responsabilidades:
    1. Autenticación de usuarios mediante múltiples métodos
    2. Generación y validación de tokens JWT
    3. Gestión de sesiones y tokens
    4. Auditoría de actividad de autenticación
    5. Rate limiting y protección contra ataques
    """
    
    def __init__(self, config_manager: Any, storage_adapter: Optional[Any] = None):
        """
        Inicializa el sistema de autenticación.
        
        Args:
            config_manager: Gestor de configuración del sistema
            storage_adapter: Adaptador para almacenamiento (opcional)
        """
        self.config_manager = config_manager
        self.storage = storage_adapter
        
        # Cargar configuración
        self.config = self._load_auth_config()
        
        # Inicializar componentes
        self.security_utils = SecurityUtils()
        self.active_sessions: Dict[str, UserSession] = {}
        self.revoked_tokens: set = set()
        self.failed_attempts: Dict[str, Dict] = {}
        
        # Inicializar métodos de autenticación
        self.auth_methods = self._initialize_auth_methods()
        
        logger.info(f"Authentication system initialized with methods: {list(self.auth_methods.keys())}")
    
    # ============================================================================
    # MÉTODOS PÚBLICOS PRINCIPALES
    # ============================================================================
    
    async def authenticate_user(self, 
                               credentials: Union[Dict, AuthRequest],
                               client_ip: str,
                               user_agent: Optional[str] = None) -> AuthResponse:
        """
        Autentica a un usuario utilizando diferentes métodos.
        
        Args:
            credentials: Credenciales de autenticación
            client_ip: Dirección IP del cliente
            user_agent: User-Agent del cliente (opcional)
            
        Returns:
            AuthResponse con tokens y datos de sesión
            
        Raises:
            AuthenticationError: Si la autenticación falla
            ValidationError: Si las credenciales son inválidas
        """
        start_time = time.time()
        
        try:
            # Convertir a AuthRequest si es necesario
            if isinstance(credentials, dict):
                auth_request = AuthRequest(**credentials)
            else:
                auth_request = credentials
            
            # Validar rate limiting
            self._check_rate_limit(auth_request.username, client_ip)
            
            # Autenticar según método
            auth_method = self.auth_methods.get(auth_request.auth_method)
            if not auth_method:
                raise AuthenticationError(f"Auth method not supported: {auth_request.auth_method}")
            
            # Ejecutar autenticación
            user_info = await auth_method(auth_request)
            
            # Crear tokens
            access_token = await self.generate_token(
                user_id=user_info["user_id"],
                token_type=TokenType.ACCESS,
                scopes=auth_request.scopes,
                client_id=auth_request.client_id
            )
            
            refresh_token = await self.generate_token(
                user_id=user_info["user_id"],
                token_type=TokenType.REFRESH,
                scopes=auth_request.scopes,
                client_id=auth_request.client_id
            )
            
            # Crear sesión
            session = await self._create_session(
                user_id=user_info["user_id"],
                auth_method=auth_request.auth_method,
                client_ip=client_ip,
                user_agent=user_agent,
                tokens=[access_token, refresh_token]
            )
            
            # Registrar actividad
            await self._audit_auth_activity({
                "event": "user_authenticated",
                "user_id": user_info["user_id"],
                "auth_method": auth_request.auth_method.value,
                "client_ip": client_ip,
                "success": True,
                "timestamp": datetime.now()
            })
            
            # Limpiar intentos fallidos
            self._clear_failed_attempts(auth_request.username, client_ip)
            
            processing_time = time.time() - start_time
            logger.info(f"User {user_info['user_id']} authenticated in {processing_time:.2f}s")
            
            return AuthResponse(
                success=True,
                access_token=access_token.token,
                refresh_token=refresh_token.token,
                user_id=user_info["user_id"],
                roles=user_info.get("roles", []),
                session_id=session.session_id,
                expires_in=int(self.config["jwt"]["access_token_expiry"])
            )
            
        except Exception as e:
            # Registrar intento fallido
            username = credentials.get("username") if isinstance(credentials, dict) else credentials.username
            await self._record_failed_attempt(username, client_ip, str(e))
            
            # Registrar en auditoría
            await self._audit_auth_activity({
                "event": "authentication_failed",
                "username": username,
                "auth_method": credentials.get("auth_method", "unknown") if isinstance(credentials, dict) else credentials.auth_method.value,
                "client_ip": client_ip,
                "error": str(e),
                "timestamp": datetime.now()
            })
            
            logger.warning(f"Authentication failed for {username}: {e}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    async def generate_token(self,
                            user_id: str,
                            token_type: TokenType = TokenType.ACCESS,
                            scopes: Optional[List[str]] = None,
                            client_id: Optional[str] = None,
                            custom_claims: Optional[Dict] = None) -> AuthToken:
        """
        Genera un token JWT.
        
        Args:
            user_id: ID del usuario
            token_type: Tipo de token
            scopes: Alcances del token
            client_id: ID del cliente (opcional)
            custom_claims: Claims personalizados (opcional)
            
        Returns:
            AuthToken generado
        """
        # Determinar expiración según tipo
        if token_type == TokenType.ACCESS:
            expiry_seconds = self.config["jwt"]["access_token_expiry"]
        elif token_type == TokenType.REFRESH:
            expiry_seconds = self.config["jwt"]["refresh_token_expiry"]
        elif token_type == TokenType.API_KEY:
            expiry_seconds = self.config["api_key"]["expiry_days"] * 24 * 3600
        else:
            expiry_seconds = 3600  # Default 1 hora
        
        # Crear payload
        issued_at = datetime.now()
        expires_at = issued_at + timedelta(seconds=expiry_seconds)
        
        payload = {
            "sub": user_id,
            "type": token_type.value,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(uuid.uuid4()),
            "scopes": scopes or [],
            "client_id": client_id
        }
        
        # Añadir claims personalizados
        if custom_claims:
            payload.update(custom_claims)
        
        # Firmar token
        secret_key = self.config["jwt"]["secret_key"]
        algorithm = self.config["jwt"]["algorithm"]
        
        token_str = jwt.encode(payload, secret_key, algorithm=algorithm)
        
        # Crear objeto AuthToken
        auth_token = AuthToken(
            token=token_str,
            token_type=token_type,
            user_id=user_id,
            client_id=client_id,
            scopes=scopes or [],
            issued_at=issued_at,
            expires_at=expires_at,
            metadata={"jti": payload["jti"]}
        )
        
        logger.debug(f"Generated {token_type.value} token for user {user_id}")
        return auth_token
    
    async def validate_token(self, 
                            token: str, 
                            token_type: TokenType = TokenType.ACCESS,
                            required_scopes: Optional[List[str]] = None) -> TokenValidationResponse:
        """
        Valida un token JWT.
        
        Args:
            token: Token a validar
            token_type: Tipo de token esperado
            required_scopes: Alcances requeridos (opcional)
            
        Returns:
            TokenValidationResponse con resultado de validación
        """
        try:
            # Verificar si el token fue revocado
            if token in self.revoked_tokens:
                return TokenValidationResponse(
                    valid=False,
                    error="Token revoked"
                )
            
            # Decodificar token
            secret_key = self.config["jwt"]["secret_key"]
            algorithm = self.config["jwt"]["algorithm"]
            
            try:
                payload = jwt.decode(
                    token, 
                    secret_key, 
                    algorithms=[algorithm],
                    options={"require": ["exp", "iat", "sub"]}
                )
            except jwt.ExpiredSignatureError:
                return TokenValidationResponse(
                    valid=False,
                    error="Token expired"
                )
            except jwt.InvalidTokenError as e:
                return TokenValidationResponse(
                    valid=False,
                    error=f"Invalid token: {str(e)}"
                )
            
            # Verificar tipo de token
            if payload.get("type") != token_type.value:
                return TokenValidationResponse(
                    valid=False,
                    error=f"Invalid token type: expected {token_type.value}, got {payload.get('type')}"
                )
            
            # Verificar alcances si se requieren
            if required_scopes:
                token_scopes = payload.get("scopes", [])
                if not all(scope in token_scopes for scope in required_scopes):
                    return TokenValidationResponse(
                        valid=False,
                        error="Insufficient scopes"
                    )
            
            # Token válido
            return TokenValidationResponse(
                valid=True,
                user_id=payload["sub"],
                roles=self._get_user_roles(payload["sub"]),
                scopes=payload.get("scopes", []),
                expires_at=datetime.fromtimestamp(payload["exp"])
            )
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return TokenValidationResponse(
                valid=False,
                error=f"Validation error: {str(e)}"
            )
    
    async def refresh_token(self, 
                           refresh_token: str,
                           client_ip: str,
                           user_agent: Optional[str] = None) -> AuthResponse:
        """
        Refresca un token de acceso usando un refresh token.
        
        Args:
            refresh_token: Refresh token válido
            client_ip: Dirección IP del cliente
            user_agent: User-Agent del cliente (opcional)
            
        Returns:
            AuthResponse con nuevo access token
        """
        # Validar refresh token
        validation = await self.validate_token(refresh_token, TokenType.REFRESH)
        
        if not validation.valid:
            raise AuthenticationError(f"Invalid refresh token: {validation.error}")
        
        # Revocar el refresh token antiguo (opcional, depende de la estrategia)
        if self.config["jwt"]["refresh_token_rotation"]:
            await self.revoke_token(refresh_token)
        
        # Generar nuevos tokens
        access_token = await self.generate_token(
            user_id=validation.user_id,
            token_type=TokenType.ACCESS,
            scopes=validation.scopes
        )
        
        new_refresh_token = await self.generate_token(
            user_id=validation.user_id,
            token_type=TokenType.REFRESH,
            scopes=validation.scopes
        )
        
        # Actualizar sesión
        session = await self._get_session_by_user(validation.user_id)
        if session:
            session.tokens = [t for t in session.tokens 
                            if t.token != refresh_token]  # Remover token antiguo
            session.tokens.extend([access_token, new_refresh_token])
            session.last_activity = datetime.now()
        
        logger.info(f"Tokens refreshed for user {validation.user_id}")
        
        return AuthResponse(
            success=True,
            access_token=access_token.token,
            refresh_token=new_refresh_token.token if self.config["jwt"]["refresh_token_rotation"] else None,
            user_id=validation.user_id,
            roles=validation.roles,
            expires_in=int(self.config["jwt"]["access_token_expiry"])
        )
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoca un token (lo añade a la lista negra).
        
        Args:
            token: Token a revocar
            
        Returns:
            bool: True si se revocó exitosamente
        """
        try:
            # Decodificar para obtener metadata
            secret_key = self.config["jwt"]["secret_key"]
            algorithm = self.config["jwt"]["algorithm"]
            
            payload = jwt.decode(
                token, 
                secret_key, 
                algorithms=[algorithm],
                options={"verify_exp": False}
            )
            
            # Añadir a lista de revocados
            self.revoked_tokens.add(token)
            
            # Si es un token de sesión, actualizar sesión
            jti = payload.get("jti")
            if jti:
                # Buscar y actualizar sesión
                for session in self.active_sessions.values():
                    for t in session.tokens:
                        if t.metadata.get("jti") == jti:
                            session.tokens.remove(t)
                            
                            # Si no quedan tokens, expirar sesión
                            if not session.tokens:
                                session.status = SessionStatus.EXPIRED
                            
                            break
            
            # Registrar auditoría
            await self._audit_auth_activity({
                "event": "token_revoked",
                "user_id": payload.get("sub"),
                "token_type": payload.get("type"),
                "jti": jti,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Token revoked for user {payload.get('sub')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def manage_sessions(self, 
                             user_id: str, 
                             action: str,
                             session_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Gestiona sesiones de usuario.
        
        Args:
            user_id: ID del usuario
            action: Acción a realizar (list, terminate, suspend, info)
            session_data: Datos adicionales de la sesión
            
        Returns:
            Dict con resultado de la operación
        """
        if action == "list":
            # Listar sesiones activas del usuario
            user_sessions = [
                session for session in self.active_sessions.values()
                if session.user_id == user_id and session.is_active()
            ]
            
            return {
                "user_id": user_id,
                "active_sessions": len(user_sessions),
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "auth_method": s.auth_method.value,
                        "client_ip": s.client_ip,
                        "issued_at": s.issued_at,
                        "last_activity": s.last_activity,
                        "expires_at": s.expires_at,
                        "status": s.status.value
                    }
                    for s in user_sessions
                ]
            }
            
        elif action == "terminate":
            # Terminar sesión específica o todas
            session_id = session_data.get("session_id") if session_data else None
            
            terminated = 0
            for session in list(self.active_sessions.values()):
                if session.user_id == user_id:
                    if not session_id or session.session_id == session_id:
                        session.status = SessionStatus.REVOKED
                        
                        # Revocar tokens asociados
                        for token in session.tokens:
                            await self.revoke_token(token.token)
                        
                        terminated += 1
            
            # Registrar auditoría
            await self._audit_auth_activity({
                "event": "sessions_terminated",
                "user_id": user_id,
                "session_id": session_id,
                "terminated_count": terminated,
                "timestamp": datetime.now()
            })
            
            return {
                "user_id": user_id,
                "terminated_sessions": terminated,
                "message": f"Terminated {terminated} session(s)"
            }
            
        elif action == "suspend":
            # Suspender sesión
            session_id = session_data.get("session_id") if session_data else None
            
            if not session_id:
                raise ValidationError("session_id required for suspend action")
            
            session = self.active_sessions.get(session_id)
            if not session or session.user_id != user_id:
                raise AuthenticationError("Session not found or access denied")
            
            session.status = SessionStatus.SUSPENDED
            
            # Registrar auditoría
            await self._audit_auth_activity({
                "event": "session_suspended",
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now()
            })
            
            return {
                "user_id": user_id,
                "session_id": session_id,
                "status": "suspended"
            }
            
        elif action == "info":
            # Obtener información de sesión
            session_id = session_data.get("session_id") if session_data else None
            
            if session_id:
                session = self.active_sessions.get(session_id)
                if not session or session.user_id != user_id:
                    raise AuthenticationError("Session not found or access denied")
                
                sessions = [session]
            else:
                sessions = [
                    session for session in self.active_sessions.values()
                    if session.user_id == user_id
                ]
            
            return {
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "auth_method": s.auth_method.value,
                        "client_ip": s.client_ip,
                        "user_agent": s.user_agent,
                        "issued_at": s.issued_at,
                        "last_activity": s.last_activity,
                        "expires_at": s.expires_at,
                        "status": s.status.value,
                        "token_count": len(s.tokens),
                        "metadata": s.metadata
                    }
                    for s in sessions
                ]
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")
    
    async def audit_auth_activity(self, activity_data: Dict[str, Any]) -> bool:
        """
        Registra actividad de autenticación para auditoría.
        
        Args:
            activity_data: Datos de la actividad
            
        Returns:
            bool: True si se registró exitosamente
        """
        return await self._audit_auth_activity(activity_data)
    
    # ============================================================================
    # MÉTODOS PRIVADOS DE IMPLEMENTACIÓN
    # ============================================================================
    
    def _load_auth_config(self) -> Dict[str, Any]:
        """Carga configuración de autenticación."""
        try:
            # Obtener configuración del sistema
            system_config = self.config_manager.get_config("system", {})
            auth_config = system_config.get("security", {}).get("authentication", {})
            
            # Configuración por defecto
            default_config = {
                "enabled": True,
                "methods": ["api_key", "jwt"],
                "jwt": {
                    "secret_key": "default-secret-change-in-production",
                    "algorithm": "HS256",
                    "access_token_expiry": 3600,  # 1 hora
                    "refresh_token_expiry": 86400 * 7,  # 7 días
                    "refresh_token_rotation": True
                },
                "api_key": {
                    "header": "X-API-Key",
                    "rotation_days": 90
                },
                "rate_limiting": {
                    "max_attempts": 5,
                    "window_minutes": 15,
                    "lockout_minutes": 30
                },
                "session": {
                    "max_sessions_per_user": 10,
                    "session_timeout": 28800,  # 8 horas
                    "inactivity_timeout": 1800  # 30 minutos
                }
            }
            
            # Combinar configuraciones
            merged_config = self._deep_merge(default_config, auth_config)
            
            # Validar configuración crítica
            if merged_config["jwt"]["secret_key"] == default_config["jwt"]["secret_key"]:
                logger.warning("Using default JWT secret key. Change in production!")
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load auth config: {e}")
            raise BrainException(f"Authentication configuration error: {e}")
    
    def _initialize_auth_methods(self) -> Dict[AuthMethod, callable]:
        """Inicializa los métodos de autenticación."""
        methods = {}
        
        # JWT Authentication
        methods[AuthMethod.JWT] = self._authenticate_jwt
        
        # API Key Authentication
        methods[AuthMethod.API_KEY] = self._authenticate_api_key
        
        # OAuth2 (placeholder - requeriría integración con proveedores)
        methods[AuthMethod.OAUTH2] = self._authenticate_oauth2
        
        # LDAP (placeholder - requeriría servidor LDAP)
        methods[AuthMethod.LDAP] = self._authenticate_ldap
        
        # Basic Auth
        methods[AuthMethod.BASIC] = self._authenticate_basic
        
        return methods
    
    async def _authenticate_jwt(self, auth_request: AuthRequest) -> Dict[str, Any]:
        """Autenticación mediante JWT (usuario/contraseña)."""
        if not auth_request.username or not auth_request.password:
            raise AuthenticationError("Username and password required for JWT auth")
        
        # Buscar usuario (en producción, esto vendría de una base de datos)
        user = await self._get_user_by_username(auth_request.username)
        
        if not user or not user.is_active:
            raise AuthenticationError("Invalid credentials or inactive user")
        
        # Verificar contraseña
        if not await self._verify_password(auth_request.password, user.password_hash):
            raise AuthenticationError("Invalid credentials")
        
        # Verificar roles y permisos
        if not self._has_required_roles(user.roles, auth_request.scopes):
            raise AuthorizationError("Insufficient privileges")
        
        # Actualizar último login
        user.last_login = datetime.now()
        await self._update_user(user)
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "email": user.email
        }
    
    async def _authenticate_api_key(self, auth_request: AuthRequest) -> Dict[str, Any]:
        """Autenticación mediante API Key."""
        if not auth_request.api_key:
            raise AuthenticationError("API key required")
        
        # Buscar API Key (en producción, esto vendría de una base de datos)
        api_key_info = await self._validate_api_key(auth_request.api_key)
        
        if not api_key_info:
            raise AuthenticationError("Invalid API key")
        
        # Verificar expiración
        if api_key_info.get("expires_at") and datetime.now() > api_key_info["expires_at"]:
            raise AuthenticationError("API key expired")
        
        # Verificar alcances
        key_scopes = api_key_info.get("scopes", [])
        if auth_request.scopes and not all(scope in key_scopes for scope in auth_request.scopes):
            raise AuthorizationError("API key does not have required scopes")
        
        return {
            "user_id": api_key_info["user_id"],
            "roles": [UserRole.SYSTEM],  # API keys generalmente tienen rol SYSTEM
            "scopes": key_scopes
        }
    
    async def _authenticate_oauth2(self, auth_request: AuthRequest) -> Dict[str, Any]:
        """Autenticación mediante OAuth2."""
        # Implementación placeholder
        # En producción, esto integraría con proveedores OAuth2 como Google, GitHub, etc.
        raise AuthenticationError("OAuth2 authentication not yet implemented")
    
    async def _authenticate_ldap(self, auth_request: AuthRequest) -> Dict[str, Any]:
        """Autenticación mediante LDAP."""
        # Implementación placeholder
        # En producción, esto integraría con servidores LDAP/Active Directory
        raise AuthenticationError("LDAP authentication not yet implemented")
    
    async def _authenticate_basic(self, auth_request: AuthRequest) -> Dict[str, Any]:
        """Autenticación HTTP Basic."""
        # Basic Auth es similar a JWT pero sin token inicial
        return await self._authenticate_jwt(auth_request)
    
    async def _create_session(self,
                             user_id: str,
                             auth_method: AuthMethod,
                             client_ip: str,
                             user_agent: Optional[str] = None,
                             tokens: Optional[List[AuthToken]] = None) -> UserSession:
        """Crea una nueva sesión de usuario."""
        # Verificar límite de sesiones
        active_sessions = [
            s for s in self.active_sessions.values()
            if s.user_id == user_id and s.is_active()
        ]
        
        max_sessions = self.config["session"]["max_sessions_per_user"]
        if len(active_sessions) >= max_sessions:
            # Terminar la sesión más antigua
            oldest_session = min(active_sessions, key=lambda s: s.issued_at)
            oldest_session.status = SessionStatus.REVOKED
            logger.info(f"Terminated oldest session for user {user_id} due to session limit")
        
        # Crear nueva sesión
        session_timeout = self.config["session"]["session_timeout"]
        session = UserSession(
            user_id=user_id,
            auth_method=auth_method,
            client_ip=client_ip,
            user_agent=user_agent,
            expires_at=datetime.now() + timedelta(seconds=session_timeout),
            tokens=tokens or []
        )
        
        # Almacenar sesión
        self.active_sessions[session.session_id] = session
        
        # Programar limpieza de sesión expirada
        asyncio.create_task(self._schedule_session_cleanup(session))
        
        logger.debug(f"Created session {session.session_id} for user {user_id}")
        return session
    
    async def _get_user_by_username(self, username: str) -> Optional[UserCredentials]:
        """Obtiene usuario por nombre de usuario."""
        # En producción, esto consultaría una base de datos
        # Aquí implementamos un usuario de ejemplo para desarrollo
        if username == "admin":
            return UserCredentials(
                username="admin",
                password_hash=bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
                user_id="admin-001",
                roles=[UserRole.ADMIN],
                email="admin@projectbrain.dev",
                full_name="System Administrator"
            )
        elif username == "developer":
            return UserCredentials(
                username="developer",
                password_hash=bcrypt.hashpw("dev123".encode(), bcrypt.gensalt()).decode(),
                user_id="dev-001",
                roles=[UserRole.DEVELOPER],
                email="dev@projectbrain.dev",
                full_name="Project Developer"
            )
        elif username == "viewer":
            return UserCredentials(
                username="viewer",
                password_hash=bcrypt.hashpw("view123".encode(), bcrypt.gensalt()).decode(),
                user_id="view-001",
                roles=[UserRole.VIEWER],
                email="viewer@projectbrain.dev",
                full_name="Project Viewer"
            )
        
        return None
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Valida una API Key."""
        # En producción, esto consultaría una base de datos de API Keys
        # Aquí implementamos keys de ejemplo para desarrollo
        
        # Formato: pb_{user_id}_{random_hash}
        if api_key.startswith("pb_"):
            parts = api_key.split("_")
            if len(parts) >= 2:
                user_id = parts[1]
                
                # Simular validación
                return {
                    "user_id": user_id,
                    "scopes": ["read", "write", "analyze"],
                    "created_at": datetime.now() - timedelta(days=30),
                    "expires_at": datetime.now() + timedelta(days=60)  # Expira en 60 días
                }
        
        return None
    
    async def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica una contraseña."""
        try:
            return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def _has_required_roles(self, user_roles: List[UserRole], required_scopes: List[str]) -> bool:
        """Verifica si el usuario tiene los roles necesarios."""
        if not required_scopes:
            return True
        
        # Mapear scopes a roles (esto podría ser configurable)
        scope_to_role = {
            "admin": [UserRole.ADMIN],
            "write": [UserRole.ADMIN, UserRole.DEVELOPER],
            "read": [UserRole.ADMIN, UserRole.DEVELOPER, UserRole.VIEWER],
            "analyze": [UserRole.ADMIN, UserRole.DEVELOPER],
            "system": [UserRole.ADMIN, UserRole.SYSTEM]
        }
        
        for scope in required_scopes:
            allowed_roles = scope_to_role.get(scope, [])
            if not any(role in allowed_roles for role in user_roles):
                return False
        
        return True
    
    def _get_user_roles(self, user_id: str) -> List[UserRole]:
        """Obtiene roles de un usuario."""
        # En producción, esto consultaría una base de datos
        # Aquí devolvemos roles de ejemplo basados en el user_id
        if "admin" in user_id:
            return [UserRole.ADMIN]
        elif "dev" in user_id:
            return [UserRole.DEVELOPER]
        else:
            return [UserRole.VIEWER]
    
    async def _update_user(self, user: UserCredentials) -> None:
        """Actualiza información de usuario."""
        # En producción, esto actualizaría una base de datos
        # Aquí solo registramos la actualización
        user.updated_at = datetime.now()
        logger.debug(f"Updated user {user.username}")
    
    async def _get_session_by_user(self, user_id: str) -> Optional[UserSession]:
        """Obtiene la sesión activa más reciente de un usuario."""
        active_sessions = [
            s for s in self.active_sessions.values()
            if s.user_id == user_id and s.is_active()
        ]
        
        if active_sessions:
            # Devolver la sesión más reciente
            return max(active_sessions, key=lambda s: s.last_activity)
        
        return None
    
    def _check_rate_limit(self, username: Optional[str], client_ip: str) -> None:
        """Verifica límites de tasa para autenticación."""
        if not username:
            username = client_ip
        
        key = f"{username}:{client_ip}"
        now = time.time()
        window = self.config["rate_limiting"]["window_minutes"] * 60
        
        # Obtener registro de intentos
        attempts = self.failed_attempts.get(key, {"count": 0, "first_attempt": now})
        
        # Verificar si está en lockout
        if attempts.get("lockout_until") and now < attempts["lockout_until"]:
            lockout_minutes = (attempts["lockout_until"] - now) / 60
            raise AuthenticationError(
                f"Too many failed attempts. Try again in {lockout_minutes:.1f} minutes"
            )
        
        # Verificar ventana de tiempo
        if now - attempts["first_attempt"] > window:
            # Reiniciar contador si la ventana ha expirado
            attempts = {"count": 0, "first_attempt": now}
        
        # Verificar límite máximo
        max_attempts = self.config["rate_limiting"]["max_attempts"]
        if attempts["count"] >= max_attempts:
            # Activar lockout
            lockout_time = self.config["rate_limiting"]["lockout_minutes"] * 60
            attempts["lockout_until"] = now + lockout_time
            
            self.failed_attempts[key] = attempts
            
            raise AuthenticationError(
                f"Too many failed attempts. Account locked for "
                f"{self.config['rate_limiting']['lockout_minutes']} minutes"
            )
    
    async def _record_failed_attempt(self, username: Optional[str], client_ip: str, error: str) -> None:
        """Registra un intento fallido de autenticación."""
        if not username:
            username = client_ip
        
        key = f"{username}:{client_ip}"
        now = time.time()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {
                "count": 0,
                "first_attempt": now,
                "last_attempt": now,
                "errors": []
            }
        
        attempts = self.failed_attempts[key]
        attempts["count"] += 1
        attempts["last_attempt"] = now
        attempts["errors"].append({
            "timestamp": datetime.now(),
            "error": error
        })
        
        # Registrar auditoría
        await self._audit_auth_activity({
            "event": "failed_authentication_attempt",
            "username": username,
            "client_ip": client_ip,
            "attempt_count": attempts["count"],
            "error": error,
            "timestamp": datetime.now()
        })
    
    def _clear_failed_attempts(self, username: Optional[str], client_ip: str) -> None:
        """Limpia intentos fallidos después de autenticación exitosa."""
        if not username:
            username = client_ip
        
        key = f"{username}:{client_ip}"
        if key in self.failed_attempts:
            del self.failed_attempts[key]
    
    async def _audit_auth_activity(self, activity_data: Dict[str, Any]) -> bool:
        """
        Registra actividad de autenticación para auditoría.
        
        Args:
            activity_data: Datos de la actividad
            
        Returns:
            bool: True si se registró exitosamente
        """
        try:
            # Añadir metadata
            activity_data["_id"] = str(uuid.uuid4())
            activity_data["_timestamp"] = datetime.now()
            
            # En producción, esto escribiría en una base de datos de auditoría
            # Aquí solo registramos en el log
            log_entry = {
                "type": "auth_audit",
                "data": activity_data
            }
            
            logger.info(f"Auth audit: {activity_data.get('event')} - {activity_data.get('user_id', 'unknown')}")
            
            # Si hay storage adapter, guardar allí también
            if self.storage and hasattr(self.storage, 'store_audit_log'):
                await self.storage.store_audit_log(activity_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to audit auth activity: {e}")
            return False
    
    async def _schedule_session_cleanup(self, session: UserSession) -> None:
        """Programa limpieza automática de sesión expirada."""
        try:
            # Calcular tiempo hasta expiración
            now = datetime.now()
            if session.expires_at > now:
                wait_seconds = (session.expires_at - now).total_seconds()
                
                # Esperar hasta expiración
                await asyncio.sleep(wait_seconds)
                
                # Marcar como expirada
                if session.session_id in self.active_sessions:
                    session.status = SessionStatus.EXPIRED
                    logger.debug(f"Session {session.session_id} expired automatically")
            
        except asyncio.CancelledError:
            # Tarea cancelada
            pass
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    # ============================================================================
    # MÉTODOS DE UTILIDAD
    # ============================================================================
    
    @staticmethod
    def _deep_merge(source: Dict, destination: Dict) -> Dict:
        """Combina dos diccionarios recursivamente."""
        for key, value in source.items():
            if key in destination:
                if isinstance(value, dict) and isinstance(destination[key], dict):
                    Authentication._deep_merge(value, destination[key])
                else:
                    destination[key] = value
            else:
                destination[key] = value
        return destination
    
    def get_active_session_count(self) -> int:
        """Obtiene el número de sesiones activas."""
        return len([s for s in self.active_sessions.values() if s.is_active()])
    
    def get_revoked_token_count(self) -> int:
        """Obtiene el número de tokens revocados."""
        return len(self.revoked_tokens)
    
    def cleanup_expired_sessions(self) -> int:
        """Limpia sesiones expiradas y devuelve el número eliminado."""
        expired_count = 0
        now = datetime.now()
        
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if not session.is_active() or session.expires_at < now:
                sessions_to_remove.append(session_id)
                expired_count += 1
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    # ============================================================================
    # MÉTODOS DE CONFIGURACIÓN EN TIEMPO DE EJECUCIÓN
    # ============================================================================
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Actualiza configuración en tiempo de ejecución.
        
        Args:
            new_config: Nueva configuración
            
        Returns:
            bool: True si se actualizó exitosamente
        """
        try:
            self.config = self._deep_merge(new_config, self.config.copy())
            logger.info("Authentication configuration updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update auth config: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de configuración."""
        return {
            "auth_methods_enabled": list(self.auth_methods.keys()),
            "active_sessions": self.get_active_session_count(),
            "revoked_tokens": self.get_revoked_token_count(),
            "rate_limiting": {
                "max_attempts": self.config["rate_limiting"]["max_attempts"],
                "lockout_minutes": self.config["rate_limiting"]["lockout_minutes"]
            },
            "jwt": {
                "algorithm": self.config["jwt"]["algorithm"],
                "access_token_expiry": self.config["jwt"]["access_token_expiry"],
                "refresh_token_rotation": self.config["jwt"]["refresh_token_rotation"]
            }
        }


# ============================================================================
# FACTORY PARA CREACIÓN DE INSTANCIAS
# ============================================================================

def create_authentication_system(config_manager: Any, 
                                storage_adapter: Optional[Any] = None) -> Authentication:
    """
    Factory para crear instancia del sistema de autenticación.
    
    Args:
        config_manager: Gestor de configuración
        storage_adapter: Adaptador de almacenamiento (opcional)
        
    Returns:
        Authentication: Instancia configurada
    """
    return Authentication(config_manager, storage_adapter)