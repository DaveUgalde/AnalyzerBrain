"""
RateLimiter - Sistema de limitación de tasa para APIs.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict
from fastapi import Request, Response, HTTPException
import redis.asyncio as redis

from ..core.exceptions import BrainException

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Estrategias de limitación de tasa."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitRule:
    """Regla de limitación de tasa."""
    key: str  # Clave para identificar el límite (ej: "ip:192.168.1.1", "user:123")
    limit: int  # Número máximo de requests
    window_seconds: int  # Ventana de tiempo en segundos
    strategy: RateLimitStrategy = RateLimitStrategy.FIXED_WINDOW
    burst_limit: Optional[int] = None  # Límite de ráfaga (para token bucket)
    
    def get_hash(self) -> str:
        """Obtiene hash único para la regla."""
        rule_str = f"{self.key}:{self.limit}:{self.window_seconds}:{self.strategy.value}"
        return hashlib.md5(rule_str.encode()).hexdigest()

@dataclass
class RateLimitResult:
    """Resultado de verificación de límite de tasa."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int  # Timestamp cuando se resetea el contador
    retry_after: Optional[int] = None  # Segundos para reintentar (si no permitido)
    current_count: int = 0

class RateLimiter:
    """
    Sistema completo de limitación de tasa.
    
    Características:
    1. Múltiples estrategias (fixed window, sliding window, token bucket)
    2. Límites por IP, usuario, endpoint, etc.
    3. Soporte para Redis para distribución
    4. Headers de rate limiting estándar
    5. Configuración dinámica de reglas
    """
    
    def __init__(self, config: Optional[Dict] = None, redis_client: Optional[redis.Redis] = None):
        """
        Inicializa el limitador de tasa.
        
        Args:
            config: Configuración del rate limiter (opcional)
            redis_client: Cliente Redis para almacenamiento distribuido (opcional)
        """
        self.config = config or {
            "enabled": True,
            "default_limit": 60,  # requests por minuto
            "default_window": 60,  # segundos
            "strategy": RateLimitStrategy.FIXED_WINDOW.value,
            "redis_prefix": "rate_limit:",
            "cleanup_interval": 300,  # segundos
            "enable_headers": True,
            "header_prefix": "X-RateLimit-",
        }
        
        self.redis_client = redis_client
        self.use_redis = redis_client is not None
        
        # Almacenamiento en memoria (si no hay Redis)
        self.counters: Dict[str, Dict] = defaultdict(dict)
        self.windows: Dict[str, List] = defaultdict(list)
        self.tokens: Dict[str, float] = defaultdict(float)  # Para token bucket
        
        # Reglas de rate limiting
        self.rules: List[RateLimitRule] = []
        self._initialize_default_rules()
        
        # Limpieza periódica
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info("RateLimiter inicializado (using %s)", 
                   "Redis" if self.use_redis else "memory")
    
    async def check_rate_limit(self, request: Request, call_next) -> Response:
        """
        Middleware de rate limiting para FastAPI.
        
        Args:
            request: Request HTTP
            call_next: Función para continuar el pipeline
            
        Returns:
            Response: Respuesta HTTP con headers de rate limiting
        """
        if not self.config["enabled"]:
            return await call_next(request)
        
        try:
            # Determinar clave de rate limiting
            limit_key = await self._get_limit_key(request)
            
            # Obtener reglas aplicables
            applicable_rules = self._get_applicable_rules(request, limit_key)
            
            # Verificar cada regla
            results = []
            for rule in applicable_rules:
                result = await self._check_rule(rule)
                results.append(result)
                
                if not result.allowed:
                    # Límite excedido
                    response = Response(
                        content=json.dumps({
                            "error": "Rate limit exceeded",
                            "message": f"Too many requests. Limit: {rule.limit} per {rule.window_seconds}s",
                            "retry_after": result.retry_after,
                        }),
                        status_code=429,
                        media_type="application/json",
                    )
                    
                    # Añadir headers
                    if self.config["enable_headers"]:
                        self._add_rate_limit_headers(response, results)
                    
                    return response
            
            # Todos los límites OK, procesar request
            response = await call_next(request)
            
            # Incrementar contadores para reglas exitosas
            for rule in applicable_rules:
                await self._increment_counter(rule)
            
            # Añadir headers de rate limiting
            if self.config["enable_headers"]:
                self._add_rate_limit_headers(response, results)
            
            return response
            
        except Exception as e:
            logger.error("Rate limiting error: %s", e, exc_info=True)
            # En caso de error, permitir la request (fail open)
            return await call_next(request)
    
    async def increment_counter(self, key: str, window_seconds: int = 60) -> Tuple[bool, int]:
        """
        Incrementa contador para una clave específica.
        
        Args:
            key: Clave del contador
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            Tuple (allowed, remaining)
        """
        rule = RateLimitRule(
            key=key,
            limit=self.config["default_limit"],
            window_seconds=window_seconds,
            strategy=RateLimitStrategy(self.config["strategy"])
        )
        
        result = await self._check_rule(rule)
        if result.allowed:
            await self._increment_counter(rule)
        
        return result.allowed, result.remaining
    
    async def reset_counters(self, key_prefix: Optional[str] = None) -> int:
        """
        Resetea contadores de rate limiting.
        
        Args:
            key_prefix: Prefijo de claves a resetear (None = todas)
            
        Returns:
            int: Número de contadores reseteados
        """
        if self.use_redis:
            return await self._reset_redis_counters(key_prefix)
        else:
            return self._reset_memory_counters(key_prefix)
    
    async def calculate_wait_time(self, key: str, window_seconds: int = 60) -> float:
        """
        Calcula tiempo de espera para una clave.
        
        Args:
            key: Clave del contador
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            float: Segundos para esperar
        """
        rule = RateLimitRule(
            key=key,
            limit=self.config["default_limit"],
            window_seconds=window_seconds,
            strategy=RateLimitStrategy(self.config["strategy"])
        )
        
        result = await self._check_rule(rule)
        if result.allowed:
            return 0.0
        
        return result.retry_after or (result.reset_time - time.time())
    
    async def handle_rate_limit_exceeded(self, request: Request) -> Response:
        """
        Maneja request que excede límite de tasa.
        
        Args:
            request: Request HTTP
            
        Returns:
            Response: Respuesta 429 con detalles
        """
        limit_key = await self._get_limit_key(request)
        applicable_rules = self._get_applicable_rules(request, limit_key)
        
        # Obtener el límite más restrictivo que fue excedido
        worst_result = None
        for rule in applicable_rules:
            result = await self._check_rule(rule)
            if not result.allowed and (not worst_result or result.retry_after > (worst_result.retry_after or 0)):
                worst_result = result
        
        if worst_result:
            retry_after = worst_result.retry_after or 1
            
            response = Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                    "limit": worst_result.limit,
                    "window": "minute" if worst_result.reset_time - time.time() <= 60 else "hour",
                }),
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                }
            )
            
            return response
        
        # No debería llegar aquí, pero por si acaso
        return Response(
            content=json.dumps({"error": "Rate limit exceeded"}),
            status_code=429,
            media_type="application/json"
        )
    
    async def optimize_rate_limiting(self) -> Dict[str, Any]:
        """
        Optimiza el sistema de rate limiting.
        
        Returns:
            Dict con métricas y optimizaciones aplicadas
        """
        metrics = await self.get_rate_limit_stats()
        
        optimizations = []
        
        # Ajustar límites basado en uso
        for rule in self.rules[:10]:  # Solo primeros 10
            usage_rate = metrics.get(rule.key, {}).get("usage_rate", 0)
            
            if usage_rate > 0.9:  # 90% de uso
                # Incrementar límite en 20%
                new_limit = int(rule.limit * 1.2)
                old_limit = rule.limit
                rule.limit = new_limit
                
                optimizations.append({
                    "rule": rule.key,
                    "action": "increase_limit",
                    "old_limit": old_limit,
                    "new_limit": new_limit,
                    "reason": f"High usage rate: {usage_rate:.1%}",
                })
            
            elif usage_rate < 0.1:  # 10% de uso
                # Reducir límite en 20%
                new_limit = max(10, int(rule.limit * 0.8))
                old_limit = rule.limit
                rule.limit = new_limit
                
                optimizations.append({
                    "rule": rule.key,
                    "action": "decrease_limit",
                    "old_limit": old_limit,
                    "new_limit": new_limit,
                    "reason": f"Low usage rate: {usage_rate:.1%}",
                })
        
        logger.info("Rate limiting optimized: %s optimizations applied", len(optimizations))
        
        return {
            "metrics": metrics,
            "optimizations": optimizations,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_rate_limit_stats(self, 
                                  key_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rate limiting.
        
        Args:
            key_filter: Filtro para claves (opcional)
            
        Returns:
            Dict con estadísticas detalladas
        """
        stats = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "rules": [],
            "top_keys": [],
        }
        
        # Estadísticas por regla
        for rule in self.rules:
            if key_filter and key_filter not in rule.key:
                continue
            
            rule_stats = await self._get_rule_stats(rule)
            stats["rules"].append(rule_stats)
            
            stats["total_requests"] += rule_stats.get("total_requests", 0)
            stats["rate_limited_requests"] += rule_stats.get("blocked_requests", 0)
        
        # Claves más activas (top 10)
        stats["top_keys"] = await self._get_top_keys(10, key_filter)
        
        # Tasa de bloqueo
        if stats["total_requests"] > 0:
            stats["block_rate"] = stats["rate_limited_requests"] / stats["total_requests"]
        else:
            stats["block_rate"] = 0.0
        
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    # Métodos de implementación
    
    async def _check_rule(self, rule: RateLimitRule) -> RateLimitResult:
        """
        Verifica una regla de rate limiting.
        
        Args:
            rule: Regla a verificar
            
        Returns:
            RateLimitResult con resultado
        """
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(rule)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(rule)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(rule)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._check_leaky_bucket(rule)
        else:
            raise BrainException(f"Unknown rate limit strategy: {rule.strategy}")
    
    async def _check_fixed_window(self, rule: RateLimitRule) -> RateLimitResult:
        """Implementa fixed window rate limiting."""
        current_time = time.time()
        window_start = int(current_time / rule.window_seconds) * rule.window_seconds
        
        if self.use_redis:
            counter_key = f"{self.config['redis_prefix']}{rule.get_hash()}:{window_start}"
            
            # Usar pipeline para atomicidad
            async with self.redis_client.pipeline() as pipe:
                await pipe.incr(counter_key)
                await pipe.expire(counter_key, rule.window_seconds)
                current_count = await pipe.execute()[0]
        else:
            counter_key = f"{rule.key}:{window_start}"
            
            # Incrementar en memoria
            self.counters[rule.key][window_start] = \
                self.counters[rule.key].get(window_start, 0) + 1
            current_count = self.counters[rule.key][window_start]
        
        remaining = max(0, rule.limit - current_count)
        reset_time = window_start + rule.window_seconds
        
        allowed = current_count <= rule.limit
        retry_after = None if allowed else reset_time - current_time
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            current_count=current_count,
        )
    
    async def _check_sliding_window(self, rule: RateLimitRule) -> RateLimitResult:
        """Implementa sliding window rate limiting."""
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        if self.use_redis:
            # Usar sorted set en Redis
            zset_key = f"{self.config['redis_prefix']}{rule.get_hash()}:timestamps"
            
            # Añadir timestamp actual
            member = f"{current_time}:{secrets.token_hex(4)}"
            await self.redis_client.zadd(zset_key, {member: current_time})
            
            # Remover timestamps fuera de la ventana
            await self.redis_client.zremrangebyscore(zset_key, 0, window_start)
            
            # Contar timestamps en ventana
            current_count = await self.redis_client.zcount(zset_key, window_start, current_time)
            
            # Expirar después de la ventana
            await self.redis_client.expire(zset_key, rule.window_seconds)
        else:
            # En memoria
            if rule.key not in self.windows:
                self.windows[rule.key] = []
            
            # Añadir timestamp actual
            self.windows[rule.key].append(current_time)
            
            # Filtrar timestamps fuera de la ventana
            self.windows[rule.key] = [
                ts for ts in self.windows[rule.key]
                if ts > window_start
            ]
            
            current_count = len(self.windows[rule.key])
        
        remaining = max(0, rule.limit - current_count)
        reset_time = current_time + rule.window_seconds
        
        allowed = current_count <= rule.limit
        retry_after = None if allowed else 1  # Simplificado para sliding window
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            current_count=current_count,
        )
    
    async def _check_token_bucket(self, rule: RateLimitRule) -> RateLimitResult:
        """Implementa token bucket rate limiting."""
        current_time = time.time()
        
        if self.use_redis:
            bucket_key = f"{self.config['redis_prefix']}{rule.get_hash()}:tokens"
            
            # Usar Lua script para atomicidad
            lua_script = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local burst = tonumber(ARGV[4])
            
            local last_update = redis.call('hget', key, 'last_update') or now
            local tokens = tonumber(redis.call('hget', key, 'tokens') or limit)
            
            -- Calcular tokens repuestos
            local elapsed = now - last_update
            local refill = (elapsed / window) * limit
            
            tokens = math.min(limit, tokens + refill)
            
            if burst and burst > limit then
                tokens = math.min(burst, tokens)
            end
            
            -- Consumir token si hay disponible
            local allowed = tokens >= 1
            if allowed then
                tokens = tokens - 1
                redis.call('hset', key, 'tokens', tokens)
                redis.call('hset', key, 'last_update', now)
                redis.call('expire', key, window * 2)
            end
            
            local remaining = math.floor(tokens)
            local reset = now + ((1 - (tokens % 1)) * window)
            
            return {allowed and 1 or 0, remaining, reset}
            """
            
            burst_limit = rule.burst_limit or rule.limit
            result = await self.redis_client.eval(
                lua_script, 1, bucket_key, 
                rule.limit, rule.window_seconds, current_time, burst_limit
            )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            reset_time = float(result[2])
        else:
            # En memoria
            if rule.key not in self.tokens:
                self.tokens[rule.key] = rule.limit
                self.counters[rule.key]["last_update"] = current_time
            
            last_update = self.counters[rule.key].get("last_update", current_time)
            tokens = self.tokens[rule.key]
            
            # Calcular tokens repuestos
            elapsed = current_time - last_update
            refill = (elapsed / rule.window_seconds) * rule.limit
            tokens = min(rule.limit, tokens + refill)
            
            # Aplicar burst limit
            burst_limit = rule.burst_limit or rule.limit
            if burst_limit > rule.limit:
                tokens = min(burst_limit, tokens)
            
            # Consumir token
            allowed = tokens >= 1
            if allowed:
                tokens -= 1
                self.tokens[rule.key] = tokens
                self.counters[rule.key]["last_update"] = current_time
            
            remaining = int(tokens)
            reset_time = current_time + ((1 - (tokens % 1)) * rule.window_seconds)
        
        retry_after = None if allowed else reset_time - current_time
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            current_count=rule.limit - remaining,
        )
    
    async def _check_leaky_bucket(self, rule: RateLimitRule) -> RateLimitResult:
        """Implementa leaky bucket rate limiting."""
        # Similar a token bucket pero diferente semántica
        # Para simplificar, usamos una implementación básica
        return await self._check_token_bucket(rule)
    
    async def _increment_counter(self, rule: RateLimitRule) -> None:
        """Incrementa contador para una regla."""
        # Esto ya se hace en _check_rule, pero lo mantenemos por compatibilidad
        pass
    
    async def _get_limit_key(self, request: Request) -> str:
        """
        Determina la clave de rate limiting para una request.
        
        Args:
            request: Request HTTP
            
        Returns:
            str: Clave para rate limiting
        """
        # Por defecto, usar IP del cliente
        client_ip = request.client.host if request.client else "unknown"
        
        # También considerar usuario autenticado si existe
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.user_id
        
        # Considerar endpoint
        endpoint = request.url.path
        
        # Construir clave compuesta
        parts = []
        
        if user_id:
            parts.append(f"user:{user_id}")
        else:
            parts.append(f"ip:{client_ip}")
        
        parts.append(f"path:{endpoint}")
        parts.append(f"method:{request.method}")
        
        return ":".join(parts)
    
    def _get_applicable_rules(self, request: Request, limit_key: str) -> List[RateLimitRule]:
        """
        Obtiene reglas aplicables para una request.
        
        Args:
            request: Request HTTP
            limit_key: Clave de rate limiting
            
        Returns:
            List[RateLimitRule]: Reglas aplicables
        """
        applicable_rules = []
        
        # Regla por defecto basada en la clave
        default_rule = RateLimitRule(
            key=limit_key,
            limit=self.config["default_limit"],
            window_seconds=self.config["default_window"],
            strategy=RateLimitStrategy(self.config["strategy"])
        )
        applicable_rules.append(default_rule)
        
        # Añadir reglas específicas si existen
        for rule in self.rules:
            if self._rule_matches(rule, request, limit_key):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_matches(self, rule: RateLimitRule, request: Request, limit_key: str) -> bool:
        """Verifica si una regla coincide con la request."""
        # Implementación simple: verificar si la clave de la regla está en limit_key
        return rule.key in limit_key
    
    def _add_rate_limit_headers(self, response: Response, results: List[RateLimitResult]) -> None:
        """Añade headers de rate limiting a la respuesta."""
        if not results:
            return
        
        # Usar el resultado más restrictivo
        worst_result = min(results, key=lambda r: r.remaining)
        
        prefix = self.config.get("header_prefix", "X-RateLimit-")
        
        response.headers[f"{prefix}Limit"] = str(worst_result.limit)
        response.headers[f"{prefix}Remaining"] = str(worst_result.remaining)
        response.headers[f"{prefix}Reset"] = str(int(worst_result.reset_time))
        
        if worst_result.current_count > worst_result.limit:
            response.headers["Retry-After"] = str(int(worst_result.retry_after or 1))
    
    def _initialize_default_rules(self):
        """Inicializa reglas por defecto."""
        # Regla para endpoints de autenticación (más estricta)
        auth_rule = RateLimitRule(
            key="path:/login",
            limit=10,  # 10 intentos por minuto
            window_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        self.rules.append(auth_rule)
        
        # Regla para análisis pesados
        analysis_rule = RateLimitRule(
            key="path:/analyze",
            limit=5,  # 5 análisis por hora
            window_seconds=3600,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        self.rules.append(analysis_rule)
        
        # Regla para consultas (más permisiva)
        query_rule = RateLimitRule(
            key="path:/query",
            limit=120,  # 120 consultas por minuto
            window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        self.rules.append(query_rule)
    
    async def _reset_redis_counters(self, key_prefix: Optional[str] = None) -> int:
        """Resetea contadores en Redis."""
        if not self.redis_client:
            return 0
        
        pattern = f"{self.config['redis_prefix']}*"
        if key_prefix:
            pattern = f"{self.config['redis_prefix']}{key_prefix}*"
        
        keys = await self.redis_client.keys(pattern)
        
        if keys:
            await self.redis_client.delete(*keys)
        
        return len(keys)
    
    def _reset_memory_counters(self, key_prefix: Optional[str] = None) -> int:
        """Resetea contadores en memoria."""
        count = 0
        
        if key_prefix:
            # Resetear solo claves con prefijo
            keys_to_delete = []
            for key in list(self.counters.keys()):
                if key.startswith(key_prefix):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.counters[key]
                count += 1
            
            for key in list(self.windows.keys()):
                if key.startswith(key_prefix):
                    del self.windows[key]
            
            for key in list(self.tokens.keys()):
                if key.startswith(key_prefix):
                    del self.tokens[key]
        else:
            # Resetear todo
            count = len(self.counters)
            self.counters.clear()
            self.windows.clear()
            self.tokens.clear()
        
        return count
    
    async def _get_rule_stats(self, rule: RateLimitRule) -> Dict[str, Any]:
        """Obtiene estadísticas para una regla."""
        current_time = time.time()
        
        if self.use_redis:
            # Implementar para Redis
            pass
        
        # En memoria, estimar basado en estructura actual
        total_requests = 0
        blocked_requests = 0
        
        if rule.key in self.counters:
            for window_start, count in self.counters[rule.key].items():
                total_requests += count
                if count > rule.limit:
                    blocked_requests += count - rule.limit
        
        return {
            "key": rule.key,
            "limit": rule.limit,
            "window_seconds": rule.window_seconds,
            "strategy": rule.strategy.value,
            "total_requests": total_requests,
            "blocked_requests": blocked_requests,
            "usage_rate": min(1.0, total_requests / (rule.limit * 10)) if rule.limit > 0 else 0.0,
        }
    
    async def _get_top_keys(self, limit: int = 10, key_filter: Optional[str] = None) -> List[Dict]:
        """Obtiene las claves más activas."""
        # Implementación simplificada
        top_keys = []
        
        for key in list(self.counters.keys())[:limit]:
            if key_filter and key_filter not in key:
                continue
            
            total_requests = sum(self.counters[key].values())
            top_keys.append({
                "key": key,
                "total_requests": total_requests,
            })
        
        # Ordenar por total de requests
        top_keys.sort(key=lambda x: x["total_requests"], reverse=True)
        
        return top_keys[:limit]
    
    def _start_cleanup_task(self):
        """Inicia tarea de limpieza periódica."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config["cleanup_interval"])
                    await self._cleanup_old_data()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Cleanup error: %s", e)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_data(self):
        """Limpia datos viejos de rate limiting."""
        if self.use_redis:
            # Redis maneja expiración automáticamente
            return
        
        current_time = time.time()
        cleanup_cutoff = current_time - 3600  # 1 hora
        
        # Limpiar contadores de fixed window
        for key in list(self.counters.keys()):
            windows_to_delete = []
            for window_start in self.counters[key]:
                if window_start < cleanup_cutoff:
                    windows_to_delete.append(window_start)
            
            for window_start in windows_to_delete:
                del self.counters[key][window_start]
            
            # Si no quedan ventanas, eliminar la clave
            if not self.counters[key]:
                del self.counters[key]
        
        # Limpiar timestamps de sliding window
        for key in list(self.windows.keys()):
            self.windows[key] = [
                ts for ts in self.windows[key]
                if ts > cleanup_cutoff
            ]
            
            if not self.windows[key]:
                del self.windows[key]
        
        logger.debug("Rate limiting data cleaned up")

# Helper para JSON (necesario para algunas funciones)
import json