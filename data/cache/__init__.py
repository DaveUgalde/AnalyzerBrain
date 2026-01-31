from pathlib import Path
from typing import Any, Optional, Dict

from .models import CacheEntry
from .memory import MemoryCache
from .redis_cache import RedisCache
from .disk import DiskCache


class CacheManager:
    """Gestor de caché multi-nivel (L1/L2/L3)."""

    def __init__(self, cache_path: Path, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "memory": {"max_size": 1000, "ttl": 300},
            "redis": {"enabled": False},
            "disk": {"max_size": 10000, "ttl": 86400},
        }

        self.memory = MemoryCache(self.config["memory"]["max_size"])
        self.redis = RedisCache(self.config["redis"])
        self.disk = DiskCache(cache_path / "l3_disk", self.config["disk"]["max_size"])

    def _key(self, key: str, namespace: str) -> str:
        return f"{namespace}:{key}"

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        k = self._key(key, namespace)

        value = self.memory.get(k)
        if value is not None:
            return value

        value = self.redis.get(k)
        if value is not None:
            self.memory.set(k, CacheEntry.create(value, self.config["memory"]["ttl"]))
            return value

        value = self.disk.get(k)
        if value is not None:
            self.memory.set(k, CacheEntry.create(value, self.config["memory"]["ttl"]))
            return value

        return None

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        level: str = "all",
    ) -> bool:
        ttl = ttl or self.config["memory"]["ttl"]
        entry = CacheEntry.create(value, ttl)
        k = self._key(key, namespace)

        ok = True
        if level in ("memory", "all"):
            ok &= self.memory.set(k, entry)
        if level in ("redis", "all"):
            ok &= self.redis.set(k, entry)
        if level in ("disk", "all"):
            ok &= self.disk.set(k, entry)

        return ok
