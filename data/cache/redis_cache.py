import pickle
from typing import Any, Optional
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .models import CacheEntry


class RedisCache:
    def __init__(self, config: dict):
        self.client = None
        if config.get("enabled") and REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=config["host"],
                    port=config["port"],
                    password=config.get("password"),
                    db=config.get("db", 0),
                    decode_responses=False,
                )
                self.client.ping()
            except Exception:
                self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None

        try:
            raw = self.client.get(key)
            if not raw:
                return None

            entry: CacheEntry = pickle.loads(raw)
            return None if entry.expired else entry.value
        except Exception:
            return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        if not self.client:
            return False
        try:
            self.client.setex(key, entry.ttl, pickle.dumps(entry))
            return True
        except Exception:
            return False
