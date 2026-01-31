from dataclasses import dataclass
import time
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    ttl: int
    created_at: float

    @property
    def expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    @classmethod
    def create(cls, value: Any, ttl: int) -> "CacheEntry":
        return cls(value=value, ttl=ttl, created_at=time.time())
