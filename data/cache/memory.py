from collections import OrderedDict
import threading
from typing import Any, Optional

from .models import CacheEntry


class MemoryCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            entry = self.store.get(key)
            if not entry:
                self.stats["misses"] += 1
                return None

            if entry.expired:
                del self.store[key]
                self.stats["evictions"] += 1
                return None

            self.store.move_to_end(key)
            self.stats["hits"] += 1
            return entry.value

    def set(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            if len(self.store) >= self.max_size:
                self.store.popitem(last=False)
                self.stats["evictions"] += 1

            self.store[key] = entry
            self.store.move_to_end(key)
        return True
