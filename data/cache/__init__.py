# data/cache/__init__.py
"""
Sistema de caché multi-nivel para Project Brain.
"""

from __future__ import annotations

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ...core.exceptions import CacheException


class CacheManager:
    """Gestor de caché multi-nivel."""

    def __init__(self, cache_path: Path, config: Optional[Dict[str, Any]] = None):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.config = config or {
            "memory": {"max_size": 1000, "ttl": 300},
            "redis": {
                "enabled": False,
                "host": "localhost",
                "port": 6379,
                "password": None,
                "db": 0,
                "ttl": 3600,
            },
            "disk": {"max_size": 10000, "ttl": 86400},
        }

        self._init_directories()
        self._init_memory_cache()
        self._init_redis()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_directories(self) -> None:
        for name in ("l1_memory", "l2_redis_backup", "l3_disk", "precomputed", "temp"):
            (self.cache_path / name).mkdir(parents=True, exist_ok=True)

    def _init_memory_cache(self) -> None:
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.memory_lock = threading.RLock()
        self.memory_stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _init_redis(self) -> None:
        self.redis_client = None

        if self.config["redis"]["enabled"] and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config["redis"]["host"],
                    port=self.config["redis"]["port"],
                    password=self.config["redis"]["password"],
                    db=self.config["redis"]["db"],
                    decode_responses=False,
                )
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        cache_key = self._make_cache_key(key, namespace)

        # L1
        with self.memory_lock:
            entry = self.memory_cache.get(cache_key)
            if entry:
                if not self._is_expired(entry):
                    self.memory_cache.move_to_end(cache_key)
                    self.memory_stats["hits"] += 1
                    return entry["value"]
                del self.memory_cache[cache_key]
                self.memory_stats["evictions"] += 1

        self.memory_stats["misses"] += 1

        # L2 Redis
        if self.redis_client:
            try:
                raw = self.redis_client.get(cache_key)
                if raw:
                    entry = pickle.loads(raw)
                    if not self._is_expired(entry):
                        self._set_memory(cache_key, entry["value"], entry["ttl"])
                        return entry["value"]
            except Exception:
                pass

        # L3 Disk
        entry = self._get_from_disk(cache_key)
        if entry and not self._is_expired(entry):
            self._set_memory(cache_key, entry["value"], entry["ttl"])
            if self.redis_client:
                self._set_redis(cache_key, entry["value"], entry["ttl"])
            return entry["value"]

        return None

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        level: str = "all",
    ) -> bool:
        cache_key = self._make_cache_key(key, namespace)
        ttl = ttl or self.config["memory"]["ttl"]

        ok = True

        if level in ("memory", "all"):
            ok &= self._set_memory(cache_key, value, ttl)

        if level in ("redis", "all") and self.redis_client:
            ok &= self._set_redis(cache_key, value, ttl)

        if level in ("disk", "all"):
            ok &= self._set_disk(cache_key, value, ttl)

        return ok

    # ------------------------------------------------------------------
    # L1 Memory
    # ------------------------------------------------------------------

    def _set_memory(self, cache_key: str, value: Any, ttl: int) -> bool:
        with self.memory_lock:
            if len(self.memory_cache) >= self.config["memory"]["max_size"]:
                self.memory_cache.popitem(last=False)
                self.memory_stats["evictions"] += 1

            self.memory_cache[cache_key] = {
                "value": value,
                "ttl": ttl,
                "created_at": time.time(),
            }
            self.memory_cache.move_to_end(cache_key)
        return True

    # ------------------------------------------------------------------
    # L2 Redis
    # ------------------------------------------------------------------

    def _set_redis(self, cache_key: str, value: Any, ttl: int) -> bool:
        try:
            entry = {"value": value, "ttl": ttl, "created_at": time.time()}
            self.redis_client.setex(cache_key, ttl, pickle.dumps(entry))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # L3 Disk
    # ------------------------------------------------------------------

    def _set_disk(self, cache_key: str, value: Any, ttl: int) -> bool:
        try:
            entry = {
                "value": value,
                "ttl": ttl,
                "created_at": time.time(),
                "accessed_at": time.time(),
            }

            key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            file = self.cache_path / "l3_disk" / f"{key_hash}.pkl"
            with open(file, "wb") as f:
                pickle.dump(entry, f)

            self._update_disk_index(cache_key, key_hash, ttl)
            return True
        except Exception:
            return False

    def _get_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        index = self._load_disk_index()
        info = index.get(cache_key)
        if not info:
            return None

        file = self.cache_path / "l3_disk" / f"{info['hash']}.pkl"
        if not file.exists():
            del index[cache_key]
            self._save_disk_index(index)
            return None

        try:
            with open(file, "rb") as f:
                entry = pickle.load(f)
            entry["accessed_at"] = time.time()
            with open(file, "wb") as f:
                pickle.dump(entry, f)
            return entry
        except Exception:
            file.unlink(missing_ok=True)
            del index[cache_key]
            self._save_disk_index(index)
            return None

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _make_cache_key(self, key: str, namespace: str) -> str:
        return f"{namespace}:{key}"

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        ttl = entry["ttl"]
        if ttl <= 0:
            return False
        return (time.time() - entry["created_at"]) > ttl

    def _load_disk_index(self) -> Dict[str, Any]:
        file = self.cache_path / "l3_disk" / "index.json"
        if file.exists():
            try:
                return json.loads(file.read_text())
            except Exception:
                pass
        return {}

    def _save_disk_index(self, index: Dict[str, Any]) -> None:
        file = self.cache_path / "l3_disk" / "index.json"
        file.write_text(json.dumps(index, indent=2))

    def _update_disk_index(self, cache_key: str, key_hash: str, ttl: int) -> None:
        index = self._load_disk_index()
        index[cache_key] = {
            "hash": key_hash,
            "ttl": ttl,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }

        if len(index) > self.config["disk"]["max_size"]:
            oldest = sorted(index.items(), key=lambda x: x[1]["last_accessed"])[:100]
            for k, v in oldest:
                (self.cache_path / "l3_disk" / f"{v['hash']}.pkl").unlink(missing_ok=True)
                del index[k]

        self._save_disk_index(index)
