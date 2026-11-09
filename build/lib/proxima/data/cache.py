"""Cache multinivel (Memoria L1 + Disco L2) con validacion por version, TTL y checksum."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    """Clave compuesta determinista para cache multinivel (Fase 3)."""

    query_signature: str
    dataset_version: str
    pipeline_version: str
    unit_system: str = "SI"

    def digest(self) -> str:
        raw = json.dumps(
            {
                "query_signature": self.query_signature,
                "dataset_version": self.dataset_version,
                "pipeline_version": self.pipeline_version,
                "unit_system": self.unit_system,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_checksum(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class MultiLevelCache:
    """Cache de dos niveles (L1 en memoria LRU + L2 en disco JSON con checksum SHA-256)."""

    def __init__(
        self,
        disk_dir: str | Path | None = None,
        memory_max_entries: int = 256,
        default_ttl_seconds: float = 86400.0,
        clock=time.time,
    ) -> None:
        self.disk_dir = Path(disk_dir) if disk_dir is not None else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.memory_max_entries = max(1, int(memory_max_entries))
        self.default_ttl_seconds = float(default_ttl_seconds)
        self._clock = clock
        self._l1: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "invalidations": 0}

    def _disk_path(self, key_digest: str) -> Path | None:
        if self.disk_dir is None:
            return None
        return self.disk_dir / f"{key_digest}.json"

    def _is_entry_valid(self, entry: dict[str, Any], key: CacheKey) -> bool:
        now = float(self._clock())
        created_at = float(entry.get("created_at", 0.0))
        ttl = float(entry.get("ttl_seconds", self.default_ttl_seconds))
        if now - created_at > ttl:
            return False

        if entry.get("dataset_version") != key.dataset_version:
            return False
        if entry.get("pipeline_version") != key.pipeline_version:
            return False
        if entry.get("unit_system") != key.unit_system:
            return False

        payload = entry.get("payload")
        expected_checksum = entry.get("checksum")
        if not expected_checksum or _compute_checksum(payload) != expected_checksum:
            return False

        return True

    def get(self, key: CacheKey) -> Any | None:
        digest = key.digest()

        # Nivel 1: Memoria
        if digest in self._l1:
            entry = self._l1[digest]
            if self._is_entry_valid(entry, key):
                self._l1.move_to_end(digest)
                self.stats["l1_hits"] += 1
                return entry["payload"]
            else:
                del self._l1[digest]
                self.stats["invalidations"] += 1

        # Nivel 2: Disco
        path = self._disk_path(digest)
        if path is not None and path.exists():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                path.unlink(missing_ok=True)
                self.stats["invalidations"] += 1
                self.stats["misses"] += 1
                return None

            if self._is_entry_valid(entry, key):
                self._put_l1(digest, entry)
                self.stats["l2_hits"] += 1
                return entry["payload"]
            else:
                path.unlink(missing_ok=True)
                self.stats["invalidations"] += 1

        self.stats["misses"] += 1
        return None

    def put(
        self,
        key: CacheKey,
        payload: Any,
        ttl_seconds: float | None = None,
    ) -> str:
        digest = key.digest()
        ttl = float(ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds)
        entry = {
            "key_digest": digest,
            "query_signature": key.query_signature,
            "dataset_version": key.dataset_version,
            "pipeline_version": key.pipeline_version,
            "unit_system": key.unit_system,
            "created_at": float(self._clock()),
            "ttl_seconds": ttl,
            "checksum": _compute_checksum(payload),
            "payload": payload,
        }
        self._put_l1(digest, entry)
        path = self._disk_path(digest)
        if path is not None:
            path.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
        return digest

    def invalidate(self, key: CacheKey) -> None:
        digest = key.digest()
        if digest in self._l1:
            del self._l1[digest]
        path = self._disk_path(digest)
        if path is not None and path.exists():
            path.unlink(missing_ok=True)
        self.stats["invalidations"] += 1

    def invalidate_by_version(self, dataset_version: str) -> int:
        """Invalida todas las entradas que no coincidan con dataset_version."""
        removed = 0
        to_delete = [
            k for k, v in self._l1.items() if v.get("dataset_version") != dataset_version
        ]
        for k in to_delete:
            del self._l1[k]
            removed += 1

        if self.disk_dir is not None and self.disk_dir.exists():
            for file_path in self.disk_dir.glob("*.json"):
                try:
                    entry = json.loads(file_path.read_text(encoding="utf-8"))
                    if entry.get("dataset_version") != dataset_version:
                        file_path.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    file_path.unlink(missing_ok=True)
                    removed += 1

        self.stats["invalidations"] += removed
        return removed

    def _put_l1(self, digest: str, entry: dict[str, Any]) -> None:
        self._l1[digest] = entry
        self._l1.move_to_end(digest)
        while len(self._l1) > self.memory_max_entries:
            self._l1.popitem(last=False)
