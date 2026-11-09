"""Conector base: cache multinivel + gobernanza + normalizacion a CelestialRecord."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable

from proxima.contracts.exceptions import DataSchemaMismatch

from ..cache import CacheKey, MultiLevelCache
from ..config import DataLayerConfig
from ..governance import RateLimiter, execute_with_retry
from ..schema import CelestialRecord


def to_python_scalar(value: Any) -> Any:
    """Convierte escalares numpy/astropy/masked a tipos JSON nativos (NaN/masked -> None)."""
    if value is None:
        return None
    # astropy masked / numpy.ma
    if getattr(value, "mask", False) is True:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return int(value) if isinstance(value, int) else f
    return str(value)


def clean_rows(rows: Any) -> list[dict[str, Any]]:
    """Valida y limpia la respuesta cruda de una fuente (lista de dicts serializables)."""
    if not isinstance(rows, list):
        raise DataSchemaMismatch(f"La fuente debe devolver list[dict], recibido {type(rows)}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise DataSchemaMismatch(f"Fila invalida (se esperaba dict): {type(row)}")
        out.append({str(k): to_python_scalar(v) for k, v in row.items()})
    return out


def table_to_rows(table: Any) -> list[dict[str, Any]]:
    """Convierte una astropy Table (o similar) a list[dict]."""
    names = list(getattr(table, "colnames", []))
    return [{n: to_python_scalar(row[n]) for n in names} for row in table]


def require_columns(row: dict[str, Any], columns: tuple[str, ...], source: str) -> None:
    missing = [c for c in columns if c not in row]
    if missing:
        raise DataSchemaMismatch(f"[{source}] faltan columnas obligatorias: {missing}")


class SourceConnector(ABC):
    """Conector abstracto por servicio.

    Cada subclase define `source_name` (clave de TTL / version) y la conversion fila -> registro.
    """

    source_name: str = "abstract"
    required_columns: tuple[str, ...] = ()

    def __init__(
        self,
        config: DataLayerConfig | None = None,
        *,
        cache: MultiLevelCache | None = None,
        backend: Callable[..., Any] | None = None,
        rate_limiter: RateLimiter | None = None,
        dataset_version: str | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or DataLayerConfig.default_config()
        self.cache = cache if cache is not None else MultiLevelCache(
            memory_max_entries=self.config.memory_max_entries,
            default_ttl_seconds=self.ttl_seconds,
        )
        self._backend = backend
        self.rate_limiter = rate_limiter or RateLimiter(
            max_calls=self.config.rate_limit_calls_per_window,
            window_seconds=self.config.rate_limit_window_seconds,
        )
        self.dataset_version = dataset_version or self.config.dataset_versions.get(
            self.source_name, "unversioned"
        )
        self._sleeper = sleeper
        self.remote_calls = 0

    # -- propiedades -----------------------------------------------------------
    @property
    def ttl_seconds(self) -> float:
        return float(self.config.ttl_seconds.get(self.source_name, 86400.0))

    @property
    def backend(self) -> Callable[..., Any]:
        if self._backend is None:
            self._backend = self.default_backend()
        return self._backend

    @abstractmethod
    def default_backend(self) -> Callable[..., Any]:
        """Backend real (import perezoso de la libreria externa)."""

    @abstractmethod
    def row_to_record(self, row: dict[str, Any]) -> CelestialRecord | None:
        """Convierte una fila cruda en CelestialRecord normalizado (None = descartar)."""

    # -- utilidades ------------------------------------------------------------
    def cache_key(self, signature: str) -> CacheKey:
        return CacheKey(
            query_signature=f"{self.source_name}:{signature}",
            dataset_version=self.dataset_version,
            pipeline_version=self.config.pipeline_version,
            unit_system=self.config.unit_system,
        )

    def cached_fetch(self, signature: str, fetch: Callable[[], Any], *, n_calls: int = 1) -> list[dict]:
        """Consulta con cache L1/L2; solo toca la red en miss y bajo rate-limit + reintentos."""
        key = self.cache_key(signature)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        self.rate_limiter.acquire(n_calls)
        kwargs: dict[str, Any] = {
            "max_retries": self.config.max_retries,
            "backoff_seconds": self.config.backoff_seconds,
            "source_name": self.source_name,
        }
        if self._sleeper is not None:
            kwargs["sleeper"] = self._sleeper
        rows = clean_rows(execute_with_retry(fetch, **kwargs))
        self.remote_calls += n_calls
        self.cache.put(key, rows, ttl_seconds=self.ttl_seconds)
        return rows

    def normalize(self, rows: list[dict[str, Any]]) -> list[CelestialRecord]:
        records: list[CelestialRecord] = []
        for row in rows:
            require_columns(row, self.required_columns, self.source_name)
            rec = self.row_to_record(row)
            if rec is not None:
                records.append(rec)
        return records
