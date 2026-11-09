"""Gobernanza de cuota, rate-limit, reintentos y proteccion anti-microconsultas."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from proxima.contracts.exceptions import DataProviderUnavailable, RateLimitExceeded

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RateLimiter:
    """Limitador de tasa por ventana deslizante."""

    max_calls: int = 50
    window_seconds: float = 60.0
    clock: Callable[[], float] = time.time
    _calls: deque[float] = field(default_factory=deque, init=False, repr=False)

    def acquire(self, count: int = 1) -> None:
        now = float(self.clock())
        cutoff = now - self.window_seconds
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

        if len(self._calls) + count > self.max_calls:
            raise RateLimitExceeded(
                f"Cuota excedida: {len(self._calls) + count} > {self.max_calls} "
                f"en ventana de {self.window_seconds}s"
            )
        for _ in range(count):
            self._calls.append(now)


@dataclass
class SimbadLoopGuard:
    """Evita loops de microconsultas secuenciales en SIMBAD (Fase 3)."""

    max_sequential_single_queries: int = 3
    _consecutive_single_queries: int = field(default=0, init=False)

    def record_query(self, batch_size: int) -> None:
        if batch_size <= 1:
            self._consecutive_single_queries += 1
            if self._consecutive_single_queries > self.max_sequential_single_queries:
                raise RateLimitExceeded(
                    "Detectado loop de microconsultas individuales a SIMBAD "
                    f"({self._consecutive_single_queries} > {self.max_sequential_single_queries}). "
                    "Use query_batch() para consultas por lote."
                )
        else:
            self._consecutive_single_queries = 0

    def reset(self) -> None:
        self._consecutive_single_queries = 0


def execute_with_retry(
    fn: Callable[[], R],
    *,
    max_retries: int = 3,
    backoff_seconds: float = 0.05,
    sleeper: Callable[[float], None] = time.sleep,
    source_name: str = "unknown_source",
) -> R:
    """Ejecuta `fn` con reintentos y backoff exponencial ante fallas transitorias."""
    last_exc: Exception | None = None
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            return fn()
        except RateLimitExceeded:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < attempts and backoff_seconds > 0.0:
                sleeper(backoff_seconds * (2**attempt))
    raise DataProviderUnavailable(
        f"Fuente '{source_name}' no disponible tras {attempts} intentos: {last_exc}"
    ) from last_exc


def chunked_batches(items: Iterable[T], max_batch_size: int) -> list[list[T]]:
    """Divide una secuencia en lotes de tamano maximo `max_batch_size`."""
    size = max(1, int(max_batch_size))
    seq = list(items)
    return [seq[i : i + size] for i in range(0, len(seq), size)]
