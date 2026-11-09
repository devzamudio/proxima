"""Extincion 3D con dustmaps / Bayestar19 (Green et al. 2019)."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from proxima.contracts.exceptions import DataProviderUnavailable

from ..cache import CacheKey, MultiLevelCache
from ..config import DataLayerConfig
from ..governance import RateLimiter, chunked_batches, execute_with_retry
from ..normalizer import PARSEC_TO_M, ebv_to_h_column_m2

# Green et al. (2019): E(B-V) = 0.884 x (unidades Bayestar19)
BAYESTAR19_TO_EBV: float = 0.884


def _dustmaps_backend(coords: list[tuple[float, float, float]]) -> list[float]:
    """coords: [(ra_deg, dec_deg, distance_pc)] -> reddening Bayestar19 (mediana)."""
    try:
        import astropy.units as u  # type: ignore[import-not-found]
        from astropy.coordinates import SkyCoord  # type: ignore[import-not-found]
        from dustmaps.bayestar import BayestarQuery  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataProviderUnavailable("Requiere dustmaps + astropy (y fetch de bayestar2019)") from exc
    query = BayestarQuery(version="bayestar2019")
    ra, dec, dist = zip(*coords)
    sc = SkyCoord(ra=list(ra) * u.deg, dec=list(dec) * u.deg, distance=list(dist) * u.pc, frame="icrs")
    values = query(sc, mode="median")
    return [0.0 if (v is None or not math.isfinite(float(v))) else float(v) for v in values]


class BayestarExtinctionProvider:
    """E(B-V) y columna de H por linea de vision; consultas por lote y cacheadas."""

    source_name = "bayestar19"

    def __init__(
        self,
        config: DataLayerConfig | None = None,
        *,
        cache: MultiLevelCache | None = None,
        backend: Callable[[list[tuple[float, float, float]]], Sequence[float]] | None = None,
        rate_limiter: RateLimiter | None = None,
        coord_precision_deg: float = 1e-3,
        distance_precision_pc: float = 1.0,
    ) -> None:
        self.config = config or DataLayerConfig.default_config()
        self.cache = cache or MultiLevelCache(memory_max_entries=self.config.memory_max_entries)
        self._backend = backend
        self.rate_limiter = rate_limiter or RateLimiter(
            self.config.rate_limit_calls_per_window, self.config.rate_limit_window_seconds
        )
        self.dataset_version = self.config.dataset_versions.get(self.source_name, "bayestar2019")
        self.coord_precision_deg = coord_precision_deg
        self.distance_precision_pc = distance_precision_pc
        self.remote_calls = 0

    @property
    def backend(self):
        if self._backend is None:
            self._backend = _dustmaps_backend
        return self._backend

    def _quantize(self, ra: float, dec: float, distance_m: float) -> tuple[float, float, float]:
        q = self.coord_precision_deg
        dq = self.distance_precision_pc
        d_pc = distance_m / PARSEC_TO_M
        return (round(ra / q) * q, round(dec / q) * q, max(dq, round(d_pc / dq) * dq))

    def _key(self, c: tuple[float, float, float]) -> CacheKey:
        return CacheKey(
            query_signature=f"{self.source_name}:{c[0]:.6f},{c[1]:.6f},{c[2]:.3f}",
            dataset_version=self.dataset_version,
            pipeline_version=self.config.pipeline_version,
            unit_system=self.config.unit_system,
        )

    def ebv_batch(self, points: Sequence[tuple[float, float, float]]) -> list[float]:
        """points: [(ra_deg, dec_deg, distance_m)] -> E(B-V) en magnitudes."""
        quantized = [self._quantize(*p) for p in points]
        result: dict[tuple[float, float, float], float] = {}
        missing: list[tuple[float, float, float]] = []
        for c in dict.fromkeys(quantized):
            hit = self.cache.get(self._key(c))
            if hit is not None:
                result[c] = float(hit)
            else:
                missing.append(c)

        ttl = float(self.config.ttl_seconds.get(self.source_name, 2592000.0))
        for chunk in chunked_batches(missing, self.config.max_batch_size):
            self.rate_limiter.acquire()
            raw = execute_with_retry(
                lambda c=chunk: list(self.backend(list(c))),
                max_retries=self.config.max_retries,
                backoff_seconds=self.config.backoff_seconds,
                source_name=self.source_name,
            )
            self.remote_calls += 1
            if len(raw) != len(chunk):
                raise DataProviderUnavailable("Bayestar devolvio un numero de valores inconsistente")
            for c, v in zip(chunk, raw):
                ebv = max(0.0, float(v) * BAYESTAR19_TO_EBV)
                result[c] = ebv
                self.cache.put(self._key(c), ebv, ttl_seconds=ttl)
        return [result[c] for c in quantized]

    def ebv(self, ra_deg: float, dec_deg: float, distance_m: float) -> float:
        return self.ebv_batch([(ra_deg, dec_deg, distance_m)])[0]

    def h_column_m2(self, ra_deg: float, dec_deg: float, distance_m: float) -> float:
        return ebv_to_h_column_m2(self.ebv(ra_deg, dec_deg, distance_m))


class ConstantExtinctionProvider:
    """Proveedor determinista sin red: E(B-V) proporcional a la distancia (mag/kpc)."""

    source_name = "constant_extinction"

    def __init__(self, ebv_per_kpc: float = 0.3) -> None:
        self.ebv_per_kpc = float(ebv_per_kpc)

    def ebv_batch(self, points: Sequence[tuple[float, float, float]]) -> list[float]:
        return [self.ebv_per_kpc * (d / (1000.0 * PARSEC_TO_M)) for _, _, d in points]

    def ebv(self, ra_deg: float, dec_deg: float, distance_m: float) -> float:
        return self.ebv_batch([(ra_deg, dec_deg, distance_m)])[0]

    def h_column_m2(self, ra_deg: float, dec_deg: float, distance_m: float) -> float:
        return ebv_to_h_column_m2(self.ebv(ra_deg, dec_deg, distance_m))


__all__: list[Any] = ["BayestarExtinctionProvider", "ConstantExtinctionProvider", "BAYESTAR19_TO_EBV"]
