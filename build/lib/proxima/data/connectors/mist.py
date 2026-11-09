"""Modelos estelares MIST via `isochrones` para estimar L y R a partir de (M, edad, [Fe/H])."""

from __future__ import annotations

import math
from typing import Any, Callable

from proxima.contracts.exceptions import DataProviderUnavailable

from ..cache import CacheKey, MultiLevelCache
from ..config import DataLayerConfig
from ..governance import execute_with_retry

DEFAULT_AGE_GYR = 4.6
DEFAULT_FEH = 0.0


def _isochrones_backend(mass_msun: float, age_gyr: float, feh: float) -> dict[str, float]:
    try:
        from isochrones import get_ichrone  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataProviderUnavailable("isochrones no instalado: pip install isochrones") from exc
    track = get_ichrone("mist", tracks=True)
    log_age = math.log10(age_gyr * 1e9)
    eep = track.get_eep(mass_msun, log_age, feh, accurate=True)
    _, _, _, mass, radius, logl, *_ = track.interp_value([eep, log_age, feh],
                                                          ["eep", "age", "feh", "mass", "radius", "logL"])
    return {"logL": float(logl), "radius_rsun": float(radius)}


def main_sequence_backend(mass_msun: float, age_gyr: float, feh: float) -> dict[str, float]:
    """Aproximacion analitica de secuencia principal (sin red): L ~ M^3.5, R ~ M^0.8."""
    m = max(float(mass_msun), 1e-3)
    return {"logL": 3.5 * math.log10(m), "radius_rsun": m**0.8}


class MistStellarModelProvider:
    source_name = "mist"

    def __init__(
        self,
        config: DataLayerConfig | None = None,
        *,
        cache: MultiLevelCache | None = None,
        backend: Callable[[float, float, float], dict[str, float]] | None = None,
    ) -> None:
        self.config = config or DataLayerConfig.default_config()
        self.cache = cache or MultiLevelCache(memory_max_entries=self.config.memory_max_entries)
        self.backend = backend or _isochrones_backend
        self.dataset_version = self.config.dataset_versions.get(self.source_name, "MIST-v1.2")

    def estimate(self, mass_msun: float, age_gyr: float | None = None, feh: float | None = None) -> dict[str, Any]:
        age = float(age_gyr) if age_gyr else DEFAULT_AGE_GYR
        z = float(feh) if feh is not None else DEFAULT_FEH
        key = CacheKey(
            query_signature=f"mist:{float(mass_msun):.4f}:{age:.4f}:{z:.3f}",
            dataset_version=self.dataset_version,
            pipeline_version=self.config.pipeline_version,
            unit_system=self.config.unit_system,
        )
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        raw = execute_with_retry(
            lambda: self.backend(float(mass_msun), age, z),
            max_retries=self.config.max_retries,
            backoff_seconds=self.config.backoff_seconds,
            source_name=self.source_name,
        )
        out = {
            "luminosity_lsun": 10.0 ** float(raw["logL"]),
            "radius_rsun": float(raw["radius_rsun"]),
            "model": self.dataset_version,
        }
        self.cache.put(key, out, ttl_seconds=float(self.config.ttl_seconds.get("mist", 2592000.0)))
        return out
