"""Proveedor de riesgo basado en catalogos astronomicos (implementa RiskDataProviderContract).

Para cada segmento recto de ruta (ICRS, o local 3+1 desplazado al origen dado por SkyCoordinates)
calcula, de forma analitica y monotona respecto a la longitud del segmento:

1. Fluencia radiante integrada  [J/m^2]:   sum_k L_k/(4 pi v) * int ds / (r_k^2 + R_k^2)
2. Columna de H                 [m^-2]:   n_ism*L + sum_gas n_k * cuerda_k  (+ N_H por linea de vision)
3. Densidad de masa local max.  [kg/m^3]: rho_ism + sum_k rho_k * [d_min,k < R_k]
4. Estres de marea max.         [1/s^2]:  sum_k 2 G M_k / max(d_min,k, R_k)^3
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from proxima.contracts.exceptions import DataSchemaMismatch
from proxima.contracts.types import (
    DataQualityMetadata,
    DataQueryContext,
    Frame,
    RiskBreakdown,
    RouteSegment,
    SkyCoordinates,
)

from .config import DataLayerConfig
from .normalizer import ebv_to_h_column_m2, pc_to_m, spherical_icrs_to_xyz_m
from .schema import CelestialRecord

G_SI = 6.67430e-11
M_H_KG = 1.6735575e-27
MEAN_MASS_PER_H = 1.4  # incluye He


def saturating_score(value: float, scale: float) -> float:
    if value <= 0.0 or scale <= 0.0 or not math.isfinite(value):
        return 0.0 if value <= 0.0 or scale <= 0.0 else 1.0
    return float(1.0 - math.exp(-value / scale))


class AstronomicalRiskProvider:
    source_name = "astronomical_catalogs"

    def __init__(
        self,
        records: list[CelestialRecord],
        config: DataLayerConfig | None = None,
        *,
        extinction_provider: Any | None = None,
        memory_cache_size: int | None = None,
    ) -> None:
        self.config = config or DataLayerConfig.default_config()
        self.records = list(records)
        self.extinction_provider = extinction_provider
        self._cache: dict[tuple, RiskBreakdown] = {}
        self._cache_size = int(memory_cache_size or self.config.memory_max_entries)

        n = len(self.records)
        self._centers = np.array([r.xyz_icrs_m for r in self.records], dtype=np.float64).reshape(n, 3)
        self._lum = np.array([r.luminosity_w for r in self.records], dtype=np.float64)
        self._mass = np.array([r.mass_kg for r in self.records], dtype=np.float64)
        self._radius = np.array([max(r.radius_m, 1.0) for r in self.records], dtype=np.float64)
        self._confidence = np.array([r.confidence_score for r in self.records], dtype=np.float64)
        volume = 4.0 / 3.0 * math.pi * self._radius**3
        self._rho = self._mass / volume
        gas = np.array([r.object_type in self.config.gas_object_types for r in self.records], dtype=bool)
        self._n_gas = np.where(gas, self._rho / (MEAN_MASS_PER_H * M_H_KG), 0.0)

    @classmethod
    def from_snapshot(cls, store, snapshot_id: str, config: DataLayerConfig | None = None, **kw: Any):
        records, _ = store.load(snapshot_id)
        return cls(records, config, **kw)

    # -- geometria ---------------------------------------------------------------
    @staticmethod
    def _origin(sky: SkyCoordinates) -> np.ndarray:
        if sky.distance_pc is None:
            return np.zeros(3)
        return np.array(spherical_icrs_to_xyz_m(sky.ra_deg, sky.dec_deg, pc_to_m(sky.distance_pc)))

    def _segment_icrs(self, segment: RouteSegment, sky: SkyCoordinates) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(segment.start_xyz_m, dtype=np.float64)
        end = np.asarray(segment.end_xyz_m, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            raise DataSchemaMismatch(f"RouteSegment invalido: {segment}")
        if segment.frame in (Frame.ICRS, Frame.BCRS):
            return start, end
        if segment.frame == Frame.LOCAL_CARTESIAN_3P1:
            o = self._origin(sky)
            return start + o, end + o
        raise DataSchemaMismatch(f"Marco de segmento no soportado: {segment.frame}")

    def physical_quantities(self, segment: RouteSegment, sky: SkyCoordinates) -> dict[str, float]:
        start, end = self._segment_icrs(segment, sky)
        d = end - start
        length = float(np.linalg.norm(d))
        n_h_bg = self.config.ism_n_h_m3
        rho_bg = n_h_bg * MEAN_MASS_PER_H * M_H_KG

        out = {
            "length_m": length,
            "fluence_j_m2": 0.0,
            "h_column_m2": n_h_bg * length,
            "mass_density_kg_m3": rho_bg,
            "tidal_s2": 0.0,
            "los_h_column_m2": 0.0,
            "n_influencing": 0.0,
            "confidence": float(self._confidence.mean()) if len(self.records) else 0.5,
        }

        if self.extinction_provider is not None and sky.distance_pc is not None:
            ebv = self.extinction_provider.ebv(sky.ra_deg, sky.dec_deg, pc_to_m(sky.distance_pc))
            out["los_h_column_m2"] = ebv_to_h_column_m2(ebv)
            out["h_column_m2"] += out["los_h_column_m2"]

        if not len(self.records):
            return out

        uhat = d / length if length > 0.0 else np.zeros(3)
        w = self._centers - start                      # (n,3)
        s_star = w @ uhat                              # proyeccion sobre el segmento
        b2 = np.maximum(np.einsum("ij,ij->i", w, w) - s_star**2, 0.0)
        s_c = np.clip(s_star, 0.0, length)
        d_min = np.sqrt(b2 + (s_c - s_star) ** 2)
        R = self._radius

        # 1) fluencia: integral cerrada con suavizado r^2 -> r^2 + R^2
        a = np.sqrt(b2 + R**2)
        path_int = (np.arctan((length - s_star) / a) - np.arctan(-s_star / a)) / a  # [1/m]
        fluence = self._lum / (4.0 * math.pi * self.config.transit_speed_m_s) * path_int
        out["fluence_j_m2"] = float(np.sum(fluence))

        # 2) cuerda dentro de esferas gaseosas
        inside_b = b2 < R**2
        half = np.sqrt(np.where(inside_b, R**2 - b2, 0.0))
        chord = np.where(
            inside_b,
            np.maximum(0.0, np.minimum(length, s_star + half) - np.maximum(0.0, s_star - half)),
            0.0,
        )
        out["h_column_m2"] += float(np.sum(self._n_gas * chord))

        # 3) densidad local maxima y 4) marea maxima (cotas conservadoras)
        inside = d_min < R
        out["mass_density_kg_m3"] += float(np.sum(np.where(inside, self._rho, 0.0)))
        out["tidal_s2"] = float(np.sum(2.0 * G_SI * self._mass / np.maximum(d_min, R) ** 3))

        influence = inside | (chord > 0.0)
        out["n_influencing"] = float(np.count_nonzero(influence))
        if np.any(influence):
            out["confidence"] = float(self._confidence[influence].mean())
        return out

    # -- contrato ----------------------------------------------------------------
    def evaluate_segment_risk(
        self,
        segment: RouteSegment,
        sky: SkyCoordinates,
        context: DataQueryContext,
    ) -> RiskBreakdown:
        key = (segment.start_xyz_m, segment.end_xyz_m, segment.frame, sky, context.dataset_version)
        if key in self._cache:
            return self._cache[key]

        q = self.physical_quantities(segment, sky)
        cfg = self.config
        scores = {
            "radiative": saturating_score(q["fluence_j_m2"], cfg.radiative_flux_scale_j_m2),
            "extinction": saturating_score(q["h_column_m2"], cfg.h_column_scale_m2),
            "mass_density": saturating_score(q["mass_density_kg_m3"], cfg.mass_density_scale_kg_m3),
            "tidal": saturating_score(q["tidal_s2"], cfg.tidal_stress_scale_s2),
        }
        weights = cfg.risk_weights
        wsum = sum(weights.get(k, 0.0) for k in scores) or 1.0
        total = sum(weights.get(k, 0.0) * v for k, v in scores.items()) / wsum

        notes = (
            f"fluence_j_m2={q['fluence_j_m2']:.6e};h_column_m2={q['h_column_m2']:.6e};"
            f"los_h_column_m2={q['los_h_column_m2']:.6e};rho_kg_m3={q['mass_density_kg_m3']:.6e};"
            f"tidal_s2={q['tidal_s2']:.6e};length_m={q['length_m']:.6e};"
            f"n_records={len(self.records)};n_influencing={int(q['n_influencing'])};"
            f"cache_key={context.cache_key}"
        )
        result = RiskBreakdown(
            radiative_flux_score=scores["radiative"],
            extinction_h_column_score=scores["extinction"],
            mass_density_score=scores["mass_density"],
            tidal_stress_score=scores["tidal"],
            total_risk_score=float(min(1.0, max(0.0, total))),
            quality=DataQualityMetadata(
                source_name=self.source_name,
                source_version=context.dataset_version or cfg.version,
                confidence_score=float(min(1.0, max(0.0, q["confidence"]))),
                notes=notes,
            ),
        )
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result
