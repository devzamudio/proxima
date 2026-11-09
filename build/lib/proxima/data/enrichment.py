"""Enriquecimiento contextual: extincion/columna de H por objeto y entorno local."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .normalizer import ebv_to_h_column_m2
from .schema import CelestialRecord


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class ContextEnricher:
    """Asocia cada objeto con el medio interestelar (dust map) y con estructuras cercanas."""

    def __init__(self, extinction_provider: Any | None = None) -> None:
        self.extinction_provider = extinction_provider

    def enrich_extinction(self, records: list[CelestialRecord]) -> list[CelestialRecord]:
        if self.extinction_provider is None or not records:
            return list(records)
        values = self.extinction_provider.ebv_batch([(r.ra_deg, r.dec_deg, r.distance_m) for r in records])
        return [
            replace(r, ebv_mag=float(v), h_column_m2=ebv_to_h_column_m2(float(v)))
            for r, v in zip(records, values)
        ]

    @staticmethod
    def local_environment(
        target: CelestialRecord,
        environment: list[CelestialRecord],
        search_radius_m: float,
    ) -> list[tuple[CelestialRecord, float]]:
        """Estructuras (nebulosas, cumulos, SNRs) cuyo borde esta a <= search_radius_m del objetivo."""
        out = []
        for env in environment:
            if env.canonical_id == target.canonical_id:
                continue
            d = _dist(target.xyz_icrs_m, env.xyz_icrs_m)
            if d - env.radius_m <= search_radius_m:
                out.append((env, d))
        out.sort(key=lambda t: t[1])
        return out

    def enrich(self, records: list[CelestialRecord]) -> list[CelestialRecord]:
        return self.enrich_extinction(records)
