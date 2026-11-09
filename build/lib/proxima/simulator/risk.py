"""Evaluacion de riesgo por tramo.

Divide la trayectoria en RouteSegment y consulta el
RiskDataProviderContract inyectado (stub por defecto).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax.numpy as jnp

from proxima.contracts.types import (
    DataQueryContext,
    DataQualityMetadata,
    Frame,
    RiskBreakdown,
    RouteSegment,
    SkyCoordinates,
)
from proxima.stubs.risk_provider import StubRiskDataProvider


def _saturating_score(value: float, scale: float) -> float:
    """Normaliza una magnitud no negativa a [0, 1] de forma estrictamente monotona."""
    if value <= 0.0 or scale <= 0.0:
        return 0.0
    return float(1.0 - math.exp(-value / scale))


class AnalyticRiskProvider:
    """Proveedor parametrico de riesgo con respuesta monotona frente a longitud y entorno.

    Implementa RiskDataProviderContract para pruebas de sensibilidad/monotonicidad
    y escenarios sinteticos antes de conectar catalogos astronomicos reales (Fase 3).
    """

    def __init__(
        self,
        radiative_flux_w_m2: float = 1.0,
        h_column_density_m2: float = 1.0e20,
        mass_density_kg_m3: float = 1.0e-21,
        cluster_tidal_s2: float = 1.0e-12,
        length_scale_m: float = 1.0e8,
    ) -> None:
        self.radiative_flux_w_m2 = float(radiative_flux_w_m2)
        self.h_column_density_m2 = float(h_column_density_m2)
        self.mass_density_kg_m3 = float(mass_density_kg_m3)
        self.cluster_tidal_s2 = float(cluster_tidal_s2)
        self.length_scale_m = float(length_scale_m)

    def evaluate_segment_risk(
        self,
        segment: RouteSegment,
        sky: SkyCoordinates,
        context: DataQueryContext,
    ) -> RiskBreakdown:
        dx = segment.end_xyz_m[0] - segment.start_xyz_m[0]
        dy = segment.end_xyz_m[1] - segment.start_xyz_m[1]
        dz = segment.end_xyz_m[2] - segment.start_xyz_m[2]
        length_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        length_factor = max(length_m / max(self.length_scale_m, 1e-12), 1e-6)

        rad_score = _saturating_score(self.radiative_flux_w_m2 * length_factor, 10.0)
        ext_score = _saturating_score(self.h_column_density_m2 * length_factor, 1.0e21)
        mass_score = _saturating_score(self.mass_density_kg_m3 * length_factor, 1.0e-20)
        tidal_score = _saturating_score(self.cluster_tidal_s2 * length_factor, 1.0e-11)
        total = 0.25 * (rad_score + ext_score + mass_score + tidal_score)

        return RiskBreakdown(
            radiative_flux_score=rad_score,
            extinction_h_column_score=ext_score,
            mass_density_score=mass_score,
            tidal_stress_score=tidal_score,
            total_risk_score=total,
            quality=DataQualityMetadata(
                source_name="analytic_model",
                source_version=context.dataset_version or "1.0.0",
                confidence_score=1.0,
                notes=f"segment_length_m={length_m:.3e}",
            ),
        )


@dataclass
class RouteSegmentRisk:
    """Riesgo de un segmento de ruta individual."""

    segment: RouteSegment
    risk_breakdown: RiskBreakdown


class RiskAssessor:
    """Evalua el riesgo compuesto de una trayectoria dividiéndola en segmentos.

    Por defecto usa StubRiskDataProvider (riesgo nulo) hasta que la Fase 3
    conecte un proveedor real de datos astronomicos.
    """

    def __init__(
        self,
        provider=None,
        sky: SkyCoordinates | None = None,
        context: DataQueryContext | None = None,
    ):
        self.provider = provider or StubRiskDataProvider()
        self.sky = sky or SkyCoordinates(
            ra_deg=0.0, dec_deg=0.0, distance_pc=None, frame=Frame.ICRS
        )
        self.context = context or DataQueryContext(
            dataset_version="stub-0.0.0",
            extraction_timestamp_utc="1970-01-01T00:00:00Z",
            cache_key="stub",
        )

    def assess_trajectory(
        self,
        trajectory: jnp.ndarray,
        valid_mask: jnp.ndarray,
        n_segments: int = 10,
    ) -> tuple[RiskBreakdown, list[RouteSegmentRisk]]:
        """Divide la trayectoria en segmentos y evalua el riesgo de cada uno.

        Retorna (risk_breakdown_agregado, lista de RouteSegmentRisk).
        """
        # Obtener puntos validos de la trayectoria
        valid_pts = trajectory[valid_mask]  # (n_valid, 8): X(4) + u(4)
        if valid_pts.shape[0] < 2:
            # No hay suficiente trayectoria
            empty_risk = RiskBreakdown(
                radiative_flux_score=0.0,
                extinction_h_column_score=0.0,
                mass_density_score=0.0,
                tidal_stress_score=0.0,
                total_risk_score=0.0,
                quality=DataQualityMetadata(
                    source_name="stub",
                    source_version="0.0.0",
                    confidence_score=0.0,
                    notes="No trajectory data",
                ),
            )
            return empty_risk, []

        positions = valid_pts[:, 1:4]  # (n_valid, 3) x, y, z

        # Dividir en n_segments segmentos
        n_pts = positions.shape[0]
        segment_indices = jnp.linspace(0, n_pts - 1, n_segments + 1).astype(jnp.int32)

        segment_risks = []
        total_risk = 0.0
        total_radiative = 0.0
        total_extinction = 0.0
        total_mass = 0.0
        total_tidal = 0.0

        for i in range(n_segments):
            start_idx = int(segment_indices[i])
            end_idx = int(segment_indices[i + 1])
            if end_idx <= start_idx:
                end_idx = start_idx + 1

            start_pos = tuple(float(x) for x in positions[start_idx])
            end_pos = tuple(float(x) for x in positions[min(end_idx, n_pts - 1)])

            segment = RouteSegment(
                start_xyz_m=start_pos,  # type: ignore[arg-type]
                end_xyz_m=end_pos,  # type: ignore[arg-type]
                frame=Frame.LOCAL_CARTESIAN_3P1,
            )

            rb = self.provider.evaluate_segment_risk(segment, self.sky, self.context)
            segment_risks.append(RouteSegmentRisk(segment=segment, risk_breakdown=rb))

            total_risk += rb.total_risk_score
            total_radiative += rb.radiative_flux_score
            total_extinction += rb.extinction_h_column_score
            total_mass += rb.mass_density_score
            total_tidal += rb.tidal_stress_score

        n_seg = max(n_segments, 1)
        first_quality = segment_risks[0].risk_breakdown.quality if segment_risks else DataQualityMetadata(
            source_name="stub",
            source_version="0.0.0",
            confidence_score=0.0,
            notes="Aggregated from stub provider",
        )
        aggregated = RiskBreakdown(
            radiative_flux_score=total_radiative / n_seg,
            extinction_h_column_score=total_extinction / n_seg,
            mass_density_score=total_mass / n_seg,
            tidal_stress_score=total_tidal / n_seg,
            total_risk_score=total_risk / n_seg,
            quality=first_quality,
        )

        return aggregated, segment_risks