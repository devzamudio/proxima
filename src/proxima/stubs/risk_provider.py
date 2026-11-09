"""
StubRiskDataProvider - implementación mínima de RiskDataProviderContract.

Devuelve un RiskBreakdown con todos los sub-puntajes en 0.0 (riesgo nulo)
y metadatos de calidad mínimos. Sirve para activar los tests de contrato.
"""

from __future__ import annotations

from proxima.contracts.types import (
    DataQualityMetadata,
    DataQueryContext,
    RiskBreakdown,
    RouteSegment,
    SkyCoordinates,
)


class StubRiskDataProvider:
    """Stub mínimo de RiskDataProviderContract.

    No consulta datos astronómicos reales. Devuelve riesgo nulo con
    metadatos de calidad de mínima confianza (confidence_score=0.0
    representa "sin datos reales disponibles").
    """

    def evaluate_segment_risk(
        self,
        segment: RouteSegment,
        sky: SkyCoordinates,
        context: DataQueryContext,
    ) -> RiskBreakdown:
        """Devuelve riesgo cero en todos los sub-puntajes."""
        quality = DataQualityMetadata(
            source_name="stub",
            source_version="0.0.0",
            confidence_score=0.0,
            notes="Stub: sin datos astronómicos reales.",
        )
        return RiskBreakdown(
            radiative_flux_score=0.0,
            extinction_h_column_score=0.0,
            mass_density_score=0.0,
            tidal_stress_score=0.0,
            total_risk_score=0.0,
            quality=quality,
        )
