"""
StubMetricCore - implementación mínima de MetricCoreContract.

Devuelve una métrica plana (Minkowski + identidad espacial) con
todos los chequeos de energía en True. Suficiente para que los
validadores de contrato pasen en verde.
"""

from __future__ import annotations

from proxima.contracts.types import (
    EnergyConditionReport,
    MetricEvaluation,
    MetricStatePoint,
    WarpParameterVector,
)


class StubMetricCore:
    """Stub mínimo de MetricCoreContract.

    Satisface la forma del contrato devolviendo magnitudes triviales
    pero finitas y bien tipadas. No calcula física real.
    """

    def evaluate_metric(
        self,
        params: WarpParameterVector,
        state: MetricStatePoint,
    ) -> MetricEvaluation:
        """Devuelve lapse=1, shift nulo, métrica espacial identidad."""
        return MetricEvaluation(
            lapse=1.0,
            shift_vector=(0.0, 0.0, 0.0),
            spatial_metric=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            valid_domain=True,
            diagnostics={"stub": True},
        )

    def evaluate_energy_conditions(
        self,
        params: WarpParameterVector,
        state: MetricStatePoint,
    ) -> EnergyConditionReport:
        """Reporta todas las condiciones de energía satisfechas."""
        return EnergyConditionReport(
            wec_ok=True,
            nec_ok=True,
            dec_ok=True,
            sec_ok=True,
            violation_magnitudes={},
        )
