"""
StubRelativisticSimulator - implementación mínima de RelativisticSimulatorContract.

Ejecuta cero pasos de integración y devuelve un resultado SUCCESS con
observables triviales. Sirve como línea base de contrato.
"""

from __future__ import annotations

from proxima.contracts.types import (
    EnergyConditionReport,
    SimulationObservables,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
)


class StubRelativisticSimulator:
    """Stub mínimo de RelativisticSimulatorContract.

    No integra geodésicas reales. Devuelve un resultado SUCCESS
    estructuralmente válido con observables de valor cero.
    """

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        """Devuelve SUCCESS con observables triviales y energía OK."""
        observables = SimulationObservables(
            delta_tau_s=0.0,
            delta_t_s=0.0,
            total_energy_j=0.0,
            tidal_stress_max_si=0.0,
        )
        energy_conditions = EnergyConditionReport(
            wec_ok=True,
            nec_ok=True,
            dec_ok=True,
            sec_ok=True,
            violation_magnitudes={},
        )
        return SimulationResult(
            status=SimulationStatus.SUCCESS,
            observables=observables,
            energy_conditions=energy_conditions,
            failure_reason=None,
            diagnostics={"stub": True, "steps_run": 0},
        )
