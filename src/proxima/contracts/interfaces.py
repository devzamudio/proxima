from __future__ import annotations

from typing import Protocol

from .types import (
    DataQueryContext,
    EnergyConditionReport,
    FitnessBreakdown,
    MetricEvaluation,
    MetricStatePoint,
    RiskBreakdown,
    RouteSegment,
    SimulationRequest,
    SimulationResult,
    SkyCoordinates,
    WarpParameterVector,
)


class MetricCoreContract(Protocol):
    """Contract consumed by the relativistic simulator."""

    def evaluate_metric(self, params: WarpParameterVector, state: MetricStatePoint) -> MetricEvaluation:
        """Return ADM metric components and diagnostics for a state point."""

    def evaluate_energy_conditions(
        self,
        params: WarpParameterVector,
        state: MetricStatePoint,
    ) -> EnergyConditionReport:
        """Return WEC/NEC/DEC/SEC checks for a state point."""


class RelativisticSimulatorContract(Protocol):
    """Contract consumed by optimization and experiment runners."""

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        """Run the full relativistic simulation and return a terminal result."""


class RiskDataProviderContract(Protocol):
    """Contract consumed by risk evaluation modules."""

    def evaluate_segment_risk(
        self,
        segment: RouteSegment,
        sky: SkyCoordinates,
        context: DataQueryContext,
    ) -> RiskBreakdown:
        """Return risk sub-scores and aggregated risk for one route segment."""


class FitnessEvaluatorContract(Protocol):
    """Contract consumed by the PSO optimizer."""

    def evaluate_fitness(self, result: SimulationResult, risk: RiskBreakdown | None) -> FitnessBreakdown:
        """Compute a decomposed objective with penalties and total fitness."""
