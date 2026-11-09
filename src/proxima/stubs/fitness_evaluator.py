"""
StubFitnessEvaluator - implementación mínima de FitnessEvaluatorContract.

Devuelve un FitnessBreakdown con todos los términos en 0.0, satisfaciendo
la restricción contable total_fitness == suma de términos.
"""

from __future__ import annotations

from proxima.contracts.types import (
    FitnessBreakdown,
    RiskBreakdown,
    SimulationResult,
)


class StubFitnessEvaluator:
    """Stub mínimo de FitnessEvaluatorContract.

    No evalúa fitness real. Devuelve todos los términos en cero,
    satisfaciendo la invariante total_fitness == suma(términos).
    """

    def evaluate_fitness(
        self,
        result: SimulationResult,
        risk: RiskBreakdown | None,
    ) -> FitnessBreakdown:
        """Devuelve fitness nulo. total_fitness == 0.0 == suma de términos."""
        return FitnessBreakdown(
            mission_term=0.0,
            physical_penalty_term=0.0,
            risk_penalty_term=0.0,
            operational_penalty_term=0.0,
            total_fitness=0.0,
        )
