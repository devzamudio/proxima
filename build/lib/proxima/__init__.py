"""PROXIMA: Motor de optimización relativista y simulación interestelar con métricas Warp."""

from __future__ import annotations

# Subpaquetes principales
from proxima import contracts, core, simulator, data, optimizer, validation

# Clases y funciones más utilizadas para importación directa de conveniencia
from proxima.contracts.types import (
    WarpParameterVector,
    MetricStatePoint,
    MetricEvaluation,
    ScientificConventions,
    SimulationRequest,
    SimulationResult,
    SimulationObservables,
    SimulationStatus,
    RouteSegment,
    SkyCoordinates,
    RiskBreakdown,
    FitnessBreakdown,
)
from proxima.core.metric_core import BobrickMartireMetricCore
from proxima.simulator.simulator import RelativisticSimulator
from proxima.simulator.config import SimulatorConfig
from proxima.data.risk_provider import AstronomicalRiskProvider
from proxima.data.pipeline import default_local_risk_provider
from proxima.optimizer.optimizer import VectorizedPSO, OptimizationResult
from proxima.optimizer.config import PSOConfig
from proxima.optimizer.fitness import RelativisticFitnessEvaluator

__version__ = "1.0.1"

__all__ = [
    # Subpaquetes
    "contracts",
    "core",
    "simulator",
    "data",
    "optimizer",
    "validation",
    # Tipos y contratos
    "WarpParameterVector",
    "MetricStatePoint",
    "MetricEvaluation",
    "ScientificConventions",
    "SimulationRequest",
    "SimulationResult",
    "SimulationObservables",
    "SimulationStatus",
    "RouteSegment",
    "SkyCoordinates",
    "RiskBreakdown",
    "FitnessBreakdown",
    # Módulos operativos
    "BobrickMartireMetricCore",
    "RelativisticSimulator",
    "SimulatorConfig",
    "AstronomicalRiskProvider",
    "default_local_risk_provider",
    "VectorizedPSO",
    "OptimizationResult",
    "PSOConfig",
    "RelativisticFitnessEvaluator",
    "__version__",
]
