"""Simulador relativista Fase 2 — integración de geodésicas en JAX puro."""

from __future__ import annotations

from .config import SimulatorConfig
from .state import pack_state, unpack_state, normalize_4velocity
from .geodesic import geodesic_rhs
from .integrator import integrate_geodesic
from .tidal import tidal_tensor, max_tidal_stress
from .energy_integral import total_energy
from .risk import AnalyticRiskProvider, RiskAssessor, RouteSegmentRisk
from .simulator import RelativisticSimulator
from .batch import simulate_batch

__all__ = [
    "SimulatorConfig",
    "pack_state",
    "unpack_state",
    "normalize_4velocity",
    "geodesic_rhs",
    "integrate_geodesic",
    "tidal_tensor",
    "max_tidal_stress",
    "total_energy",
    "AnalyticRiskProvider",
    "RiskAssessor",
    "RouteSegmentRisk",
    "RelativisticSimulator",
    "simulate_batch",
]