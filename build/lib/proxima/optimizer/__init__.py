"""Optimizador PSO vectorizado (Fase 4)."""

from __future__ import annotations

from .config import PSOConfig
from .fitness import RelativisticFitnessEvaluator, jax_fitness_breakdown
from .topology import get_global_best, get_ring_best
from .swarm import SwarmState, init_swarm, update_particle_dynamics, update_pbest
from .optimizer import VectorizedPSO, OptimizationResult, OptimizationHistory

__all__ = [
    "PSOConfig",
    "RelativisticFitnessEvaluator",
    "jax_fitness_breakdown",
    "get_global_best",
    "get_ring_best",
    "SwarmState",
    "init_swarm",
    "update_particle_dynamics",
    "update_pbest",
    "VectorizedPSO",
    "OptimizationResult",
    "OptimizationHistory",
]
