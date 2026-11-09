"""Prueba de integración end-to-end: optimización de parámetros Warp acoplada a simulate_batch."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from proxima.optimizer import VectorizedPSO, PSOConfig
from proxima.core.params import PARAM_ORDER


def test_warp_pso_end_to_end_optimization():
    """Ejecuta una corrida real de PSO acoplada a la simulación relativista."""
    # Configuración de prueba ligera
    cfg = PSOConfig(
        n_particles=6,
        n_iterations=5,
        seed=101,
    )
    pso = VectorizedPSO(config=cfg)

    # Correr optimización
    result = pso.optimize_warp(
        seed=101,
        n_particles=6,
        n_iterations=5,
    )

    # Validaciones del resultado
    assert result.converged is True
    assert len(result.best_vector) == 5
    assert result.best_fitness > 0.0

    # Verificar que el vector óptimo esté dentro de los límites canónicos
    for i, name in enumerate(PARAM_ORDER):
        val = result.best_vector[i]
        low, high = float(pso.bounds[i, 0]), float(pso.bounds[i, 1])
        assert low <= val <= high, f"{name}={val} fuera de [{low}, {high}]"

    # Verificar monotonicidad del historial gbest
    history = result.history.gbest_fitness_history
    assert len(history) == 6  # inicial + 5 iteraciones
    for it in range(len(history) - 1):
        assert history[it + 1] <= history[it] + 1e-9
