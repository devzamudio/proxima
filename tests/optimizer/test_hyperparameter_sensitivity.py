"""Pruebas de sensibilidad a hiperparametros y reproducibilidad entre seeds."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from proxima.optimizer import VectorizedPSO, PSOConfig


def quad_fn(P: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((P - 1.0)**2, axis=1)


def test_seed_determinism():
    """Con la misma semilla, el proceso y la trayectoria de optimizacion deben ser exactamente identicos."""
    bounds = jnp.asarray([[-5.0, 5.0]] * 3, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)

    res1 = pso.optimize_function(quad_fn, seed=12345, n_particles=20, n_iterations=25)
    res2 = pso.optimize_function(quad_fn, seed=12345, n_particles=20, n_iterations=25)

    assert res1.best_fitness == pytest.approx(res2.best_fitness, rel=1e-12)
    assert res1.best_vector == pytest.approx(res2.best_vector, rel=1e-12)
    assert res1.history.gbest_fitness_history == pytest.approx(res2.history.gbest_fitness_history, rel=1e-12)


def test_seed_variability():
    """Distintas semillas exploran trayectorias diferentes pero ambas convergen."""
    bounds = jnp.asarray([[-5.0, 5.0]] * 3, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)

    res_a = pso.optimize_function(quad_fn, seed=1, n_particles=20, n_iterations=30)
    res_b = pso.optimize_function(quad_fn, seed=2, n_particles=20, n_iterations=30)

    # Ambos deben converger cerca del optimo f(1,1,1)=0
    assert res_a.best_fitness < 0.1
    assert res_b.best_fitness < 0.1
    # Pero las trayectorias iniciales o intermedias no son identicas
    assert res_a.history.gbest_fitness_history[0] != res_b.history.gbest_fitness_history[0]


def test_hyperparameter_sensitivity_inertia():
    """Inercia alta (w=0.9 constante) vs inercia con decremento lineal."""
    bounds = jnp.asarray([[-5.0, 5.0]] * 4, dtype=jnp.float64)
    
    cfg_static = PSOConfig(w_max=0.9, w_min=0.9)
    cfg_decay = PSOConfig(w_max=0.9, w_min=0.4)

    pso_static = VectorizedPSO(config=cfg_static, bounds=bounds)
    pso_decay = VectorizedPSO(config=cfg_decay, bounds=bounds)

    res_static = pso_static.optimize_function(quad_fn, seed=77, n_particles=25, n_iterations=35)
    res_decay = pso_decay.optimize_function(quad_fn, seed=77, n_particles=25, n_iterations=35)

    # El cronograma decreciente tipicamente afina mejor en las etapas finales
    assert res_decay.best_fitness < 0.05
    assert res_static.best_fitness < 0.2


def test_hyperparameter_particle_count_scaling():
    """Mas particulas aumentan la probabilidad de convergencia rapida en menos iteraciones."""
    bounds = jnp.asarray([[-5.0, 5.0]] * 5, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)

    res_small = pso.optimize_function(quad_fn, seed=42, n_particles=5, n_iterations=15)
    res_large = pso.optimize_function(quad_fn, seed=42, n_particles=40, n_iterations=15)

    assert res_large.best_fitness <= res_small.best_fitness
