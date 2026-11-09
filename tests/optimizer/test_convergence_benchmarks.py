"""Pruebas de referencia de convergencia en funciones sinteticas multimodales y convexas."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from proxima.optimizer import VectorizedPSO, PSOConfig


# Funciones de benchmark clasicas
def sphere_fn(P: jnp.ndarray) -> jnp.ndarray:
    """Sphere: f(x) = sum(x^2), minimo global en 0 con f(0)=0."""
    return jnp.sum(P**2, axis=1)


def rosenbrock_fn(P: jnp.ndarray) -> jnp.ndarray:
    """Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2), minimo en 1 con f(1)=0."""
    return jnp.sum(100.0 * (P[:, 1:] - P[:, :-1]**2)**2 + (1.0 - P[:, :-1])**2, axis=1)


def rastrigin_fn(P: jnp.ndarray) -> jnp.ndarray:
    """Rastrigin: altamente multimodal, f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i)), minimo en 0."""
    d = P.shape[1]
    return 10.0 * d + jnp.sum(P**2 - 10.0 * jnp.cos(2.0 * jnp.pi * P), axis=1)


def test_pso_sphere_convergence():
    bounds = jnp.asarray([[-5.0, 5.0]] * 5, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)
    res = pso.optimize_function(sphere_fn, n_particles=30, n_iterations=40, seed=42)

    # Monotonicidad no creciente del mejor global
    history = res.history.gbest_fitness_history
    for i in range(len(history) - 1):
        assert history[i + 1] <= history[i] + 1e-12

    # Reducción drástica del error
    assert history[-1] < history[0] * 1e-3
    assert history[-1] < 0.05


def test_pso_rosenbrock_convergence():
    bounds = jnp.asarray([[-2.0, 2.0]] * 4, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)
    res = pso.optimize_function(rosenbrock_fn, n_particles=40, n_iterations=60, seed=10)

    history = res.history.gbest_fitness_history
    assert history[-1] < history[0]
    assert history[-1] < 1.0


def test_pso_rastrigin_multimodal():
    bounds = jnp.asarray([[-5.12, 5.12]] * 3, dtype=jnp.float64)
    pso = VectorizedPSO(bounds=bounds)
    res = pso.optimize_function(rastrigin_fn, n_particles=50, n_iterations=50, seed=99)

    history = res.history.gbest_fitness_history
    assert history[-1] < history[0]
    # Encuentra un minimo local o global muy cercano
    assert history[-1] < 5.0


def test_pso_ring_topology_convergence():
    bounds = jnp.asarray([[-5.0, 5.0]] * 5, dtype=jnp.float64)
    cfg = PSOConfig(topology="ring", ring_neighborhood_size=3)
    pso = VectorizedPSO(config=cfg, bounds=bounds)
    res = pso.optimize_function(sphere_fn, n_particles=30, n_iterations=40, seed=42)

    history = res.history.gbest_fitness_history
    assert history[-1] <= history[0]
    assert history[-1] < 0.1
