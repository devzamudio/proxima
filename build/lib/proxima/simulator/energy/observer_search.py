"""Busqueda de observadores para tipos II-IV de Hawking-Ellis.

Optimiza sobre rapidez y direccion para encontrar un observador
que viola las condiciones de energia (cota conservadora).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _boost_matrix(rapidity: float, direction: jnp.ndarray) -> jnp.ndarray:
    """Matrix de Lorentz boost en 4D (firma -+++)."""
    beta = jnp.tanh(rapidity)
    gamma = 1.0 / jnp.sqrt(1.0 - beta * beta)
    n = direction / jnp.linalg.norm(direction)

    Lambda = jnp.eye(4, dtype=jnp.float64)
    # Boost en direccion n
    Lambda = Lambda.at[0, 0].set(gamma)
    Lambda = Lambda.at[0, 1:].set(-gamma * beta * n)
    Lambda = Lambda.at[1:, 0].set(-gamma * beta * n)
    spatial = jnp.eye(3, dtype=jnp.float64) + (gamma - 1.0) * jnp.outer(n, n)
    Lambda = Lambda.at[1:, 1:].set(spatial)
    return Lambda


def search_observer_violation(T_mn: jnp.ndarray, g: jnp.ndarray, n_samples: int = 32) -> dict:
    """Busca un observador (boost) que maximice la violacion de NEC.

    Para tipos II-IV, la violacion de NEC es equivalente a existir un
    vector nulo k con T_{ab} k^a k^b < 0. Parametrizamos k via boost
    sobre un vector nulo de referencia.

    Retorna dict con mejor rapidez, direccion y NEC marginal.
    """
    # Vector nulo de referencia: k = (1, 1, 0, 0)
    k0 = jnp.asarray([1.0, 1.0, 0.0, 0.0])

    # Muestreo de direcciones espaciales y rapididades
    directions = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
         [1.0, 1.0, 1.0], [-1.0, 1.0, 0.0]]
    )
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    rapidities = jnp.linspace(0.0, 3.0, n_samples)

    best_nec = jnp.inf
    best_rapidity = 0.0
    best_dir = directions[0]

    for dir_idx in range(directions.shape[0]):
        n = directions[dir_idx]
        for rap_idx in range(rapidities.shape[0]):
            rap = rapidities[rap_idx]
            Lambda = _boost_matrix(rap, n)
            k = Lambda @ k0
            # Asegurar que la componente temporal esté normalizada a 1 para comparar escalas
            k0_abs = jnp.maximum(jnp.abs(k[0]), 1e-12)
            k = k / k0_abs

            nec_val = jnp.einsum("ij,i,j->", T_mn, k, k)
            is_better = nec_val < best_nec
            best_nec = jnp.where(is_better, nec_val, best_nec)
            best_rapidity = jnp.where(is_better, rap, best_rapidity)
            best_dir = jnp.where(is_better, n, best_dir)

    return {
        "nec_min": float(best_nec),
        "nec_violated": bool(best_nec < -1e-12),
        "best_rapidity": float(best_rapidity),
        "best_direction": [float(x) for x in best_dir],
    }