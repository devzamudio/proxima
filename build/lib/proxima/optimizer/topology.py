"""Topologias de comunicacion para PSO (Global Star y Local Ring)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def get_global_best(
    pbest_positions: jnp.ndarray,
    pbest_fitness: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Topologia Global (Star): cada particula es guiada por la mejor global del enjambre.

    pbest_positions: (N, D)
    pbest_fitness: (N,)
    Retorna: (best_positions_for_each_particle: (N, D), best_fitness_for_each_particle: (N,))
    """
    gbest_idx = jnp.argmin(pbest_fitness)
    gbest_pos = pbest_positions[gbest_idx]      # (D,)
    gbest_fit = pbest_fitness[gbest_idx]        # ()
    
    n_particles = pbest_positions.shape[0]
    rep_positions = jnp.repeat(gbest_pos[None, :], n_particles, axis=0)  # (N, D)
    rep_fitness = jnp.repeat(gbest_fit[None], n_particles, axis=0)       # (N,)
    return rep_positions, rep_fitness


def get_ring_best(
    pbest_positions: jnp.ndarray,
    pbest_fitness: jnp.ndarray,
    neighborhood_size: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Topologia Local (Ring): cada particula i interactua solo con sus vecinos inmediatos en el anillo.

    Para neighborhood_size = 3, los vecinos de i son i-1, i, i+1 (con condiciones de frontera periodicas).
    """
    n_particles = pbest_positions.shape[0]
    radius = neighborhood_size // 2
    
    # Matriz de indices de vecindad circular (N, neighborhood_size)
    base_idx = jnp.arange(n_particles)[:, None]
    offsets = jnp.arange(-radius, radius + 1)[None, :]
    neighbor_indices = (base_idx + offsets) % n_particles

    # Obtener fitness de vecinos (N, neighborhood_size)
    neighbor_fit = pbest_fitness[neighbor_indices]
    
    # Indice local del mejor vecino (N,)
    best_local_idx = jnp.argmin(neighbor_fit, axis=1)
    
    # Indice global del mejor vecino para cada particula (N,)
    best_global_neighbor = jnp.take_along_axis(neighbor_indices, best_local_idx[:, None], axis=1).squeeze(axis=1)

    lbest_positions = pbest_positions[best_global_neighbor]
    lbest_fitness = pbest_fitness[best_global_neighbor]
    return lbest_positions, lbest_fitness
