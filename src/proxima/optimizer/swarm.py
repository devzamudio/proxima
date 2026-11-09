"""Estado del enjambre PSO y operadores vectorizados de actualizacion."""

from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp


class SwarmState(NamedTuple):
    """Estado inmutable del enjambre compatible con jax.jit y jax.lax.scan."""
    positions: jnp.ndarray        # (N, D)
    velocities: jnp.ndarray       # (N, D)
    pbest_positions: jnp.ndarray  # (N, D)
    pbest_fitness: jnp.ndarray    # (N,)
    gbest_position: jnp.ndarray   # (D,)
    gbest_fitness: jnp.ndarray    # ()
    iteration: jnp.ndarray        # () int32
    key: jax.random.PRNGKey       # PRNGKey


def init_swarm(
    key: jax.random.PRNGKey,
    n_particles: int,
    bounds: jnp.ndarray,
    v_clamp_ratio: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """Inicializa posiciones y velocidades dentro de los límites del dominio.

    bounds: (D, 2) con [lower_bound, upper_bound]
    """
    k_pos, k_vel, key_next = jax.random.split(key, 3)
    dim = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]
    span = high - low

    # Posiciones uniformes en [low, high]
    positions = low + jax.random.uniform(k_pos, (n_particles, dim)) * span

    # Velocidades uniformes en [-v_max, v_max]
    v_max = v_clamp_ratio * span
    velocities = (jax.random.uniform(k_vel, (n_particles, dim)) * 2.0 - 1.0) * v_max

    return positions, velocities, key_next


@jax.jit
def update_particle_dynamics(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    pbest_positions: jnp.ndarray,
    attractor_positions: jnp.ndarray,
    bounds: jnp.ndarray,
    w: float,
    c1: float,
    c2: float,
    v_clamp_ratio: float,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """Actualiza la velocidad y posición de todas las partículas en lote con jax.vmap/vectorizado.

    v_new = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (attractor - x)
    x_new = clamp(x + v_new, bounds)
    """
    n_particles, dim = positions.shape
    low = bounds[:, 0]
    high = bounds[:, 1]
    v_max = v_clamp_ratio * (high - low)

    k1, k2, key_next = jax.random.split(key, 3)
    r1 = jax.random.uniform(k1, (n_particles, dim))
    r2 = jax.random.uniform(k2, (n_particles, dim))

    # Actualizar velocidad
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (attractor_positions - positions)
    new_v = w * velocities + cognitive + social

    # Clamping de velocidad
    new_v_clamped = jnp.clip(new_v, -v_max, v_max)

    # Actualizar posición y clamping de dominio
    new_positions = jnp.clip(positions + new_v_clamped, low, high)

    return new_positions, new_v_clamped, key_next


@jax.jit
def update_pbest(
    current_positions: jnp.ndarray,
    current_fitness: jnp.ndarray,
    pbest_positions: jnp.ndarray,
    pbest_fitness: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Actualiza la mejor posición histórica de cada partícula si current < pbest."""
    improved = (current_fitness < pbest_fitness)[:, None]  # (N, 1)
    new_pbest_pos = jnp.where(improved, current_positions, pbest_positions)
    new_pbest_fit = jnp.where(current_fitness < pbest_fitness, current_fitness, pbest_fitness)
    return new_pbest_pos, new_pbest_fit
