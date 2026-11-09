"""Funciones de forma de la burbuja warp."""

from __future__ import annotations

import jax.numpy as jnp


def tanh_profile(r: jnp.ndarray, radius: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Perfil top-hat suave de Alcubierre.

    f(r) = [tanh(sigma (r + R)) - tanh(sigma (r - R))] / (2 tanh(sigma R)).
    f(0) = 1 ; f -> 0 para r >> R. Adimensional.
    """
    return (jnp.tanh(sigma * (r + radius)) - jnp.tanh(sigma * (r - radius))) / (
        2.0 * jnp.tanh(sigma * radius)
    )
