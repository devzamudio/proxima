"""Empaquetado / desempaquetado del estado 8D y normalizacion de 4-velocidad.

Estado: y = (X^0, X^1, X^2, X^3, u^0, u^1, u^2, u^3)
  X^mu en metros (x0 = c*t), u^mu = dX^mu/dlambda adimensional (lambda = c*tau)
  Normalizacion: g_mu_nu u^mu u^nu = -1
"""

from __future__ import annotations

import jax.numpy as jnp


def pack_state(X: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Concatena X (4,) y u (4,) en estado (8,)."""
    return jnp.concatenate([X, u])


def unpack_state(y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Separa estado (8,) en X (4,) y u (4,)."""
    return y[:4], y[4:]


def normalize_4velocity(
    u: jnp.ndarray, g: jnp.ndarray, c: float = 299792458.0
) -> jnp.ndarray:
    """Normaliza u de modo que g_mu_nu u^mu u^nu = -1 (adimensional).

    Dado que X está en metros y lambda = c*tau en metros,
    u = dX/dlambda es adimensional y u.u = -1.
    """
    norm_sq = jnp.einsum("ij,i,j->", g, u, u)
    # norm_sq debe ser -1; dividimos por sqrt(-norm_sq)
    denom = jnp.sqrt(jnp.maximum(-norm_sq, 1e-30))
    return u / denom