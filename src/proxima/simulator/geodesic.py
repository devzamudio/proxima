"""Ecuacion de geodesica: dX/dlambda = u, du/dlambda = -Gamma u u.

Usa el nucleo metrico de Fase 1 para las conexiones Christoffel.
"""

from __future__ import annotations

import jax.numpy as jnp

from proxima.core.kernels import geometry


def geodesic_rhs(
    y: jnp.ndarray,
    p: jnp.ndarray,
    eps_rel: float,
    c: float,
) -> jnp.ndarray:
    """Lado derecho del sistema de geodesicas en parametro lambda = c*tau.

    y = (X^0..3, u^0..3) con X en metros, u adimensional.
    Retorna dy/dlambda con la misma estructura.
    """
    X, u = y[:4], y[4:]

    # Conexiones Christoffel Gamma^mu_{alpha beta} (shape 4,4,4)
    Gamma = geometry.christoffel(p, X, eps_rel)

    # du^mu/dlambda = - Gamma^mu_{alpha beta} u^alpha u^beta
    du = -jnp.einsum("mab,a,b->m", Gamma, u, u)

    # dX^mu/dlambda = u^mu
    dX = u

    return jnp.concatenate([dX, du])