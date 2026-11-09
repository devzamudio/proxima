"""Guardas de seguridad numerica (puras JAX)."""

from __future__ import annotations

import jax.numpy as jnp


def regularized_radius(dx: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    """|dx| regularizado: sqrt(|dx|^2 + eps^2). Evita gradientes NaN en r -> 0."""
    return jnp.sqrt(jnp.sum(dx * dx) + eps * eps)


def floored_lapse(alpha_raw: jnp.ndarray, floor: jnp.ndarray) -> jnp.ndarray:
    """Impone alpha >= lapse_floor (evita colapso del lapse)."""
    return jnp.maximum(alpha_raw, floor)


def all_finite(*arrays: jnp.ndarray) -> jnp.ndarray:
    ok = jnp.asarray(True)
    for a in arrays:
        ok = ok & jnp.all(jnp.isfinite(a))
    return ok


def spatial_metric_ok(gamma: jnp.ndarray) -> jnp.ndarray:
    """gamma_ij debe ser definida positiva (criterio de Sylvester)."""
    m1 = gamma[0, 0]
    m2 = gamma[0, 0] * gamma[1, 1] - gamma[0, 1] * gamma[1, 0]
    m3 = jnp.linalg.det(gamma)
    return (m1 > 0) & (m2 > 0) & (m3 > 0)
