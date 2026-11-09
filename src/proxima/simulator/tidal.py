"""Matriz de marea en el marco ortonormal de la nave.

E_ij = R_{mu alpha nu beta} u^mu u^nu e_i^alpha e_j^beta
"""

from __future__ import annotations

import jax.numpy as jnp

from proxima.core.kernels import adm, geometry


def _spatial_triad(u, X, p, eps_rel):
    """3 vectores espaciales ortonormales ortogonales a u (firma -+++).

    Retorna array (3, 4) con e_i^alpha.
    """
    g = adm.metric4(p, X, eps_rel)
    cand = jnp.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    u_lower = jnp.einsum("ij,j->i", g, u)
    guv = jnp.einsum("m,im->i", u_lower, cand)
    e_unnorm = cand + guv[:, None] * u[None, :]

    def g_inner(a, b):
        return jnp.einsum("ij,i,j->", g, a, b)

    e0 = e_unnorm[0]
    n0 = jnp.sqrt(jnp.abs(g_inner(e0, e0)))
    e0 = e0 / n0
    e1_raw = e_unnorm[1] - g_inner(e_unnorm[1], e0) * e0
    n1 = jnp.sqrt(jnp.abs(g_inner(e1_raw, e1_raw)))
    e1 = e1_raw / n1
    e2_raw = e_unnorm[2] - g_inner(e_unnorm[2], e0) * e0 - g_inner(e_unnorm[2], e1) * e1
    n2 = jnp.sqrt(jnp.abs(g_inner(e2_raw, e2_raw)))
    e2 = e2_raw / n2
    return jnp.stack([e0, e1, e2])


def tidal_tensor(u, X, p, eps_rel):
    """E_ij = R_{mu alpha nu beta} u^mu u^nu e_i^alpha e_j^beta.

    Retorna matriz (3, 3) en 1/s^2.
    """
    R = geometry.riemann(p, X, eps_rel)
    g = adm.metric4(p, X, eps_rel)
    R_cov = jnp.einsum("rl,lsmu->rsmu", g, R)
    e = _spatial_triad(u, X, p, eps_rel)
    E = jnp.einsum("manb,m,n,ia,jb->ij", R_cov, u, u, e, e)
    return 0.5 * (E + E.T)


def max_tidal_stress(E: jnp.ndarray) -> jnp.ndarray:
    """Valor maximo absoluto de los autovalores de E (1/s^2)."""
    E_sym = 0.5 * (E + E.T)
    evals = jnp.linalg.eigvalsh(E_sym)
    return jnp.max(jnp.abs(evals))