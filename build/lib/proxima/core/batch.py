"""API vectorizada del nucleo (consumida por simulador y PSO).

evaluate_point(p, X, bounds, eps_rel, c, G) -> dict de arrays
evaluate_batch: jit(vmap particulas x vmap puntos).

Errores como mascaras (valid_domain, finite_ok) para no romper lotes jit/vmap.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .kernels import adm, geometry, safety
from .params import domain_mask


def evaluate_point(p, X, bounds, eps_rel, c, G_newton):
    """Evalua el nucleo en un punto X=(ct,x,y,z)[m] para parametros p (canonicos)."""
    alpha = adm.lapse(p, X, eps_rel)
    beta = adm.shift(p, X, eps_rel)
    gamma = adm.spatial_metric(p, X, eps_rel)
    g4 = adm.metric4(p, X, eps_rel)
    f = adm.shift_function(p, X, eps_rel)
    r_s = adm.bubble_radius_coord(p, X, eps_rel)

    Gmn, ricci_scalar = geometry.einstein_tensor(p, X, eps_rel)
    T = (c**4 / (8.0 * jnp.pi * G_newton)) * Gmn
    n = geometry.eulerian_normal(p, X, eps_rel)
    rho = jnp.einsum("m,n,mn->", n, n, T)
    _, trK, KK = geometry.extrinsic_curvature(p, X, eps_rel)

    valid = domain_mask(p, bounds) & safety.spatial_metric_ok(gamma) & (jnp.linalg.det(g4) < 0)
    finite = safety.all_finite(alpha, beta, gamma, g4, T, rho, trK, KK, ricci_scalar)
    return {
        "lapse": alpha,
        "shift": beta,
        "spatial_metric": gamma,
        "metric4": g4,
        "shift_function": f,
        "r_s_m": r_s,
        "ricci_scalar_per_m2": ricci_scalar,
        "einstein_tensor_per_m2": Gmn,
        "stress_energy_J_m3": T,
        "rho_eulerian_J_m3": rho,
        "trace_K_per_m": trK,
        "KijKij_per_m2": KK,
        "valid_domain": valid,
        "finite_ok": finite,
    }


@partial(jax.jit, static_argnames=("eps_rel", "c", "G_newton"))
def evaluate_batch(P, Xs, bounds, eps_rel, c, G_newton):
    """P: (n_particulas, 5), Xs: (n_puntos, 4) -> dict con ejes (n_particulas, n_puntos, ...)."""
    per_point = jax.vmap(evaluate_point, in_axes=(None, 0, None, None, None, None))
    per_particle = jax.vmap(per_point, in_axes=(0, None, None, None, None, None))
    return per_particle(P, Xs, bounds, eps_rel, c, G_newton)


evaluate_point_jit = jax.jit(evaluate_point, static_argnames=("eps_rel", "c", "G_newton"))
