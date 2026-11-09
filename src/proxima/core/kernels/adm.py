"""Campos ADM de la clase BM (caso canonico: shift tipo Alcubierre).

Convencion de coordenadas internas: X = (x0, x, y, z) con x0 = c t, todo en metros.
Asi g_mu_nu es adimensional y el shift beta^i es adimensional (fraccion de c).
La conversion a SI (m/s) del shift se hace en el adaptador de contrato.

Campos:
  alpha      = max(1, lapse_floor)                (lapse)
  beta^x     = -A v f(r_s),  beta^y = beta^z = 0   (shift; f = "shift function")
  gamma_ij   = delta_ij                            (metrica espacial)
  x_s(x0)    = v x0                                (centro de burbuja sobre eje x)
"""

from __future__ import annotations

import jax.numpy as jnp

from ..params import I_A, I_LF, I_R, I_V, I_W
from .safety import floored_lapse, regularized_radius
from .shape import tanh_profile


def bubble_radius_coord(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    v = p[I_V]
    center = jnp.stack([v * X[0], 0.0, 0.0])
    return regularized_radius(X[1:] - center, eps_rel * p[I_R])


def shift_function(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    r_s = bubble_radius_coord(p, X, eps_rel)
    return tanh_profile(r_s, p[I_R], 1.0 / p[I_W])


def lapse(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    return floored_lapse(jnp.asarray(1.0, dtype=X.dtype), p[I_LF])


def shift(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    """beta^i (adimensional)."""
    f = shift_function(p, X, eps_rel)
    return jnp.stack([-p[I_A] * p[I_V] * f, 0.0 * f, 0.0 * f])


def spatial_metric(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    return jnp.eye(3, dtype=X.dtype)


def metric4(p: jnp.ndarray, X: jnp.ndarray, eps_rel: float) -> jnp.ndarray:
    """g_mu_nu (4x4, firma -+++) construida desde los campos ADM."""
    a = lapse(p, X, eps_rel)
    b_up = shift(p, X, eps_rel)
    gam = spatial_metric(p, X, eps_rel)
    b_dn = gam @ b_up
    g00 = -a * a + b_up @ b_dn
    top = jnp.concatenate([g00[None], b_dn])
    bottom = jnp.concatenate([b_dn[:, None], gam], axis=1)
    return jnp.concatenate([top[None, :], bottom], axis=0)
