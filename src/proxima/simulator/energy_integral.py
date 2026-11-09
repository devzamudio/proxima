"""Energia total Euleriana y energia negativa por cuadratriz vectorizada 3D.

E  = int rho * sqrt(gamma) d^3x   sobre hipersuperficie t = cte
E_- = int min(rho, 0) d^3x         (solo contribucion negativa)

La malla se centra en la burbuja con extension configurable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxima.core.batch import evaluate_point


def _build_grid(n: int, extent: float, center: jnp.ndarray, bubble_radius: float):
    """Construye malla 3D centrada en la burbuja.

    Retorna: Xs (n^3, 4) con x0=center[0], pesos (n^3,)
    """
    half = extent * bubble_radius
    c_arr = jnp.asarray(center, dtype=jnp.float64)
    if c_arr.shape[0] == 4:
        x0_val = c_arr[0]
        cx, cy, cz = c_arr[1], c_arr[2], c_arr[3]
    else:
        x0_val = jnp.asarray(0.0, dtype=jnp.float64)
        cx, cy, cz = c_arr[0], c_arr[1], c_arr[2]

    xs = jnp.linspace(cx - half, cx + half, n)
    ys = jnp.linspace(cy - half, cy + half, n)
    zs = jnp.linspace(cz - half, cz + half, n)

    grid = jnp.meshgrid(xs, ys, zs, indexing="ij")
    pts = jnp.stack([g.ravel() for g in grid], axis=1)  # (n^3, 3)

    x0_col = jnp.full((pts.shape[0], 1), x0_val, dtype=jnp.float64)
    Xs = jnp.concatenate([x0_col, pts], axis=1)  # (n^3, 4)

    dx = xs[1] - xs[0] if n > 1 else 2.0 * half
    cell_vol = dx**3
    pesos = jnp.full((pts.shape[0],), cell_vol, dtype=jnp.float64)

    return Xs, pesos


def total_energy(
    p: jnp.ndarray,
    Xs: jnp.ndarray,
    pesos: jnp.ndarray,
    eps_rel: float,
    c: float,
    G_newton: float,
) -> tuple[float, float, jnp.ndarray]:
    """Calcula E total y E_- en la malla.

    Retorna (E_total_J, E_negative_J, rho_values)
    """
    bounds = jnp.zeros((5, 2))  # domain check desactivado para malla de energia

    def eval_fn(X):
        out = evaluate_point(p, X, bounds, eps_rel, c, G_newton)
        det_gamma = jnp.linalg.det(out["spatial_metric"])
        return out["rho_eulerian_J_m3"], jnp.sqrt(jnp.maximum(det_gamma, 0.0))

    rho_vals, sqrt_gamma = jax.vmap(eval_fn)(Xs)

    E_total = jnp.sum(rho_vals * sqrt_gamma * pesos)
    E_negative = jnp.sum(jnp.minimum(rho_vals, 0.0) * sqrt_gamma * pesos)

    return float(E_total), float(E_negative), rho_vals


def total_energy_fast(
    p: jnp.ndarray,
    Xs: jnp.ndarray,
    pesos: jnp.ndarray,
    eps_rel: float,
    c: float,
    G_newton: float,
    max_points: int = 64,
) -> tuple[float, float]:
    """Version rapida: subsempla la malla para evitar compilacion costosa."""
    n = Xs.shape[0]
    if n <= max_points:
        return total_energy(p, Xs, pesos, eps_rel, c, G_newton)[:2]

    # Subsemplar uniformemente
    idx = jnp.linspace(0, n - 1, max_points).astype(jnp.int32)
    Xs_sub = Xs[idx]
    pesos_sub = pesos[idx]

    return total_energy(p, Xs_sub, pesos_sub, eps_rel, c, G_newton)[:2]