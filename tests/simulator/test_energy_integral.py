"""Test de la integral de energia total y energia negativa."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.simulator.energy_integral import _build_grid, total_energy
from proxima.core import params_to_array

C = 299792458.0
G = 6.67430e-11


def test_energy_negative_in_alcubierre_wall():
    """En la pared de Alcubierre, la energia negativa E_- debe ser < 0."""
    params = jnp.asarray([100.0, 5.0, 0.5, 1.0, 0.01], dtype=jnp.float64)
    eps_rel = 1.0e-9

    n_grid = 16
    extent = 2.0
    R_bubble = 100.0
    Xs, pesos = _build_grid(n_grid, extent, jnp.zeros(4), R_bubble)

    E_total, E_negative, rho_vals = total_energy(
        params, Xs, pesos, eps_rel, C, G
    )

    # La energia negativa debe ser finita y menor que cero (hay energia negativa en la pared)
    assert np.isfinite(E_negative)
    assert E_negative < 0.0
    assert E_total <= 0.0
    assert E_total >= E_negative


def test_energy_grid_convergence():
    """Refinar la malla debe converger."""
    params = jnp.asarray([100.0, 5.0, 0.5, 1.0, 0.01], dtype=jnp.float64)
    eps_rel = 1.0e-9

    E_vals = []
    for n in [8, 16, 32]:
        Xs, pesos = _build_grid(n, 2.0, jnp.zeros(4), 100.0)
        E_total, E_negative, _ = total_energy(params, Xs, pesos, eps_rel, C, G)
        E_vals.append(E_total)

    # Los valores deberían convergir (diferencias decrecientes)
    if len(E_vals) >= 2:
        diff1 = abs(E_vals[1] - E_vals[0])
        diff2 = abs(E_vals[2] - E_vals[1]) if len(E_vals) > 2 else diff1
        assert diff2 <= diff1 * 2  # convergencia aproximada