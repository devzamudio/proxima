"""Test del tensor de marea."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.core import params_to_array
from proxima.core.kernels import adm, geometry
from proxima.simulator.tidal import tidal_tensor, max_tidal_stress
from proxima.simulator.state import pack_state, normalize_4velocity

C = 299792458.0
G = 6.67430e-11


def test_minkowski_tidal_is_zero():
    """En Minkowski, el tensor de Riemann es cero, por lo que la marea es cero."""
    params = jnp.asarray([100.0, 5.0, 0.0, 0.0, 0.01], dtype=jnp.float64)
    X = jnp.asarray([0.0, 10.0, 20.0, 30.0], dtype=jnp.float64)
    u = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    eps_rel = 1.0e-9

    R = geometry.riemann(params, X, eps_rel)
    # En Minkowski, R deberia ser ~0
    np.testing.assert_allclose(R, 0.0, atol=1e-10)

    E = tidal_tensor(u, X, params, eps_rel)
    np.testing.assert_allclose(E, 0.0, atol=1e-10)

    stress = max_tidal_stress(E)
    assert abs(stress) < 1e-10


def test_tidal_tensor_shape_and_symmetry():
    """La matriz de marea debe ser 3x3 y simetrica."""
    params = jnp.asarray([100.0, 5.0, 0.5, 1.0, 0.01], dtype=jnp.float64)
    X = jnp.asarray([0.0, 50.0, 0.0, 0.0], dtype=jnp.float64)
    u = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    eps_rel = 1.0e-9

    E = tidal_tensor(u, X, params, eps_rel)
    assert E.shape == (3, 3)
    # Simetrica (en teoria)
    np.testing.assert_allclose(E, E.T, rtol=1e-6)