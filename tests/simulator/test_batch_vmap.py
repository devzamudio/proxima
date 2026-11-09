"""Test de batch vmap: el lote debe ser igual a ejecuciones individuales y determinista."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.simulator.batch import simulate_batch
from proxima.simulator.config import SimulatorConfig


def test_batch_deterministic():
    """Dos ejecuciones del mismo lote deben dar el mismo resultado."""
    cfg = SimulatorConfig.default_config()
    P = jnp.asarray([
        [100.0, 5.0, 0.1, 1.0, 0.01],
        [100.0, 5.0, 0.3, 1.0, 0.01],
    ], dtype=jnp.float64)

    # Estados iniciales: reposo en el origen (X + u empaquetados)
    X0_list = jnp.asarray([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)

    result1 = simulate_batch(P, X0_list, cfg)
    result2 = simulate_batch(P, X0_list, cfg)

    np.testing.assert_array_equal(result1["delta_tau"], result2["delta_tau"])
    np.testing.assert_array_equal(result1["term_code"], result2["term_code"])


def test_batch_output_shapes():
    """Los arrays de salida deben tener la forma correcta."""
    cfg = SimulatorConfig.default_config()
    P = jnp.asarray([
        [100.0, 5.0, 0.1, 1.0, 0.01],
        [100.0, 5.0, 0.3, 1.0, 0.01],
        [100.0, 5.0, 0.5, 1.0, 0.01],
    ], dtype=jnp.float64)

    X0_list = jnp.zeros((3, 8), dtype=jnp.float64)
    X0_list = X0_list.at[:, 4].set(1.0)  # u^0 = 1 (reposo)

    result = simulate_batch(P, X0_list, cfg)

    n = P.shape[0]
    assert result["delta_tau"].shape == (n,)
    assert result["delta_t"].shape == (n,)
    assert result["tidal_max"].shape == (n,)
    assert result["term_code"].shape == (n,)
    assert result["success"].shape == (n,)
    assert result["nec_margin"].shape == (n,)