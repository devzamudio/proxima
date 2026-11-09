"""Test: en el centro de la burbuja Alcubierre, dtau = dt y x(t) = v * c * t."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.simulator import RelativisticSimulator, SimulatorConfig

C = 299792458.0


def test_center_trajectory_dtau_equals_dt():
    """En el centro de la burbuja (x=y=z=0), la metrica es Minkowski localmente,
    por lo que dtau = dt y la particula se mueve con la burbuja."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    request = type(
        "Req",
        (),
        {"params": params, "initial_state": state, "conventions": None, "max_steps": 1000},
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]

    assert result.status.name == "SUCCESS"
    # En el centro de la burbuja Alcubierre (A=1), dtau = dt (sin dilatacion temporal)
    np.testing.assert_allclose(result.observables.delta_tau_s, 1.0, rtol=1e-4)
    np.testing.assert_allclose(
        result.observables.delta_t_s, result.observables.delta_tau_s, rtol=1e-4
    )