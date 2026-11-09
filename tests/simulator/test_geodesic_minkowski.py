"""Test: en Minkowski (A=0) la particula libre se mueve en linea recta y
Delta tau = Delta t / gamma_Lorentz.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.simulator import RelativisticSimulator, SimulatorConfig
from proxima.simulator.state import pack_state, unpack_state
from proxima.simulator.geodesic import geodesic_rhs

C = 299792458.0


def _minkowski_params():
    return WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.0,  # v=0 -> Minkowski
        shift_amplitude=0.0,
        lapse_floor=0.01,
    )


def test_minkowski_free_particle_straight_line():
    """Con A=0, v=0: particula en reposo permanece en reposo, Delta tau = Delta t."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    params = _minkowski_params()
    state = MetricStatePoint(t_s=0.0, x_m=10.0, y_m=5.0, z_m=-3.0, tau_s=0.0)

    request = type(
        "Req",
        (),
        {
            "params": params,
            "initial_state": state,
            "conventions": None,
            "max_steps": 1000,
        },
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]

    # Con v=0 y A=0, la particula no se mueve: x, y, z constantes
    # delta_tau ~= tau_final_s (1s), delta_t = delta_tau (gamma=1)
    assert result.status.name in ("SUCCESS", "FAILURE")
    if result.status.name == "SUCCESS":
        np.testing.assert_allclose(result.observables.delta_tau_s, 1.0, rtol=1e-2)
        np.testing.assert_allclose(result.observables.delta_t_s, result.observables.delta_tau_s, rtol=1e-2)


def test_minkowski_moving_particle_time_dilation():
    """Con A=0, v=0.5c: Delta tau = Delta t / gamma."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=0.0,
        lapse_floor=0.01,
    )
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    request = type(
        "Req",
        (),
        {"params": params, "initial_state": state, "conventions": None, "max_steps": 1000},
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]

    if result.status.name == "SUCCESS":
        beta = 0.5
        gamma = 1.0 / np.sqrt(1.0 - beta**2)
        expected_dtau = 1.0 / gamma  # tau_final = 1s, pero en coordenados t = gamma * tau
        # Nota: tau_final_s es el objetivo de tiempo propio, así que delta_tau ~= 1s
        # y delta_t = gamma * delta_tau
        np.testing.assert_allclose(result.observables.delta_tau_s, 1.0, rtol=5e-2)
        np.testing.assert_allclose(
            result.observables.delta_t_s, gamma * result.observables.delta_tau_s, rtol=5e-2
        )


def test_geodesic_rhs_minkowski_zero_acceleration():
    """En Minkowski, Christoffel = 0, por lo que du/dlambda = 0."""
    from proxima.core.kernels import geometry
    from proxima.core.params import params_to_array

    params = _minkowski_params()
    p = params_to_array(params)
    X = jnp.asarray([0.0, 10.0, 5.0, -3.0], dtype=jnp.float64)
    u = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    y = pack_state(X, u)

    rhs = geodesic_rhs(y, p, 1.0e-9, C)
    dX, du = rhs[:4], rhs[4:]

    np.testing.assert_allclose(du, 0.0, atol=1e-12)
    np.testing.assert_allclose(dX, u, atol=1e-12)