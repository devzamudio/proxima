"""Test: conservacion de u.u = -c^2 y E = -p_t en metrica estacionaria."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.simulator import RelativisticSimulator, SimulatorConfig

C = 299792458.0


def test_four_velocity_norm_preserved():
    """u.u = -1 (adimensional) se conserva a lo largo de la integracion."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.3,
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
    assert result.diagnostics.get("termination_code") == 0


def test_killing_energy_conserved_in_static_metric():
    """Cuando v=0 la metrica es estacionaria y E_killing = -g_{0 mu} u^mu se conserva."""
    from proxima.core.kernels import adm
    from proxima.core.params import params_to_array
    from proxima.simulator.geodesic import geodesic_rhs
    from proxima.simulator.integrator import integrate_geodesic
    from proxima.simulator.state import normalize_4velocity, pack_state

    cfg = SimulatorConfig.default_config()
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.0,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    p = params_to_array(params)
    eps_rel = 1.0e-9
    X0 = jnp.asarray([0.0, 20.0, 5.0, 0.0], dtype=jnp.float64)
    g0 = adm.metric4(p, X0, eps_rel)
    u0 = normalize_4velocity(jnp.asarray([1.0, 0.2, 0.0, 0.0], dtype=jnp.float64), g0, C)
    y0 = pack_state(X0, u0)

    res = integrate_geodesic(y0, p, eps_rel, C, cfg, f=geodesic_rhs, max_steps=500)
    assert res.termination_code == 0

    Xf, uf = res.y_final[:4], res.y_final[4:]
    gf = adm.metric4(p, Xf, eps_rel)

    E_init = -float(jnp.einsum("j,j->", g0[0], u0))
    E_final = -float(jnp.einsum("j,j->", gf[0], uf))
    np.testing.assert_allclose(E_final, E_init, rtol=1e-8)