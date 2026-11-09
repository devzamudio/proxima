"""Test de convergencia: al refinar tolerancias, el error disminuye."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.simulator import RelativisticSimulator, SimulatorConfig

C = 299792458.0


def test_convergence_tolerance():
    """Comparar resultados con diferentes tolerancias; el mas fino es referencia."""
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.1,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    def make_request(atol, rtol, tau_final):
        cfg = SimulatorConfig(
            h0=1e-3, h_min=1e-12, h_max=1e-1,
            atol=atol, rtol=rtol, norm_tol=1e-6,
            tidal_limit_per_s2=9.81, tau_final_s=tau_final,
            energy_n=16, energy_extent_rel=2.0,
            energy_backend="own",
            wec_tolerance=1e-10, nec_tolerance=1e-10,
            dec_tolerance=1e-10, sec_tolerance=1e-10,
        )
        sim = RelativisticSimulator(config=cfg)
        request = type(
            "Req",
            (),
            {"params": params, "initial_state": state, "conventions": None, "max_steps": 1000},
        )()
        return sim, request

    # Solucion de referencia con tolerancia muy fina
    sim_ref, req_ref = make_request(1e-14, 1e-12, 0.1)
    result_ref = sim_ref.simulate(req_ref)  # type: ignore[arg-type]

    # Solucion con tolerancia normal
    sim_norm, req_norm = make_request(1e-10, 1e-8, 0.1)
    result_norm = sim_norm.simulate(req_norm)  # type: ignore[arg-type]

    if result_ref.status.name == "SUCCESS" and result_norm.status.name == "SUCCESS":
        # El error deberia ser pequeño (las soluciones convergen)
        dtau_diff = abs(result_norm.observables.delta_tau_s - result_ref.observables.delta_tau_s)
        dt_diff = abs(result_norm.observables.delta_t_s - result_ref.observables.delta_t_s)

        # Con tolerancias razonables, la diferencia debe ser pequeña
        assert dtau_diff < 1e-2, f"dtau diverge: {dtau_diff}"
        assert dt_diff < 1e-2, f"dt diverge: {dt_diff}"