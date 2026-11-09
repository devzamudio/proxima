"""Test de guardas y codigos de falla."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.core.params import load_core_config
from proxima.simulator import RelativisticSimulator, SimulatorConfig


def test_domain_error_invalid_params():
    """Parametros fuera de dominio deben producir DOMAIN."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    params = WarpParameterVector(
        bubble_radius_m=-1.0,  # invalido
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    request = type(
        "Req",
        (),
        {"params": params, "initial_state": state, "conventions": None, "max_steps": 100},
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]
    assert result.status.name == "FAILURE"
    assert result.failure_reason is not None
    assert "DOMAIN" in result.failure_reason or "domain" in result.failure_reason.lower()


def test_max_steps_failure():
    """Con muy pocos pasos, se alcanza MAX_STEPS."""
    sim = RelativisticSimulator(
        config=SimulatorConfig(
            h0=1e-3, h_min=1e-12, h_max=1e-1,
            atol=1e-10, rtol=1e-8, norm_tol=1e-6,
            tidal_limit_per_s2=9.81, tau_final_s=100.0,  # muy largo
            energy_n=16, energy_extent_rel=2.0,
            energy_backend="own",
            wec_tolerance=1e-10, nec_tolerance=1e-10,
            dec_tolerance=1e-10, sec_tolerance=1e-10,
        )
    )
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.1,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    request = type(
        "Req",
        (),
        {"params": params, "initial_state": state, "conventions": None, "max_steps": 3},  # muy pocos pasos
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]
    # Con solo 3 pasos permitidos, debe fallar por MAX_STEPS
    assert result.status.name == "FAILURE"
    assert result.failure_reason is not None
    assert "MAX_STEPS" in result.failure_reason


def test_tidal_limit_guard():
    """Si la marea en la pared supera tidal_limit_per_s2, falla con TIDAL_LIMIT."""
    cfg = SimulatorConfig(
        h0=1e-3, h_min=1e-12, h_max=1e-2,
        atol=1e-8, rtol=1e-6, norm_tol=1e-3,
        tidal_limit_per_s2=1e-12,  # umbral muy estricto
        tau_final_s=0.01,
        energy_n=8, energy_extent_rel=2.0,
        energy_backend="own",
        wec_tolerance=1e-10, nec_tolerance=1e-10,
        dec_tolerance=1e-10, sec_tolerance=1e-10,
    )
    sim = RelativisticSimulator(config=cfg)
    params = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    # Punto inicial en la pared de la burbuja donde la curvatura no es nula
    state = MetricStatePoint(t_s=0.0, x_m=100.0, y_m=10.0, z_m=0.0, tau_s=0.0)
    request = type(
        "Req",
        (),
        {"params": params, "initial_state": state, "conventions": None, "max_steps": 100},
    )()

    result = sim.simulate(request)  # type: ignore[arg-type]
    assert result.status.name == "FAILURE"
    assert result.failure_reason is not None
    assert "TIDAL_LIMIT" in result.failure_reason


def test_superluminal_guard_and_flag():
    """Velocidad superluminal requiere allow_superluminal=True."""
    sim = RelativisticSimulator(config=SimulatorConfig.default_config())
    state = MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)

    p_no_flag = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=2.0,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    req1 = type("Req", (), {"params": p_no_flag, "initial_state": state, "conventions": None, "max_steps": 100})()
    res1 = sim.simulate(req1)  # type: ignore[arg-type]
    assert res1.status.name == "FAILURE"
    assert "DOMAIN" in (res1.failure_reason or "")

    p_with_flag = WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=2.0,
        shift_amplitude=1.0,
        lapse_floor=0.01,
        metadata={"allow_superluminal": True},
    )
    req2 = type("Req", (), {"params": p_with_flag, "initial_state": state, "conventions": None, "max_steps": 100})()
    res2 = sim.simulate(req2)  # type: ignore[arg-type]
    assert res2.status.name == "SUCCESS"