"""Test de condiciones de energia WEC/NEC/DEC/SEC."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.contracts import EnergyCondition
from proxima.core import params_to_array
from proxima.core.batch import evaluate_point
from proxima.core.kernels import adm, geometry
from proxima.simulator.energy.hawking_ellis import classify_energy_conditions

C = 299792458.0
G = 6.67430e-11


def test_minkowski_all_energy_conditions():
    """En Minkowski, T_mn = 0, todas las condiciones se cumplen."""
    params = jnp.asarray([100.0, 5.0, 0.0, 0.0, 0.01], dtype=jnp.float64)  # v=0, A=0
    X = jnp.asarray([0.0, 10.0, 20.0, 30.0], dtype=jnp.float64)
    eps_rel = 1.0e-9

    g4 = adm.metric4(params, X, eps_rel)
    g_inv = jnp.linalg.inv(g4)
    Gmn, _ = geometry.einstein_tensor(params, X, eps_rel)
    T_mn = (C**4 / (8.0 * jnp.pi * G)) * Gmn

    result = classify_energy_conditions(T_mn, g_inv)
    assert result["wec"] is True
    assert result["nec"] is True
    assert result["dec"] is True
    assert result["sec"] is True


def test_alcubierre_wall_nec_violated():
    """En la pared de Alcubierre, rho < 0, asi que NEC y WEC se violan."""
    params = jnp.asarray([100.0, 5.0, 0.5, 1.0, 0.01], dtype=jnp.float64)
    # Punto en la pared (r_s ~= R)
    X = jnp.asarray([0.0, 100.0, 0.0, 0.0], dtype=jnp.float64)
    eps_rel = 1.0e-9

    g4 = adm.metric4(params, X, eps_rel)
    g_inv = jnp.linalg.inv(g4)
    Gmn, _ = geometry.einstein_tensor(params, X, eps_rel)
    T_mn = (C**4 / (8.0 * jnp.pi * G)) * Gmn

    result = classify_energy_conditions(T_mn, g_inv)
    # En la pared de Alcubierre con v > 0, NEC deberia violarse
    assert result["nec"] is False or result["violation_magnitudes"].get("NEC", 0) > 0


def test_energy_report_aggregation_and_observer_search():
    """Verifica reporte por punto, por tramo, agregado y busqueda de observador."""
    from proxima.simulator.energy import (
        EnergyReport,
        aggregate_point_reports,
        aggregate_segment_reports,
        energy_margins,
        search_observer_violation,
    )

    params = jnp.asarray([100.0, 5.0, 0.5, 1.0, 0.01], dtype=jnp.float64)
    X_wall = jnp.asarray([0.0, 98.0, 10.0, 0.0], dtype=jnp.float64)
    eps_rel = 1.0e-9

    g4 = adm.metric4(params, X_wall, eps_rel)
    g_inv = jnp.linalg.inv(g4)
    Gmn, _ = geometry.einstein_tensor(params, X_wall, eps_rel)
    T_mn = (C**4 / (8.0 * jnp.pi * G)) * Gmn

    margins = energy_margins(T_mn, g_inv)
    assert float(margins["WEC"]) < 0.0

    obs = search_observer_violation(T_mn, g4, n_samples=8)
    assert obs["nec_violated"] is True
    assert obs["nec_min"] < 0.0

    r_ok = EnergyReport(0, True, True, True, True, {"WEC": 0.0})
    r_bad = EnergyReport(1, False, False, False, False, {"WEC": 10.0, "NEC": 5.0})
    agg = aggregate_point_reports([r_ok, r_bad])
    assert agg.wec_ok_all is False
    assert agg.wec_violated_points == 1
    assert agg.max_violation_magnitudes["WEC"] == 10.0

    segs = aggregate_segment_reports([r_ok, r_bad], n_segments=2)
    assert len(segs) == 2
    assert segs[0].wec_ok is True
    assert segs[1].wec_ok is False