from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from proxima.contracts import DomainError
from proxima.core import evaluate_batch, params_to_array
from proxima.core.batch import evaluate_point
from proxima.core.kernels import adm
from proxima.core.params import domain_mask

from .conftest import C, G, alcubierre_rho, make_params, state


# ---------------------------------------------------------------- dtype / params
def test_float64_everywhere(core, canonical):
    out = core._evaluate(canonical, state(x=100.0, y=10.0))
    for k, v in out.items():
        if hasattr(v, "dtype") and v.dtype != jnp.bool_:
            assert v.dtype == jnp.float64, k


@pytest.mark.parametrize(
    "kw",
    [
        {"bubble_radius_m": -1.0},
        {"shell_thickness_m": 0.0},
        {"shift_amplitude": 1.5},
        {"lapse_floor": 0.0},
        {"velocity_fraction_c": 2.0},
        {"velocity_fraction_c": float("nan")},
    ],
)
def test_invalid_domain_raises(core, kw):
    with pytest.raises(DomainError):
        core.evaluate_metric(make_params(**kw), state())


def test_superluminal_requires_flag(core):
    p = make_params(velocity_fraction_c=2.0, metadata={"allow_superluminal": True})
    assert core.evaluate_metric(p, state()).valid_domain


def test_domain_mask_vmap(config):
    bounds = config.bounds_array()
    P = jnp.stack([params_to_array(make_params()), params_to_array(make_params(shift_amplitude=3.0))])
    assert list(np.asarray(jax.vmap(domain_mask, in_axes=(0, None))(P, bounds))) == [True, False]


# ---------------------------------------------------------------- Minkowski
@pytest.mark.parametrize("kw", [{"shift_amplitude": 0.0}, {"velocity_fraction_c": 0.0}])
def test_minkowski_limit(core, kw):
    out = core._evaluate(make_params(**kw), state(x=97.0, y=8.0, z=-3.0))
    np.testing.assert_allclose(out["metric4"], np.diag([-1.0, 1, 1, 1]), atol=1e-14)
    np.testing.assert_allclose(out["einstein_tensor_per_m2"], 0.0, atol=1e-14)
    assert float(out["rho_eulerian_J_m3"]) == 0.0
    assert float(out["trace_K_per_m"]) == 0.0


# ---------------------------------------------------------------- referencias
SAMPLE_POINTS = [(98.0, 10.0, 0.0), (60.0, 75.0, 20.0), (0.0, 0.0, 101.0), (-90.0, 30.0, -20.0)]


@pytest.mark.parametrize("xyz", SAMPLE_POINTS)
def test_rho_matches_alcubierre_analytic(core, canonical, xyz):
    rho = core.evaluate_metric(canonical, state(*xyz)).diagnostics["rho_eulerian_J_m3"]
    ref = float(alcubierre_rho(0.5, 100.0, 5.0, *xyz))
    np.testing.assert_allclose(rho, ref, rtol=1e-8, atol=1e-10 * abs(C**4 / (8 * np.pi * G * 100.0**2)))


def test_rho_matches_sympy_reference(config, canonical):
    """Referencia independiente: rho = (3R + K^2 - KijKij)/(16 pi G) c^4 (vinculo hamiltoniano)."""
    x0, x, y, z = sp.symbols("x0 x y z", real=True)
    v, R, s = sp.Rational(1, 2), 100, sp.Rational(1, 5)
    r = sp.sqrt((x - v * x0) ** 2 + y**2 + z**2)
    f = (sp.tanh(s * (r + R)) - sp.tanh(s * (r - R))) / (2 * sp.tanh(s * R))
    beta = [-v * f, 0, 0]
    X3 = [x, y, z]
    K = sp.Matrix(3, 3, lambda i, j: (sp.diff(beta[j], X3[i]) + sp.diff(beta[i], X3[j])) / 2)
    rho_expr = (K.trace() ** 2 - sum(K[i, j] ** 2 for i in range(3) for j in range(3))) * C**4 / (16 * sp.pi * G)
    pt = {x0: 0, x: 60, y: 75, z: 20}
    ref = float(rho_expr.subs(pt).evalf(30))
    out = evaluate_point(
        params_to_array(canonical), jnp.asarray([0.0, 60.0, 75.0, 20.0]), config.bounds_array(),
        config.radius_regularization_rel, C, G,
    )
    np.testing.assert_allclose(float(out["rho_eulerian_J_m3"]), ref, rtol=1e-8)


# ---------------------------------------------------------------- signos / unidades
def test_signature_and_units(core, canonical):
    far = core._evaluate(canonical, state(x=1e4))
    assert float(far["metric4"][0, 0]) < 0
    assert float(jnp.linalg.det(far["metric4"])) < 0
    ev = core.evaluate_metric(canonical, state())  # centro: f=1, beta^x = -v c (m/s)
    np.testing.assert_allclose(ev.shift_vector[0], -0.5 * C, rtol=1e-12)
    assert ev.diagnostics["traceability"]["unidades_activas"]["shift_vector"] == "m/s"


def test_wall_has_negative_energy_wec_violated(core, canonical):
    rep = core.evaluate_energy_conditions(canonical, state(98.0, 10.0, 0.0))
    assert rep.violation_magnitudes["rho_eulerian_J_m3"] < 0
    assert rep.wec_ok is False
    inside = core.evaluate_energy_conditions(canonical, state(0.0, 0.0, 0.0))
    assert inside.wec_ok is True


# ---------------------------------------------------------------- estabilidad
@pytest.mark.parametrize("R", [1e-2, 1.0, 1e3, 1e6])
@pytest.mark.parametrize("w_rel", [1e-3, 0.05, 1.0])
def test_stability_sweep(core, R, w_rel):
    p = make_params(bubble_radius_m=R, shell_thickness_m=max(R * w_rel, 1e-4))
    for xyz in [(0, 0, 0), (R, 0, 0), (R * 0.7, R * 0.7, 0), (10 * R, 0, 0)]:
        out = core._evaluate(p, state(*xyz))
        assert bool(out["finite_ok"]) and bool(out["valid_domain"])


def test_lapse_floor_respected(config):
    p = params_to_array(make_params(lapse_floor=1.0))
    assert float(adm.lapse(p, jnp.zeros(4), 1e-9)) >= 1.0


def test_center_is_regular(core, canonical):
    """r_s -> 0 no produce NaN gracias a la regularizacion."""
    out = core._evaluate(canonical, state(0.0, 0.0, 0.0))
    assert bool(out["finite_ok"])


# ---------------------------------------------------------------- sensibilidad
def test_rho_scales_with_v_squared(core):
    pt = state(98.0, 10.0, 0.0)
    r1 = core.evaluate_metric(make_params(velocity_fraction_c=0.1), pt).diagnostics["rho_eulerian_J_m3"]
    r2 = core.evaluate_metric(make_params(velocity_fraction_c=0.2), pt).diagnostics["rho_eulerian_J_m3"]
    np.testing.assert_allclose(r2 / r1, 4.0, rtol=1e-9)


def test_gradients_wrt_params_finite(config, canonical):
    def rho(p):
        return evaluate_point(p, jnp.asarray([0.0, 98.0, 10.0, 0.0]), config.bounds_array(),
                              config.radius_regularization_rel, C, G)["rho_eulerian_J_m3"]

    g = jax.grad(rho)(params_to_array(canonical))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(g[2]) < 0  # mas velocidad => rho mas negativo


# ---------------------------------------------------------------- jit / vmap
def test_batch_matches_pointwise_and_is_deterministic(config):
    P = jnp.stack([params_to_array(make_params(velocity_fraction_c=v)) for v in (0.1, 0.5, 0.9)])
    Xs = jnp.asarray([[0.0, x, 10.0, 0.0] for x in (0.0, 50.0, 98.0, 200.0)])
    args = (config.bounds_array(), config.radius_regularization_rel, C, G)
    out1 = evaluate_batch(P, Xs, *args)
    out2 = evaluate_batch(P, Xs, *args)
    assert out1["rho_eulerian_J_m3"].shape == (3, 4)
    np.testing.assert_array_equal(out1["rho_eulerian_J_m3"], out2["rho_eulerian_J_m3"])
    single = evaluate_point(P[2], Xs[2], *args)
    np.testing.assert_allclose(out1["rho_eulerian_J_m3"][2, 2], single["rho_eulerian_J_m3"], rtol=1e-12)
