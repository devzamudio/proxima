from __future__ import annotations

import jax.numpy as jnp
import pytest

import proxima.core  # noqa: F401  (activa x64)
from proxima.contracts import MetricStatePoint, WarpParameterVector
from proxima.core import BobrickMartireMetricCore, load_core_config

C = 299792458.0
G = 6.67430e-11


@pytest.fixture(scope="session")
def config():
    return load_core_config()


@pytest.fixture(scope="session")
def core(config):
    return BobrickMartireMetricCore(config=config)


@pytest.fixture
def canonical() -> WarpParameterVector:
    return WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )


def make_params(**kw) -> WarpParameterVector:
    base = dict(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )
    base.update(kw)
    return WarpParameterVector(**base)


def state(x=0.0, y=0.0, z=0.0, t=0.0) -> MetricStatePoint:
    return MetricStatePoint(t_s=t, x_m=x, y_m=y, z_m=z, tau_s=0.0)


def alcubierre_rho(v, R, w, x, y, z):
    """Densidad Euleriana analitica de Alcubierre (t=0)."""
    sigma = 1.0 / w
    r = jnp.sqrt(x * x + y * y + z * z)
    sech2 = lambda u: 1.0 / jnp.cosh(u) ** 2  # noqa: E731
    df = sigma * (sech2(sigma * (r + R)) - sech2(sigma * (r - R))) / (2.0 * jnp.tanh(sigma * R))
    return -(C**4 / (32.0 * jnp.pi * G)) * v**2 * df**2 * (y * y + z * z) / (r * r)
