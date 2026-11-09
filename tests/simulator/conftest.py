from __future__ import annotations

import jax.numpy as jnp
import pytest

import proxima.core  # noqa: F401  (activa x64)
from proxima.contracts import MetricStatePoint, ScientificConventions, WarpParameterVector, Frame, MetricSignature
from proxima.core import BobrickMartireMetricCore, load_core_config
from proxima.simulator import RelativisticSimulator
from proxima.simulator.config import SimulatorConfig, default_config

C = 299792458.0
G = 6.67430e-11


@pytest.fixture(scope="session")
def config():
    return load_core_config()


@pytest.fixture(scope="session")
def sim_config():
    return SimulatorConfig.default_config()


@pytest.fixture
def sim():
    return RelativisticSimulator(config=default_config())


@pytest.fixture
def conventions():
    return ScientificConventions(
        phase0_version="1.0.0",
        metric_signature=MetricSignature.MINUS_PLUS_PLUS_PLUS,
        global_frame=Frame.ICRS,
        local_frame=Frame.LOCAL_CARTESIAN_3P1,
        c_m_per_s=C,
        g_m3_per_kg_s2=G,
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.fixture
def canonical_params():
    return WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=1.0,
        lapse_floor=0.01,
    )


@pytest.fixture
def initial_state():
    return MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)