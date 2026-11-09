from __future__ import annotations

import pytest

from proxima.contracts import (
    DataQueryContext,
    Frame,
    MetricSignature,
    MetricStatePoint,
    RouteSegment,
    ScientificConventions,
    SkyCoordinates,
    WarpParameterVector,
)
from proxima.core import BobrickMartireMetricCore
from proxima.stubs import (
    StubFitnessEvaluator,
    StubRelativisticSimulator,
    StubRiskDataProvider,
)


@pytest.fixture
def conventions() -> ScientificConventions:
    return ScientificConventions(
        phase0_version="1.0.0",
        metric_signature=MetricSignature.MINUS_PLUS_PLUS_PLUS,
        global_frame=Frame.ICRS,
        local_frame=Frame.LOCAL_CARTESIAN_3P1,
        c_m_per_s=299792458.0,
        g_m3_per_kg_s2=6.67430e-11,
        atol=1e-10,
        rtol=1e-8,
    )


@pytest.fixture
def sample_params() -> WarpParameterVector:
    return WarpParameterVector(
        bubble_radius_m=100.0,
        shell_thickness_m=5.0,
        velocity_fraction_c=0.5,
        shift_amplitude=0.1,
        lapse_floor=0.01,
    )


@pytest.fixture
def sample_state() -> MetricStatePoint:
    return MetricStatePoint(t_s=0.0, x_m=0.0, y_m=0.0, z_m=0.0, tau_s=0.0)


@pytest.fixture
def sample_segment() -> RouteSegment:
    return RouteSegment(
        start_xyz_m=(0.0, 0.0, 0.0),
        end_xyz_m=(1000.0, 0.0, 0.0),
        frame=Frame.LOCAL_CARTESIAN_3P1,
    )


@pytest.fixture
def sample_sky() -> SkyCoordinates:
    return SkyCoordinates(ra_deg=217.4292, dec_deg=-62.6795, distance_pc=1.301, frame=Frame.ICRS)


@pytest.fixture
def sample_data_context() -> DataQueryContext:
    return DataQueryContext(
        dataset_version="snapshot-2026-11-08",
        extraction_timestamp_utc="2026-11-08T00:00:00Z",
        cache_key="example-key",
    )


# Stubs mínimos — reemplaza cada fixture con la implementación real cuando esté lista.


@pytest.fixture
def metric_core_impl(conventions) -> BobrickMartireMetricCore:
    # Fase 1: implementacion real del nucleo metrico.
    return BobrickMartireMetricCore(conventions=conventions)


@pytest.fixture
def simulator_impl():
    # Fase 2: implementacion real del simulador relativista.
    from proxima.simulator import RelativisticSimulator
    from proxima.simulator.config import default_config

    return RelativisticSimulator(config=default_config())


@pytest.fixture
def risk_provider_impl():
    # Fase 3: proveedor real basado en catalogos astronomicos locales (sin red).
    from proxima.data.pipeline import default_local_risk_provider

    return default_local_risk_provider()


@pytest.fixture
def fitness_evaluator_impl():
    # Fase 4: evaluador real de fitness descompuesto y penalizaciones fisicas/riesgo
    from proxima.optimizer.fitness import RelativisticFitnessEvaluator

    return RelativisticFitnessEvaluator()
