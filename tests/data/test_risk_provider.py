"""Proveedor de riesgo astronomico: contrato, monotonicidad y fisica basica."""

from __future__ import annotations

import math

import numpy as np
import pytest

from proxima.contracts.exceptions import DataSchemaMismatch
from proxima.contracts.testing import assert_risk_breakdown_contract
from proxima.contracts.types import DataQueryContext, Frame, RouteSegment, SkyCoordinates
from proxima.data.connectors import ConstantExtinctionProvider
from proxima.data.normalizer import PARSEC_TO_M, build_normalized_record
from proxima.data.pipeline import default_local_risk_provider
from proxima.data.risk_provider import G_SI, AstronomicalRiskProvider
from proxima.simulator.risk import RiskAssessor

CTX = DataQueryContext("snap-test", "2026-11-09T00:00:00Z", "k")
SUN_SKY = SkyCoordinates(0.0, 0.0, None, Frame.ICRS)


def _seg(length_m, direction=(1.0, 0.0, 0.0), start=(0.0, 0.0, 0.0), frame=Frame.ICRS):
    d = np.asarray(direction) / np.linalg.norm(direction)
    end = tuple(float(s + length_m * di) for s, di in zip(start, d))
    return RouteSegment(tuple(map(float, start)), end, frame)


def _cloud(data_config, d_pc=10.0, r_pc=2.0, otype="HII_region"):
    rec = build_normalized_record(canonical_id="cloud", source_name="local_csv", source_version="t",
                                  object_type=otype, ra_deg=0.0, dec_deg=0.0, distance_pc=d_pc,
                                  radius_pc=r_pc, mass_msun=1000.0, luminosity_lsun=1e4)
    return AstronomicalRiskProvider([rec], data_config)


def test_local_provider_satisfies_contract(data_config):
    prov = default_local_risk_provider(data_config)
    sky = SkyCoordinates(217.4292, -62.6795, 1.301, Frame.ICRS)
    risk = prov.evaluate_segment_risk(_seg(1e15, frame=Frame.LOCAL_CARTESIAN_3P1), sky, CTX)
    assert_risk_breakdown_contract(risk)
    assert risk.quality.source_version == "snap-test"
    assert "fluence_j_m2=" in risk.quality.notes


def test_risk_monotonic_in_segment_length(data_config):
    prov = _cloud(data_config)
    prev = None
    for L_pc in [0.0, 1.0, 5.0, 8.0, 9.5, 10.0, 12.0, 20.0]:
        r = prov.evaluate_segment_risk(_seg(L_pc * PARSEC_TO_M), SUN_SKY, CTX)
        cur = (r.radiative_flux_score, r.extinction_h_column_score, r.mass_density_score,
               r.tidal_stress_score, r.total_risk_score)
        if prev is not None:
            assert all(c >= p - 1e-15 for c, p in zip(cur, prev)), (L_pc, prev, cur)
        prev = cur


def test_crossing_cloud_increases_column_by_chord(data_config):
    prov = _cloud(data_config, d_pc=10.0, r_pc=2.0)
    q_full = prov.physical_quantities(_seg(20 * PARSEC_TO_M), SUN_SKY)
    q_miss = prov.physical_quantities(_seg(20 * PARSEC_TO_M, direction=(0, 1, 0)), SUN_SKY)
    n_gas = prov._n_gas[0]
    expected_extra = n_gas * 4.0 * PARSEC_TO_M  # cuerda = diametro
    assert q_full["h_column_m2"] - q_miss["h_column_m2"] == pytest.approx(expected_extra, rel=1e-9)
    assert q_full["mass_density_kg_m3"] > q_miss["mass_density_kg_m3"]


def test_tidal_matches_point_mass_formula(data_config):
    prov = _cloud(data_config, d_pc=10.0, r_pc=0.01, otype="open_cluster")
    q = prov.physical_quantities(_seg(1.0, direction=(0, 0, 1)), SUN_SKY)  # ~en el Sol
    m = prov._mass[0]
    expected = 2 * G_SI * m / (10.0 * PARSEC_TO_M) ** 3
    assert q["tidal_s2"] == pytest.approx(expected, rel=1e-6)


def test_fluence_matches_numeric_integration(data_config):
    prov = _cloud(data_config, d_pc=10.0, r_pc=1.0)
    seg = _seg(30 * PARSEC_TO_M, direction=(1, 0.2, 0))
    q = prov.physical_quantities(seg, SUN_SKY)
    s = np.linspace(0, 30 * PARSEC_TO_M, 200001)
    u = np.array([1, 0.2, 0]) / np.linalg.norm([1, 0.2, 0])
    pts = s[:, None] * u
    r2 = np.sum((pts - prov._centers[0]) ** 2, axis=1) + prov._radius[0] ** 2
    numeric = prov._lum[0] / (4 * math.pi * data_config.transit_speed_m_s) * np.trapezoid(1 / r2, s)
    assert q["fluence_j_m2"] == pytest.approx(numeric, rel=1e-6)


def test_line_of_sight_extinction_added(data_config):
    rec = build_normalized_record(canonical_id="s", source_name="gaia_dr3", source_version="t",
                                  object_type="star", ra_deg=0, dec_deg=0, distance_pc=1000)
    sky = SkyCoordinates(10.0, 10.0, 500.0, Frame.ICRS)
    base = AstronomicalRiskProvider([rec], data_config)
    dusty = AstronomicalRiskProvider([rec], data_config, extinction_provider=ConstantExtinctionProvider(1.0))
    seg = _seg(1e12, frame=Frame.LOCAL_CARTESIAN_3P1)
    assert (dusty.evaluate_segment_risk(seg, sky, CTX).extinction_h_column_score
            > base.evaluate_segment_risk(seg, sky, CTX).extinction_h_column_score)


def test_invalid_segment_rejected(data_config):
    prov = _cloud(data_config)
    with pytest.raises(DataSchemaMismatch):
        prov.evaluate_segment_risk(RouteSegment((0, 0, float("nan")), (1, 1, 1), Frame.ICRS), SUN_SKY, CTX)


def test_integrates_with_simulator_risk_assessor(data_config):
    import jax.numpy as jnp

    prov = default_local_risk_provider(data_config)
    sky = SkyCoordinates(83.8, 9.9, 400.0, Frame.ICRS)  # dentro de Lambda Orionis
    traj = jnp.zeros((11, 8)).at[:, 1].set(jnp.linspace(0.0, 1e16, 11))
    assessor = RiskAssessor(provider=prov, sky=sky, context=CTX)
    agg, segs = assessor.assess_trajectory(traj, jnp.ones(11, dtype=bool), n_segments=5)
    assert len(segs) == 5
    assert_risk_breakdown_contract(agg)
    assert agg.mass_density_score > 0.0
