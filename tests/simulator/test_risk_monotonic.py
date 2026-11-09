"""Test de monotonicidad del riesgo: mas longitud o mas flujo => mas riesgo."""

from __future__ import annotations

import pytest

from proxima.contracts import (
    DataQueryContext, DataQualityMetadata, RiskBreakdown,
    RouteSegment, SkyCoordinates,
)
from proxima.simulator.risk import RiskAssessor


def test_risk_monotonic_with_stub():
    """Con el stub, el riesgo es cero independientemente del segmento.
    Este test verifica que la estructura de riesgo funcione."""
    assessor = RiskAssessor()

    import jax.numpy as jnp

    # Trayectoria simple: 2 puntos en el origen
    traj = jnp.zeros((2, 8), dtype=jnp.float64)
    traj = traj.at[0, 0].set(0.0)   # t=0
    traj = traj.at[1, 0].set(1.0)   # t=1 (en unidades de lambda/c ~ s)

    mask = jnp.asarray([True, True])

    risk, segments = assessor.assess_trajectory(traj, mask, n_segments=1)

    # Con el stub, riesgo total = 0
    assert risk.total_risk_score == 0.0
    assert risk.radiative_flux_score == 0.0
    assert len(segments) == 1


def test_risk_monotonic_length_and_environment():
    """Con AnalyticRiskProvider, mayor longitud de tramo o mayor intensidad
    ambiental produce estrictamente mayor puntaje de riesgo en [0, 1]."""
    import jax.numpy as jnp
    from proxima.contracts.testing import assert_risk_breakdown_contract
    from proxima.simulator.risk import AnalyticRiskProvider

    traj_short = jnp.zeros((2, 8), dtype=jnp.float64)
    traj_short = traj_short.at[1, 1].set(1.0e7)  # 10,000 km en x

    traj_long = jnp.zeros((2, 8), dtype=jnp.float64)
    traj_long = traj_long.at[1, 1].set(1.0e8)  # 100,000 km en x

    mask = jnp.asarray([True, True])

    provider_low = AnalyticRiskProvider(
        radiative_flux_w_m2=1.0,
        h_column_density_m2=1.0e20,
        mass_density_kg_m3=1.0e-21,
        cluster_tidal_s2=1.0e-12,
    )
    provider_high = AnalyticRiskProvider(
        radiative_flux_w_m2=5.0,
        h_column_density_m2=5.0e20,
        mass_density_kg_m3=5.0e-21,
        cluster_tidal_s2=5.0e-12,
    )

    risk_short, _ = RiskAssessor(provider=provider_low).assess_trajectory(traj_short, mask, n_segments=2)
    risk_long, _ = RiskAssessor(provider=provider_low).assess_trajectory(traj_long, mask, n_segments=2)
    risk_high_env, _ = RiskAssessor(provider=provider_high).assess_trajectory(traj_long, mask, n_segments=2)

    assert_risk_breakdown_contract(risk_short)
    assert_risk_breakdown_contract(risk_long)
    assert_risk_breakdown_contract(risk_high_env)

    # Monotonicidad por longitud de ruta
    assert risk_long.radiative_flux_score > risk_short.radiative_flux_score
    assert risk_long.extinction_h_column_score > risk_short.extinction_h_column_score
    assert risk_long.mass_density_score > risk_short.mass_density_score
    assert risk_long.tidal_stress_score > risk_short.tidal_stress_score
    assert risk_long.total_risk_score > risk_short.total_risk_score

    # Monotonicidad por severidad del entorno
    assert risk_high_env.total_risk_score > risk_long.total_risk_score