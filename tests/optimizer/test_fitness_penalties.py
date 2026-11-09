"""Pruebas de la evaluacion de fitness relativista, restricciones fisicas y penalizaciones."""

from __future__ import annotations

import math
import jax.numpy as jnp
import pytest

from proxima.contracts.types import (
    EnergyConditionReport,
    FitnessBreakdown,
    RiskBreakdown,
    DataQualityMetadata,
    SimulationObservables,
    SimulationResult,
    SimulationStatus,
)
from proxima.contracts.testing import assert_fitness_breakdown_contract
from proxima.optimizer import RelativisticFitnessEvaluator, PSOConfig, jax_fitness_breakdown


def test_fitness_contract_compliance():
    evaluator = RelativisticFitnessEvaluator()

    obs = SimulationObservables(
        delta_tau_s=0.5,
        delta_t_s=1.0,
        total_energy_j=1.0e15,
        tidal_stress_max_si=1.2,
    )
    ec = EnergyConditionReport(
        wec_ok=True,
        nec_ok=True,
        dec_ok=True,
        sec_ok=True,
    )
    res = SimulationResult(
        status=SimulationStatus.SUCCESS,
        observables=obs,
        energy_conditions=ec,
        failure_reason=None,
    )
    risk = RiskBreakdown(
        radiative_flux_score=0.1,
        extinction_h_column_score=0.2,
        mass_density_score=0.05,
        tidal_stress_score=0.0,
        total_risk_score=0.1,
        quality=DataQualityMetadata(source_name="test", source_version="1", confidence_score=1.0),
    )

    fb = evaluator.evaluate_fitness(res, risk)
    assert_fitness_breakdown_contract(fb)
    assert fb.mission_term > 0.0
    assert fb.physical_penalty_term == 0.0
    assert fb.operational_penalty_term == 0.0
    assert fb.risk_penalty_term > 0.0


def test_fitness_penalizes_energy_condition_violations():
    evaluator = RelativisticFitnessEvaluator()
    obs = SimulationObservables(delta_tau_s=1.0, delta_t_s=1.0, total_energy_j=0.0, tidal_stress_max_si=0.0)
    
    # Caso sin violacion
    ec_clean = EnergyConditionReport(wec_ok=True, nec_ok=True, dec_ok=True, sec_ok=True)
    res_clean = SimulationResult(status=SimulationStatus.SUCCESS, observables=obs, energy_conditions=ec_clean, failure_reason=None)
    fb_clean = evaluator.evaluate_fitness(res_clean, None)

    # Caso con violacion WEC y NEC
    ec_viol = EnergyConditionReport(
        wec_ok=False,
        nec_ok=False,
        dec_ok=True,
        sec_ok=True,
        violation_magnitudes={"WEC": 2.5, "NEC": 1.0},
    )
    res_viol = SimulationResult(status=SimulationStatus.SUCCESS, observables=obs, energy_conditions=ec_viol, failure_reason=None)
    fb_viol = evaluator.evaluate_fitness(res_viol, None)

    assert fb_viol.physical_penalty_term > 0.0
    assert fb_viol.total_fitness > fb_clean.total_fitness


def test_fitness_penalizes_simulation_failure():
    evaluator = RelativisticFitnessEvaluator()
    res_fail = SimulationResult(
        status=SimulationStatus.FAILURE,
        observables=None,
        energy_conditions=None,
        failure_reason="Numerical divergence",
    )
    fb = evaluator.evaluate_fitness(res_fail, None)
    assert_fitness_breakdown_contract(fb)
    assert fb.operational_penalty_term >= evaluator.config.penalty_simulation_failure
    assert fb.total_fitness > 1.0e6


def test_jax_fitness_breakdown_matches_logic():
    delta_t = jnp.asarray([1.0, 2.0])
    delta_tau = jnp.asarray([0.8, 1.5])
    total_energy = jnp.asarray([0.0, 1.0e10])
    tidal_max = jnp.asarray([2.0, 15.0])  # Segundo caso viola 9.81
    wec_margin = jnp.asarray([0.1, -0.5]) # Segundo caso viola WEC
    nec_margin = jnp.asarray([0.1, 0.2])
    sec_margin = jnp.asarray([0.1, 0.2])
    dec_margin = jnp.asarray([0.1, 0.2])
    success = jnp.asarray([True, True])
    risk_score = jnp.asarray([0.0, 0.5])

    total, mission, physical, risk, oper = jax_fitness_breakdown(
        delta_t, delta_tau, total_energy, tidal_max,
        wec_margin, nec_margin, sec_margin, dec_margin,
        success, risk_score,
        mission_time_s=1.0, mission_tau_weight=0.5,
        energy_scale_j=1e20, energy_weight=0.1,
        penalty_wec=1e4, penalty_nec=1e4, penalty_sec=1e3, penalty_dec=1e3,
        penalty_tidal=1e5, penalty_risk=1e3, penalty_sim_failure=1e6
    )

    assert total.shape == (2,)
    # Partícula 0 limpia de penalización física
    assert physical[0] == 0.0
    # Partícula 1 con penalización física por WEC y marea
    assert physical[1] > 0.0
    # Consistencia contable
    assert jnp.allclose(total, mission + physical + risk + oper)
