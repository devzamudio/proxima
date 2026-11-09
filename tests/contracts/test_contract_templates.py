from __future__ import annotations

import pytest

from proxima.contracts import SimulationRequest, SimulationStatus
from proxima.contracts.testing import (
    assert_energy_report_contract,
    assert_fitness_breakdown_contract,
    assert_metric_evaluation_contract,
    assert_risk_breakdown_contract,
    assert_simulation_result_contract,
)


pytestmark = [pytest.mark.contract, pytest.mark.template]


def test_metric_core_contract_template(metric_core_impl, sample_params, sample_state):
    if metric_core_impl is None:
        pytest.skip("Template: define la fixture metric_core_impl con tu implementacion")

    metric_eval = metric_core_impl.evaluate_metric(sample_params, sample_state)
    energy_report = metric_core_impl.evaluate_energy_conditions(sample_params, sample_state)

    assert_metric_evaluation_contract(metric_eval)
    assert_energy_report_contract(energy_report)


def test_simulator_contract_template(simulator_impl, sample_params, sample_state, conventions):
    if simulator_impl is None:
        pytest.skip("Template: define la fixture simulator_impl con tu implementacion")

    request = SimulationRequest(
        params=sample_params,
        initial_state=sample_state,
        conventions=conventions,
        max_steps=100,
    )
    result = simulator_impl.simulate(request)

    assert_simulation_result_contract(result)
    assert result.status in (SimulationStatus.SUCCESS, SimulationStatus.FAILURE)


def test_risk_provider_contract_template(
    risk_provider_impl,
    sample_segment,
    sample_sky,
    sample_data_context,
):
    if risk_provider_impl is None:
        pytest.skip("Template: define la fixture risk_provider_impl con tu implementacion")

    risk = risk_provider_impl.evaluate_segment_risk(sample_segment, sample_sky, sample_data_context)
    assert_risk_breakdown_contract(risk)


def test_fitness_contract_template(fitness_evaluator_impl, simulator_impl, risk_provider_impl, sample_params, sample_state, conventions, sample_segment, sample_sky, sample_data_context):
    if fitness_evaluator_impl is None:
        pytest.skip("Template: define la fixture fitness_evaluator_impl con tu implementacion")
    if simulator_impl is None:
        pytest.skip("Template: define la fixture simulator_impl con tu implementacion")

    request = SimulationRequest(
        params=sample_params,
        initial_state=sample_state,
        conventions=conventions,
        max_steps=100,
    )
    sim_result = simulator_impl.simulate(request)

    risk = None
    if risk_provider_impl is not None:
        risk = risk_provider_impl.evaluate_segment_risk(sample_segment, sample_sky, sample_data_context)

    fitness = fitness_evaluator_impl.evaluate_fitness(sim_result, risk)
    assert_fitness_breakdown_contract(fitness)
