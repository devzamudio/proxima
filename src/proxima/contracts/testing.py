from __future__ import annotations

import math

from .types import (
    EnergyConditionReport,
    FitnessBreakdown,
    MetricEvaluation,
    RiskBreakdown,
    SimulationResult,
    SimulationStatus,
)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def assert_metric_evaluation_contract(result: MetricEvaluation) -> None:
    if not isinstance(result.lapse, (float, int)):
        raise AssertionError("lapse debe ser numerico")
    if not _is_finite(float(result.lapse)):
        raise AssertionError("lapse debe ser finito")

    if len(result.shift_vector) != 3:
        raise AssertionError("shift_vector debe tener 3 componentes")
    for component in result.shift_vector:
        if not _is_finite(float(component)):
            raise AssertionError("shift_vector contiene valores no finitos")

    if len(result.spatial_metric) != 3:
        raise AssertionError("spatial_metric debe ser matriz 3x3")
    for row in result.spatial_metric:
        if len(row) != 3:
            raise AssertionError("spatial_metric debe ser matriz 3x3")
        for element in row:
            if not _is_finite(float(element)):
                raise AssertionError("spatial_metric contiene valores no finitos")



def assert_energy_report_contract(report: EnergyConditionReport) -> None:
    for flag in (report.wec_ok, report.nec_ok, report.dec_ok, report.sec_ok):
        if not isinstance(flag, bool):
            raise AssertionError("flags WEC/NEC/DEC/SEC deben ser bool")
    for key, value in report.violation_magnitudes.items():
        if not isinstance(key, str):
            raise AssertionError("claves de violation_magnitudes deben ser string")
        if not _is_finite(float(value)):
            raise AssertionError("violation_magnitudes contiene valores no finitos")



def assert_simulation_result_contract(result: SimulationResult) -> None:
    if result.status not in (SimulationStatus.SUCCESS, SimulationStatus.FAILURE):
        raise AssertionError("status invalido")

    if result.status is SimulationStatus.SUCCESS:
        if result.observables is None:
            raise AssertionError("success requiere observables")
        if result.energy_conditions is None:
            raise AssertionError("success requiere energy_conditions")
        if result.failure_reason is not None:
            raise AssertionError("success no debe tener failure_reason")

        observables = result.observables
        for value in (
            observables.delta_tau_s,
            observables.delta_t_s,
            observables.total_energy_j,
            observables.tidal_stress_max_si,
        ):
            if not _is_finite(float(value)):
                raise AssertionError("observables contiene valores no finitos")

        assert_energy_report_contract(result.energy_conditions)

    if result.status is SimulationStatus.FAILURE:
        if not result.failure_reason:
            raise AssertionError("failure requiere failure_reason")



def assert_risk_breakdown_contract(risk: RiskBreakdown) -> None:
    for name, score in (
        ("radiative_flux_score", risk.radiative_flux_score),
        ("extinction_h_column_score", risk.extinction_h_column_score),
        ("mass_density_score", risk.mass_density_score),
        ("tidal_stress_score", risk.tidal_stress_score),
        ("total_risk_score", risk.total_risk_score),
    ):
        if not _is_finite(float(score)):
            raise AssertionError(f"{name} no es finito")
        if score < 0.0 or score > 1.0:
            raise AssertionError(f"{name} debe estar en [0, 1]")

    quality = risk.quality
    if not quality.source_name:
        raise AssertionError("quality.source_name es obligatorio")
    if not quality.source_version:
        raise AssertionError("quality.source_version es obligatorio")
    if quality.confidence_score < 0.0 or quality.confidence_score > 1.0:
        raise AssertionError("quality.confidence_score debe estar en [0, 1]")



def assert_fitness_breakdown_contract(fitness: FitnessBreakdown, *, atol: float = 1e-12) -> None:
    for value in (
        fitness.mission_term,
        fitness.physical_penalty_term,
        fitness.risk_penalty_term,
        fitness.operational_penalty_term,
        fitness.total_fitness,
    ):
        if not _is_finite(float(value)):
            raise AssertionError("fitness contiene valores no finitos")

    expected_total = (
        fitness.mission_term
        + fitness.physical_penalty_term
        + fitness.risk_penalty_term
        + fitness.operational_penalty_term
    )
    if abs(expected_total - fitness.total_fitness) > atol:
        raise AssertionError("total_fitness debe ser suma de terminos")
