"""Pruebas del motor de calibracion: barrido de hiperparametros, sensibilidad y robustez."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from proxima.validation import CalibrationEngine


def test_sweep_pso_hyperparameters():
    engine = CalibrationEngine()

    grid = {
        "w_max": [0.8, 0.9],
        "c1": [1.2, 1.5],
        "c2": [1.5],
    }

    # Funcion objetivo convexa cuadrática
    res = engine.sweep_pso_hyperparameters(
        lambda P: jnp.sum((P - 0.5)**2, axis=1),
        grid,
        n_seeds=2,
        base_seed=10,
    )

    assert len(res.records) == 4  # 2 x 2 x 1
    assert res.best_fitness >= 0.0
    assert len(res.ranking) == 4
    # El ranking debe estar ordenado de menor a mayor fitness
    for i in range(len(res.ranking) - 1):
        assert res.ranking[i]["mean_fitness"] <= res.ranking[i + 1]["mean_fitness"]


def test_analyze_warp_sensitivity():
    engine = CalibrationEngine()

    # Vector nominal canónico
    nominal = (100.0, 5.0, 0.5, 1.0, 0.01)

    # Funcion de prueba con dependencia fuerte en el radio y la velocidad
    def dummy_eval(vec: tuple[float, ...]) -> float:
        r, w, v, a, lf = vec
        return 10.0 * (r / 100.0)**2 + 50.0 * (v / 0.5) + (a - 1.0)**2

    sens = engine.analyze_warp_sensitivity(dummy_eval, nominal, delta_rel=0.05)

    assert sens.baseline_params == nominal
    assert sens.baseline_fitness > 0.0
    assert "bubble_radius_m" in sens.sensitivities
    assert "velocity_fraction_c" in sens.sensitivities
    # Ambas sensibilidades deben ser no nulas
    assert sens.sensitivities["bubble_radius_m"] > 0.0
    assert sens.sensitivities["velocity_fraction_c"] > 0.0
    # Cada parametro tiene sus registros de perturbacion (-delta, 0, +delta)
    assert len(sens.perturbations["bubble_radius_m"]) == 3


def test_evaluate_data_robustness():
    engine = CalibrationEngine()

    # Simulador de riesgo que responde con pequeña varianza ante ruido gaussiano
    def eval_with_noise(noise: float) -> float:
        base_risk = 0.25
        return float(base_risk + 0.02 * noise)

    res = engine.evaluate_data_robustness(
        eval_with_noise,
        nominal_noise=0.0,
        perturbation_sigma=0.1,
        n_mc_samples=25,
        seed=123,
    )

    assert res.nominal_fitness == pytest.approx(0.25, rel=1e-6)
    assert res.perturbed_fitness_mean == pytest.approx(0.25, abs=0.01)
    assert res.perturbed_fitness_std >= 0.0
    assert 0.0 <= res.stability_score <= 1.0
    # Al ser un ruido pequeño, la estabilidad debe ser alta (> 0.9)
    assert res.stability_score > 0.9
